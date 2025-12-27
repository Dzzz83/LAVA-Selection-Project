import argparse
import time
import os
import logging
import numpy as np
import lava

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from dataloader import create_dataloader 
from _imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from _augmentation import get_Trival_Augmentation, get_week_augmentation

# Modified Resnet 18
class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = models.resnet18(weights=None)
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x, return_features=False):
        x = self.net.conv1(x) 
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x) 

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.net.fc(features)

        if return_features:
            return features, logits
        return logits

def validate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets, *rest in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total

def train_one_epoch(model, loader, optimizer, device, criterion, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets, *rest in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type=="cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total

def setup_logger(args, exp_name):
    os.makedirs("results", exist_ok=True)
    log_filename = f"results/{exp_name}.log"
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_filename

def run_lava_valuation(model, train_ds, val_ds, logger):
    """Computes LAVA dual solutions as sample quality scores."""
    # Ensure model is in eval mode for valuation
    model.eval() 
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    
    logger.info(">>> [LAVA] Computing Transport Costs...")
    
    # Wrap in no_grad to prevent the OTDD library from breaking the grad chain
    with torch.no_grad():
        dual_sol, trained_with_flag = lava.compute_dual(
            model, train_loader, val_loader, 
            len(train_ds), [], resize=32
        )
    
    # Extract and flatten scores
    if isinstance(dual_sol, (list, tuple)):
        scores = dual_sol[0].detach().cpu().numpy().flatten()
    else:
        scores = dual_sol.detach().cpu().numpy().flatten()
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--num-workers", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)
    args = parser.parse_args()

    datasets = ["cifar10","cifar100","imbalance_cifar10","imbalance_cifar100"] 
    augmentations = ["trivial", "weak"] 
    select_ratios = [None, 0.7, 0.8, 0.9]

    if args.dataset: datasets = [args.dataset]
    if args.augmentation: augmentations = [args.augmentation]
    if args.ratio is not None: select_ratios = [args.ratio]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exp_id = 0

    for dataset_name in datasets:
        for augmentation in augmentations:
            for select_ratio in select_ratios:
                exp_id += 1
                ratio_str = f"r{select_ratio}" if select_ratio is not None else "Baseline"
                exp_name = f"train_{dataset_name}_{augmentation}_{ratio_str}"
                
                # Check for completion
                potential_log = f"results/{exp_name}.log"
                if os.path.exists(potential_log):
                    with open(potential_log, 'r') as f:
                        if "Experiment Finished" in f.read():
                            continue

                logger, logfile = setup_logger(args, exp_name)
                logger.info(f"STARTING EXP {exp_id}: {exp_name}")

                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                clean_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                train_transform, _ = get_Trival_Augmentation() if augmentation == "trivial" else get_week_augmentation()

                # Dataset Loading
                if "cifar10" in dataset_name:
                    from _cifar10 import cifar10_dataset
                    DS_Class = IMBALANCECIFAR10 if "imbalance" in dataset_name else cifar10_dataset
                    num_classes = 10
                else:
                    DS_Class = IMBALANCECIFAR100
                    num_classes = 100

                dataset = DS_Class(root=args.data_dir, train=True, download=True, transform=clean_transform)
                val_dataset = DS_Class(root=args.data_dir, train=False, download=True, transform=clean_transform)

                # Initialize Model
                model = ResNet18_CIFAR(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                scaler = torch.amp.GradScaler('cuda')

                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

                # Initial Loader State
                dataset.transform = train_transform
                current_indices = None
                train_loader, _ = create_dataloader(train_dataset=dataset, val_dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, select_indices=current_indices)

                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    
                    # --- LAVA SELECTION PHASE ---
                    if select_ratio is not None and (epoch == 1 or epoch % 5 == 0):
                        logger.info(f"[Epoch {epoch}] Updating Selection with LAVA OT...")
                        
                        dataset.transform = clean_transform
                        num_total = len(dataset)
                        
                        # Call the valuation
                        lava_scores = run_lava_valuation(model, dataset, val_dataset, logger)

                        # --- THE CRITICAL GRADIENT RESTORE ---
                        torch.set_grad_enabled(True)
                        for param in model.parameters():
                            param.requires_grad = True
                        model.train() 
                        # -------------------------------------

                        if len(lava_scores) != num_total:
                            logger.warning(f"Shape Mismatch! Resizing scores from {len(lava_scores)} to {num_total}")
                            lava_scores = np.resize(lava_scores, num_total)

                        sorted_indices = np.argsort(lava_scores)[::-1]
                        num_keep = int(num_total * select_ratio)
                        current_indices = sorted_indices[:num_keep]
                        
                        logger.info(f"LAVA Selected {len(current_indices)}/{num_total} samples.")

                        dataset.transform = train_transform
                        train_loader, _ = create_dataloader(train_dataset=dataset, val_dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, select_indices=current_indices)
                    
                    # --- TRAINING PHASE ---
                    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
                    val_loss, val_acc = validate(model, val_loader, device, criterion)
                    
                    scheduler.step()
                    elapsed = time.time() - t0
                    msg = (f"Ep {epoch:03d} | TrLoss: {train_loss:.4f} Acc: {(train_acc*100):.2f}% | ValLoss: {val_loss:.4f} Acc: {(val_acc*100):.2f}% | {elapsed:.1f}s")
                    logger.info(msg)

                logger.info("Experiment Finished.")

if __name__ == "__main__":
    main()