import argparse
import time
import os
import logging
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from dataloader import create_dataloader 
from _density import compute_density_probability
from _consistency import ConsistencyCalculator
from selection import select_samples, get_class_adaptive_ratios, get_targets_safe
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
        # Standard ResNet18 forward
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
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        # CLEAN: Use *rest to ignore indices if they exist
        for inputs, targets, *rest in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total

def train_one_epoch(model, loader, optimizer, device, criterion, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # CLEAN: Use *rest to ignore indices if they exist
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
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total

def extract_features(model, dataset, device):
    """Extracts features using the current dataset transform."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    all_feats = []
    with torch.no_grad():
        # CLEAN: We only need inputs, ignore everything else
        for inputs, *rest in loader:
            inputs = inputs.to(device)
            features, _ = model(inputs, return_features=True)
            all_feats.append(features.cpu())
    return torch.cat(all_feats, dim=0).numpy()

def setup_logger(args, exp_name):
    os.makedirs("results", exist_ok=True)
    
    # Filename matches exp_name for easy skipping
    log_filename = f"results/{exp_name}.log"
    
    logger = logging.getLogger("train_logger")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=200)
    
    # === OPTIMIZED DEFAULTS ===
    parser.add_argument("--batch-size", type=int, default=128) 
    parser.add_argument("--num-workers", type=int, default=16) 
    parser.add_argument("--lr", type=float, default=0.001) 
    parser.add_argument("--device", default="cuda:1")
    # ==========================
    
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=None)

    args = parser.parse_args()


    
    datasets = ["cifar10","cifar100","imbalance_cifar10","imbalance_cifar100"] 
    augmentations = ["trivial", "weak"] 
    select_ratios = [None]

    if args.dataset: datasets = [args.dataset]
    if args.augmentation: augmentations = [args.augmentation]
    if args.ratio is not None: select_ratios = [args.ratio]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    exp_id = 0

    for dataset_name in datasets:
        for augmentation in augmentations:
            for select_ratio in select_ratios:
                exp_id += 1
                args.dataset = dataset_name
                args.augmentation = augmentation
                args.ratio = select_ratio
                
                ratio_str = f"r{select_ratio}" if select_ratio is not None else "Baseline"
                exp_name = f"train_{dataset_name}_{augmentation}_{ratio_str}"
                
                potential_log = f"results/{exp_name}.log"
                if os.path.exists(potential_log):
                    with open(potential_log, 'r') as f:
                        if "Experiment Finished" in f.read():
                            print(f"[SKIP] Experiment {exp_name} already completed.")
                            continue

                logger, logfile = setup_logger(args, exp_name)
                logger.info(f"STARTING EXP {exp_id}: {exp_name}")

                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                clean_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
                
                if augmentation == "trivial":
                    train_transform, _ = get_Trival_Augmentation()
                else:
                    train_transform, _ = get_week_augmentation()

                if dataset_name == "cifar10":
                    from _cifar10 import cifar10_dataset
                    dataset = cifar10_dataset(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = cifar10_dataset(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10
                elif dataset_name == "imbalance_cifar10":
                    dataset = IMBALANCECIFAR10(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR10(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 10

                elif dataset_name == "imbalance_cifar100":
                    dataset = IMBALANCECIFAR100(root=args.data_dir, train=True, download=True, transform=clean_transform)
                    val_dataset = IMBALANCECIFAR100(root=args.data_dir, train=False, download=True, transform=clean_transform)
                    num_classes = 100

                # CLIP Calculation
                p_con = np.ones(len(dataset)) 
                if select_ratio is not None:
                    dataset.transform = clean_transform 
                    logger.info("Calculating CLIP Consistency Scores...")
                    consistency_calc = ConsistencyCalculator(device=device)
                    p_con = consistency_calc.calculate(dataset)
                    del consistency_calc
                    torch.cuda.empty_cache()

                model = ResNet18_CIFAR(num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
                criterion = nn.CrossEntropyLoss()
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
                scaler = torch.amp.GradScaler('cuda')

                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
                )

                # Initialize Train Loader (Initial state)
                dataset.transform = train_transform
                current_indices = None
                
                train_loader, _ = create_dataloader(
                    train_dataset=dataset, val_dataset=val_dataset,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    select_indices=current_indices 
                )


                for epoch in range(1, args.epochs + 1):
                    t0 = time.time()
                    
                    # 1. Dynamic Selection Phase
                    if select_ratio is not None:
                        # Update ONLY every 5 epochs (or first epoch)
                        if epoch == 1 or epoch % 5 == 0:
                            logger.info(f"[Epoch {epoch}] Updating Selection...")
                            
                            # A. Switch to Clean Transform for Feature Extraction
                            dataset.transform = clean_transform
                            feats = extract_features(model, dataset, device)
                            p_rho = compute_density_probability(feats)
                            
                            # === NEW CODE START: Calculate Adaptive Ratios ===
                            logger.info("Calculating Adaptive Selection Ratios...")
                            
                            # 1. Get targets to count classes
                            all_targets = get_targets_safe(dataset)
                            
                            # 2. Compute ratios 
                            # base_ratio = args.ratio (e.g. 0.7 for Majority)
                            # max_ratio = 1.0 (Keep 100% of Minority)
                            adaptive_ratios = get_class_adaptive_ratios(
                                all_targets, 
                                base_ratio=select_ratio, 
                                max_ratio=1.0
                            )
                            if epoch == 1:
                                logger.info(f"VERIFY RATIOS: Class 0 (Head): {adaptive_ratios.get(0)} | Class 9 (Tail): {adaptive_ratios.get(9)}")
                                
                            current_indices = select_samples(dataset, p_rho, p_con, adaptive_ratios)
                            
                            logger.info(f"Selected {len(current_indices)} samples.")
                            
                            # C. Switch back to Train Transform
                            dataset.transform = train_transform

                            # Update Train Loader
                            train_loader, _ = create_dataloader(
                                train_dataset=dataset, val_dataset=val_dataset,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                select_indices=current_indices 
                            )
                    
                    # 2. Training Phase
                    # (Note: We use the existing train_loader, we don't recreate it if not needed)
                    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, criterion, scaler)
                    val_loss, val_acc = validate(model, val_loader, device, criterion)
                    
                    scheduler.step()
                    elapsed = time.time() - t0
                    
                    msg = (f"[EXP {exp_id}] Ep {epoch:03d} | "
                           f"TrLoss: {train_loss:.4f} Acc: {(train_acc*100):.2f}% | "
                           f"ValLoss: {val_loss:.4f} Acc: {(val_acc*100):.2f}% | Time: {elapsed:.1f}s")
                    print(msg)
                    logger.info(msg)

                logger.info("Experiment Finished.")

if __name__ == "__main__":
    main()
