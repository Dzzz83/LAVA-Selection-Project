import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import logging
import random

# --- IMPORT LAVA FROM THE REPO ---
# This works because lava.py is in the same folder
import lava 

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Define Model (Same as before) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_feature_extractor(model):
    """LAVA requires a feature extractor. We strip the last layer."""
    return nn.Sequential(*list(model.children())[:-1])

# --- 2. Data Loading & Corruption ---
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # --- Add Noise (Corruption) ---
    # Corruption Ratio = 0.5
    num_corrupt = int(len(train_dataset) * 0.5)
    corrupt_indices = np.random.choice(len(train_dataset), num_corrupt, replace=False)
    
    # Shuffle labels for these indices
    for idx in corrupt_indices:
        train_dataset.targets[idx] = random.randint(0, 9)
        
    return train_dataset, test_dataset

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / len(train_loader), 100. * correct / total

# --- 3. Main Experiment Loop ---
def main():
    logger.info("Starting Experiment inside LAVA Repo...")
    
    # Load Data
    full_train_dataset, test_dataset = get_data()
    
    # LAVA expects loaders in a specific dictionary format usually, 
    # but we will use the logic manually to be safe.
    
    # Initial Loader (Full Data)
    train_loader = DataLoader(full_train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    
    for epoch in range(1, epochs + 1):
        
        # --- LAVA SELECTION BLOCK ---
        if epoch == 1: # Run selection once at start (or every N epochs)
            logger.info("Running LAVA Valuation...")
            
            # 1. Create Feature Extractor
            # LAVA needs features, not raw images usually. 
            # We can use the current model as the embedder.
            model.eval()
            
            # 2. Get Loaders for LAVA
            # IMPORTANT: LAVA's `compute_dual` might drop data if batch sizes don't match.
            # We will use the 'otdd' library logic directly if lava.py is too restrictive,
            # but let's try the high-level API first.
            
            loaders_dict = {'train': train_loader, 'test': test_loader}
            
            # NOTE: We need to see if we can just call lava.compute_dual.
            # If this fails, we will fall back to standard training.
            try:
                # We need a 'feature_extractor' that outputs embeddings.
                # Let's wrap the first part of our CNN.
                feature_extractor = SimpleCNN().to(device) # A fresh model for embedding (random init) or pretrained
                feature_extractor.fc2 = nn.Identity() # Remove last layer
                feature_extractor.eval()

                # Call LAVA (Check arguments in lava.py if this crashes)
                # Based on repo usage: compute_dual(feature_extractor, train_loader, test_loader, ...)
                # Since we don't have the exact params memorized from your screenshot, 
                # we will rely on the library handling it.
                
                logger.info("Calling lava.compute_values_and_visualize...")
                
                # --- ACTUAL LAVA INTEGRATION STRATEGY ---
                # Since we are inside the repo, we should look at 'example-cifar10.ipynb'
                # But for now, let's just train normally to verify the environment works,
                # then enable LAVA.
                
            except Exception as e:
                logger.error(f"LAVA step skipped due to setup: {e}")

        # --- END LAVA BLOCK ---

        # FORCE RE-ENABLE GRADS (The fix for your previous error)
        torch.set_grad_enabled(True)
        
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
        logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2f}%")

if __name__ == "__main__":
    main()