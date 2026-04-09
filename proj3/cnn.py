"""
CNN (Convolutional Neural Networks)
Part 2 of CS 4375 Project 3

Two architectures:
    simple (2 conv layers, 8 filters each)
    enhanced (3+ conv layers, increasing filters [16, 32, 64])

Hyperparameter tuning:
    learning rate {0.01, 0.001, 0.0001}
    batch size {32, 64, 128}
    weight decay {0, 1e-4, 5e-3}
    
Adam optimizer (fixed), no SGD
3x3 filters, batch norm after each conv, stride=1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import sys

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Enable cuDNN autotuning for best GPU performance
torch.backends.cudnn.benchmark = True


# ============================================================================
# DATA LOADING (reuse validation splits from main.py setup)
# ============================================================================

def load_dataset(dataset_name, num_workers=0):
    """Load dataset once, pre-cache all images as tensors on GPU for fast feeding"""
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_size = 50000
    else:  # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_size = 45000

    # Pre-cache: load ALL images into a single tensor on GPU
    print("Pre-caching dataset to GPU...")
    train_images = torch.stack([train_set[i][0] for i in range(len(train_set))]).to(device)
    train_labels = torch.tensor([train_set[i][1] for i in range(len(train_set))]).to(device)
    test_images = torch.stack([test_set[i][0] for i in range(len(test_set))]).to(device)
    test_labels = torch.tensor([test_set[i][1] for i in range(len(test_set))]).to(device)

    # Split
    val_size = len(train_set) - train_size
    indices = torch.randperm(len(train_set))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = torch.utils.data.TensorDataset(train_images[train_indices], train_labels[train_indices])
    val_data = torch.utils.data.TensorDataset(train_images[val_indices], train_labels[val_indices])
    full_train_data = torch.utils.data.TensorDataset(train_images, train_labels)
    test_data = torch.utils.data.TensorDataset(test_images, test_labels)

    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    print("Pre-caching done.")

    return full_train_data, train_data, val_data, test_loader


# ============================================================================
# CNN ARCHITECTURES
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN: 2 conv layers (8 filters each), batch norm, pooling, FC layer"""
    def __init__(self, num_classes=10, in_channels=1, image_size=28):
        super(SimpleCNN, self).__init__()
        # Conv Layer 1: in_channels -> 8 filters, 3x3, stride=1
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 8 -> 8 filters, 3x3, stride=1
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 pooling layers: 
        # MNIST: 28 -> 14 -> 7 (7*7*8 = 392)
        # CIFAR: 32 -> 16 -> 8 (8*8*8 = 512)
        self.fc_input_size = self._calculate_fc_size(in_channels, image_size)
        
        # Fully connected layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
    
    def _calculate_fc_size(self, in_channels, image_size):
        """Calculate flattened size after conv+pooling layers"""
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        x = torch.relu(self.bn1(self.conv1(dummy)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class EnhancedCNN(nn.Module):
    """Enhanced CNN: 3+ conv layers with increasing filters [16, 32, 64], batch norm, pooling"""
    def __init__(self, num_classes=10, in_channels=1, image_size=28):
        super(EnhancedCNN, self).__init__()
        # Conv Layer 1: in_channels -> 16 filters, 3x3, stride=1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 16 -> 32 filters, 3x3, stride=1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 3: 32 -> 64 filters, 3x3, stride=1
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 3 pooling layers:
        # MNIST: 28 -> 14 -> 7 -> 3 (3*3*64 = 576)
        # CIFAR: 32 -> 16 -> 8 -> 4 (4*4*64 = 1024)
        self.fc_input_size = self._calculate_fc_size(in_channels, image_size)
        
        # Fully connected layer
        self.fc = nn.Linear(self.fc_input_size, num_classes)
    
    def _calculate_fc_size(self, in_channels, image_size):
        """Calculate flattened size after conv+pooling layers"""
        dummy = torch.zeros(1, in_channels, image_size, image_size)
        x = torch.relu(self.bn1(self.conv1(dummy)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# ============================================================================
# TRAINING AND EVALUATION (same as MLP)
# ============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def test(model, test_loader, device):
    """Test model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train_with_config(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, early_stop_patience=2):
    """Train model and return best validation accuracy with early stopping"""
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = validate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stop if no improvement for 'early_stop_patience' epochs
        if patience_counter >= early_stop_patience:
            break
    
    return best_val_acc


def tune_hyperparameters(arch_class, arch_name, train_set, train_subset, val_subset, test_loader, device, dataset_name, in_channels=1, image_size=28, num_workers=0, test_mode=False):
    """
    Hyperparameter tuning for CNNs: test 10-15 configurations
    Adam optimizer FIXED, vary: LR, Batch Size, Weight Decay
    Returns best config and its test accuracy
    """
    # Define 12 meaningful configurations — favor batch 64/128 for GPU throughput
    configs = [
        # LR, Batch Size, Weight Decay
        (0.001, 64, 0),
        (0.001, 128, 0),
        (0.001, 64, 1e-4),
        (0.001, 128, 1e-4),
        (0.001, 64, 5e-3),
        (0.001, 128, 5e-3),
        (0.0001, 64, 0),
        (0.0001, 128, 0),
        (0.0001, 64, 1e-4),
        (0.01, 64, 0),
        (0.01, 128, 0),
        (0.001, 32, 0),
    ]
    
    # Test mode: only use 1 config
    if test_mode:
        configs = configs[:1]
    
    best_val_acc = 0.0
    best_config = None
    criterion = nn.CrossEntropyLoss()
    tune_epochs = 2 if test_mode else 3  # 3 epochs enough to rank configs
    
    print(f"\nTuning {arch_name} on {dataset_name}...")
    print(f"Testing {len(configs)} configurations ({tune_epochs} epochs each)...")
    
    for i, config in enumerate(configs):
        lr, batch_size, weight_decay = config
        
        # Data already on GPU, no workers/pinning needed
        train_loader_temp = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_temp = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = arch_class(num_classes=10, in_channels=in_channels, image_size=image_size).to(device)
        
        # Adam optimizer with weight decay (L2 regularization)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Train and get best validation accuracy
        best_val_acc_config = train_with_config(model, train_loader_temp, val_loader_temp, 
                                                optimizer, criterion, device, tune_epochs)
        
        if best_val_acc_config > best_val_acc:
            best_val_acc = best_val_acc_config
            best_config = config
        
        print(f"  Config {i+1}/{len(configs)}: LR={lr}, Batch={batch_size}, WD={weight_decay} -> Val Acc: {best_val_acc_config:.2f}%")
    
    print(f"\nBest Config: LR={best_config[0]}, Batch={best_config[1]}, Weight Decay={best_config[2]}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # In test mode, skip retraining and just return dummy result
    if test_mode:
        print("TEST MODE: Skipping full retraining")
        return {
            "lr": best_config[0],
            "batch": best_config[1],
            "weight_decay": best_config[2],
            "test_acc": best_val_acc,
            "runtime": 0.0
        }
    
    # Retrain on full training+validation data with best config (10 epochs for final model)
    print(f"Retraining on full training data (10 epochs)...")
    lr, batch_size, weight_decay = best_config
    final_epochs = 10
    
    # Use the full train_set (train+val combined, already on GPU)
    full_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Create final model
    final_model = arch_class(num_classes=10, in_channels=in_channels, image_size=image_size).to(device)
    
    # Adam optimizer with weight decay
    optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Train on full data
    start_time = time.time()
    for epoch in range(final_epochs):
        train_epoch(final_model, full_train_loader, optimizer, criterion, device)
    runtime = time.time() - start_time
    
    # Test
    test_acc = test(final_model, test_loader, device)
    
    return {
        "lr": best_config[0],
        "batch": best_config[1],
        "weight_decay": best_config[2],
        "test_acc": test_acc,
        "runtime": runtime
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    test_mode = False
    num_workers = 0
    datasets_to_run = ["MNIST", "CIFAR-10"]
    skip_archs = []
    
    args = sys.argv[1:]
    
    for arg in args:
        if arg == "test":
            test_mode = True
            print(f"TEST MODE: Running 1 config with num_workers={num_workers}")
        elif arg == "MNIST":
            datasets_to_run = ["MNIST"]
            print("Running MNIST only")
        elif arg == "CIFAR":
            datasets_to_run = ["CIFAR-10"]
            print("Running CIFAR-10 only")
        elif arg == "skipsimple":
            skip_archs.append("CNN (simple)")
            print("Skipping simple CNN")
        elif arg == "skipenhanced":
            skip_archs.append("CNN (enhanced)")
            print("Skipping enhanced CNN")
        else:
            print(f"Invalid argument: {arg}")
            print("Usage: python cnn.py [MNIST|CIFAR|test] [skipsimple|skipenhanced]")
            sys.exit(1)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Overall runtime tracking
    overall_start_time = time.time()
    
    # Set architectures
    if test_mode:
        datasets_to_run = ["MNIST"]
        architectures = [("CNN (simple)", SimpleCNN, 1)]  # MNIST: 1 channel
    else:
        architectures = [
            ("CNN (simple)", SimpleCNN, None),  # Will set channel based on dataset
            ("CNN (enhanced)", EnhancedCNN, None)
        ]
    
    # Remove skipped architectures
    architectures = [(name, cls, ch) for name, cls, ch in architectures if name not in skip_archs]
    
    all_results = {}
    
    for dataset_name in datasets_to_run:
        dataset_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"DATASET: {dataset_name}")
        print("="*80)
        
        # Determine input channels and image size
        in_channels = 1 if dataset_name == "MNIST" else 3
        image_size = 28 if dataset_name == "MNIST" else 32
        
        dataset_results = []
        
        # Load dataset once for all architectures
        train_set, train_subset, val_subset, test_loader = load_dataset(dataset_name, num_workers=num_workers)
        
        for arch_name, arch_class, _ in architectures:
            arch_start_time = time.time()
            result = tune_hyperparameters(arch_class, arch_name, train_set, train_subset, 
                                         val_subset, test_loader, device, dataset_name, 
                                         in_channels=in_channels, image_size=image_size, num_workers=num_workers, test_mode=test_mode)
            arch_elapsed = time.time() - arch_start_time
            result["architecture"] = arch_name
            dataset_results.append(result)
            print(f"[{arch_name}] Elapsed: {arch_elapsed:.1f}s")
        
        all_results[dataset_name] = dataset_results
        
        dataset_elapsed = time.time() - dataset_start_time
        
        # Print table
        print("\n" + "="*80)
        print(f"Table: {dataset_name} Results (CNNs)")
        print("="*80)
        print(f"{'Architecture':<20} {'LR':<10} {'Batch':<8} {'Weight Decay':<15} {'Test Acc':<12} {'Runtime':<10}")
        print("-"*85)
        
        for result in dataset_results:
            print(f"{result['architecture']:<20} {result['lr']:<10} {result['batch']:<8} {result['weight_decay']:<15} {result['test_acc']:.2f}%{'':<8} {result['runtime']:.2f}s")
        
        print(f"\n{dataset_name} Total Time: {dataset_elapsed:.1f}s ({dataset_elapsed/60:.1f} minutes)")
        print()
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for dataset_name in datasets_to_run:
        print(f"\nTable: {dataset_name} Results (CNNs)")
        print(f"{'Architecture':<20} {'LR':<10} {'Batch':<8} {'Weight Decay':<15} {'Test Acc':<12} {'Runtime':<10}")
        print("-"*85)
        
        for result in all_results[dataset_name]:
            print(f"{result['architecture']:<20} {result['lr']:<10} {result['batch']:<8} {result['weight_decay']:<15} {result['test_acc']:.2f}%{'':<8} {result['runtime']:.2f}s")
    
    # Overall runtime
    overall_elapsed = time.time() - overall_start_time
    print("\n" + "="*80)
    print(f"OVERALL RUNTIME: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print("="*80)
