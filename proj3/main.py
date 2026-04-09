"""
MLP
Need to test 3 architectures:
    shallow (1 hidden layer, ~128 units)
    medium (3 hl, [512,256,128])
    deep (>=5 hl, my choice)

hyper parameter tuning (10-20 configurations per architecture) instead of all 36 combinations:
    learning rate (.01, .001, .0001)
    batch size (32, 64, 128)
    optimizer: sgd or adam
    dropout {.2, .5}

validation splits
    45k/5k for cifar-10
    50k/10k for mnist
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
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_dataset(dataset_name, num_workers=0):
    """Load dataset once, pre-cache all images as tensors for fast GPU feeding"""
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_size = 50000
        input_size = 784
    else:  # CIFAR-10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_size = 45000
        input_size = 3072

    # Pre-cache: load ALL images into a single tensor on GPU
    # This eliminates per-batch transform + CPU->GPU transfer overhead
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

    return full_train_data, train_data, val_data, test_loader, input_size


# ============================================================================
# MLP ARCHITECTURES
# ============================================================================

class MLPShallow(nn.Module):
    """Shallow MLP: 1 hidden layer with 128 units"""
    def __init__(self, input_size=784, num_classes=10, dropout_rate=0.2):
        super(MLPShallow, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class MLPMedium(nn.Module):
    """Medium MLP: 3 hidden layers [512, 256, 128]"""
    def __init__(self, input_size=784, num_classes=10, dropout_rate=0.2):
        super(MLPMedium, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


class MLPDeep(nn.Module):
    """Deep MLP: 5+ hidden layers"""
    def __init__(self, input_size=784, num_classes=10, dropout_rate=0.2):
        super(MLPDeep, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(64, 32)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc6 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = self.fc6(x)
        return x


# ============================================================================
# TRAINING AND EVALUATION
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


def tune_hyperparameters(arch_class, arch_name, train_set, train_subset, val_subset, test_loader, input_size, device, dataset_name, num_workers=0, test_mode=False):
    """
    Hyperparameter tuning: test 10 meaningful configurations
    Returns best config and its test accuracy
    """
    # Define 10 meaningful configurations — favor batch 64/128 for GPU throughput
    configs = [
        # LR, Batch Size, Optimizer, Dropout Rate
        (0.001, 64, "Adam", 0.2),
        (0.001, 128, "Adam", 0.2),
        (0.001, 64, "Adam", 0.5),
        (0.0001, 64, "Adam", 0.2),
        (0.0001, 128, "Adam", 0.2),
        (0.001, 64, "SGD", 0.2),
        (0.001, 128, "SGD", 0.2),
        (0.01, 128, "SGD", 0.2),
        (0.001, 32, "Adam", 0.2),
        (0.0001, 32, "Adam", 0.5),
    ]
    
    # Test mode: only use 1 config
    if test_mode:
        configs = configs[:1]
    
    best_val_acc = 0.0
    best_config = None
    criterion = nn.CrossEntropyLoss()
    tune_epochs = 2 if test_mode else 3  # 3 epochs enough to separate good from bad configs
    
    print(f"\nTuning {arch_name} on {dataset_name}...")
    print(f"Testing {len(configs)} configurations ({tune_epochs} epochs each)...")
    
    for i, config in enumerate(configs):
        lr, batch_size, opt_name, dropout = config
        
        # Only rebuild DataLoaders (data already on GPU, no workers/pinning needed)
        train_loader_temp = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_temp = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Create model
        model = arch_class(input_size=input_size, num_classes=10, dropout_rate=dropout).to(device)
        
        # Create optimizer
        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:  # SGD
            optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Train and get best validation accuracy
        best_val_acc_config = train_with_config(model, train_loader_temp, val_loader_temp, 
                                                optimizer, criterion, device, tune_epochs)
        
        if best_val_acc_config > best_val_acc:
            best_val_acc = best_val_acc_config
            best_config = config
        
        print(f"  Config {i+1}/{len(configs)}: LR={lr}, Batch={batch_size}, Opt={opt_name}, Dropout={dropout} -> Val Acc: {best_val_acc_config:.2f}%")
    
    print(f"\nBest Config: LR={best_config[0]}, Batch={best_config[1]}, Opt={best_config[2]}, Dropout={best_config[3]}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # In test mode, skip retraining and just return dummy result
    if test_mode:
        print("TEST MODE: Skipping full retraining")
        return {
            "lr": best_config[0],
            "batch": best_config[1],
            "opt": best_config[2],
            "dropout": best_config[3],
            "test_acc": best_val_acc,  # Use validation acc as proxy
            "runtime": 0.0
        }
    
    # Retrain on full training+validation data with best config (10 epochs for final model)
    print(f"Retraining on full training data (10 epochs)...")
    lr, batch_size, opt_name, dropout = best_config
    final_epochs = 10  # Full training for final model
    
    # Use the full train_set (train+val combined, already on GPU)
    full_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Create final model
    final_model = arch_class(input_size=input_size, num_classes=10, dropout_rate=dropout).to(device)
    
    # Create optimizer
    if opt_name == "Adam":
        optimizer = optim.Adam(final_model.parameters(), lr=lr)
    else:  # SGD
        optimizer = optim.SGD(final_model.parameters(), lr=lr)
    
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
        "opt": best_config[2],
        "dropout": best_config[3],
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
    datasets_to_run = ["MNIST", "CIFAR-10"]  # Default: run both
    skip_archs = []  # Architectures to skip
    
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
        elif arg == "skipshallow":
            skip_archs.append("MLP (shallow)")
            print("Skipping shallow architecture")
        elif arg == "skipmedium":
            skip_archs.append("MLP (medium)")
            print("Skipping medium architecture")
        elif arg == "skipdeep":
            skip_archs.append("MLP (deep)")
            print("Skipping deep architecture")
        else:
            print(f"Invalid argument: {arg}")
            print("Usage: python main.py [MNIST|CIFAR|test] [skipshallow|skipmedium|skipdeep]")
            sys.exit(1)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Overall runtime tracking
    overall_start_time = time.time()
    
    # In test mode, only run MNIST and shallow arch
    if test_mode:
        datasets_to_run = ["MNIST"]
        architectures = [("MLP (shallow)", MLPShallow)]
    else:
        architectures = [
            ("MLP (shallow)", MLPShallow),
            ("MLP (medium)", MLPMedium),
            ("MLP (deep)", MLPDeep)
        ]
    
    # Remove skipped architectures
    architectures = [(name, cls) for name, cls in architectures if name not in skip_archs]
    
    all_results = {}
    
    for dataset_name in datasets_to_run:
        dataset_start_time = time.time()
        
        print("\n" + "="*80)
        print(f"DATASET: {dataset_name}")
        print("="*80)
        
        dataset_results = []
        
        # Load dataset once for all architectures
        train_set, train_subset, val_subset, test_loader, input_size = load_dataset(dataset_name, num_workers=num_workers)
        
        for arch_name, arch_class in architectures:
            arch_start_time = time.time()
            result = tune_hyperparameters(arch_class, arch_name, train_set, train_subset, 
                                         val_subset, test_loader, input_size, device, dataset_name, 
                                         num_workers=num_workers, test_mode=test_mode)
            arch_elapsed = time.time() - arch_start_time
            result["architecture"] = arch_name
            dataset_results.append(result)
            print(f"[{arch_name}] Elapsed: {arch_elapsed:.1f}s")
        
        all_results[dataset_name] = dataset_results
        
        dataset_elapsed = time.time() - dataset_start_time
        
        # Print table
        print("\n" + "="*80)
        print(f"Table: {dataset_name} Results (MLPs)")
        print("="*80)
        print(f"{'Architecture':<20} {'LR':<10} {'Batch':<8} {'Opt':<8} {'Dropout':<10} {'Test Acc':<12} {'Runtime':<10}")
        print("-"*88)
        
        for result in dataset_results:
            print(f"{result['architecture']:<20} {result['lr']:<10} {result['batch']:<8} {result['opt']:<8} {result['dropout']:<10} {result['test_acc']:.2f}%{'':<8} {result['runtime']:.2f}s")
        
        print(f"\n{dataset_name} Total Time: {dataset_elapsed:.1f}s ({dataset_elapsed/60:.1f} minutes)")
        print()
    
    # Print final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for dataset_name in datasets_to_run:
        print(f"\nTable: {dataset_name} Results (MLPs)")
        print(f"{'Architecture':<20} {'LR':<10} {'Batch':<8} {'Opt':<8} {'Dropout':<10} {'Test Acc':<12} {'Runtime':<10}")
        print("-"*88)
        
        for result in all_results[dataset_name]:
            print(f"{result['architecture']:<20} {result['lr']:<10} {result['batch']:<8} {result['opt']:<8} {result['dropout']:<10} {result['test_acc']:.2f}%{'':<8} {result['runtime']:.2f}s")
    
    # Overall runtime
    overall_elapsed = time.time() - overall_start_time
    print("\n" + "="*80)
    print(f"OVERALL RUNTIME: {overall_elapsed:.1f} seconds ({overall_elapsed/60:.1f} minutes)")
    print("="*80)
