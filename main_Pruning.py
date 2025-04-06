import torch.nn as nn
import torch.nn.utils.prune as prune
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import *
from KD_Lib.models import ResNet18
from torch.utils.data import Subset, DataLoader
import torch_pruning as tp

# Load data
train_set = datasets.MNIST(
    "mnist_data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

# train_set = Subset(train_set, range(1000))

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,  # Can still shuffle
    num_workers=0  # Recommended to disable multi-threading for debugging
)

test_set = datasets.MNIST(
    "mnist_data",
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

# test_set = Subset(test_set, range(1000))

test_loader = DataLoader(test_set, batch_size=16)


def train_test_pipeline(model, train_loader, test_loader, lr=0.01, epochs=5, is_pruned=False):
    def train(model, loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) if not is_pruned \
            else optim.Adam(model.parameters(), lr=lr)
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def test(model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    best_acc = 0.0
    for epoch in range(epochs):
        train(model, train_loader)
        acc = test(model, test_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
    return best_acc


def structured_pruning(model, iterative_steps=3, pruning_ratio=0.3):
    # 1. Build dependency graph
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 1, 28, 28))

    #######
    imp = tp.importance.TaylorImportance()
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    example_inputs = torch.randn(1, 1, 28, 28)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,  # Remove 50% of channels
        ignored_layers=ignored_layers,
        global_pruning=True
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Before pruning | MACs: {base_macs:,} | Parameters: {base_nparams:,}")
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = model(example_inputs).sum()  # A dummy loss for TaylorImportance
            loss.backward()  # Before pruner.step()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"Iteration {i} | Rate:{macs / base_macs:.4f}  {nparams / base_nparams:.4f}")

    pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"After pruning | MACs: {pruned_macs:,} | Parameters: {pruned_nparams:,}")

    return model


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# 1. Train model
print("=== Load pre-trained model ===")
student_params = [4, 4, 4, 4, 4]
model = ResNet18(student_params, 1, 10)
pretrained_acc = train_test_pipeline(model, train_loader, test_loader, lr=0.001, epochs=5)

# 2. Prune model
print("\n=== Performing model pruning ===")
pruned_model = structured_pruning(model, iterative_steps=3, pruning_ratio=0.6)
pruned_acc = train_test_pipeline(pruned_model, train_loader, test_loader, lr=0.001, epochs=5)

# 3. Compare results
print("\n=== Result comparison ===")
print(f"Original accuracy: {pretrained_acc:.4f}")
print(f"Pruned accuracy: {pruned_acc:.4f}")
print(f"Accuracy change: {pretrained_acc:.4f} → {pruned_acc:.4f}")