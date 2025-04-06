import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from KD_Lib.KD import *
from KD_Lib.models import ResNet18, ResNet50
from torch.utils.data import Subset, DataLoader


def train_test_pipeline(model, train_loader, test_loader, lr=0.01, epochs=5, is_pruned=False):
    def train(model, loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr)
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
        # print(f"Epoch {epoch + 1}/{epochs} | Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
    return best_acc


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
    shuffle=True,
    num_workers=0
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

# 1. Load model and optimizer
student_params = [4, 4, 4, 4, 4]
teacher_model = ResNet50(student_params, 1, 10)
student_model = ResNet18(student_params, 1, 10)

teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
student_optimizer = optim.SGD(student_model.parameters(), 0.01)


# 2. train student model without KD
print("Training Student without KD...")
without_acc = train_test_pipeline(student_model, train_loader, test_loader, lr=0.01, epochs=5)
print(f"Without KD accuracy: {without_acc:.4f}")

# 3. Train student model with KD
# distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
#                       teacher_optimizer, student_optimizer)
distiller = Attention(teacher_model, student_model, train_loader, test_loader,
                      teacher_optimizer, student_optimizer)
distiller.train_teacher(epochs=5, plot_losses=True, save_model=False)
print()
distiller.train_student(epochs=5, plot_losses=True, save_model=False)

tea_acc = distiller.evaluate(teacher=True, verbose=False)
with_acc = distiller.evaluate(teacher=False, verbose=False)
print(f"\nTeacher accuracy: {tea_acc:.4f}")
print(f"Student without KD accuracy: {without_acc:.4f}")
print(f"Student with KD accuracy: {with_acc:.4f}")

# distiller.get_parameters()

