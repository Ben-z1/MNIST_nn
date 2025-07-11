import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

input_size = 28 * 28  # 784 pixels
hidden_size = 128
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 5

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_size = int(0.8 * len(train_dataset))
val_size   = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)       
        out = self.relu(out)    
        out = self.fc2(out)     
        return out

model = NN(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()           
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(loader, model):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.reshape(images.shape[0], -1)

            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)

            # Necessary for .cpu() to move back tensors to CPU, as Numpy doesn't work on GPU Tensors
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, cm

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # NECESSARY STEP for FCNN! Flattening
        images = images.reshape(images.shape[0], -1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad() #Clearing gradients from previous iterations
        loss.backward() #Backpropagation step, calculating all partial derivatives for all paras
        optimizer.step() #ACTUAL updating of the model's weights

    val_acc, _ = evaluate(val_loader, model)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.2f}, Validation Accuracy: {val_acc:.2f}")


test_acc, test_cm = evaluate(test_loader, model)

print(f"\nTest Accuracy: {test_acc:.2f}")


plt.figure(figsize=(8,6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("predictions")
plt.ylabel("Actual")
plt.title("Confusion Matrix on MNIST Test Set")
plt.show()
