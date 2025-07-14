import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class NN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate):
        super(NN, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EngineMNIST:
    def __init__(self, model, optimizer, device, criterion):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion

    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        total_samples = 0
        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            images = images.reshape(images.shape[0], -1)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def validate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                images = images.reshape(images.shape[0], -1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        return avg_loss

    def test(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                images = images.reshape(images.shape[0], -1)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / total_samples
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc
