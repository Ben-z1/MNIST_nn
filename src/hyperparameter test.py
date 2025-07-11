import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import plotly
import plotly.io as pio
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import optuna.integration


# -----------------------------
# Fixed constants
# -----------------------------
pio.renderers.default = "browser" 

input_size = 28 * 28
num_classes = 10
num_epochs = 20
batch_size = 1024
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])

def get_data_loaders(batch_size):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    return train_loader, val_loader, test_loader

# -----------------------------
# Model definition
# -----------------------------
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

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_loss(loader, model, device, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.reshape(images.shape[0], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss

# -----------------------------
# Objective function for Optuna
# -----------------------------
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_sizes = [trial.suggest_int(f"n_units_l{i}", 32, 128, step=16) for i in range(n_layers)]
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    momentum = 0.0
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)  #Play around with dropout


    print(f"Trial details - {n_layers} layers, Hidden layer sizes: {hidden_sizes}, Optimizer: {optimizer_name}, Dropout rate: {dropout_rate}")

    model = NN(input_size, hidden_sizes, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    train_loader, val_loader, _ = get_data_loaders(batch_size)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.reshape(images.shape[0], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(val_loader, model, device, criterion)
        trial.report(val_loss, epoch)

        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}, GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if trial.should_prune():
            raise optuna.TrialPruned()
            

    return best_loss

# -----------------------------
# Main Execution Block (Windows Safe)
# -----------------------------
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), sampler=optuna.samplers.TPESampler(seed=70))
    study.optimize(objective, n_trials=30)

    print("\nBest hyperparameters:", study.best_params)

    # Visualization
    plot_optimization_history(study).show()
    plot_param_importances(study).show()

    # Retrain final model
    best_params = study.best_params
    hidden_sizes = [best_params[f"n_units_l{i}"] for i in range(best_params["n_layers"])]
    model = NN(input_size, hidden_sizes, num_classes, dropout_rate=best_params["dropout_rate"]).to(device)
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    elif best_params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=best_params['lr'])
    else: 
        momentum = best_params.get('momentum', 0.0)
        optimizer = optim.SGD(model.parameters(), lr=best_params['lr'], momentum=momentum, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            images = images.reshape(images.shape[0], -1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate_loss(val_loader, model, device, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    test_loss = evaluate_loss(test_loader, model, device, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()