'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import optuna.integration


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
print("PyTorch version:", torch.__version__)
print(torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
'''

import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

x = torch.randn(5000, 5000, device=device)
y = torch.randn(5000, 5000, device=device)

start = time.time()
for _ in range(100):
    z = torch.matmul(x, y)
torch.cuda.synchronize()  # wait for ops to finish
end = time.time()

print(f"Time taken for matrix multiplications: {end - start:.2f} seconds")
