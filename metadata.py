# ML model for Kaggle Pawpularity Contest

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Dataset from metadata
class PawpularityMetadata(Dataset):
    def __init__(self, file, tr_test, split=0.8):
        self.df = pd.read_csv(file)
        if tr_test == 'train':
            self.df = self.df.truncate(after=np.floor(len(self.df) * split))
        else:
            self.df = self.df.truncate(before=np.floor(len(self.df) * split))
        self.data = self.df.drop(columns=['Pawpularity', 'Id'])
        self.data = self.data.to_numpy(dtype="float32")
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)
        self.labels = self.df['Pawpularity']
        self.labels = self.labels.to_numpy(dtype="float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[item])
        label = torch.tensor(self.labels[item].reshape(-1))
        return data, label


train_dataset = PawpularityMetadata('train.csv', tr_test='train')
test_dataset = PawpularityMetadata('train.csv', tr_test='test')

batch_sz = 100
D = len(train_dataset.data[0])

# Instantiate loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)

# Model, just for metadata
model = nn.Sequential(
    nn.Linear(D, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 1)
)

# Move to GPU
device = torch.device("cuda:0")
model.to(device)

# Instantiate criterion and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
train_losses = []
test_losses = []
metadata_training = True

# Main training loop
if metadata_training:
    for epoch in range(epochs):
        train_loss = []
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_losses.append(np.mean(train_loss))

        test_loss = []
        model.eval()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_losses.append(np.mean(test_loss))

        # if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train loss: {train_losses[-1]:.4f}, Test loss: {test_losses[-1]:.4f}")

    plt.plot(train_losses, label="Train losses")
    plt.plot(test_losses, label="Test losses")
    plt.show()

    with torch.no_grad():
        train_outputs = []
        train_targets = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).cpu().numpy().flatten().tolist()
            train_targets += targets.cpu().numpy().flatten().tolist()
            train_outputs += outputs

        train_outputs = np.array(train_outputs)
        train_targets = np.array(train_targets)
        train_RMSE = np.sqrt(((train_targets - train_outputs) ** 2).mean())

        test_outputs = []
        test_targets = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).cpu().numpy().flatten().tolist()
            test_targets += targets.cpu().numpy().flatten().tolist()
            test_outputs += outputs

        test_outputs = np.array(test_outputs)
        test_targets = np.array(test_targets)
        test_RMSE = np.sqrt(((test_targets - test_outputs) ** 2).mean())
        print(f"Train RMSE: {train_RMSE:.4f}, Test RMSE: {test_RMSE:.4f}")






