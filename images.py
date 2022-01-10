import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cProfile
import pstats
from pstats import SortKey


IMG_BOX = 1280


class PawpularityImages(Dataset):
    def __init__(self, targets_csv, img_dir, tr_test, split=0.8):
        self.img_dir = img_dir
        self.df = pd.read_csv(targets_csv)
        if tr_test == 'train':
            self.df = self.df.truncate(after=np.floor(len(self.df) * split))
        else:
            self.df = self.df.truncate(before=np.floor(len(self.df) * split))
        self.targets = self.df['Pawpularity']
        self.targets = self.targets.to_numpy(dtype='float32')
        self.indexes = self.df['Id']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.indexes.iloc[item])
        image = read_image(img_path + ".jpg")
        target = torch.tensor(self.targets[item].reshape(-1))
        """Image reshaping and padding."""
        if image.shape[1] != IMG_BOX or image.shape[2] != IMG_BOX:
            resize = transforms.Resize(IMG_BOX - 1, max_size=IMG_BOX)
            image = resize(image)
        if image.shape[1] < IMG_BOX:
            padding = int(np.floor((IMG_BOX - image.shape[1]) / 2))
            pad = transforms.Pad(padding=(0, padding))
            image = pad(image)
        if image.shape[2] < IMG_BOX:
            padding = int(np.floor((IMG_BOX - image.shape[2]) / 2))
            pad = transforms.Pad(padding=(padding, 0))
            image = pad(image)
        # final_resize = transforms.Resize((IMG_BOX//2, IMG_BOX//2))
        final_resize = transforms.Resize((192, 192))  # Get rid of FP errors, VRAM saving
        image = final_resize(image)
        image = image.type(torch.float32)
        image = (image - torch.mean(image)) / torch.std(image)
        return image, target


train_dataset = PawpularityImages(targets_csv='train.csv',
                                  img_dir='train',
                                  tr_test='train')

test_dataset = PawpularityImages(targets_csv='train.csv',
                                 img_dir='train',
                                 tr_test='test')


# image, target = test_dataset.__getitem__(0)
# print(image)

# im_arr = test_dataset.__getitem__(5)[0].numpy()
# print(im_arr.shape)
# im_arr = np.transpose(im_arr, (1, 2, 0))
# plt.imshow(im_arr)
# plt.show()


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(294912, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1)
        )

    def forward(self, X):
        out = self.conv1(X)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.dense_layers(out)
        return out


batch_sz = 50
batches = train_dataset.__len__() / batch_sz
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)

model = CNN()
device = torch.device("cuda:0")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train_losses = []
test_losses = []


def train(epochs, grade):
    epochs = epochs
    t0 = datetime.now()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            if (batch % 5) - 1 == 0:
                batch_t0 = datetime.now()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if batch % 5 == 0:
                dt = datetime.now() - batch_t0
                print(f"Processed batch {batch}/{int(np.ceil(batches))}, duration {dt}")

        train_losses.append(np.mean(train_loss))

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_losses.append(np.mean(test_loss))
        dt = datetime.now() - t0

        print(f"Epoch: {epoch + 1}/{epochs}, Train loss: {train_losses[-1]:.4f}, Test loss: {test_losses[-1]:.4f}, \
        Duration: {dt}")

    plt.plot(train_losses, label="Train losses")
    plt.plot(test_losses, label="Test losses")
    plt.show()

    if grade:
        with torch.no_grad():
            train_outputs = []
            train_targets = []
            model.train()
            batch = 0
            for inputs, targets in train_loader:
                batch += 1
                if batch % 5 == 0:
                    print(f"Grading batch {batch}/{int(np.ceil(batches))}...")
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).cpu().numpy().flatten().tolist()
                train_targets += targets.cpu().numpy().flatten().tolist()
                train_outputs += outputs

            train_outputs = np.array(train_outputs)
            train_targets = np.array(train_targets)
            train_RMSE = np.sqrt(((train_targets - train_outputs) ** 2).mean())

            test_outputs = []
            test_targets = []
            model.eval()
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).cpu().numpy().flatten().tolist()
                test_targets += targets.cpu().numpy().flatten().tolist()
                test_outputs += outputs

            test_outputs = np.array(test_outputs)
            test_targets = np.array(test_targets)
            test_RMSE = np.sqrt(((test_targets - test_outputs) ** 2).mean())
            print(f"Train RMSE: {train_RMSE:.4f}, Test RMSE: {test_RMSE:.4f}")


# cProfile.run('train(1, grade=False)', 'stats')
# p = pstats.Stats('stats')
# p.strip_dirs().sort_stats(-1).print_stats()
# p.sort_stats(SortKey.TIME).print_stats(10)
