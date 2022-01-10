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
from PIL import Image


def process_image(im, final_size, destination, index):
    bg = (0, 0, 0)
    w, h = im.size
    if w == h:
        image = im.resize(final_size)
        image.save(os.path.join(destination, index + ".jpg"))
    elif w > h:
        image = Image.new(im.mode, (w, w), bg)
        image.paste(im, (0, (w - h) // 2))
        image = image.resize(final_size)
        image.save(os.path.join(destination, index + ".jpg"))
    elif h > w:
        image = Image.new(im.mode, (h, h), bg)
        image.paste(im, ((h - w) // 2, 0))
        image = image.resize(final_size)
        image.save(os.path.join(destination, index + ".jpg"))


def preprocess_images(labels_csv, source, destination, final_size):
    print("Starting image preprocessing...")
    if not os.path.isdir(destination):
        os.mkdir(destination)
    df = pd.read_csv(labels_csv)
    indexes = df['Id']
    i = 0
    n = len(indexes)
    t0 = datetime.now()
    processed = 0
    skipped = 0
    for index in indexes:
        if (i + 1) % 1000 == 0:
            print(f"Checking image {i + 1}/{n}")
        if os.path.isfile(os.path.join(destination, index + ".jpg")):
            with Image.open(os.path.join(destination, index + ".jpg")) as dst_im:
                if dst_im.size == final_size:
                    skipped += 1
                    pass
                else:
                    with Image.open(os.path.join(source, index + ".jpg")) as src_im:
                        process_image(src_im, final_size, destination, index)
                        processed += 1
        else:
            with Image.open(os.path.join(source, index + ".jpg")) as src_im:
                process_image(src_im, final_size, destination, index)
                processed += 1
        i += 1
    print(f"Image preprocessing took {datetime.now() - t0}, processed {processed} images, skipped {skipped}")


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
        image = image.type(torch.float32)
        image = (image - torch.mean(image)) / torch.std(image)
        return image, target


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


def train(epochs, grade, model, device, criterion, optimizer, batches):
    train_losses = []
    test_losses = []
    epochs = epochs
    t0 = datetime.now()
    for epoch in range(epochs):
        model.train()
        train_loss = []
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            if (batch % 10) - 1 == 0:
                batch_t0 = datetime.now()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if batch % 10 == 0:
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


img_size = (192, 192)
preprocess_images('train.csv', 'train', 'train-post', img_size)

train_dataset = PawpularityImages(targets_csv='train.csv',
                                  img_dir='train-post',
                                  tr_test='train', )

test_dataset = PawpularityImages(targets_csv='train.csv',
                                 img_dir='train-post',
                                 tr_test='test')

# image, target = test_dataset.__getitem__(0)
# print(image)

# im_arr = test_dataset.__getitem__(5)[0].numpy()
# print(im_arr.shape)
# im_arr = np.transpose(im_arr, (1, 2, 0))
# plt.imshow(im_arr)
# plt.show()

batch_sz = 50
batches = train_dataset.__len__() / batch_sz
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)

model = CNN()
device = torch.device("cuda:0")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

train(5, True, model, device, criterion, optimizer, batches)

# cProfile.run('train(1, grade=False, model, device, criterion, optimizer, batches)', 'stats')
# p = pstats.Stats('stats')
# p.strip_dirs().sort_stats(-1).print_stats()
# p.sort_stats(SortKey.TIME).print_stats(10)
