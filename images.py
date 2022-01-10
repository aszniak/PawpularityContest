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
from sklearn.preprocessing import StandardScaler


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


class PawpularityDataset(Dataset):
    def __init__(self, csv, img_dir, tr_test, split=0.8, transformations=None):
        self.transformations = transformations
        self.img_dir = img_dir
        self.df = pd.read_csv(csv)
        if tr_test == 'train':
            self.df = self.df.truncate(after=np.floor(len(self.df) * split))
        else:
            self.df = self.df.truncate(before=np.floor(len(self.df) * split))
        self.targets = self.df['Pawpularity']
        self.targets = self.targets.to_numpy(dtype='float32')
        self.indexes = self.df['Id']
        self.metadata = self.df.drop(columns=['Pawpularity', 'Id'])
        self.metadata = self.metadata.to_numpy(dtype="float32")
        self.scaler = StandardScaler()
        self.metadata = self.scaler.fit_transform(self.metadata)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.indexes.iloc[item])
        image = read_image(img_path + ".jpg")
        if self.transformations:
            image = self.transformations(image)
        image = image.type(torch.float32)
        image = (image - torch.mean(image)) / torch.std(image)
        metadata = torch.from_numpy(self.metadata[item])
        target = torch.tensor(self.targets[item].reshape(-1))
        data = (image, metadata)
        return data, target


class PawpularityModel(nn.Module):

    def __init__(self, img_size):
        super(PawpularityModel, self).__init__()
        self.convolution_1 = nn.Sequential(
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
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.dense_metadata = nn.Sequential(
            nn.Linear(12, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024)
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(img_size[0] * img_size[1] * 2 + 1024, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 1)
        )

    def forward(self, X):
        image, metadata = X
        image, metadata = image.to(device), metadata.to(device)
        conv_out = self.convolution_1(image)
        conv_out = self.convolution_2(conv_out)
        conv_out = self.convolution_3(conv_out)
        image_flattened = conv_out.view(conv_out.size(0), -1)  # Flatten
        metadata_out = self.dense_metadata(metadata)
        dense_input = torch.cat((image_flattened, metadata_out), dim=1)
        out = self.dense_layers(dense_input)
        return out


def train(model, device, criterion, optimizer, batches, train_loader, test_loader, epochs):
    train_losses = []
    test_losses = []
    epochs = epochs
    t0 = datetime.now()
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = []
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            if (batch % 50) - 1 == 0:
                batch_t0 = datetime.now()
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if batch % 50 == 0:
                dt = datetime.now() - batch_t0
                print(f"""Processed batch {batch}/{int(np.ceil(batches))}, duration: {dt}""")

        train_losses.append(np.mean(train_loss))

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

        test_losses.append(np.mean(test_loss))
        dt = datetime.now() - t0

        print(f"""|| Epoch: {epoch + 1}/{epochs}
|| Train loss: {train_losses[-1]:.4f}
|| Test loss: {test_losses[-1]:.4f}
|| Total duration: {dt}""")

    plt.plot(train_losses, label="Train losses")
    plt.plot(test_losses, label="Test losses")
    plt.show()


def grade(model, device, batches, train_loader, test_loader):
    print("Starting grading...")
    with torch.no_grad():
        train_outputs = []
        train_targets = []
        model.train()
        batch = 0
        for inputs, targets in train_loader:
            batch += 1
            if batch % 20 == 0:
                print(f"Grading batch {batch}/{int(np.ceil(batches))}...")
            targets = targets.to(device)
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
            targets = targets.to(device)
            outputs = model(inputs).cpu().numpy().flatten().tolist()
            test_targets += targets.cpu().numpy().flatten().tolist()
            test_outputs += outputs

        test_outputs = np.array(test_outputs)
        test_targets = np.array(test_targets)
        test_RMSE = np.sqrt(((test_targets - test_outputs) ** 2).mean())
        print(f"Train RMSE: {train_RMSE:.4f}, Test RMSE: {test_RMSE:.4f}")


# Image size, preprocessing
img_size = (96, 96)
preprocess_images('train.csv', 'train', 'train-post', img_size)

# Data augmentation transforms
hFlip = transforms.RandomHorizontalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=45)

train_transforms = transforms.Compose([hFlip, rotate])

# Instantiating datasets
train_dataset = PawpularityDataset(csv='train.csv',
                                   img_dir='train-post',
                                   tr_test='train',
                                   transformations=train_transforms)

test_dataset = PawpularityDataset(csv='train.csv',
                                  img_dir='train-post',
                                  tr_test='test')

# Debug image showing
# im_arr = test_dataset.__getitem__(5)[0].numpy()
# im_arr = np.transpose(im_arr, (1, 2, 0))
# plt.imshow(im_arr)
# plt.show()

# Training parameters
batch_sz = 25
batches = train_dataset.__len__() / batch_sz

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)

model = PawpularityModel(img_size)
device = torch.device("cuda:0")
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

# Training run
train(model, device, criterion, optimizer, batches, train_loader, test_loader, epochs=20)

torch.save(model.state_dict(), 'model.pth')
print("Saved model.")

# Grading
grade(model, device, batches, train_loader, test_loader)

# print("Loading model...")
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# Create submissions
# with torch.no_grad():
#     preprocess_images('test.csv', 'test', 'test-post', img_size)
#     submission = pd.DataFrame(columns=['Id', 'Pawpularity'])
#     df = pd.read_csv('test.csv')
#     metadata_df = df.drop(columns=['Id'])
#     metadata = metadata_df.to_numpy(dtype="float32")
#     scaler = StandardScaler()
#     metadata = scaler.fit_transform(metadata)
#     indexes = df['Id']
#     ids = []
#     pawpularities = []
#     i = 0
#     for index in indexes:
#         ids.append(index)
#         image = read_image(os.path.join('test-post', index + ".jpg"))
#         image = image.type(torch.float32)
#         image = (image - torch.mean(image)) / torch.std(image)
#         image = image.reshape(1, 3, 240, 240)
#         md = metadata[i]
#         md = md.reshape(1, 12)
#         md = torch.from_numpy(md)
#         data = (image, md)
#         output = model(data).cpu().item()
#         pawpularities.append(output)
#         i += 1
#
#     submission['Id'] = ids
#     submission['Pawpularity'] = pawpularities
#     submission.to_csv('submission.csv', index=False)

# Profiling
# cProfile.run('train(model, device, criterion, optimizer, batches, train_loader, test_loader, epochs=1)', 'stats')
# p = pstats.Stats('stats')
# p.strip_dirs().sort_stats(-1).print_stats()
# p.sort_stats(SortKey.TIME).print_stats(10)
