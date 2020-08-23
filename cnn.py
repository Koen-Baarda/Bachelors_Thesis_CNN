import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL.Image as Image
import cv2
import glob, os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score as js
import random


class ImageDataset(Dataset):

    def __init__(self, csv_file, root_dir):
        self.root_dir = root_dir
        self.frame = pd.read_csv(csv_file)
        self.transform = TF.Compose([TF.RandomHorizontalFlip(), TF.RandomVerticalFlip()])
        self.totensor = TF.ToTensor()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        img_name = str(self.root_dir) + str(self.frame.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = Image.fromarray(image, 'RGB')
        padded_image = Image.new('RGB', (1053, 795), (0, 0, 0))
        padded_image.paste(image, image.getbbox())

        seg_name = str(self.root_dir) + str(self.frame.iloc[idx, 2])
        segmentation = io.imread(seg_name)
        if len(segmentation.shape) > 2 and segmentation.shape[2] == 4:
            segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGRA2GRAY)

        segmentation = Image.fromarray(segmentation)
        padded_segmentation = Image.new('1', (1053, 795), 0)
        padded_segmentation.paste(segmentation, (0, 0))

        IR_name = str(self.root_dir) + str(self.frame.iloc[idx, 1])
        IR = io.imread(IR_name)
        IR = cv2.cvtColor(IR, cv2.COLOR_BGRA2GRAY)
        if len(IR.shape) > 2 and IR.shape[2] == 4:
            IR = cv2.cvtColor(IR, cv2.COLOR_BGRA2GRAY)

        IR = Image.fromarray(IR, '1')
        padded_IR = Image.new('1', (1053, 795), 0)
        padded_IR.paste(IR, (0, 0))

        if random.randint(0, 1) > 0.5:
            padded_image = TFF.hflip(padded_image)
            padded_segmentation = TFF.hflip(padded_segmentation)
            padded_IR = TFF.hflip(padded_IR)

        if random.randint(0, 1) > 0.5:
            padded_image = TFF.vflip(padded_image)
            padded_segmentation = TFF.vflip(padded_segmentation)
            padded_IR = TFF.vflip(padded_IR)

        sample = {'image': self.totensor(padded_image), 'segmentation': self.totensor(padded_segmentation),
                  'IR': self.totensor(padded_IR)}

        return sample


testset = ImageDataset(csv_file="test.csv", root_dir="")

trainset = ImageDataset(csv_file="train.csv", root_dir="")

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=1, shuffle=False)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(dilation) for dilation in [1, 2, 4, 8, 16]])
        self.conv2 = nn.Conv2d(128, 160, 1)
        self.conv3 = nn.Conv2d(160, 1, 1)

    def forward(self, x, z):
        _h = [self.conv1(x)]
        h = []
        for i in range(0, (len(self.blocks) - 1)):
            _h = self.blocks[i](_h[0], z)
            h += [F.relu(_h[1])]
        h = self.conv2(torch.cat(h, 1))
        return self.conv3(F.relu(h))


class ResidualBlock(nn.Module):

    def __init__(self, dilation):
        super().__init__()
        self.b1 = nn.Conv2d(32, 64, 3, dilation=dilation, padding=dilation)
        self.b2 = nn.Conv2d(1, 64, 3, padding=1)
        self.b3 = nn.Conv2d(32, 64, 1)

    def forward(self, x, z):
        h = [self.b1(x), self.b2(z)]
        h = torch.chunk(h[0], 2, 1) + torch.chunk(h[1], 2, 1)
        h = torch.chunk(self.b3(torch.sigmoid(h[0] + h[2]) * torch.tanh(h[1] + h[3])), 2, 1)

        return h[0] + x, h[1]


posweight = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net()
model.to(device)
weight = torch.tensor([posweight])

loss = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
loss.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
running_loss = 0.0
loss_matrix = np.zeros(0)
loss_index = np.zeros(0)

for epoch in range(20):
    for i, data in enumerate(train_loader):
        x, y, z = data['image'].to(device), data['segmentation'].to(device), data['IR'].to(device)
        print(epoch, ": ", i)

        x = model.forward(x, z)
        output = loss(x, y)
        print(output)

        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += output.item()
        if i % 50 == 49:
            loss_matrix = np.append(loss_matrix, running_loss / 50)

            loss_index = np.append(loss_index, epoch * len(train_loader) + i)

            running_loss = 0.0


print(np.average(loss_matrix[-4:]))
plt.plot(loss_index, loss_matrix)
plt.savefig('loss.png')

ious = []
for i, data in enumerate(test_loader):
    x, y, z = data['image'].to(device), data['segmentation'].to(device), data['IR'].to(device)
    x = model.forward(x, z)
    s = x
    s = torch.sigmoid(s)
    s = s.cpu().detach().numpy()
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    x = x[0, 0, :, :]
    y = y[0, 0, :, :]
    s = s[0, 0, :, :]
    thresholded = np.where(x > 0.5, 1, 0)
    ious = np.append(ious, js(y.astype(bool), thresholded.astype(bool), average='micro'))
    if 0 < ious[i] < 0.99:
        im = Image.fromarray((y * 255).astype(np.uint8))
        im = im.convert("L")
        im.save("original" + str(i) + ".jpeg")

        im = Image.fromarray((s * 255).astype(np.uint8))
        im = im.convert("L")
        im.save("newgraysig" + str(i) + ".jpeg")

        im = Image.fromarray((thresholded * 255).astype(np.uint8))
        im = im.convert("L")
        im.save("new" + str(i) + ".jpeg")


print("average: ", np.average(ious))

torch.save(model, "/home/k_baarda/model.pt")
