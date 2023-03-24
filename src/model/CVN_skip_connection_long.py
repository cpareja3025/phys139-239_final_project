import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py

from pathlib import Path
proj_dir = Path.cwd()
csv_dir = proj_dir.joinpath('csv_logs').resolve()
h5_dir = proj_dir.joinpath('data/hdf5').resolve()
model_dir = proj_dir.joinpath('models/cvn').resolve()

train_path = h5_dir.joinpath('train_norm.h5')
test_path = h5_dir.joinpath('test_norm.h5')
log_path = csv_dir.joinpath('ResNet2View_norm.csv')
best_path = model_dir.joinpath('cvn_skip_trainnorm_best.pt')
latest_path = model_dir.joinpath('cvn_skip_trainnorm_latest.pt')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class hdf5Dataset(Dataset):
    def __init__(self, h5_path, x_name, y_name):
        super().__init__()
        self.h5_path = h5_path
        self._data = None
        self.x_name = x_name
        self.y_name = y_name

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.h5_path, "r")
        return self._data

    def __getitem__(self, index):
        return self.data[self.x_name][index], self.data[self.y_name][index]

    def __len__(self):
        return len(self.data[self.x_name])

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x))


'''
class Inception_Block(nn.Module):
    def __init__(
            self,
            in_channels,
            output_1x1,
            output_1x1_block2,
            output_3x3,
            output_5x5_reduce,
            output_5x5,
            output_pool,
    ):
            super(Inception_Block, self).__init__()
            self.branch1 = ConvBlock(in_channels, output_1x1, kernel_size = 1)
            self.branch2 = nn.Sequential(
            ConvBlock(in_channels, out_channels = output_1x1_block2, kernel_size = 1),
            ConvBlock(output_1x1_block2, output_3x3, kernel_size = 3, padding = 1)
            )
            self.branch3 = nn.Sequential(
                  ConvBlock(in_channels, out_channels = output_5x5_reduce, kernel_size = 1),
                  ConvBlock(in_channels= output_5x5_reduce, out_channels=output_5x5, kernel_size = 5, padding = 2),
            )
            self.branch4 = nn.Sequential(
                  nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                  ConvBlock(in_channels, output_pool, kernel_size = 1)
            )
    def forward(self, x):
        first_block = self.branch1(x)
        second_block = self.branch2(x)
        third_block = self.branch3(x)
        fourth_block = self.branch4(x)
        output_concat = torch.cat([first_block, second_block, third_block, fourth_block], dim  = 1)

        return output_concat

'''


class Skip_Connection_Block(nn.Module):
    def __init__( self, in_channels ):

        super(Skip_Connection_Block, self).__init__()


        #  in_channels = 256
        self.through_connection = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels, out_channels = in_channels,
                kernel_size=3, padding='same' ),

            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d( in_channels = in_channels, out_channels = in_channels,
                      kernel_size=3, padding='same' ),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)
        )


    def forward(self, x):
        y = self.through_connection(x)
        output = torch.add(x, y)

        return output





class x_model(nn.Module):
    def __init__(self):
        super(x_model,self).__init__()
        self.conv_7x7 = nn.Conv2d(in_channels=1, out_channels=32,  kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride = 2)
        self.lrn_norm = nn.LocalResponseNorm(size=5,  alpha=0.0001, beta=0.75)
        self.conv_1x1 = nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=1)
        self.conv3_3x3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)


        self.fake_inception = Skip_Connection_Block( in_channels = 128 )

        # Shape here ....
        # self.inception3a = Inception_Block(
        #     in_channels=128,output_1x1=32, output_1x1_block2=48, output_3x3=54,
        #     output_5x5_reduce=8, output_5x5= 16, output_pool=16
        # )
        #
        # self.inception3b = Inception_Block(
        #     in_channels=128,output_1x1=32, output_1x1_block2=48, output_3x3=54,
        #     output_5x5_reduce=8, output_5x5= 16, output_pool=16
        # )
        #
        # self.inception4a = Inception_Block(
        #     in_channels=128,output_1x1=32, output_1x1_block2=48, output_3x3=54,
        #     output_5x5_reduce=8, output_5x5= 16, output_pool=16
        # )

        # self.linear = nn.Linear()


    def forward(self, x):
        x = self.max_pool(F.relu(self.conv_7x7(x)))
        x = self.lrn_norm(x)
        x = F.relu(self.conv_1x1(x))
        x = F.relu(self.conv3_3x3(x))
        x = self.max_pool(self.lrn_norm(x))


        # ================= Original CVN ===================
        # x = self.inception3a(x)     #
        # x = self.inception3b(x)     #
        # x = self.max_pool(x)        #
        # x = self.inception4a(x)     #


        # =========== Modified CVN: Skip Connection =============
        x = self.fake_inception(x)
        x = self.fake_inception(x)
        x = self.max_pool(x)
        x = self.fake_inception(x)

        return x


class combineXY(nn.Module):
    def __init__(self):
        super(combineXY, self).__init__()
        self.x_model = x_model()
        self.y_model = x_model()
        # concatenating both models gives us channels of 512

        # self.final_inception = Inception_Block(
        #     in_channels=512,output_1x1=128, output_1x1_block2=192,
        #     output_3x3=256, output_5x5_reduce=32, output_5x5= 64, output_pool=64
        # )

        self.final_fake_inception = Skip_Connection_Block( in_channels = 256 )


        self.avg_pooling = nn.AvgPool2d(kernel_size=(6,5))
        self.linear = nn.Linear(1024, 5)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, data):
        print(f'data shape: {data.shape}')
        split = torch.tensor_split(data, 2, dim = 1)
        x_data = split[0]
        y_data = split[1]
        print(f'x data shape: {x_data.shape}')
        print(f'y data shape: {y_data.shape}')
        x = self.x_model(x_data)
        y = self.y_model(y_data)
        print(f'x data shape: {x.shape}')
        print(f'y data shape: {y.shape}')
        concat =  torch.cat([x, y], dim  = 1)
        print(f'concat data shape: {concat.shape}')
        # combined_data = self.final_inception(concat)

        combined_data = self.final_fake_inception(concat)


        combined_data = self.avg_pooling(combined_data)
        print(f'output shape after pooling {combined_data.shape}')
        combined_data = combined_data.reshape(combined_data.shape[0],-1)
        print(f'output after reshape {combined_data.shape}')
        combined_data = self.linear(combined_data)
        # combined_data = self.softmax(combined_data)
        return combined_data



epochs = 500
batch_size = 64
learning_rate = 0.0005
dataset_train = hdf5Dataset(train_path, "X_train", "y_train")
dataset_test = hdf5Dataset(test_path, "X_test", "y_test")
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
n_total_steps = len(train_loader)

combined_model = combineXY().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)

from pytorchsummary import summary
summary(combined_model, input_size=(3, 256, 256))

best = 100000

f = open(log_path, "w+")
f.write("epoch,loss,accuracy,val_loss,val_acc\n")
f.close()

for epoch in range(epochs):
    running_loss = 0.0
    correct = 0.0
    n_samples = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward Pass

        outputs = combined_model(images[:, :2, :, :])
#         target = target.long()
#         labels = labels.long
        loss = criterion(outputs, torch.max(labels, 1)[1])

        #Backward and optimize

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        outputs = torch.softmax(outputs,1)
        preds = torch.argmax(outputs, dim=1)
        truths = torch.argmax(labels, dim=1)

        n_samples += truths.size(0)
        correct += (preds == truths).sum().item()
        # correct += (outputs == labels).float().sum()
        print(f'{epoch}')

    epoch_loss = running_loss / len(train_loader)
    torch.save(combined_model.state_dict(), latest_path)
    if(epoch_loss < best):
        best = epoch_loss
        torch.save(combined_model.state_dict(), best_path)
    epoch_accuracy = 100.0 * correct / n_samples
    print(f"{epoch},{epoch_loss},{epoch_accuracy}\n")

    running_loss = 0.0
    correct = 0.0
    n_samples = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = combined_model(images[:, :2, :, :])
        loss = criterion(outputs, torch.max(labels, 1)[1])

        running_loss += loss.item() * images.size(0)
        outputs = torch.softmax(outputs,1)
        preds = torch.argmax(outputs, dim=1)
        truths = torch.argmax(labels, dim=1)

        n_samples += labels.size(0)
        correct += (preds == truths).sum().item()
    val_loss = running_loss / len(test_loader)
    val_accuracy = 100.0 * correct / n_samples

    f = open(log_path, "a")
    print(f"{epoch},{epoch_loss},{epoch_accuracy},{val_loss},{val_accuracy}\n")
    f.write(f"{epoch},{epoch_loss},{epoch_accuracy},{val_loss},{val_accuracy}\n")
    f.close()
