import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

hep_x_data = torch.randn(1,1, 256, 256)
hep_y_data = torch.randn(1,1, 256, 256)

N = 320
epochs = 5
batch_size = 32
train_arr_x = np.random.rand(N,2,256,256)
train_arr_y = np.random.rand(N, 5)
tensor_x = torch.Tensor(train_arr_x)
tensor_y = torch.Tensor(train_arr_y)
dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
learning_rate = 0.001
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return F.relu(self.conv(x))
    
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

class x_model(nn.Module):
    def __init__(self):
        super(x_model,self).__init__()
        self.conv_7x7 = nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride = 2)        
        self.lrn_norm = nn.LocalResponseNorm(size=5,  alpha=0.0001, beta=0.75)
        self.conv_1x1 = nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=1)
        self.conv3_3x3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.inception3a = Inception_Block(in_channels=256,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        self.inception3b = Inception_Block(in_channels=256,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        self.inception4a = Inception_Block(in_channels=256,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        # self.linear = nn.Linear()
        

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv_7x7(x)))
        x = self.lrn_norm(x)
        x = F.relu(self.conv_1x1(x))
        x = F.relu(self.conv3_3x3(x))
        x = self.max_pool(self.lrn_norm(x))
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.max_pool(x)
        x = self.inception4a(x)
        return x
    
class combineXY(nn.Module):
    def __init__(self):
        super(combineXY, self).__init__()
        self.x_model = x_model()
        self.y_model = x_model()
        # concatenating both models gives us channels of 512
        self.final_inception = Inception_Block(in_channels=512,output_1x1=128, output_1x1_block2=192, output_3x3=256, output_5x5_reduce=32, output_5x5= 64, output_pool=64)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(6,5))
        self.linear = nn.Linear(2048, 5)
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
        combined_data = self.final_inception(concat)
        combined_data = self.avg_pooling(combined_data)
        print(f'output shape after pooling {combined_data.shape}')
        combined_data = combined_data.reshape(combined_data.shape[0],-1)
        print(f'output after reshape {combined_data.shape}')
        combined_data = self.linear(combined_data)
        # combined_data = self.softmax(combined_data)
        return combined_data


combined_model = combineXY()
# output_combined = combined_model(hep_x_data, hep_y_data)

# print(f'Final Shape of both x and y models: {output_combined.shape}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(combined_model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
images, labels = next(iter(train_loader))
outputs = combined_model(images)
print(outputs.shape)

for i, (images, labels) in enumerate(train_loader):
    images = images
    labels = labels
        # Forward Pass
    outputs = combined_model(images)
    loss = criterion(outputs, labels)

        #Backward and optimize

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
