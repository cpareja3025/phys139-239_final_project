import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

hep_x_data = torch.randn(1,1, 256, 256)
hep_y_data = torch.randn(1,1, 256, 256)

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

class xy_model(nn.Module):
    def __init__(self):
        super(xy_model,self).__init__()
        self.conv_7x7 = nn.Conv2d(in_channels=1, out_channels=6,  kernel_size=7, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride = 2)        
        self.lrn_norm = nn.LocalResponseNorm(size=5,  alpha=0.0001, beta=0.75)
        self.conv_1x1 = nn.Conv2d(in_channels=6, out_channels= 12, kernel_size=1)
        self.conv3_3x3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3)
        self.inception3a = Inception_Block(in_channels=6,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        self.inception3b = Inception_Block(in_channels=256,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        self.inception4a = Inception_Block(in_channels=256,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        

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
        self.x_model = xy_model()
        self.y_model = xy_model()
        # concatenating both models gives us channels of 512
        self.final_inception = Inception_Block(in_channels=512,output_1x1=64, output_1x1_block2=96, output_3x3=128, output_5x5_reduce=16, output_5x5= 32, output_pool=32)
        self.avg_pooling = nn.AvgPool2d(kernel_size=(6,5))
        # self.softmax = nn.Softmax(dim=2)
    def forward(self, x_data, y_data):
        x = self.x_model(x_data)
        y = self.y_model(y_data)
        concat =  torch.cat([x, y], dim  = 1)
        combined_data = self.final_inception(concat)
        combined_data = self.avg_pooling(combined_data)
        #combined_data = self.softmax(combined_data)
        return combined_data


combined_model = combineXY()
output_combined = combined_model(hep_x_data, hep_y_data)

print(f'Final Shape of both x and y models: {output_combined.shape}')
         