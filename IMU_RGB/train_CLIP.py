"""
This file runs CLIP style training to align IMU and RGB data representations
"""
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import RGB_IMU_Dataset
import os
from argparse import ArgumentParser
import wandb
from tabulate import tabulate
from torchvision.models import resnet18, ResNet18_Weights
from dataset import RGB_IMU_Dataset
import torchvision.transforms as transforms

# From rgb train.py

# Define the 3D CNN model 
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes, video_length=16):
        super(ActionRecognitionModel, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-4])
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.fc_block = nn.Sequential(
            nn.Linear(int(video_length/2)*64*14*14, 512), # 64C,video_len/2 T,28H,28W
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        # self.residual_block = nn.Sequential(
        #     nn.Linear(video_length*128*28*28,int(video_length/2)*64*14*14),
        #     nn.ReLU()
        # )

    def forward(self, x):
         # Apply ResNet to each frame
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.feature_extractor(x)
        post_fe_x = x.clone()

        #insert time dim after b
        x = x.view(b, t, *x.shape[1:])

        x = x.permute(0,2,1,3,4) # permute to BCTHW bc conv3d expects C before T

        # Apply 3D convolutional layers
        x = self.conv3d_block(x)
        # print("after conv3d",x.shape)
        x = x.view(b,-1) # flatten preserve batch_size
        # print(x.shape)

        #residual connection
        post_fe_x = post_fe_x.view(b,-1).chunk(16,dim=1)
        # print("post_fe_x shape",len(post_fe_x),post_fe_x[0].shape)
        chunks = torch.stack(post_fe_x,dim=1)
        # print("chunks shape",chunks.shape)
        summed_chunks = chunks.sum(dim=1)
        # print("summed chunks shape",summed_chunks.shape)
        x = x+summed_chunks #maybe this will help with gradient propogation


        #Fully connected for class prediction
        x = self.fc_block(x)
        return x
    


def main():
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)

    FE_rgb = ActionRecognitionModel(num_classes=128, video_length = 16).to(device)
    FE_imu = 
    for epoch in range(num_epochs):
        

if __name__ == '__main__':
    main()

