"""
NOTE: For ViT to work with imu i had to copy the vision_transformer.py file from the pytorch library and make some changes to it.
Thus i import it as custom_vision_transformer.py
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import IMUDataset
# from torchvision.models import vit_b_16
from custom_vision_transformer import VisionTransformer

class ViT(nn.Module):
    def __init__(self, num_classes, hidden_size=768, num_layers=12, num_heads=12):
        super(ViT, self).__init__()
        self.hidden_size = hidden_size
        self.model = VisionTransformer(
        image_size=180,
        patch_size=6,
        num_layers=num_layers, 
        num_heads=num_heads, 
        hidden_dim=hidden_size,
        mlp_dim=1024, # 3072
        num_classes=num_classes,
        )
        # self.model.conv_proj = nn.Conv1d(6, 768, kernel_size=(6), stride=(6)) # looking at patches of 6 times steps at 50 hz, so 0.12 seconds   
        # self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes, bias=True)

    def forward(self, x):
        # Maybe do some convolutions before feeding into ViT
        # https://arxiv.org/pdf/2106.14881.pdf

        x = x.permute(0,2,1) # permute to bs x channels x timesteps
        x = self.model(x)
        return x

if __name__ == '__main__':
    """ NOTE: This main function was just for debugging let's import and train this in train.py"""
    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Define transforms
    transforms = Compose([])

    # Load dataset
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    train_dataset = IMUDataset(dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = ViT(num_classes=27)
    print(model.model.conv_proj)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            print("inputs.shape:", inputs.shape)
            # Forward pass
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print statistics
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
