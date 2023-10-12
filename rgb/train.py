import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import CustomVideoDataset  # Import your custom dataset class
from tqdm import tqdm
import os
from torchvision.models import resnet18, ResNet18_Weights
# import torchvideo.transforms as VT
import wandb
from timm.models.vision_transformer import vit_small_patch32_224
from argparse import ArgumentParser

# Parse command-line arguments
parser = ArgumentParser()
# parser.add_argument('--sweep', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=.001)
args = parser.parse_args()

# Define hyperparameters (note sweep overrides them)
video_length = 16
# below are in confiug
# batch_size = 1
# num_epochs = 50
# optimizer = 'Adam'
# learning_rate=.001

# Define the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)
# torch.cuda.empty_cache()



# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="toy-HAR",
    
    # track hyperparameters and run metadata
    config={
    'num_epochs': args.num_epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'optimizer': args.optimizer
}
)
config = wandb.config

# Define hyperparameters
num_epochs = config.num_epochs
batch_size = config.batch_size
learning_rate = config.learning_rate
optimizer = config.optimizer



# Define the custom dataset and data loader
# https://torchvideo.readthedocs.io/en/latest/transforms.html
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize frames
    transforms.ToTensor(),           # Convert frames to tensors
])

train_dataset = CustomVideoDataset(root_dir="/home/abhi/data/utd-mhad/train.txt", video_length=video_length, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomVideoDataset(root_dir="/home/abhi/data/utd-mhad/val.txt", video_length=video_length, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Define the 3D CNN model 
class BasicActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.vit = vit_small_patch32_224(pretrained=True)
        self.conv_block = nn.Sequential(
            nn.Conv3d(1000, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.fc_block = nn.Sequential(
            nn.Linear(int(video_length/2)*64*28*28, 512), # 64C,video_len/2 T,28H,28W
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Apply ViT to each frame
        b, t, c, h, w = x.shape
        print("Shape before view:",x.shape)
        x = x.view(-1, c, h, w)
        print("Shape after view:",x.shape)
        x = self.vit(x)
        print("Shape after vit:",x.shape)
        x = x.view(b, t, -1)
        print("Shape after view:",x.shape)

        # Apply 3D convolutional layers
        x = x.unsqueeze(2)
        print("Shape after unsqueeze:",x.shape)
        x = self.conv_block(x)
        print("Shape after conv:",x.shape)
        x = x.view(b, -1)
        print("Shape after view:",x.shape)
        
        # Apply fully connected layers
        x = self.fc_block(x)
        print("Shape after fc:",x.shape)
        return x
    

# Define the 3D CNN model 
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
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
        y = x.clone()

        # print(x.shape)
        # print(x.view(1,-1).shape)
        #insert time dim after b
        x = x.view(b, t, *x.shape[1:])

        x = x.permute(0,2,1,3,4) # permute to BCTHW bc conv3d expects C before T

        # Apply 3D convolutional layers
        x = self.conv3d_block(x)
        # print("after conv3d",x.shape)
        x = x.view(batch_size,-1) # flatten preserve batch_size
        # print(x.shape)

        #residual connection
        y = y.view(batch_size,-1).chunk(16,dim=1)
        # print("y shape",len(y),y[0].shape)
        chunks = torch.stack(y,dim=1)
        # print("chunks shape",chunks.shape)
        summed_chunks = chunks.sum(dim=1)
        # print("summed chunks shape",summed_chunks.shape)
        x = x+summed_chunks #maybe this will help with gradient propogation


        #Fully connected for class prediction
        x = self.fc_block(x)
        return x
    

model = ActionRecognitionModel(num_classes=27).to(device)  # Replace 27 with the number of action classes
# print(model)

# Define the model, loss function, and optimizer
model = ActionRecognitionModel(num_classes=27).to(device)
criterion = nn.CrossEntropyLoss()
if optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Define a variable to keep track of the highest validation accuracy achieved
best_val_acc = 0.0

if __name__ == '__main__':
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs.cpu(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print the average loss for this epoch
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss / len(train_loader)}')
        # Log the loss to wandb
        wandb.log({'train_loss': running_loss / len(train_loader)})

        if (epoch+1)%2 == 0:
            model.eval()
            total_acc=0.
            for val_inputs,val_labels in tqdm(val_loader):
                out = model(val_inputs.to(device))
                pred = out.cpu().type(torch.int).argmax(dim=-1)
                acc = sum(torch.eq(val_labels,pred))/float(len(val_labels))
                total_acc+=acc
            total_acc/=len(val_loader)
            print(f'Val on [{epoch+1}] Acc: {total_acc}')
            wandb.log({'val_acc': total_acc})

            # Check if this is the best validation accuracy achieved so far
            if total_acc > best_val_acc:
                best_val_acc = total_acc
                # Save the model state with the best validation accuracy
                checkpoint_path = f'rgb/best_checkpoint_FE_{best_val_acc:.2f}.pt'
                torch.save(model.state_dict(), checkpoint_path)

    print(f'Training finished, best validation accuracy: {best_val_acc:.2f}, saved model checkpoint: {checkpoint_path}')



"""
Notes to self:
1) Edit module to use sequential blocks
2) Insert a vit model or resnet18 backbone on each image
"""
