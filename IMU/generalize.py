import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from IMU_VAE import VariationalAutoencoder
from dataset import IMUDataset
from train import MLP

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    # Load model
    vae = VariationalAutoencoder(256).to(device).float() # GPU
    vae.load_state_dict(torch.load("vae_256.pt"))

    # Freeze VAE Model:
    for param in vae.parameters():
        param.requires_grad = False

    # Load dataset
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    train_dataset = IMUDataset(dir,dataset_name="USC")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Now training har model:
    print("Training HAR model with VAE latent inputs")
    num_epochs = 100
    # latent_dims = 10 #given above
    learning_rate = 1e-3

    latent_dims=256
    # model = nn.Linear(256, 27).to(device)
    model = MLP(latent_dims,latent_dims*2, 27).to(device).float() # GPU
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = vae.encoder(F.normalize(inputs, dim=1).to(device))
            outputs = model(inputs.to(device).type(torch.float32))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        

    # Test model
    # datapath = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1"
    datapath = "/home/abhi/data/USC-HAD/splits"
    val_dir = os.path.join(datapath,"val.txt")
    val_dataset = IMUDataset(val_dir, time_invariance_test=False, dataset_name="USC")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("Testing model")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs = vae.encoder(F.normalize(inputs, dim=1).to(device))
            outputs = model(inputs.to(device).type(torch.float32))
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print("Accuracy: ", 100*correct.item()/total, "%")