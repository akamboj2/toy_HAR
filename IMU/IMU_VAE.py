#https://avandekleut.github.io/vae/ 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import IMUDataset
from train import MLP, IMU_CNN
import os

class IMU_DeCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_channels):
        super(IMU_DeCNN, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size//4*15),
        )
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(hidden_size//4, hidden_size//2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.ConvTranspose1d(hidden_size//2, hidden_size, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, output_channels, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels),
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.shape[0], self.hidden_size//4, -1)
        out = self.layers(out)
        out = out.permute(0,2,1) # permute to bs x timesteps x channels
        return out

class Decoder_DeConv(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder_DeConv, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 256)
        self.linear2 = nn.Linear(256, 512)
        self.DeCNN = IMU_DeCNN(input_size=512, hidden_size=512, output_channels=6)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        # print("Before DeCNN:", z.shape)
        z = self.DeCNN(z)
        # print("After DeCNN:", z.shape)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 180*6)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z
    
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(180*6, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x.float(), start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.fe = IMU_CNN(input_channels=6, hidden_size=256, output_size=786)
        self.linear1 = nn.Linear(786, 512)
        # self.linear1 = nn.Linear(180*6, 512)
        self.linear2 = nn.Linear(512, latent_dims) # for means
        self.linear3 = nn.Linear(512, latent_dims) # for stds

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1).float()
        x = self.fe(x.float())
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder_DeConv(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    



if __name__=='__main__':
    """ NOTE: This main function was just for debugging let's import and train this in train.py"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs=500 #1000
    latent_dims= 4 #512
    batch_size = 32 #lower bs seems to decrease mse loss and increase kl loss
    beta = 1/2000 #weight of kl divergence
    """
    total 370, 93, 277: 100,50, 32
    150: 100,10,32
    140, 95, 44: 100,8,32
    104, mse 93, kl 11: 100,2, 32
    """
     # Load dataset
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    # dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    train_dataset = IMUDataset(dir, dataset_name="UTD")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    vae = VariationalAutoencoder(latent_dims).to(device).float() # GPU
    # vae = Autoencoder(latent_dims).to(device).float() # GPU
    opt = optim.Adam(vae.parameters())
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device) # GPU
            x = F.normalize(x, dim=1) # prevents output x_hat from going to infinity
            opt.zero_grad()
            x_hat = vae(x)
            loss_mse = ((torch.flatten(x, start_dim=1).float() - torch.flatten(x_hat, start_dim=1))**2).sum()
            loss_kl =  vae.encoder.kl
            loss = loss_mse + beta*loss_kl
            loss.backward()
            opt.step()
            
            # exit()
        if epoch % 10 == 0:
            print("MSE Loss:", loss_mse.item(), "KL Loss:", loss_kl.item())
            # print("MSE Loss:", loss_mse.item())
            print(f"Epoch {epoch+1}/{epochs}")

    # Freeze VAE Model:
    for param in vae.parameters():
        param.requires_grad = False

    #Test VAE model on unseen data
    val_dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/val.txt"
    # val_dir = "/home/abhi/data/USC-HAD/splits/val.txt"
    val_dataset = IMUDataset(val_dir, time_invariance_test=False, dataset_name="UTD")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print("Testing VAE model")
    vae.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in val_loader:
            x = x.to(device) # GPU
            x = F.normalize(x, dim=1) # prevents output x_hat from going to infinity
            x_hat = vae(x)
            error = ((torch.abs(torch.flatten(x, start_dim=1).float() - torch.flatten(x_hat, start_dim=1)))).sum()
            print("Absolute Error:", error.item())
            
    # Now training har model:
    print("Training HAR model with VAE latent inputs")
    num_epochs = 100
    # latent_dims = 10 #given above
    learning_rate = 1e-3

    model = MLP(latent_dims,latent_dims*2, 27).to(device).float() # GPU
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
    datapath = "Inertial_splits/action_80_20_#1"
    val_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"val.txt")
    val_dataset = IMUDataset(val_dir, time_invariance_test=False)
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

    # Save model
    torch.save(model.state_dict(), 'model.pt')
    torch.save(vae.state_dict(), f'vae_{latent_dims}.pt')