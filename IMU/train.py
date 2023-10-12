import torch
import torch.nn as nn
import torch.optim as optim
from dataset import IMUDataset
import os
from argparse import ArgumentParser
import wandb

# Define the MLP model
class IMU_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IMU_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            MLP(input_size, hidden_size, int(hidden_size/4)),
            nn.Dropout(0.5),
            MLP(int(hidden_size/4), int(hidden_size/4), int(hidden_size/16)),
            nn.Linear(int(hidden_size/16), output_size, dtype=torch.float32)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.layers(x)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float32),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, dtype=torch.float32),
            nn.BatchNorm1d(output_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.layers(x)
        return out

# Define the training loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, label_category):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device).type(torch.float32))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        
        # Log the loss to wandb
        wandb.log({'train_loss': running_loss / len(train_loader)})

        if (epoch+1) % 2 == 0:
            acc = evaluate(model, val_loader, device)
            print('Test accuracy: {:.4f} %'.format(acc))
            wandb.log({'val_acc': acc})
            best_val_acc=0
            best_val_file= None
            for f in os.listdir('./models/'):
                if f.startswith(label_category+'_best_model'):
                    best_val_acc = float(f.split("_")[3][:-3]) #get the accuracy from the filename, remove .pth in the end
                    best_val_file = os.path.join("./models",f)
            if acc > best_val_acc:
                if best_val_file: os.remove(best_val_file)
                best_val_acc = acc
                torch.save(model.state_dict(), f'./models/{label_category}_best_model{model.hidden_size}_{acc:.4f}.pt')

# Define evaluation loop
def evaluate(model, val_loader,device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device).type(torch.float32))
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        return 100 * correct / total

# Define the main function
def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    # parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--hidden_size', type=int, default=1024)
    args = parser.parse_args()

    # Set the hyperparameters (note most set in argparser abpve)
    batch_size = 16
    label_category =  'pid' # or 'action'
    num_classes = 27 if label_category == 'action' else 8

    #MLP Specficic:
    input_size = 180*6
    hidden_size = args.hidden_size
    output_size = num_classes

    if not args.test:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="toy-HAR-IMU-"+label_category,
            
            # track hyperparameters and run metadata
            config={
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'hidden_size': args.hidden_size
        })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)


    # Load the dataset
    datapath = "Inertial_splits/action_80_20_#1" if label_category == 'action' else "Inertial_splits/pid_80_20_#1"
    train_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"train.txt")
    val_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"val.txt")
    train_dataset = IMUDataset(train_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = IMUDataset(val_dir)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Define the model, loss function, and optimizer
    model = IMU_MLP(input_size, hidden_size, output_size).to(device).float()
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)



    # Test or train the model
    if args.test:
        models = os.listdir('./models/')
        for m in models:
            if m.startswith(label_category+'_best_model'):
                model_path = os.path.join('./models/', models[0])
                break
        print("Evaluating model: ", model_path)
        model.load_state_dict(torch.load(model_path))
        acc = evaluate(model, val_loader, device)
        print('Test accuracy: {:.4f} %'.format(acc))
    else:
        train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, label_category=label_category)

if __name__ == '__main__':
    main()
