import torch
import torch.nn as nn
import torch.optim as optim
from dataset import IMUDataset
import os
from argparse import ArgumentParser
import wandb
from tabulate import tabulate


actions_dict = {
    1: 'Swipe left',
    2: 'Swipe right',
    3: 'Wave',
    4: 'Clap',
    5: 'Throw',
    6: 'Arm cross',
    7: 'Basketball shoot',
    8: 'Draw X',
    9: 'Draw circle (clockwise)',
    10: 'Draw circle (counter clockwise)',
    11: 'Draw triangle',
    12: 'Bowling',
    13: 'Boxing',
    14: 'Baseball swing',
    15: 'Tennis swing',
    16: 'Arm curl',
    17: 'Tennis serve',
    18: 'Push',
    19: 'Knock',
    20: 'Catch',
    21: 'Pickup and throw',
    22: 'Jog',
    23: 'Walk',
    24: 'Sit to stand',
    25: 'Stand to sit',
    26: 'Lunge',
    27: 'Squat'
}


# Define the MLP model
class joint_IMU_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(joint_IMU_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.pid = IMU_MLP(input_size, hidden_size, output_size=8) #pid has 8 classes
        model_path = "./models/pid_best_model2048_68.7861.pt"
        self.pid.load_state_dict(torch.load(model_path))
        self.pid.layers[3] = nn.Identity() #remove the last linear layer, output should be bs x hidden_size/16
        # Freeze pid layers
        for param in self.pid.parameters():
            param.requires_grad = False

        self.action_features = MLP(input_size, hidden_size, hidden_size//4)

        self.action = IMU_MLP(hidden_size//4+hidden_size//16, hidden_size, output_size)
        #basically 2/3 contriubtion from inputs and 1/3 from pid (1/4/(1/4+1/16)=2/3)... shouldn't it be swapped? idk can test that

    def forward(self, x):
        pid_out = self.pid(x)
        action_features = self.action_features(x)
        out = self.action(torch.cat((pid_out, action_features), dim=1)) #cat on dim1 to keep batch size
        return out
    
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
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, label_category, joint):
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
                prefix = label_category+'Joint_best_model' if joint else label_category+'_best_model'
                if f.startswith(prefix):
                    best_val_acc = float(f.split("_")[3][:-3]) #get the accuracy from the filename, remove .pth in the end
                    best_val_file = os.path.join("./models",f)
            if acc > best_val_acc:
                if best_val_file: os.remove(best_val_file)
                best_val_acc = acc
                fname = f'./models/{label_category}Joint_best_model{model.hidden_size}_{acc:.4f}.pt' if joint else f'./models/{label_category}_best_model{model.hidden_size}_{acc:.4f}.pt'
                torch.save(model.state_dict(), fname)

# Define evaluation loop
def evaluate(model, val_loader,device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels, path in val_loader:
            outputs = model(inputs.to(device).type(torch.float32))
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            """ NOTE: For this chunk need to update dataset.py to return labels as well as path"""
            # Print incorrect predictions
            headers = ["Predicted, Actual, Path"]
            data = []
            if (predicted == labels).sum() != val_loader.batch_size:
                incorrect_indices = (predicted != labels).nonzero()[:,0]
                for i in incorrect_indices:
                    print("Predicted:", actions_dict[predicted[i].item()+1], ", Actual:", actions_dict[labels[i].item()+1])#, ", Path:", path[i])
                    # data.append([actions_dict[predicted[i].item()+1], actions_dict[labels[i].item()+1], path[i]])
            # #output to txt file
            # with open("incorrect_predictions.txt", "w") as f:
            #     f.write(tabulate(data, headers=headers))
            #     f.close()

        return 100 * correct / total

# Define the main function
def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    # parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--hidden_size', type=int, default=2048)
    args = parser.parse_args()

    # Set the hyperparameters (note most set in argparser abpve)
    label_category =  'action' # 'pid' or 'action'
    num_classes = 27 if label_category == 'action' else 8
    joint = True

    #MLP Specficic:
    input_size = 180*6
    hidden_size = args.hidden_size
    output_size = num_classes

    if not args.test:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="toy-HAR-IMU-"+label_category if not joint else "toy-HAR-IMU-joint-"+label_category,
            
            # track hyperparameters and run metadata
            config={
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'hidden_size': args.hidden_size
        })
        wandb.run.name = f"IMU_Joint{joint}_{label_category}_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.num_epochs}"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)
    print("Args: ", args)


    # Load the dataset
    datapath = "Inertial_splits/action_80_20_#1" if label_category == 'action' else "Inertial_splits/pid_80_20_#1"
    train_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"train.txt")
    val_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"val.txt")
    train_dataset = IMUDataset(train_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = IMUDataset(val_dir)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Define the model, loss function, and optimizer
    if joint:
        model = joint_IMU_MLP(input_size, hidden_size, output_size).to(device).float()
    else:
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
            prefix = label_category+'Joint_best_model' if joint else label_category+'_best_model'
            if m.startswith(prefix):
                model_path = os.path.join('./models/', m)
                break
        print("Evaluating model: ", model_path)
        model.load_state_dict(torch.load(model_path))
        acc = evaluate(model, val_loader, device)
        print('Test accuracy: {:.4f} %'.format(acc))
    else:
        train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, label_category=label_category, joint=joint)

if __name__ == '__main__':
    main()
