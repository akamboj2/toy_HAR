import torch
import torch.nn as nn
import torch.optim as optim
from dataset import IMUDataset
import os
from argparse import ArgumentParser
import wandb
from tabulate import tabulate

from IMU_ViT import ViT
#    2 Args:  Namespace(num_epochs=267, batch_size=32, learning_rate=0.04962918927073625, optimizer='SGD', test=False, hidden_size=128, model_type='ViT', num_layers=18, num_heads=8)
# python train.py --batch_size=32 --num_epochs=267 --learning_rate=0.04962918927073625 --optimizer='SGD' --hidden_size=128 --model_type='ViT' --num_layers=18 --num_heads=8

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

#Define 1D CNN model
class IMU_CNN(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super(IMU_CNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(hidden_size, hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(hidden_size//2, hidden_size//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        # assuming starting dim is 6x180 output should be hidden_size/4 x 15

        self.fc = nn.Sequential(
            nn.Linear(hidden_size//4*15, output_size, dtype=torch.float32),
        )

    def forward(self, x):
        x = x.permute(0,2,1) # permute to bs x channels x timesteps
        out = self.layers(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

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
        # NOTE THIS CORRESPONDS TO THE BEST MODEL
        # model_path = "/home/abhi/research/action_recognition/toy_HAR/IMU/models/best_saves/action_best_model1024_91.3295.pt"
        # self.layers = nn.Sequential(
        #     MLP(input_size, hidden_size, 256),
        #     nn.Dropout(0.5),
        #     MLP(256, 128, 64),
        #     nn.Linear(64, output_size, dtype=torch.float32)
        # )
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
                prefix = label_category+f'Joint_best_model{MODEL_TYPE}' if joint else label_category+f'_best_model{MODEL_TYPE}'
                if f.startswith(prefix):
                    best_val_acc = float(f.split("_")[3][:-3]) #get the accuracy from the filename, remove .pth in the end
                    best_val_file = os.path.join("./models",f)
            if acc > best_val_acc:
                if best_val_file: os.remove(best_val_file)
                best_val_acc = acc
                fname = f'./models/{label_category}Joint_best_model{model.hidden_size}_{acc:.4f}.pt' if joint else f'./models/{label_category}_best_model{MODEL_TYPE}{model.hidden_size}_{acc:.4f}.pt'
                torch.save(model.state_dict(), fname)

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

            """ NOTE: For this chunk need to update dataset.py to return labels as well as path"""
            # Print incorrect predictions
            # headers = ["Predicted, Actual, Path"]
            # data = []
            # if (predicted == labels).sum() != val_loader.batch_size:
            #     incorrect_indices = (predicted != labels).nonzero()[:,0]
            #     for i in incorrect_indices:
            #         print("Predicted:", actions_dict[predicted[i].item()+1], ", Actual:", actions_dict[labels[i].item()+1])#, ", Path:", path[i])
            #         # data.append([actions_dict[predicted[i].item()+1], actions_dict[labels[i].item()+1], path[i]])
            # #output to txt file
            # with open("incorrect_predictions.txt", "w") as f:
            #     f.write(tabulate(data, headers=headers))
            #     f.close()

        return 100 * correct / total

#GLOBAL VARIABLE to help with checkpoints saving and loading
MODEL_TYPE = "ViT" #   "CNN" or "ViT" or "" (for MLP)
# Define the main function
def main():
    global MODEL_TYPE
    # Parse command-line arguments
    parser = ArgumentParser()
    # parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='ViT')
    parser.add_argument('--num_layers', type=int, default=7) 
    parser.add_argument('--num_heads', type=int, default=12) #note, needs to evenly divide 768 (default embedding/hidden size for ViT): 2,3,4,6,8,12,16,24
    args = parser.parse_args()

    # Set the hyperparameters (note most set in argparser abpve)
    label_category =  'action' # 'pid' or 'action'
    num_classes = 27 if label_category == 'action' else 8
    joint = False

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
        wandb.run.name = f"IMU_{MODEL_TYPE}_Joint{joint}_{label_category}_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.num_epochs}"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)
    print("Args: ", args)


    # Load the dataset
    # datapath = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1" if label_category == 'action' else "/home/abhi/data/utd-mhad/Inertial_splits/pid_80_20_#1"
    datapath = "/home/abhi/data/USC-HAD/splits"
    train_dir = os.path.join(datapath,"train.txt")
    val_dir = os.path.join(datapath,"val.txt")
    train_dataset = IMUDataset(train_dir, dataset_name="USC")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = IMUDataset(val_dir, time_invariance_test=False, dataset_name="USC")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Define the model, loss function, and optimizer
    if joint:
        model = joint_IMU_MLP(input_size, hidden_size, output_size).to(device).float()
    else:
        if MODEL_TYPE=="CNN":
            model = IMU_CNN(6, hidden_size, output_size).to(device).float()
        elif MODEL_TYPE=="ViT":
            # NOTE: "embed_dim (Hidden_size) must be divisible by num_heads"
            model = ViT(num_classes=output_size, hidden_size = 768, num_layers=args.num_layers, num_heads=args.num_heads).to(device).float()
        else:
            model = IMU_MLP(input_size, hidden_size, output_size).to(device).float()
        
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)



    # Test or train the model
    if args.test:
        print("Testing model: ", model) 
 
        models = os.listdir('./models/')
        for m in models:
            prefix = label_category+f'Joint_best_model{MODEL_TYPE}' if joint else label_category+f'_best_model{MODEL_TYPE}'
            if m.startswith(prefix):
                model_path = os.path.join('./models/', m)
                break

        # model_path = "/home/abhi/research/action_recognition/toy_HAR/IMU/models/best_saves/action_best_model1024_91.3295.pt"
        print("Evaluating model: ", model_path)

        # # Load the state dictionary from the file
        # state_dict = torch.load(model_path)
        # # Print the keys in the state dictionary
        # print(state_dict.keys())   

        model.load_state_dict(torch.load(model_path))
        acc = evaluate(model, val_loader, device)
        print('Test accuracy: {:.4f} %'.format(acc))
    else:
        train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, label_category=label_category, joint=joint)

if __name__ == '__main__':
    main()
