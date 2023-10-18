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

# from models import Early_Fusion, Middle_Fusion, Late_Fusion, IMU_MLP, joint_IMU_MLP
# from train_utils import train, evaluate


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

#Fusion is adding
class Middle_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rgb_video_length):
        super(Middle_Fusion, self).__init__()
        self.hidden_size = hidden_size #here hiddent size will be the size the two features join at (addition)
        self.rgb_model = RGB_Action(hidden_size, rgb_video_length)
        self.imu_model = IMU_MLP(input_size, hidden_size*2, hidden_size)
        self.joint_processing = IMU_MLP(hidden_size, hidden_size//2, output_size)

    def forward(self, x):
        x_rgb = self.rgb_model(x[0])
        x_imu = self.imu_model(x[1])
        x_sum = x_rgb+x_imu
        out = self.joint_processing(x_sum)
        return out

class Early_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rgb_video_length):
        super(Early_Fusion, self).__init__()
        self.hidden_size = hidden_size #here hidden size will be the size the two models squentially join at
        self.rgb_model = RGB_Action(hidden_size, rgb_video_length) #here we use hiddent size to connect rgb and imu into one big model
        self.imu_model = IMU_MLP(hidden_size, hidden_size//2, output_size)
        

    def forward(self, x):
        # Just flatten, add the data, and unflatten
        shape = x[0].shape
        x0flat = x[0].flatten() #rgb
        x1flat = x[1].flatten() #imu
        if len(x0flat) > len(x1flat):
            padding = torch.zeros(x0flat.shape[0] - x1flat.shape[0], dtype=x1flat.dtype, device=x1flat.device)
            x1flat = torch.cat((x1flat, padding))
        else:
            raise ValueError("RGB is bigger than IMU data, early fusion failed")
        x_sum = x0flat + x1flat
        x_sum = x_sum.view(shape)
        
        # For simplicity/reusability right now the full model is just the rgb followed by the IMU
        y_rgb = self.rgb_model(x_sum)
        out = self.imu_model(y_rgb)
        return out
    
class Late_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rgb_video_length):
        super(Late_Fusion, self).__init__()
        self.hidden_size = hidden_size
        self.rgb_model = RGB_Action(output_size, rgb_video_length)
        self.imu_model = IMU_MLP(input_size, hidden_size, output_size)

    def forward(self, x):
        rgb_out = self.rgb_model(x[0])
        imu_out = self.imu_model(x[1])
        out = (imu_out+rgb_out)/2
        return out

# Define the 3D CNN model 
class RGB_Action(nn.Module):
    def __init__(self, num_classes, video_length): #num_classes is the output size
        super(RGB_Action, self).__init__()
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
    
class joint_IMU_MLP(nn.Module): #This is IMU to PID+Action
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

def decouple_inputs(data_batch, model_info, device):
    # Extract the inputs and labels
    #THERE is probs a better way to do this...
    if 'IMU' in model_info['sensors'] and 'RGB' in model_info['sensors']:
        inputs = (data_batch[0].to(device).type(torch.float32), data_batch[1].to(device).type(torch.float32))
    else:
        if 'IMU' in model_info['sensors']:
            inputs = data_batch[1].to(device).type(torch.float32)
        elif 'RGB' in model_info['sensors']:
            inputs = data_batch[0].to(device).type(torch.float32)
    if 'HAR' in model_info['tasks'] and 'PID' in model_info['tasks']:
        labels = (data_batch[2], data_batch[3])
    elif 'HAR' in model_info['tasks']:
        labels = data_batch[2]
    elif 'PID' in model_info['tasks']:
        labels = data_batch[3]
    

    return inputs, labels

# Define the training loop
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_info):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data_batch in enumerate(train_loader):
            
            inputs, labels = decouple_inputs(data_batch, model_info, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if len(model_info['tasks']) == 2:
                loss = criterion(outputs[0], labels[0].to(device)) + criterion(outputs[1], labels[1].to(device))
            else:
                loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        
        # Log the loss to wandb
        wandb.log({'train_loss': running_loss / len(train_loader)})


#NOTE: EVALUATION CURRENTLY ASSUMES ONE OUTPUT LABEL
        if (epoch+1) % 2 == 0:
            acc = evaluate(model, val_loader, device, model_info)
            print('Test accuracy: {:.4f} %'.format(acc))
            wandb.log({'val_acc': acc})
            best_val_acc=0
            best_val_file= None
            if not os.path.exists("./models"):
                os.mkdir("./models")
            for f in os.listdir('./models/'):
                prefix = model_info['project_name']+'_best_model'
                if f.startswith(prefix):
                    best_val_acc = float(f.split("_")[-1][:-3]) #get the accuracy from the filename, remove .pth in the end
                    best_val_file = os.path.join("./models",f)
            if acc > best_val_acc:
                if best_val_file: os.remove(best_val_file)
                best_val_acc = acc
                fname = f'./models/{model_info["project_name"]}_best_model{model.hidden_size}_{acc:.4f}.pt'
                torch.save(model.state_dict(), fname)



#NOTE: EVALUATION CURRENTLY ASSUMES ONE OUTPUT LABEL
# Define evaluation loop
def evaluate(model, val_loader, device, model_info):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data_batch in val_loader:
            inputs, labels = decouple_inputs(data_batch, model_info, device=device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

            # # Print incorrect predictions
            # headers = ["Predicted, Actual, Path"]
            # data = []
            # if (predicted == labels).sum() != val_loader.batch_size:
            #     incorrect_indices = (predicted != labels).nonzero()[:,0]
            #     for i in incorrect_indices:
            #         print("Predicted:", actions_dict[predicted[i].item()+1], ", Actual:", actions_dict[labels[i].item()+1])#, ", Path:", path[i])
                    

        return 100 * correct / total


# Define the custom dataset and data loader
# https://torchvideo.readthedocs.io/en/latest/transforms.html
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize frames
    transforms.ToTensor(),           # Convert frames to tensors
])

def main():
    # Parse command-line arguments
    parser = ArgumentParser()
    # parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--hidden_size', type=int, default=2048)
    args = parser.parse_args()

    # Set the hyperparameters (note most set in argparser above)
    model_info = {
        'sensors' : ['RGB', 'IMU'], #['RGB', 'IMU'] #NOTE: Keep the order here consistent for naming purposes
        'tasks' : ['HAR'], #['HAR', 'PID'],
        'fusion_type' : 'early', # 'early', 'middle', 'late'
        'num_classes' : 8,
        'project_name' : ""
    }
    rgb_video_length = 30
    if "PID" in model_info['tasks']:
        model_info['num_classes'] = 8
    elif "HAR" in model_info['tasks']:
        model_info['num_classes'] = 27

    model_info['project_name'] = "toy-"+"-".join(model_info['sensors']+model_info['tasks'])+'-'+model_info['fusion_type']
    #MLP Specficic:
    input_size = 180*6  #imu length * 6 sensors see IMU/dataset.py for more info
    hidden_size = args.hidden_size
    output_size = model_info['num_classes']

    if not args.test:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=model_info['project_name'],
            
            # track hyperparameters and run metadata
            config={
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'optimizer': args.optimizer,
            'hidden_size': args.hidden_size
        })
        wandb.run.name = f"{args.hidden_size}_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.num_epochs}"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)
    print("Args: ", args)


    # Load the dataset
    # datapath = "Inertial_splits/action_80_20_#1" if label_category == 'action' else "Inertial_splits/pid_80_20_#1"
    datapath = "Both_splits/both_80_20_#1"
    train_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"train.txt")
    val_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"val.txt")
    train_dataset = RGB_IMU_Dataset(train_dir, video_length=rgb_video_length, transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = RGB_IMU_Dataset(val_dir, video_length=rgb_video_length, transform=transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define the model, loss function, and optimizer
    if 'IMU' in model_info['sensors'] and 'RGB' in model_info['sensors']:
        if model_info['fusion_type'] == 'early':
            model = Early_Fusion(input_size, hidden_size, output_size, rgb_video_length=rgb_video_length).to(device).float()
        elif model_info['fusion_type'] == 'middle':
            model = Middle_Fusion(input_size, hidden_size, output_size, rgb_video_length=rgb_video_length).to(device).float()
        elif model_info['fusion_type'] == 'late':
            model = Late_Fusion(input_size, hidden_size, output_size, rgb_video_length=rgb_video_length).to(device).float()
    elif 'IMU' in model_info['sensors']:
        if 'HAR' in model_info['tasks'] and 'PID' in model_info['tasks']: #this doesn't make full sense rn, bc my joint imu -> 2 task model doesn't output pid rn
            model = joint_IMU_MLP(input_size, hidden_size, output_size).to(device).float()
            train_loader = ((x[1], x[2], x[3]) for x in train_loader)
            val_loader  = ((x[1], x[2], x[3]) for x in val_loader)
        else:
            model = IMU_MLP(input_size, hidden_size, output_size).to(device).float()
            if 'PID' in model_info['tasks']:
                train_loader = ((x[1], x[2]) for x in train_loader)
                val_loader  = ((x[1], x[2]) for x in val_loader)
            elif 'HAR' in model_info['tasks']:
                train_loader = ((x[1], x[3]) for x in train_loader)
                val_loader  = ((x[1], x[3]) for x in val_loader)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Test or train the model
    if args.test:
        models = os.listdir('./models/')
        for m in models:
            prefix = model_info['project_name']+'_best_model'
            if m.startswith(prefix):
                model_path = os.path.join('./models/', m)
                break
        print("Evaluating model: ", model_path)
        model.load_state_dict(torch.load(model_path))
        acc = evaluate(model, val_loader, device)
        print('Test accuracy: {:.4f} %'.format(acc))
    else:
        train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info=model_info)


if __name__ == '__main__':
    main()

