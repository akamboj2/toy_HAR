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
import numpy as np

# from models import Early_Fusion, Middle_Fusion, Late_Fusion, IMU_MLP, joint_IMU_MLP
# from train_utils import train, evaluate

args = None #main will fill in args 

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

# Cross-modal Fusion Method
class CM_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rgb_video_length):
        super(CM_Fusion, self).__init__()
        self.hidden_size = hidden_size #here hiddent size will be the size the two features join at (addition)
        self.FE_rgb = RGB_Action(hidden_size, rgb_video_length)
        self.FE_imu = IMU_MLP(input_size, hidden_size*2, hidden_size)
        self.joint_processing = IMU_MLP(hidden_size, hidden_size//2, output_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, sensors):
        if 'RGB' in sensors and 'IMU' in sensors:
            z_rgb = self.FE_rgb(x[0])
            z_imu = self.FE_imu(x[1])
            z = (z_rgb+z_imu)/2
        elif 'RGB' in sensors:
            z = self.FE_rgb(x) #the decouple_inputs function called in train.py  will only give us rgb here no need for x[0]
        elif 'IMU' in sensors:
            z = self.FE_imu(x)
        out = self.joint_processing(z)
        return out

#Fusion is adding
class Middle_Fusion(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rgb_video_length):
        super(Middle_Fusion, self).__init__()
        self.hidden_size = hidden_size #here hiddent size will be the size the two features join at (addition)
        self.rgb_model = RGB_Action(hidden_size, rgb_video_length)
        self.imu_model = IMU_MLP(input_size, hidden_size*2, hidden_size)
        self.joint_processing = IMU_MLP(hidden_size, hidden_size//2, output_size)

    def forward(self, x):
        z_rgb = self.rgb_model(x[0])
        z_imu = self.imu_model(x[1])
        z_sum = (z_rgb+z_imu)/2
        out = self.joint_processing(z_sum)
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
            if model_info['fusion_type'] == 'cross_modal':
                outputs = model(inputs, model_info['sensors'])
            else:
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
        if not args.no_wandb: wandb.log({'train_loss'+(f'_{model_info["sensors"]}' if model_info['fusion_type']== "cross_modal" else ''): running_loss / len(train_loader)})

        #NOTE: EVALUATION CURRENTLY ASSUMES ONE OUTPUT LABEL
        if (epoch+1) % 2 == 0:
            acc = evaluate(model, val_loader, device, model_info)
            print('Test accuracy: {:.4f} %'.format(acc))
            if not args.no_wandb: wandb.log({'val_acc'+(f'_{model_info["sensors"]}' if model_info['fusion_type']== "cross_modal" else ''): acc})
            #The snippet below is to save the best model
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

# CLIP based cosine similarity representation alignment training
def CLIP_train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_info):

    loss_imu = nn.CrossEntropyLoss()
    loss_rgb = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data_batch in enumerate(train_loader):
            
            inputs, labels = decouple_inputs(data_batch, model_info, device)

            optimizer.zero_grad()
            # outputs = model(inputs)
            z_rgb = model.FE_rgb(inputs[0]) # z is the latent feature vector
            z_imu = model.FE_imu(inputs[1])
            z_rgb = z_rgb / z_rgb.norm(dim=-1, keepdim=True)
            z_imu = z_imu / z_imu.norm(dim=-1, keepdim=True)
            # loss = 1 - (z_rgb * z_imu).sum(dim=-1).mean()
            #using clip style training: https://github.com/openai/CLIP/issues/83
            # cosine similarity as logits
            # logit_scale = model.logit_scale.exp()
            # logits_per_rgb = logit_scale * z_rgb @ z_imu.t()
            # logit_scale is the temperature parameter, probs help stabilize training with lots of data
            # in our case it makes it not train, or train very slowly...
            logits_per_rgb =  z_rgb @ z_imu.t()
            logits_per_imu = logits_per_rgb.t()

            ground_truth = torch.arange(len(inputs[0]),dtype=torch.long,device=device)
            total_loss = (loss_rgb(logits_per_rgb,ground_truth) + loss_imu(logits_per_imu,ground_truth))/2
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), total_loss.item()))
        
        # Log the loss to wandb
        if not args.no_wandb: wandb.log({'CLIP_loss': running_loss / len(train_loader)})

        #Eval and save best model
        if (epoch+1) % 2 == 0:
            acc = CLIP_evaluate(model, val_loader, device, model_info)
            print('Test accuracy: {:.4f} %'.format(acc))
            if not args.no_wandb: wandb.log({'val_acc'+(f'_{model_info["sensors"]}' if model_info['fusion_type']== "cross_modal" else ''): acc})
            #The snippet below is to save the best model
            best_val_acc=0
            best_val_file= None
            if not os.path.exists("./models"):
                os.mkdir("./models")
            for f in os.listdir('./models/'):
                prefix = model_info['project_name']+"-FEs"+'_best_model'
                if f.startswith(prefix):
                    best_val_acc = float(f.split("_")[-1][:-3]) #get the accuracy from the filename, remove .pth in the end
                    best_val_file = os.path.join("./models",f)
            if acc > best_val_acc:
                if best_val_file: os.remove(best_val_file)
                best_val_acc = acc
                fname = f'./models/{model_info["project_name"]}-FEs_best_model{model.hidden_size}_{acc:.4f}.pt'
                torch.save(model.state_dict(), fname)

# Shared representation space alignment training
def shared_train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_info):

    loss_imu = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data_batch in enumerate(train_loader):
            
            inputs, labels = decouple_inputs(data_batch, model_info, device)

            optimizer.zero_grad()
            # outputs = model(inputs)
            z_rgb = model.FE_rgb(inputs[0]) # z is the latent feature vector
            z_imu = model.FE_imu(inputs[1])
            z_rgb = z_rgb / z_rgb.norm(dim=-1, keepdim=True)
            z_imu = z_imu / z_imu.norm(dim=-1, keepdim=True)

            total_loss = loss_imu(z_imu, z_rgb)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
            if (i+1) % 5 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), total_loss.item()))
        
        # Log the loss to wandb
        if not args.no_wandb: wandb.log({'Shared_loss': running_loss / len(train_loader)})

#NOTE: EVALUATION CURRENTLY ASSUMES ONE OUTPUT LABEL
# Define evaluation loop
def evaluate(model, val_loader, device, model_info):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data_batch in val_loader:
            inputs, labels = decouple_inputs(data_batch, model_info, device=device)
            if model_info['fusion_type'] == 'cross_modal':
                outputs = model(inputs, model_info['sensors'])
            else:
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

def CLIP_evaluate(model, val_loader, device, model_info):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data_batch in val_loader:
            inputs, labels = decouple_inputs(data_batch, model_info, device=device)
            z_rgb = model.FE_rgb(inputs[0]) # z is the latent feature vector
            z_imu = model.FE_imu(inputs[1])
            z_rgb = z_rgb / z_rgb.norm(dim=-1, keepdim=True)
            z_imu = z_imu / z_imu.norm(dim=-1, keepdim=True)
            # cosine similarity as logits
            similarity = z_rgb @ z_imu.t()
            probs = torch.nn.functional.softmax(similarity, dim=-1).max(-1)[1]
            ground_truth = torch.arange(len(inputs[0]),dtype=torch.long,device=device)
            correct += (probs == ground_truth).sum()
            total += ground_truth.size(0)

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
    # Best Performance: 2048_Adam_0.0001485682045159312_8_240 under toy-rgb-imu-har-middle
    #    2 Args:  Namespace(num_epochs=240, batch_size=8, learning_rate=0.0001485682045159312, optimizer='Adam', test=False, hidden_size=2048)
    # peaked 94 % at 149 steps (or 150 steps)
    # 95.95% epochs at 347 steps
    # python train.py --batch_size=8 --learning_rate=0.0001485682045159312 --optimizer=Adam --hidden_size=2048 --num_epochs=240

    # Parse command-line arguments
    parser = ArgumentParser()
    # parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--test', action='store_true',default=False)
    parser.add_argument('--no_wandb', action='store_true',default=False)
    parser.add_argument('--hidden_size', type=int, default=2048)
    parser.add_argument('--experiment', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    global args
    args = parser.parse_args()

    # Set the hyperparameters (note most set in argparser above)
    model_info = {
        'sensors' : ['RGB', 'IMU'], #['RGB', 'IMU'] #NOTE: Keep the order here consistent for naming purposes
        'tasks' : ['HAR'], #['HAR', 'PID'],
        'fusion_type' : 'cross_modal', #'cross_modal', # 'early', 'middle', 'late', 'cross_modal'
        'num_classes' : -1,
        'project_name' : ""
    }
    rgb_video_length = 30
    if "PID" in model_info['tasks']:
        model_info['num_classes'] = 8
    elif "HAR" in model_info['tasks']:
        model_info['num_classes'] = 27

    model_info['project_name'] = "toy-"+"-".join(model_info['sensors']+model_info['tasks'])+'-'+model_info['fusion_type']+(f'{args.experiment}' if model_info['fusion_type'] == 'cross_modal' else '')
    #MLP Specficic:
    input_size = 180*6  #imu length * 6 sensors see IMU/dataset.py for more info
    hidden_size = args.hidden_size
    output_size = model_info['num_classes']

    if not args.test and not args.no_wandb: #don't run this if we are just running eval or way say no_wandb=true
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
            'hidden_size': args.hidden_size,
            'experiment': args.experiment
        })
        wandb_name = f"{args.hidden_size}_{args.optimizer}_{args.learning_rate}_{args.batch_size}_{args.num_epochs}"
        wandb.run.name = wandb_name
        print("Creating a wandb run:",wandb_name)


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    print("Using device: ",device)
    print("Args: ", args)


    # Load the dataset
    # datapath = "Inertial_splits/action_80_20_#1" if label_category == 'action' else "Inertial_splits/pid_80_20_#1"
    if model_info['fusion_type'] == 'cross_modal':
        datapath = "Both_splits/both_45_45_10_#1"
    else:
        datapath = "Both_splits/both_80_20_#1"
    base_path = "/home/akamboj2/data/utd-mhad/"
    train_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"train.txt")
    val_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"val.txt")
    train_dataset = RGB_IMU_Dataset(train_dir, video_length=rgb_video_length, transform=transforms, base_path=base_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = RGB_IMU_Dataset(val_dir, video_length=rgb_video_length, transform=transforms, base_path=base_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    if model_info['fusion_type'] == 'cross_modal':
        train_2_dir = os.path.join("/home/akamboj2/data/utd-mhad/",datapath,"train_2.txt")
        train_2_dataset = RGB_IMU_Dataset(train_2_dir, video_length=rgb_video_length, transform=transforms, base_path=base_path)
        train_2_loader = torch.utils.data.DataLoader(train_2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Define the model, loss function, and optimizer
    if 'IMU' in model_info['sensors'] and 'RGB' in model_info['sensors']:
        if model_info['fusion_type'] == 'cross_modal':
            model = CM_Fusion(input_size, hidden_size, output_size, rgb_video_length=rgb_video_length).to(device).float()
        elif model_info['fusion_type'] == 'early':
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
        if model_info['fusion_type'] == 'cross_modal':
            

            if args.experiment==1:

                # ----------- EXPERIMENT 1 ----------- 
                #First align modalities representations
                print("Aligning Modalities")
                fname = f'./models/cross-modal/trained-FEs-{model_info["project_name"]}.pt'
                CLIP_train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info)
                torch.save(model.state_dict(), fname)
                # torch.load(fname)
                print("Loaded encoders")

                #Perform CLIP Eval:
                print("Evaluating on RGB and IMU CLIP representations")
                acc = CLIP_evaluate(model, val_loader, device, model_info)
                print('\tTest accuracy: {:.4f} %'.format(acc))
                exit()

                #Freeze the encoders
                #Freeze weights of model.FE_rgb and model.FE_imu
                for param in model.FE_rgb.parameters():
                    param.requires_grad = False
                for param in model.FE_imu.parameters():
                    param.requires_grad = False

                #Then train the model with camera
                print("Training on RGB only")
                camera_only = model_info.copy()
                camera_only['sensors'] = ['RGB']
                train(model, train_2_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info=camera_only)
                ## or load it from the best model
                # models = os.listdir('./models/')
                # for m in models:
                #     prefix = model_info['project_name']+'_best_model'
                #     if m.startswith(prefix):
                #         model_path = os.path.join('./models/', m)
                #         break
                # print("Evaluating model: ", model_path)
                # model.load_state_dict(torch.load(model_path))

                #Finally evaluate on camera, imu, camera+imu
                imu_only = model_info.copy()
                imu_only['sensors'] = ['IMU']

                print("Evaluating on RGB only")
                acc = evaluate(model, val_loader, device, model_info=camera_only)
                print('\tTest accuracy: {:.4f} %'.format(acc))

                print("Evaluating on IMU only")
                acc = evaluate(model, val_loader, device, model_info=imu_only)
                print('\tTest accuracy: {:.4f} %'.format(acc))

                print("Evaluating on RGB and IMU")
                acc = evaluate(model, val_loader, device, model_info=model_info)
                print('\tTest accuracy: {:.4f} %'.format(acc))

            elif args.experiment==2:
                # ----------- EXPERIMENT 2 ----------- 
                # First train RGB HAR model
                print("Training RGB HAR")
                camera_only = model_info.copy()
                camera_only['sensors'] = ['RGB']
                train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info=camera_only)

                # Then Freeze RGB
                for param in model.FE_rgb.parameters():
                    param.requires_grad = False

                # Align the representations directly not through cosine similarity
                print("Aligning Modalities")
                shared_train(model, train_2_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info)

               #Finally evaluate on camera, imu, camera+imu
                imu_only = model_info.copy()
                imu_only['sensors'] = ['IMU']

                print("Evaluating on RGB only")
                acc = evaluate(model, val_loader, device, model_info=camera_only)
                print('Test accuracy: {:.4f} %'.format(acc))

                print("Evaluating on IMU only")
                acc = evaluate(model, val_loader, device, model_info=imu_only)
                print('Test accuracy: {:.4f} %'.format(acc))

                print("Evaluating on RGB and IMU")
                acc = evaluate(model, val_loader, device, model_info=model_info)
                print('Test accuracy: {:.4f} %'.format(acc))
                

        else:
            train(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, model_info=model_info)


if __name__ == '__main__':
    main()

