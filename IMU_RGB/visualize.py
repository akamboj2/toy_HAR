import os
import torch
from train import CM_Fusion, transforms, decouple_inputs
from dataset import RGB_IMU_Dataset
# import torchvision.transforms as transforms
import numpy as np

class args:
    fusion_type = 'cross_modal'
    experiment = 4
    batch_size = 8

if __name__=='__main__':

    # DEFINE THE MODEL
    model_info = {
        'sensors' : ['RGB', 'IMU'], #['RGB', 'IMU'] #NOTE: Keep the order here consistent for naming purposes
        'tasks' : ['HAR'], #['HAR', 'PID'],
        'fusion_type' : args.fusion_type, #'middle', #'cross_modal', # 'early', 'middle', 'late', 'cross_modal'
        'num_classes' : -1,
        'project_name' : ""
    }
    if "PID" in model_info['tasks']:
        model_info['num_classes'] = 8
    elif "HAR" in model_info['tasks']:
        model_info['num_classes'] = 27
    model_info['project_name'] = "toy-"+"-".join(model_info['sensors']+model_info['tasks'])+'-'+model_info['fusion_type']+(f'{args.experiment}' if model_info['fusion_type'] == 'cross_modal' else '')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 180*6  #imu length * 6 sensors see IMU/dataset.py for more info
    hidden_size = 2048
    output_size = model_info['num_classes']
    rgb_video_length = 30
    model = CM_Fusion(input_size, hidden_size, output_size, rgb_video_length = rgb_video_length).to(device).float()

    # LOAD THE MODEL
    models = os.listdir('./models/')
    for m in models:
        prefix = model_info['project_name']+'-FEs_best_model'
        if m.startswith(prefix):
            model_path = os.path.join('./models/', m)
            break
    print("Loaded model: ", model_path)
    model.load_state_dict(torch.load(model_path))

    # LOAD THE DATASET

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

    all_zs = None
    all_labels = None
    model.eval()
    # Gather Embeddings
    with torch.no_grad():
        for i, data_batch in enumerate(train_loader):
            inputs, labels = decouple_inputs(data_batch, model_info, device)
            imu, rgb = inputs
            imu = imu.to(device).float()
            # rgb = rgb.to(device).float()
            # labels = labels.to(device).float()
            print("IMU: ", imu.shape)
            # print("RGB: ", rgb.shape)
            print("Label: ", labels.shape)
            # print("labels: ", labels)
            print("Model: ", model(inputs,model_info['sensors']).shape)
            z_rgb = model.FE_rgb(inputs[0]) # z is the latent feature vector
            z_imu = model.FE_imu(inputs[1])
            z_rgb = z_rgb / z_rgb.norm(dim=-1, keepdim=True)
            z_imu = z_imu / z_imu.norm(dim=-1, keepdim=True)
            # print("z_rgb: ", z_rgb.shape)
            print("z_imu: ", z_imu.shape)
            #let's just focus on visualizing z_IMU by doing pca on it
            z_imu = z_imu.detach().cpu().numpy()
            # all_zs = np.vstack((all_zs, z_imu))
            if all_zs is None:
                all_zs = z_imu
                all_labels = labels.detach().cpu().numpy()
            else:
                all_zs = np.concatenate((all_zs, z_imu), axis=0)
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)

            torch.cuda.empty_cache()
            if i ==50:
                break

    print("all_zs: ", all_zs.shape)
    # VISUALIZE THE MODEL
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    
    import random

    # Function to generate a random hex color
    def random_hex_color():
        return '#{:06x}'.format(random.randint(0, 0xFFFFFF))

    # Creating dictionary with keys from 0 to 26 and random colors
    color_map = {i: random_hex_color() for i in range(27)}
    # print(color_map)

    all_zs = pca.fit_transform(all_zs.T)
    # plt.scatter(all_zs[:,0], all_zs[:,1], c=all_labels)
    for i in range(27):
        indices = np.where(all_labels == i)
        plt.scatter(all_zs[indices,0], all_zs[indices,1], c=color_map[i], label=i)

    # plt.legend()
    plt.savefig("pca.png")
