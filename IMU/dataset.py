import torch
from torch.utils.data import Dataset
import scipy.io
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, split_file):
        self.videos = []
        self.vid_length = 180
        # Read the lines from split file
        f = open(split_file,'r')
        for line in f.readlines():
            video_path, class_idx = line.split(" ")
            class_idx = int(class_idx)-1
            if class_idx > 26 or class_idx < 0:
                raise ValueError(f"{class_idx} is an invalid class index")
        
            self.videos.append((video_path, class_idx))  # (video path, class index)
        f.close()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        file_path, label = self.videos[idx]
        data = scipy.io.loadmat(file_path)
        # xyz = np.array(data['d_iner'])
        # Convert to torch tensor
        accel_data = torch.tensor(data['d_iner'])
        
        t,xyz = accel_data.shape
        if t>self.vid_length:
            accel_data = accel_data[:self.vid_length,:]
        elif t<self.vid_length:
            # Pad accel_data with zeros to make them the same length
            accel_data = torch.cat([accel_data, torch.zeros(self.vid_length - len(accel_data), *accel_data.shape[1:])])
            
        
        return accel_data, int(label)

if __name__=='__main__':
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/train.txt"
    d = IMUDataset(dir)

    sizes = []
    for itm in d:
        print("input:", itm[0].shape, "action label:", itm[1])
        print(itm[0])
        sizes.append(itm[0].shape[0])

    print("Average size:", np.mean(sizes)) #180 frames
