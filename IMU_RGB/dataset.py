import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import normalize
import torch.nn.utils.rnn as rnn_utils
import scipy.io


class RGB_IMU_Dataset(Dataset):
    def __init__(self, split_file, video_length=50, imu_length=180, transform=None, base_path ="/home/abhi/data/utd-mhad/", return_path=False):
        self.split_file = split_file
        self.transform = transform
        self.videos = []
        self.vid_length = video_length
        self.imu_length = imu_length
        self.return_path = return_path
        
        #assume we are reading "path label_action label_PID" from a file
        f = open(self.split_file,'r')
        for line in f.readlines():
            video_name, class_idx, pid_idx = line.split(" ")
            class_idx = int(class_idx)-1
            if class_idx > 26 or class_idx < 0:
                raise ValueError(f"{class_idx} is an invalid class index")
            pid_idx = int(pid_idx)-1
            if pid_idx > 8 or pid_idx < 0:
                raise ValueError(f"{pid_idx} is an invalid PID index")
        
            
            rgb_path = os.path.join(base_path,"RGB", video_name+"_color.avi")
            imu_path = os.path.join(base_path,"Inertial", video_name+"_inertial.mat")
            self.videos.append((rgb_path, imu_path, class_idx, pid_idx))  # (video path, IMU path, class index)
        f.close()

            

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        rgb_path, imu_path, class_idx, pid_idx = self.videos[idx]
        frames, audio, info = read_video(rgb_path, pts_unit="sec") # Tensor Shape THWC: 57,480,640,3
        

        """ VIDEO PROCESSING"""
        # # Normalize video frames (you can adjust the mean and std)
        # frames = normalize(frames.float(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        
        # frames = torch.stack(frames) # conver list to tensor
        # print(len(frames),frames[0].shape)

        # Pad or shorten video to vid_length
        t,h,w,c = frames.shape
        if t>self.vid_length:
            frames = frames[:self.vid_length,:,:,:]
        elif t<self.vid_length:
            # Pad frames with zeros to make them the same length
            frames = torch.cat([frames, torch.zeros(self.vid_length - len(frames), *frames.shape[1:])]) 
        

        #perform tansforms on each frame
        frames = frames.permute(0,3,1,2) # permute to TCHW to perform image-wise transforms
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        # Time, channel makes more sense bc channel describes img, and t describes multiple imgs
        # # # Permute to CTHW for 3d convs
        # frames = frames.permute(1,0,2,3)
        
        # return frames, class_idx #returns TCHW and class idx

        """IMU PROCESSING"""
        data = scipy.io.loadmat(imu_path)
        # xyz = np.array(data['d_iner'])
        # Convert to torch tensor
        accel_data = torch.tensor(data['d_iner'])
        
        t,xyz = accel_data.shape
        if t>self.imu_length:
            accel_data = accel_data[:self.imu_length,:]
        elif t<self.imu_length:
            # Pad accel_data with zeros to make them the same length
            accel_data = torch.cat([accel_data, torch.zeros(self.imu_length - len(accel_data), *accel_data.shape[1:])])

        # return accel_data, int(label), file_path
        if self.return_path:
            return frames, accel_data, class_idx, pid_idx, rgb_path, imu_path
        else:
            return frames, accel_data, class_idx, pid_idx #returns TCHW video

if __name__=='__main__':
    dir = "/home/abhi/data/utd-mhad/Both_splits/both_80_20_#1/train.txt"
    dir = "/home/abhi/data/utd-mhad/Both_splits/both_80_20_#1/val.txt"
    
    d = RGB_IMU_Dataset(dir)
    
    video_lengths = []
    for itm in d:
        print("Input RGB:", itm[0].shape, "Input IMU:", itm[1].shape, "action label:", itm[2], "PID label:", itm[3])
        video_lengths.append(itm[0].shape[0])
        continue

    print(len(d))
    print("average video length:", sum(video_lengths)/len(video_lengths))
    # average video length: 52.55668604651163 (in training)
    
