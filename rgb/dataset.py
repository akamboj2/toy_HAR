import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import normalize
import torch.nn.utils.rnn as rnn_utils

class CustomVideoDataset(Dataset):
    def __init__(self, root_dir, video_length=30, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.videos = []
        self.vid_length = video_length

        
        #check if it a valid path
        if os.path.isdir(self.root_dir):
            video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4') or f.endswith('.avi')]
            for video_file in video_files:
                video_path = os.path.join(self.root_dir, video_file)
                class_idx  = int(video_file.split("_")[0][1:])-1 #a26_s6_t3_color.avi indicates action 26, subject 6, trial 3
                if class_idx > 26 or class_idx < 0:
                    raise ValueError(f"{class_idx} is an invalid class index")
                
                self.videos.append((video_path, class_idx))  # (video path, class index)

        elif root_dir.split(".")[-1]=="txt":
            #assume we are reading int path, label from a file
            f = open(root_dir,'r')
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
        video_path, class_idx = self.videos[idx]
        frames, audio, info = read_video(video_path, pts_unit="sec") # Tensor Shape THWC: 57,480,640,3
        

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
        
        return frames, class_idx #returns TCHW and class idx

if __name__=='__main__':
    dir = "/home/abhi/data/utd-mhad/RGB"
    dir = "/home/akamboj2/data/utd-mhad/RGB_splits/Action_80_20_#1/train.txt"
    
    d = CustomVideoDataset(dir)
    
    video_lengths = []
    for itm in d:
        print("input:", itm[0].shape, "action label:", itm[1])
        video_lengths.append(itm[0].shape[0])
        continue

    print(len(d))
    print("average video length:", sum(video_lengths)/len(video_lengths))
    # average video length: 52.55668604651163 (in training)
    
