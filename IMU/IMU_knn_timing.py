"""
Perform KNN classificaiton on the IMU data using dynamic time warping algorithm for distance measurement: 

https://github.com/wannesm/dtaidistance 

"""

from dataset import IMUDataset
from torch.utils.data import DataLoader
from dtaidistance import dtw
import time

if __name__=='__main__':
    
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    # dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    train_dataset = IMUDataset(dir, dataset_name="UTD")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # avgerage_data = []
    sample = train_dataset[0]
    start_time = time.time()
    for x, y in train_loader:
        # print(x.shape, y)
        for i in range(x.shape[-1]):
            # print(type(sample),type(x))
            # print(sample)
            a = dtw.distance_fast(sample[0][:,i].numpy(), x[0,:,i].numpy())
            # print(i,a)
            
        break
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    # 861 *.2 = 172 sequences for testing
    # 861 *.8 = 689 sequences for training
    # 172*689 = 118348
    print("Total number of comparisons:", 118348)
    print("Estimated Time:", 118348*execution_time/60, "minutes which is", 118348*execution_time/3600, "hours")
    # 2863 minutes, e.g. 35.6 hours for dtw.distance version
    # 2.56 minutes for dtw.distance_fast version