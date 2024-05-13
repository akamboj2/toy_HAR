"""
Perform KNN classificaiton on the IMU data using dynamic time warping algorithm for distance measurement: 
https://github.com/wannesm/dtaidistance 

note: 6 comparisons take about 1.4 seconds, 172*689 would take 35.6 hours
instead attempting to average all the data and classify like that.

"""

from dataset import IMUDataset
from torch.utils.data import DataLoader
from dtaidistance import dtw
from tqdm import tqdm
import time, os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


if __name__=='__main__':
    
    # When k=0 it averages all the training data and classifies based on that
    # When k=1 it classifies based on the nearest neighbour (searching through entire training data)
    k = 4

    dir = "/home/abhi/data/utd-mhad/Inertial_splits/action_80_20_#1/train.txt"
    # dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    train_dataset = IMUDataset(dir, dataset_name="UTD")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    if k==0:
        # Average all the data
        avgerage_data = torch.zeros((27,180,6))
        counts = torch.zeros((27))
        for x, y in train_loader:
            # print(x.shape, y) # [1,180,6]
            avgerage_data[y] += x[0]
            counts[y] += 1
                
        avgerage_data = avgerage_data / counts[:,None, None]

        print("Averaged data:", avgerage_data.shape)

    # Test model
    datapath = "Inertial_splits/action_80_20_#1"
    val_dir = os.path.join("/home/abhi/data/utd-mhad/",datapath,"val.txt")
    val_dataset = IMUDataset(val_dir, time_invariance_test=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print("Testing model")
    correct = 0
    confusion_matrix = np.zeros((27, 27), dtype=int)  # Initialize confusion matrix
    for x, y in tqdm(val_loader):
        x = x.squeeze(0)
        min_dist = float('inf')
        min_class = -1

        # if k==0:
        #     for i in range(27):
        #         dist = 0
        #         for j in range(6):
        #             dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), avgerage_data[i, :, j].numpy().astype(np.double))
        #         dist /= 6
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_class = i
        # else:
        #     for x_train, y_train in train_dataset:
        #         dist = 0
        #         for j in range(6):
        #             dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), x_train[:, j].numpy().astype(np.double))
        #         dist /= 6
        #         if dist < min_dist:
        #             min_dist = dist
        #             min_class = y_train

        # if k == 0:
        #             for i in range(27):
        #                 dist = 0
        #                 for j in range(6):
        #                     dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), avgerage_data[i, :, j].numpy().astype(np.double))
        #                 dist /= 6
        #                 if dist < min_dist:
        #                     min_dist = dist
        #                     min_class = i
        #         else:
        #             top_k = []  # Initialize a list to store the top k minimum distances and classes
        #             for x_train, y_train in train_dataset:
        #                 dist = 0
        #                 for j in range(6):
        #                     dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), x_train[:, j].numpy().astype(np.double))
        #                 dist /= 6
        #                 if len(top_k) < k:
        #                     top_k.append((dist, y_train))
        #                 else:
        #                     max_dist = max(top_k, key=lambda x: x[0])[0]
        #                     if dist < max_dist:
        #                         max_dist_index = top_k.index((max_dist, None))
        #                         top_k[max_dist_index] = (dist, y_train)
        if k == 0:
            for i in range(27):
                dist = 0
                for j in range(6):
                    dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), avgerage_data[i, :, j].numpy().astype(np.double))
                dist /= 6
                if dist < min_dist:
                    min_dist = dist
                    min_class = i
        else:
            top_k = []  # Initialize a list to store the top k minimum distances and classes
            for x_train, y_train in train_dataset:
                dist = 0
                for j in range(6):
                    dist += dtw.distance_fast(x[:, j].numpy().astype(np.double), x_train[:, j].numpy().astype(np.double))
                dist /= 6
                if len(top_k) < k:
                    top_k.append((dist, y_train))
                else:
                    max_dist,max_y = max(top_k, key=lambda x: x[0])
                    if dist < max_dist:
                        max_dist_index = top_k.index((max_dist, max_y))
                        top_k[max_dist_index] = (dist, y_train)
            min_dist_classes = [c[1] for c in top_k]  # Get the class values from the top k set
            min_class = max(set(min_dist_classes), key=min_dist_classes.count)  # Take the mode of the class values
            min_dist = min([c[0] for c in top_k])  # Get the minimum distance



        if min_class == y:
            correct += 1
        confusion_matrix[y, min_class] += 1

    # Print accuracy
    accuracy = correct / len(val_dataset)
    print("Accuracy:", accuracy)

    # Print confusion matrix
    # print("Confusion Matrix:")
    # print(confusion_matrix)

    plt.figure(figsize=(24, 20))

    # Create a heatmap of the confusion matrix using seaborn
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", square=True)

    # Set the x-axis and y-axis labels using the actions_dict
    plt.xlabel("Predicted Action")
    plt.ylabel("True Action")
    if k==0:
        plt.title(f"Average NN Confusion Matrix: Accuracy {accuracy*100:.2f}%")
    else:
        plt.title(f"k={k}-KNN Confusion Matrix: Accuracy {accuracy*100:.2f}%")
    plt.xticks(ticks=range(27), labels=[actions_dict[i] for i in range(1, 28)], rotation=90)
    plt.yticks(ticks=range(27), labels=[actions_dict[i] for i in range(1, 28)], rotation=0)

    # Show the plot
    if k==0:
        plt.savefig("plots/avg_nn_confusion_matrix.png")
    else:
        plt.savefig(f"plots/k={k}_knn_confusion_matrix.png")

