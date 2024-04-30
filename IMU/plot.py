import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset import IMUDataset

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

def plot_one(action, subject, plt):
    # Plot
    file = f"/home/abhi/data/utd-mhad/Inertial/a{action}_s{subject}_t1_inertial.mat"

    # Load data from .mat file
    data = scipy.io.loadmat(file)
    sn = np.array(data['d_iner'])

    # Extract accelerometer data
    accel_x = sn[:, 0]
    accel_y = sn[:, 1]
    accel_z = sn[:, 2]

    # Create time array
    time = np.arange(len(accel_x)) / 30 # 30 Hz sampling rate


    # Plot accelerometer data
    # plt.figure()
    plt.plot(time, accel_x, label='X')
    plt.plot(time, accel_y, label='Y')
    plt.plot(time, accel_z, label='Z')
    plt.set_xlabel('Time (s)')
    plt.set_ylabel('Acceleration (m/s^2)')
    plt.set_title(actions_dict[action])
    # plt.set_title(actions_dict[action] + ' subject ' + str(subject) + ' trial 1')
    plt.legend()
    # plt.savefig(f"plots/scratch/accel_a{action}_s{subject}.png")

    return plt

def plot_dataset():
    dir = "/home/abhi/data/utd-mhad/Inertial_splits/all/all.txt"
    dataset = IMUDataset(dir,time_invariance_test=True,dataset_name="UTD")
    # dir = "/home/abhi/data/USC-HAD/splits/train.txt"
    # dataset = IMUDataset(dir,time_invariance_test=True,dataset_name="USC")

    sizes = []
    for data, label in dataset:
        print("input:", data.shape, "action label:", label) 

        # Extract accelerometer data
        accel_x = data[:, 0]
        accel_y = data[:, 1]
        accel_z = data[:, 2]

        # Create time array
        time = range(len(accel_x))

        # Plot accelerometer data
        plt.plot(time, accel_x, label='X')
        plt.plot(time, accel_y, label='Y')
        plt.plot(time, accel_z, label='Z')
        plt.xlabel('Time')
        plt.ylabel('Acceleration (m/s^2)')
        plt.title('IMU Accelerometer Data')
        plt.legend()
        plt.savefig("plots/test.png")
        break



if __name__ == "__main__":
    # Loop through and call plot one for a1-a27

    name = "a*_s1_t1" #test
    fig, axs = plt.subplots(3, 9, figsize=(50, 15))
    for i in range(1, 28):
        # plt.subplot(3, 9, i)
        plt = plot_one(i, 1, axs[(i-1)//9,(i-1)%9])  # always plot subject 1
    # plt.tight_layout()
    fig.savefig(f"plots/scratch/{name}.png")


    # TODO: 
    #Plot all actions on one page -- done

    # plot one person with avg of 4 trials and std 
    # Plot avg of all trials of each action and std

