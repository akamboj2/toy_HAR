import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

def get_data(action, subject, trial=1, trim=False):
    # Plot
    file = f"/home/abhi/data/utd-mhad/Inertial/a{action}_s{subject}_t{trial}_inertial.mat"

    # Load data from .mat file
    data = scipy.io.loadmat(file)
    sn = np.array(data['d_iner'])

    # Extract accelerometer data
    accel_x = sn[:, 0]
    accel_y = sn[:, 1]
    accel_z = sn[:, 2]
    if trim:
        accel_x = trim_data(accel_x)
        accel_y = trim_data(accel_y)
        accel_z = trim_data(accel_z)

    # Create time array
    time = np.arange(len(accel_x)) / 30 # 30 Hz sampling rate

    return accel_x, accel_y, accel_z, time

def plot_one(action, subject, plt):
    # Get data
    accel_x, accel_y, accel_z, time = get_data(action, subject)

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

def plot_trials(action, subject):
    # Plot all trials
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for trial in range(1, 5):
        _ = plot_one(action, subject, axs[trial-1])
    fig.savefig(f"plots/scratch/accel_a{action}_s{subject}_all_trials.png")

def trim_data(accel_data, vid_length=180):
    t = len(accel_data)
    if t>=vid_length:
            accel_data = accel_data[:vid_length]
    elif t<vid_length:
        # Pad accel_data with zeros to make them the same length
        accel_data = np.concatenate([accel_data, np.zeros(vid_length - t, *accel_data.shape[1:])])
    return accel_data

def plot_trials_avg(action, subject, ax = None):
    # Plot trials avg and fill with std
    accel_xs = []
    accel_ys = []
    accel_zs = []
    for trial in range(1,5):
        try :
            accel_x, accel_y, accel_z, time = get_data(action, subject, trial, trim=True)
        except FileNotFoundError:
            print(f"File not found: a{action}_s{subject}_t{trial}")
            continue
        accel_xs.append(accel_x)
        accel_ys.append(accel_y)
        accel_zs.append(accel_z)
    
    x_avg = np.array(accel_xs).mean(axis=0)
    y_avg = np.array(accel_ys).mean(axis=0)
    z_avg = np.array(accel_zs).mean(axis=0)
    x_std = np.array(accel_xs).std(axis=0)
    y_std = np.array(accel_ys).std(axis=0)
    z_std = np.array(accel_zs).std(axis=0)

    if ax:
        # Create time array
        # fig, ax = plt.subplots()
        time = np.arange(len(accel_x)) / 30 # 30 Hz sampling rate
        ax.plot(time, x_avg, label='X')
        ax.plot(time, y_avg, label='Y')
        ax.plot(time, z_avg, label='Z')
        ax.fill_between(time, x_avg - x_std, x_avg + x_std, alpha=0.3)
        ax.fill_between(time, y_avg - y_std, y_avg + y_std, alpha=0.3)
        ax.fill_between(time, z_avg - z_std, z_avg + z_std, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s^2)')
        ax.set_title(actions_dict[action] + ' subject ' + str(subject) + ' avg of 4 trials')
        ax.legend()
        # fig.savefig(f"plots/scratch/accel_a{action}_s{subject}_avg.png")

    accel_xs = np.array(accel_xs)
    accel_ys = np.array(accel_ys)
    accel_zs = np.array(accel_zs)
    return accel_xs, accel_ys, accel_zs

def plot_users_avg(action, ax):
    # Plot the average accelerometers across all users and all trials
    # fig, ax = plt.subplots()
    accel_xs = np.array([])
    accel_ys = np.array([])
    accel_zs = np.array([])
    for subject in range(1,9):
        accel_x, accel_y, accel_z = plot_trials_avg(action, subject)
        if accel_xs.size == 0:
            accel_xs = accel_x
            accel_ys = accel_y
            accel_zs = accel_z
        else:
            accel_xs = np.concatenate((accel_xs, accel_x), axis=0)
            accel_ys = np.concatenate((accel_ys, accel_y), axis=0)
            accel_zs = np.concatenate((accel_zs, accel_z), axis=0)
    
    x_avg = np.array(accel_xs).mean(axis=0)
    y_avg = np.array(accel_ys).mean(axis=0)
    z_avg = np.array(accel_zs).mean(axis=0)
    x_std = np.array(accel_xs).std(axis=0)
    y_std = np.array(accel_ys).std(axis=0)
    z_std = np.array(accel_zs).std(axis=0)

    if ax:
        # Create time array
        time = np.arange(180) / 30
        ax.plot(time, x_avg, label='X')
        ax.plot(time, y_avg, label='Y')
        ax.plot(time, z_avg, label='Z')
        ax.fill_between(time, x_avg - x_std, x_avg + x_std, alpha=0.3)
        ax.fill_between(time, y_avg - y_std, y_avg + y_std, alpha=0.3)
        ax.fill_between(time, z_avg - z_std, z_avg + z_std, alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s^2)')
        ax.set_title(actions_dict[action])
        ax.legend()
        # fig.savefig(f"plots/scratch/accel_a{action}_s{subject}_avg.png")

    return accel_xs, accel_ys, accel_zs


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


    
def plot_all_2d():
    name = "a_s_t_avg" # a*_s1_t*_avgtrials" #"a*_s1_t1" #test
    # name = "a_s1_t_avg" # a*_s1_t*_avgtrials" #"a*_s1_t1" #test

    # plot_trials_avg(1,1)
    fig, axs = plt.subplots(3, 9, figsize=(50, 15))

    fig.suptitle('Average Acceleration for Each Action Across 8 Subjects and 4 Trials Each')
    # fig.suptitle('Average Acceleration for Each Action Across 1 Subject and 4 Trials Each')
    for i in range(1, 28):
        # plt.subplot(3, 9, i)
        plt = plot_users_avg(i, axs[(i-1)//9,(i-1)%9])
        # plt = plot_trials_avg(i, 1, axs[(i-1)//9,(i-1)%9])
        # plt = plot_one(i, 1, axs[(i-1)//9,(i-1)%9])  # always plot subject 1
    
    fig.savefig(f"plots/scratch/{name}.png")


def plot_position(action, subject, ax = None):

    accel_x, accel_y, accel_z, time = get_data(action=action, subject=subject, trim=True)

    # Calculate velocity and position
    vel_x = np.zeros_like(accel_x)
    for i in range(1, len(accel_x)): 
        vel_x[i] = vel_x[i-1] + accel_x[i] / 30
    vel_y = np.zeros_like(accel_y)
    for i in range(1, len(accel_y)): 
        vel_y[i] = vel_y[i-1] + accel_y[i] / 30
    vel_z = np.zeros_like(accel_z)
    for i in range(1, len(accel_z)): 
        vel_z[i] = vel_z[i-1] + accel_z[i] / 30
    
    pos_x = np.zeros_like(accel_x)
    for i in range(1, len(vel_x)):
        pos_x[i] = .5 * accel_x[i-1] / 30**2 + vel_x[i-1] / 30 + pos_x[i-1]
    pos_y = np.zeros_like(accel_y)
    for i in range(1, len(vel_y)):
        pos_y[i] = .5 * accel_y[i-1] / 30**2 + vel_y[i-1] / 30 + pos_y[i-1]
    pos_z = np.zeros_like(accel_z)
    for i in range(1, len(vel_z)):
        pos_z[i] = .5 * accel_z[i-1] / 30**2 + vel_z[i-1] / 30 + pos_z[i-1]

    
    # Plot position animation
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax.add_subplot(111, projection='3d')
    point, = ax.plot([], [], [], color='r', marker=".")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Position from Acceleration')

    def update(frame):
        point.set_data(pos_x[frame], pos_y[frame])
        point.set_3d_properties(pos_z[frame])
        ax.set_xlim(min(pos_x), max(pos_x))
        ax.set_ylim(min(pos_y), max(pos_y))
        ax.set_zlim(min(pos_z), max(pos_z))

    ani = FuncAnimation(fig, update, frames=len(pos_x), interval=30)
    ani.save("plots/scratch/position_animation.gif", writer='pillow')


def plot_all_3d():
    fig = plt.figure()
    fig.suptitle('Position of Accelerometer Data in 3D Space')

    fig, axs = plt.subplots(3, 9, figsize=(50, 15))

    for i in range(1, 28):



if __name__ == "__main__":
    # Loop through and call plot one for a1-a27

    plot_all_2d()
    # plot_all_3d()




    # TODO: 
    #Plot all actions on one page -- done

    # plot one person with avg of 4 trials and std --done
    # Plot avg of all trials, all people of each action and std --done

