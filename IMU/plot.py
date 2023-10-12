import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# Path File:
file= "/home/abhi/data/utd-mhad/Inertial/a1_s1_t1_inertial.mat"

# Load data from .mat file
data = scipy.io.loadmat(file)
sn = np.array(data['d_iner'])

# Extract accelerometer data
accel_x = sn[:, 0]
accel_y = sn[:, 1]
accel_z = sn[:, 2]

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
plt.show()
