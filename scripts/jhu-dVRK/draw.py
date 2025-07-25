import numpy as np
import matplotlib.pyplot as plt

# Load the data from two text files
file1 = '/home/xle6/dvrk_teleop_data/July_11/MTML_internal.txt'
file2 = '/home/xle6/dvrk_teleop_data/July_11/MTML_total.txt'

predicted_force = '/home/xle6/dvrk_teleop_data/July_11/MTML_force.txt'
measured_force = '/home/xle6/dvrk_teleop_data/July_11/MTML_total_force.txt'

# Load assuming space-separated values
data1 = np.loadtxt(file1, usecols=(0, 1, 2, 3, 4, 5))
data2 = np.loadtxt(file2)

# Compute the difference
diff = data1 - data2
rmse = np.sqrt(np.mean(diff**2, axis=0))
max = np.max(data2, axis = 0)
min = np.min(data2, axis=0)

nrmse = 100 * rmse/(max-min)
print(nrmse)
# Plotting
joint_labels = [f'Joint{i+1}' for i in range(data1.shape[1])]
timesteps = np.arange(data1.shape[0])

fig, axs = plt.subplots(data1.shape[1], 1, figsize=(10, 2 * data1.shape[1]), sharex=True)
fig.suptitle('Joint Angles and Their Differences', fontsize=16)

for i in range(data1.shape[1]):
    axs[i].plot(timesteps, data1[:, i], label='predicted', color='blue')
    axs[i].plot(timesteps, data2[:, i], label='measured', color='orange')
    axs[i].plot(timesteps, diff[:, i], label='Difference', color='green', linestyle='--')
    axs[i].set_ylabel(joint_labels[i])
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

axs[-1].set_xlabel('Timestep')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.show()


# Load assuming space-separated values
data3 = np.loadtxt(predicted_force, usecols=(0, 1, 2))
data4 = np.loadtxt(measured_force, usecols=(0, 1, 2))

# Compute the difference
diff = data3 - data4
rmse = np.sqrt(np.mean(diff**2, axis=0))
max = np.max(data4, axis = 0)
min = np.min(data4, axis=0)

nrmse = 100 * rmse/(max-min)
print(nrmse)
# Plotting
joint_labels = [f'Axis{i+1}' for i in range(data3.shape[1])]
timesteps = np.arange(data3.shape[0])

fig, axs = plt.subplots(data3.shape[1], 1, figsize=(10, 2 * data3.shape[1]), sharex=True)
fig.suptitle('Cartisian force', fontsize=16)

for i in range(data3.shape[1]):
    axs[i].plot(timesteps, data3[:, i], label='predicted', color='blue')
    axs[i].plot(timesteps, data4[:, i], label='measured', color='orange')
    #axs[i].plot(timesteps, diff[:, i], label='Difference', color='green', linestyle='--')
    axs[i].set_ylabel(joint_labels[i])
    axs[i].legend(loc='upper right')
    axs[i].grid(True)

axs[-1].set_xlabel('Timestep')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
