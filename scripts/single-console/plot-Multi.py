import matplotlib.pyplot as plt
import matplotlib
import numpy as np

true_pos = np.loadtxt(f'/home/xle6/dvrk_teleop_data/July_29/multi_array.txt', skiprows=1)
expect_pos = np.loadtxt(f'/home/xle6/dvrk_teleop_data/July_29/multi_array_exp.txt', skiprows=1)
gamma = -1.0
beta = 0.5

m1_force = np.loadtxt(f'/home/xle6/dvrk_teleop_data/July_29/MTML_total_force.txt', skiprows=1, usecols=(0, 1, 2))
m2_force = np.loadtxt(f'/home/xle6/dvrk_teleop_data/July_29/MTMR_total_force.txt', skiprows=1, usecols=(0, 1, 2))
puppet_force = np.loadtxt(f'/home/xle6/dvrk_teleop_data/July_29/PSM_total_force.txt', skiprows=1, usecols=(0, 1, 2))
total_force = beta * m1_force + (1-beta) * m2_force
puppet_force = gamma * puppet_force

# puppet_force[:,1] = -1 * puppet_force[:,1]
# # capture when contacting
# true_pos = true_pos[750:]
# expect_pos = expect_pos[750:]
# total_force = total_force[750:]
# puppet_force = puppet_force[750:]

m1_force_norm = np.linalg.norm(m1_force, axis=1)
m2_force_norm = np.linalg.norm(m2_force, axis=1)
puppet_force_norm = np.linalg.norm(puppet_force, axis=1)
total_force_norm = beta * m1_force_norm + (1-beta) * m2_force_norm
num = len(m1_force)
x = np.arange(num) * 0.00066

# RMSE for pos
std_pos = np.std(true_pos, axis=0)
rmse_pos = np.sqrt(np.mean((true_pos - expect_pos)**2, axis=0))
print(f"rmse of pos: {rmse_pos*1000} mm")
print("")

# Normalized-RMSE for force (divided by range)
force_max = np.max(puppet_force, axis=0)
force_min = np.min(puppet_force, axis=0)

rmse_force = np.sqrt(np.mean((total_force - puppet_force)**2, axis=0))
nrmse_force = rmse_force / (force_max - force_min)
print(f"rmse_force: {rmse_force} N")
print(f"nrmse of force: {nrmse_force*100} %")

matplotlib.rcParams['axes.linewidth'] = 1.5
lw = 2


# plot position
fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# X-axis force
axs[0].plot(x, true_pos[:,0]*1000, linewidth=lw, label='true_psm_x')
axs[0].plot(x, expect_pos[:,0]*1000, linewidth=lw, label='expect_psm_x')
axs[0].set_ylabel("Position X (mm)", fontsize=14, fontweight='bold')
axs[0].set_title("Position in X-axis", fontsize=16, fontweight='bold')
axs[0].legend()
axs[0].tick_params(labelsize=14)

# Y-axis force
axs[1].plot(x, true_pos[:,1]*1000, linewidth=lw, label='true_psm_y')
axs[1].plot(x, expect_pos[:,1]*1000, linewidth=lw, label='expect_psm_y')
axs[1].set_ylabel("Position Y (mm)", fontsize=14, fontweight='bold')
axs[1].set_title("Position in Y-axis", fontsize=16, fontweight='bold')
axs[1].legend()
axs[1].tick_params(labelsize=14)

# Z-axis force
axs[2].plot(x, true_pos[:,2]*1000, linewidth=lw, label='true_psm_z')
axs[2].plot(x, expect_pos[:,2]*1000, linewidth=lw, label='expect_psm_z')
axs[2].set_xlabel("Time (s)", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Position Z (mm)", fontsize=14, fontweight='bold')
axs[2].set_title("Position in Z-axis", fontsize=16, fontweight='bold')
axs[2].legend()
axs[2].tick_params(labelsize=14)

fig.tight_layout()



# plot force
fig, axs = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

# X-axis force
axs[0].plot(x, total_force[:,0], linewidth=lw, label='m1+m2_force_x')
axs[0].plot(x, puppet_force[:,0], linewidth=lw, label='puppet_force_x')
axs[0].set_ylabel("Force X (N)", fontsize=14, fontweight='bold')
axs[0].set_title("Force in X-axis", fontsize=16, fontweight='bold')
axs[0].legend()
axs[0].tick_params(labelsize=14)

# Y-axis force
axs[1].plot(x, total_force[:,1], linewidth=lw, label='m1+m2_force_y')
axs[1].plot(x, puppet_force[:,1], linewidth=lw, label='puppet_force_y')
axs[1].set_ylabel("Force Y (N)", fontsize=14, fontweight='bold')
axs[1].set_title("Force in Y-axis", fontsize=16, fontweight='bold')
axs[1].legend()
axs[1].tick_params(labelsize=14)

# Z-axis force
axs[2].plot(x, total_force[:,2], linewidth=lw, label='m1+m2_force_z')
axs[2].plot(x, puppet_force[:,2], linewidth=lw, label='puppet_force_z')
axs[2].set_xlabel("Time (s)", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Force Z (N)", fontsize=14, fontweight='bold')
axs[2].set_title("Force in Z-axis", fontsize=16, fontweight='bold')
axs[2].legend()
axs[2].tick_params(labelsize=14)

fig.tight_layout()


""" plot error histogram """
from scipy.stats import skew
hist_axis = ['X', 'Y', 'Z']
for i in range(puppet_force.shape[1]):
    diff_force = total_force[:,i] - puppet_force[:,i]
    mu = np.mean(diff_force)
    sigma = np.std(diff_force)

    bin_edges = np.histogram_bin_edges(diff_force, bins='auto')

    plt.figure(figsize=(6,4))
    counts, bins, patches = plt.hist(
        diff_force,
        bins = bin_edges,
        density=False,
        color = 'skyblue'
    )    
    plt.axvline(mu, color = 'red', linestyle='--',
                label = f"mean = {mu:.2f}")
    plt.axvline(mu + sigma, color = 'green', linestyle=':',
                label = f"+sigma = {sigma:.2f}")
    plt.axvline(mu - sigma, color = 'green', linestyle=':',
                label = f"-sigma = {sigma:.2f}")
    plt.xlabel("Force error (N)")
    plt.ylabel("Counts")
    plt.title(f"Force tracking error histogram {hist_axis[i]} axis\n mean={mu:.2f}, std={sigma:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

