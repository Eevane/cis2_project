import matplotlib.pyplot as plt
import matplotlib
import numpy as np

true_pos = np.loadtxt('multi_array_0605.txt', skiprows=1)
expect_pos = np.loadtxt('multi_array_exp_0605.txt', skiprows=1)
error = np.abs(true_pos - expect_pos) * 1000

m1_force = np.loadtxt('multi_m1_force_0605.txt', skiprows=1)
m2_force = np.loadtxt('multi_m2_force_0605.txt', skiprows=1)
puppet_force = np.loadtxt('multi_puppet_force_0605.txt', skiprows=1)
total = 0.5 * (m1_force + m2_force)

m1_force_norm = np.linalg.norm(m1_force, axis=1)
m2_force_norm = np.linalg.norm(m2_force, axis=1)
puppet_force_norm = np.linalg.norm(puppet_force, axis=1)
num = len(true_pos)
x = np.arange(num) * 0.002

# rmse = np.sqrt(np.mean((true_pos - expect_pos)**2))

matplotlib.rcParams['axes.linewidth'] = 1.5
lw = 2

plt.figure()
# plt.subplot(311)
plt.plot(x, true_pos[:,0]*1000, linewidth=lw, label='true_pos_x')
plt.plot(x, true_pos[:,1]*1000, linewidth=lw, label='true_pos_y')
plt.plot(x, true_pos[:,2]*1000, linewidth=lw, label='true_pos_z')
# plt.plot(expect_pos)
plt.plot(x, expect_pos[:,0]*1000, linewidth=lw, label='expect_pos_x')
plt.plot(x, expect_pos[:,1]*1000, linewidth=lw, label='expect_pos_y')
plt.plot(x, expect_pos[:,2]*1000, linewidth=lw, label='expect_pos_z')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel("Time (s)", fontsize=14, fontweight='bold')
plt.ylabel("Position (mm)", fontsize=14, fontweight='bold')
plt.title("Position tracking", fontsize=16, fontweight='bold')
plt.legend()
# plt.figure()
# # # plt.subplot(312)
# # plt.plot(x, error[:,0], linewidth=lw, label='error in x axis')
# # plt.plot(x, error[:,1], linewidth=lw, label='error in y axis')
# # plt.plot(x, error[:,2], linewidth=lw, label='error in z axis')
# # plt.xticks(fontsize=14, fontweight='bold')
# # plt.yticks(fontsize=14, fontweight='bold')
# # plt.xlabel("Time (s)", fontsize=14, fontweight='bold')
# # plt.ylabel("Position (mm)", fontsize=14, fontweight='bold')
# # plt.title("Absolute position error", fontsize=16, fontweight='bold')
# # plt.legend()
# # plt.figure()
# # plt.subplot(313)
# plt.plot(x, total[:,2], linewidth=lw, label='total_force')
# # plt.plot(x, m2_force[:,2], linewidth=lw, label='m2_force')
# plt.plot(x, puppet_force[:,2], linewidth=lw, label='puppet_force')
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.xlabel("Time (s)", fontsize=14, fontweight='bold')
# plt.ylabel("Force (N)", fontsize=14, fontweight='bold')
# plt.title("Force in Z-axis", fontsize=16, fontweight='bold')
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.figure()
# # # plt.subplot(224)
# # plt.plot(m1_force_norm, label='m1_force')
# # plt.plot(m2_force_norm, label='m2_force')
# # plt.plot(puppet_force_norm, label='puppet_force')
# # plt.legend()
# # plt.show()




fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# X-axis force
axs[0].plot(x, total[:,0], linewidth=lw, label='m1+m2_force_x')
axs[0].plot(x, puppet_force[:,0], linewidth=lw, label='puppet_force_x')
axs[0].set_ylabel("Force X (N)", fontsize=14, fontweight='bold')
axs[0].set_title("Force in X-axis", fontsize=16, fontweight='bold')
axs[0].legend()
axs[0].tick_params(labelsize=14)

# Y-axis force
axs[1].plot(x, total[:,1], linewidth=lw, label='m1+m2_force_y')
axs[1].plot(x, puppet_force[:,1], linewidth=lw, label='puppet_force_y')
axs[1].set_ylabel("Force Y (N)", fontsize=14, fontweight='bold')
axs[1].set_title("Force in Y-axis", fontsize=16, fontweight='bold')
axs[1].legend()
axs[1].tick_params(labelsize=14)

# Z-axis force
axs[2].plot(x, total[:,2], linewidth=lw, label='m1+m2_force_z')
axs[2].plot(x, puppet_force[:,2], linewidth=lw, label='puppet_force_z')
axs[2].set_xlabel("Time (s)", fontsize=14, fontweight='bold')
axs[2].set_ylabel("Force Z (N)", fontsize=14, fontweight='bold')
axs[2].set_title("Force in Z-axis", fontsize=16, fontweight='bold')
axs[2].legend()
axs[2].tick_params(labelsize=14)

fig.tight_layout()
plt.show()




