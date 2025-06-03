import matplotlib.pyplot as plt
import matplotlib
import numpy as np

true_pos = np.loadtxt('bi_array_0603.txt', skiprows=1)
expect_pos = np.loadtxt('bi_array_exp_0603.txt', skiprows=1)
error = np.abs(true_pos - expect_pos) * 1000

master_force = np.loadtxt('bi_m1_force_0603.txt', skiprows=1)
puppet_force = np.loadtxt('bi_puppet_force_0603.txt', skiprows=1)

master_force_norm = np.linalg.norm(master_force, axis=1)
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
plt.figure()
# # plt.subplot(312)
# plt.plot(x, error[:,0], linewidth=lw, label='error in x axis')
# plt.plot(x, error[:,1], linewidth=lw, label='error in y axis')
# plt.plot(x, error[:,2], linewidth=lw, label='error in z axis')
# plt.xticks(fontsize=14, fontweight='bold')
# plt.yticks(fontsize=14, fontweight='bold')
# plt.xlabel("Time (s)", fontsize=14, fontweight='bold')
# plt.ylabel("Position (mm)", fontsize=14, fontweight='bold')
# plt.title("Absolute position error", fontsize=16, fontweight='bold')
# plt.legend()
# plt.figure()
# plt.subplot(313)
plt.plot(x, master_force[:,2], linewidth=lw, label='master_force')
plt.plot(x, puppet_force[:,2], linewidth=lw, label='puppet_force')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.xlabel("Time (s)", fontsize=14, fontweight='bold')
plt.ylabel("Force (N)", fontsize=14, fontweight='bold')
plt.title("Force in Z-axis", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()
# plt.figure()
# # plt.subplot(224)
# plt.plot(m1_force_norm, label='m1_force')
# plt.plot(m2_force_norm, label='m2_force')
# plt.plot(puppet_force_norm, label='puppet_force')
# plt.legend()
# plt.show()




