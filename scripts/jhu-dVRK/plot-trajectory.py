import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

""" plot 3D trajectory of each component. """
image_root = '../../../Plot_data/'
true_pos = np.loadtxt(image_root + 'vel_array_0603.txt', skiprows=1)
expect_pos = np.loadtxt(image_root + 'vel_array_exp_0603.txt', skiprows=1)
gamma = -0.714
beta = 0.5

m1_force = np.loadtxt(image_root + 'vel_m1_force_0603.txt', skiprows=1, usecols=(0, 1, 2))
m2_force = np.loadtxt(image_root + 'vel_m2_force_0603.txt', skiprows=1, usecols=(0, 1, 2))
puppet_force = np.loadtxt(image_root + 'vel_puppet_force_0603.txt', skiprows=1, usecols=(0, 1, 2))
total_force = beta * m1_force + (1-beta) * m2_force
puppet_force = gamma * puppet_force

fig = plt.figure()
ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot(true_pos[:,0], true_pos[:,1], true_pos[:,2], color='blue', linestyle='-', marker='o')
ax.plot(expect_pos[:,0], expect_pos[:,1], expect_pos[:,2], color='orange', linestyle='-', marker='o')

# ax = fig.add_subplot(1,2,2, projection='3d')
# ax.plot(expect_pos[:,0], expect_pos[:,1], expect_pos[:,2], color='blue', linestyle='-', marker='o')
plt.show()

# import plotly.graph_objects as go
# fig = go.Figure(data=go.Scatter3d(
#     x=true_pos[:,0], y=true_pos[:,1], z=true_pos[:,2],
#     mode='lines+markers',
#     marker=dict(size=4, color=true_pos[:,2], colorscale='Viridis'),
#     line=dict(width=2, color='blue')
# ))

# fig.update_layout(
#     title='3D Trajectory',
#     scene=dict(
#         xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
#     ),
#     width=700, height=700
# )
# fig.show()