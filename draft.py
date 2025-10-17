import torch
import torch.nn as nn
import numpy as np
from mayavi import mlab
from mayavi.mlab import *
from Psatz_inner_bound_matlab import *
import superp
import prob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LinearLocator

def gen_act_data(model,S_overl):#####################################
	act_set_point = []
	sample_x = torch.linspace(prob.DOMAIN[0][0], prob.DOMAIN[0][1], int(superp.PLOT_LEN_B[0]))
	sample_y = torch.linspace(prob.DOMAIN[1][0], prob.DOMAIN[1][1], int(superp.PLOT_LEN_B[1]))
	sample_z = torch.linspace(prob.DOMAIN[2][0], prob.DOMAIN[2][1], int(superp.PLOT_LEN_B[2]))
	grid_xyz = torch.meshgrid([sample_x, sample_y, sample_z])
	Layers = get_layers(model)
	H1w = Layers[1]  # Hidden layer pre-act = Layer 1
	H1a = Layers[2]  # Hidden layer active = Layer 2
	flatten_xyz = [torch.flatten(grid_xyz[i]) for i in range(len(grid_xyz))]
	for i in range(len(flatten_xyz[0])):
		x_test = torch.Tensor([flatten_xyz[0][i], flatten_xyz[1][i], flatten_xyz[2][i]])
		[out_w, out_a, act] = output_forward_activation(x_test, H1w, H1a)
		if any(all(act == s_i) for s_i in S_overl):
			act_set_point.append(x_test)
	plot_input = torch.stack(act_set_point, 1)

	return plot_input

def mlab_plt_activated(S_overl):##############################################################
	# generating nn_output for plotting
	model = ann.gen_nn()
	model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)
	plot_input = gen_act_data(model, S_overl)
	# Plot the surface.
	xyz_input = np.array(plot_input)
	x=xyz_input[0]
	y=xyz_input[1]
	z=xyz_input[2]

	xyz=[]

	for index, ele in enumerate(x):
		tem=[x[index],y[index],z[index]]
		xyz.append(tem)

	xyz=np.array(xyz)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	surf = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], cmap=cm.jet, linewidth=0)
	fig.colorbar(surf)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(6))
	ax.zaxis.set_major_locator(MaxNLocator(5))

	# For x-axis limit
	matplotlib.axis.Axis.set_xlim(-2, 2)

	# For y-axis limit
	matplotlib.axis.Axis.set_ylim(-2, 2)

	# For z-axis limit
	matplotlib.axis.Axis.set_zlim(-3.14/2, 3.14/2)

	fig.tight_layout()

	plt.show()

model = ann.gen_nn()
model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)
#S_overl=torch.load('S_overl.pt')
#plot_input = gen_act_data(model, S_overl)
plot_input=torch.load('plot_input.pt')
xyz_input = np.array(plot_input)
x = xyz_input[0]
y = xyz_input[1]
z = xyz_input[2]

xyz = []
for index, ele in enumerate(x):
    tem = [x[index], y[index], z[index]]
    xyz.append(tem)

xyz = np.array(xyz)
xyz_tensor=torch.from_numpy(xyz)

xyz_bx_eq0=[]
for i in xyz_tensor:
	tem=model(i)
	if abs(tem) <0.1:
		xyz_bx_eq0.append(i.detach().numpy() )
xyz_bx_eq0=np.array(xyz_bx_eq0)
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(projection='3d')

#surf = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2], cmap=cm.jet, linewidth=0)
#fig.colorbar(surf)
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2] ,marker=".",c='b')
#ax.xaxis.set_major_locator(MaxNLocator(5))
#ax.yaxis.set_major_locator(MaxNLocator(6))
#ax.zaxis.set_major_locator(MaxNLocator(5))
ax.scatter(xyz_bx_eq0[:, 0], xyz_bx_eq0[:, 1], xyz_bx_eq0[:, 2] ,marker="1",c='y')
fig.tight_layout()

plt.show()