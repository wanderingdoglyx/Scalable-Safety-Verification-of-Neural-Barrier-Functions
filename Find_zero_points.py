# Load pre-trained model
import torch
import ann
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dtype=torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_eg1_prajna_original(number):
    x1 = np.linspace(-3.5, 2.0, num=number, endpoint=True)
    x2 = np.linspace(-2.0, 1.0, num=number, endpoint=True)

    return x1,x2
# load model
model = ann.gen_nn()
model.load_state_dict(torch.load('pre-trained.pt'), strict=True)
model.zero_grad()
# The initial range
x1 = np.linspace(-3.5, 2.0, num=100, endpoint=True)
x2 = np.linspace(-2.0, 1.0, num=100, endpoint=True)
xx = []
bx = []
bx0 = []
for account_x1, ele_x1 in enumerate(x1):
    for account_x2, ele_x2 in enumerate(x2):
        temp_xx = [ele_x1, ele_x2]
        xx.append(temp_xx)
        temp_xi = torch.tensor(temp_xx)
        temp_yi = model(temp_xi)
        temp_yi = temp_yi.detach().numpy()
        temp_yi = -temp_yi[0]
        bx.append([ele_x1, ele_x2, temp_yi])
        if abs(temp_yi) < 0.01:
            bx0.append([ele_x1, ele_x2, temp_yi])

# Plot the surface.
xyz_test=np.array(bx)


#print(xyz_test,"xyz_test")

fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')

surf = ax.plot_trisurf(xyz_test[:,0], xyz_test[:,1], xyz_test[:,2], cmap=cm.jet, linewidth=0)
fig.colorbar(surf)

ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(5))

fig.tight_layout()

plt.show() #
