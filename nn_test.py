import  torch
from  torch import nn
from    torch.nn import functional as F
from    torch import optim
import numpy as np
import  torchvision
from    matplotlib import pyplot as plt
from rectangle_data import rectangle_data
from    utils import plot_image, plot_curve, one_hot
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x,y,xy=rectangle_data()
x1=x[:]
y1=y[:]
xy=np.array(xy)
xy1=xy[:]
xy2_t=xy[0:-1]
xy2=np.concatenate(([xy1[-1]], xy2_t))

xy_x=xy1
xy_y=xy2

x= torch.tensor(x).float()
y= torch.tensor(y).float()
xy_x=torch.tensor(xy_x).float()
xy_y=torch.tensor(xy_y).float()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        return x



net = Net().to(device)
net=net.float()
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []

for epoch in range(1):
    #print(epoch,"epoch")
    for batch_idx, x_t in enumerate(xy_x):
        #print(x_t,"x-t")
        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        #x = x.view(x.size(0), 28*28)
        # => [b, 10]
        y_t = xy_y[batch_idx]

        x_t=x_t.to(device)
        y_t=y_t.to(device)
        y_t.float()
        x_t.float()
        out = net(x_t.float())
        # [b, 10]

        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_t.float())

        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10==0:
            print(epoch, batch_idx, loss.item())

#plot_curve(train_loss)
# we get optimal [w1, b1, w2, b2, w3, b3]
FILE="model.pth"
torch.save(net.state_dict(),FILE)

#for x,y in enumerate(xy_x,xy_y):
    #x  = x.view(x.size(0), 28*28)
    #out = net(xy_x)
    # out: [b, 10] => pred: [b]
xy_test=random.choices(xy_x, k=100)


total_correct = 0
pre=[]
x_pre=[]
y_pre=[]
for x in xy_test:

    out = net(x)
    # out: [b, 10] => pred: [b]
    pre.append(out.detach().numpy())
    #x_pre.extend(out.detach().numpy()[0])
    #y_pre.extend(out.detach().numpy()[1])
#print(y_pre)
pre=np.array(pre[:])
x_pre=pre[:,0]
y_pre=pre[:,1]
print(pre,"pre")
plt.plot(x1,y1)
print(x_pre,"xpre")
print(y_pre,"ypre")
#plt.plot(pre)
plt.plot(x_pre,y_pre)
#plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()
