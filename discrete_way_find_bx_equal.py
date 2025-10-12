# Load pre-trained model
import torch
import torch.nn as nn
import ann
import matplotlib.pyplot as plt
import numpy as np
import random
import prob
import superp

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

# randomly pick initial points
x1_select=np.random.choice(x1,4)
x2_select=np.random.choice(x2,4)
xx=np.stack((x1_select,x2_select),axis=1)

# back propagation training
learning_rate=1e-3

for count, ele in enumerate(xx):
    x_i=torch.tensor(ele, requires_grad=True)
    #x_i.requires_grad= True
    print(x_i,"x_i")###################################
    y_i=model(x_i)
    #y_i_abs=torch.abs(y_i)
    #loss=y_i_abs
    loss=(y_i).pow(2).sum()
    epoch=0.0
    while loss>0.0001:
        loss.backward()
        print(x_i.grad,'grad')##############################################
        epoch=epoch+1
        print(x_i,'xi_value')######################################
        beta=0.01
        gamma=1.0
        rate = learning_rate / (1 + beta * epoch ** gamma)
        with torch.no_grad():
            x_i= x_i -rate*x_i.grad
            x_i.grad=None
        print(x_i.requires_grad,"xi")####################################
        x_i.requires_grad = True
        y_i = model(x_i)
       # y_i_abs = torch.abs(y_i)
        #print(y_i_abs.sum(),"y_i")#########################################
        #loss = y_i_abs
        loss = (y_i).pow(2).sum()
        print(loss,"loss")

# Compute range of initial data set



