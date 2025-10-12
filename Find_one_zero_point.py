# Load pre-trained model
import torch
import numpy as np
import ann

dtype=torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_eg1_prajna_original(number):
    x1 = np.linspace(-3.5, 2.0, num=number, endpoint=True)
    x2 = np.linspace(-2.0, 1.0, num=number, endpoint=True)
    return x1,x2

# load model
model = ann.gen_nn()
model.load_state_dict(torch.load('pre-trained.pt'), strict=True)

# The initial range
x1 = np.linspace(-3.5, 2.0, num=100, endpoint=True)
x2 = np.linspace(-2.0, 1.0, num=100, endpoint=True)
x1_select=np.random.choice(x1,4)
x2_select=np.random.choice(x2,4)
xx=np.stack((x1_select,x2_select),axis=1)

# randomly pick initial points
index=np.random.randint(0,len(xx))
xi=xx[index]
# back propagation training
learning_rate=1e-1
x_restart=xx[index]
x_i=torch.tensor(xi, requires_grad=True)
y_i = model(x_i)
loss = abs(y_i)
epoch = 0.0 ##########################################################
while loss>0.00001:
    epoch=epoch+1
    loss.backward()
    #print(x_i.grad, 'grad')  ##############################################
    beta = 0.01
    gamma = 1.0
    rate = learning_rate / (1 + beta * epoch ** gamma)
    #print(x_i, 'xi_value')  ######################################
    with torch.no_grad():
        x_i= x_i -rate*x_i.grad
        x_i.grad=None

    #print(x_i.requires_grad, "xi")  ####################################
    x_i.requires_grad = True
    y_i = model(x_i)
    loss = abs(y_i)
    if epoch>600:
        with torch.no_grad():
            x_i = torch.tensor(x_restart, requires_grad=True)
            print(x_i,'Please restart the function and run again')
            epoch=0
    #print(epoch,"epoch")
    # y_i_abs = torch.abs(y_i)
    #print(loss,"loss")#########################################
    # loss = y_i_abs



def find_one_zero_point(data,model):

    model.zero_grad()

    # randomly pick initial points
    index = np.random.randint(0, len(data))
    xi = data[index]
    # back propagation training
    learning_rate = 1e-3

    x_i = torch.tensor(xi, requires_grad=True)

    delta = x_i*0+0.0001
    y_i = model(x_i)
    loss = abs(y_i)
    while loss > 0.000001:
        x_i_min = x_i - delta
        x_i_plus = x_i + delta

        y_i = model(x_i)
        y_i_min = model(x_i_min)
        y_i_plus = model(x_i_plus)

        loss = abs(y_i)

       # print(loss, 'loss')
        grad = (abs(y_i_plus) - abs(y_i_min)) / delta
       # print(grad,'grad')
        x_i = x_i - learning_rate * grad
       # print(x_i, "x_i")
    return x_i


def find_one_zero_point_autograd(data,model_input):
    model=model_input
    # randomly pick initial points
    index = np.random.randint(0, len(data))
    xi = data[index]

    # back propagation training
    learning_rate = 1e-1
    x_restart = xx[index]
    x_i = torch.tensor(xi, requires_grad=True)
    y_i = model(x_i)
    loss = abs(y_i)
    epoch = 0.0
    while loss > 0.00001:
        epoch = epoch + 1
        loss.backward()
        beta = 0.01
        gamma = 1.0
        rate = learning_rate / (1 + beta * epoch ** gamma)
        with torch.no_grad():
            x_i = x_i - rate * x_i.grad
            x_i.grad = None
        y_i = model(x_i)
        loss = abs(y_i)
        if epoch > 10000:
            with torch.no_grad():
                x_i = torch.tensor(x_restart, requires_grad=True)
                print(x_i, 'Please restart the function and run again')
                epoch = 0

    return x_i