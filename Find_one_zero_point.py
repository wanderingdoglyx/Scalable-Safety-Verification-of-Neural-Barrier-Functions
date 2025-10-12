# Load pre-trained model
import torch
import numpy as np
#import ann

dtype=torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
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
'''
def find_one_zero_point_autograd(data,model_input):
    model=model_input

    # randomly pick initial points
    index = np.random.randint(0, len(data))
    xi = data[index]

    # back propagation training
    learning_rate = 1e-1
    x_restart = data[index]
    x_i = torch.tensor(xi, requires_grad=True)
    y_i = model(x_i)
    loss = abs(y_i)
    epoch = 0.0
    while loss > 0.000001:
        epoch = epoch + 1
        loss.backward()

        beta = 0.01
        gamma = 1.0
        rate = learning_rate / (1 + beta * epoch ** gamma)
        with torch.no_grad():
            x_i = x_i - rate * x_i.grad
            x_i.grad = None
        x_i.requires_grad = True
        y_i = model(x_i)
        loss = abs(y_i)
        if epoch > 1000:
            with torch.no_grad():
                x_i = torch.tensor(x_restart, requires_grad=True)
                print(x_i, 'Please restart the function and run again')
                epoch = 0

    return x_i

