'''
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
from numpy import array
from numpy.linalg import norm
import torch
import torch.nn as nn
import superp
import ann
import data
import train
import time
# comment this line if matplotlib, mayavi, or PyQt5 was not successfully installed
import plot
'''
'''
a=torch.tensor([([[ True],
        [ True],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [ True],
        [ True],
        [False]]), ([[ True],
        [ True],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[False],
        [ True],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[ True],
        [ True],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[False],
        [ True],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False]]), ([[False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [ True],
        [False],
        [False]]), ([[ True],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [ True],
        [ True],
        [False]]),([[ True],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [ True],
        [ True],
        [ True],
        [False]]),([[ True],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [ True],
        [False],
        [False],
        [False],
        [False],
        [ True],
        [ True],
        [ True],
        [False]])])
s_label=[]
s=[]

b=a*1

for index,ele in enumerate(b):
    s_label.append(index)
    s.append(ele)
location=s_label[:]
print(location,'location')

for i in s_label:
    for j in s_label[i+1:]:
        di=s[i]-s[j]
        len_dis=torch.linalg.vector_norm(di)
        location[j] = location[i]+len_dis


i=9
for j in s_label:
    di = s[i] - s[j]
    #len_dis = torch.linalg.vector_norm(di).numpy()
    norm_l1 = norm(di)
    location[j] = location[i] + norm_l1

def find_bx_eq_zero_activated_set():

    a=1

model = ann.gen_nn()
model.load_state_dict(torch.load('pre-trained.pt'), strict=True)
'''
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
from numpy import array
from numpy.linalg import norm
import torch
import torch.nn as nn
import superp



a=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
b=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, True, True, True, True, True, True, True, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False]
c=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, True, True, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, True, False, True, True, True, True, True, True, False, False, True, False, False, False, True, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True, True, False, False, True, True, False, False, False, False, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, True, False, False, False, False, False, False]
judge_data=[a,b,c]

def judge_NCBF(judge_data):
    overall_judge_decision=[]
    for index, ele in enumerate(judge_data):

        judge_data[index]=np.array(judge_data[index])
        judge_data[index]=judge_data[index]*1
        judge_index=abs(sum(judge_data[index]))
        if judge_index>0:
            judge_decision=[False]
        else:
            judge_decision=[True]

        overall_judge_decision.append(judge_decision)
        print(judge_decision,'Subject decision')

    overall_judge_decision=np.array(overall_judge_decision)*1
    pass_length=len(overall_judge_decision)
    Overall_judge_decision=abs(sum(overall_judge_decision))
    if Overall_judge_decision < pass_length:
        final_decision=[False]
    else:
        final_decision=[True]
    print(final_decision,'Final_decision')
    return overall_judge_decision,final_decision
