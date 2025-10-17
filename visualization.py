import torch
import torch.nn as nn
import ann
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import itertools
import prob
import superp
from veri_util import *

# sampling data for plotting barrier, vector field and scattering sample points
def gen_plot_data(region, len_sample):
    grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in
                   range(prob.DIM)]  # gridding each dimension
    mesh = torch.meshgrid(grid_sample)  # mesh the gridding of each dimension
    flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))]  # flatten the list of meshes
    plot_data = torch.stack(flatten, 1)  # stack the list of flattened meshes
    return plot_data

# Todo activated_set visualization
def visualize_activated_set(model,S_overl,fig=plt,color='#17becf'):
    # Get layers' output
    Layers = get_layers(model)
    # Prajna 07 example
    H1w = Layers[1]  # Hidden layer pre-act = Layer 1
    H1a = Layers[2]  # Hidden layer active = Layer 2
    BoundaryMode = False # 0 for activated based 1 for boundary based
    barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
    for i in barrier_plot_nn_input:
        [out_w, out_a, activated] = output_forward_activation(i, H1w, H1a)
        activated = torch.reshape(activated, [len(activated), 1])
        activated_set = activated
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, activated_set)
        W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
        x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
        if not BoundaryMode:
            if any(all(activated_set == s_i) for s_i in S_overl):
                if -0.05 < model(i) < 0.05:
                    fig.scatter(i[0],i[1],c=color)
        else:
            if (torch.reshape(torch.matmul(W_Bound,i),[len(W_Bound),1])<=-r_Bound).all():
                fig.scatter(i[0], i[1], c=color)
    return fig

def visualize_point(model,activated,fig=plt,color='#17becf'):
    barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
    for i in barrier_plot_nn_input:
        activated = torch.reshape(activated, [len(activated), 1])
        activated_set = activated
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, activated_set)
        W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        if (torch.reshape(torch.matmul(W_Bound,i),[5,1])<=-r_Bound).all():
            fig.scatter(i[0], i[1], c=color)
    return fig

def Input_Partition(model,activated):
    l = [False, True]
    List = [list(i) for i in itertools.product(l, repeat=5)]
    color = ['b','g','r','c','m','y','k','w']
    plt1 = visualize_point(model, activated)
    for i in List:
        act = torch.reshape(torch.Tensor(i), [len(i), 1])
        c = color[np.random.randint(0, 7)]
        plt1 = visualize_point(model, act, plt1, color=c)
    plt1.xlim([prob.DOMAIN[0][0], prob.DOMAIN[0][1]])
    plt1.ylim([prob.DOMAIN[1][0], prob.DOMAIN[1][1]])
    plt1.show()
    return plt1

def visualize_zero(model,fig,color='#2ca02c'):
    barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
    for i in barrier_plot_nn_input:
        if -0.05 < model(i) < 0.05:
            fig.scatter(i[0],i[1],c=color)
        # else:
        #     plt.scatter(i[0], i[1], c='#17becf')
    return fig