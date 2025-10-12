import torch
import torch.nn as nn
import ann
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
    # plot_sample_x = np.linspace(prob.DOMAIN[0][0], prob.DOMAIN[0][1], superp.PLOT_LEN_B[0])
    # plot_sample_y = np.linspace(prob.DOMAIN[1][0], prob.DOMAIN[1][1], superp.PLOT_LEN_B[1])
    # barrier_plot_nn_input = np.meshgrid(plot_sample_x, plot_sample_y)
    barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
    for i in barrier_plot_nn_input:
        [out_w, out_a, activated] = output_forward_activation(i, H1w, H1a)
        activated = torch.reshape(activated, [len(activated), 1])
        activated_set = activated
        # if any(all(activated_set == s_i) for s_i in S_overl):
        if -0.05 < model(i) < 0.05:
            fig.scatter(i[0],i[1],c=color)
        # else:
        #     plt.scatter(i[0], i[1], c='#17becf')
    return fig

def visualize_zero(model,fig,color='#2ca02c'):
    # plot_sample_x = np.linspace(prob.DOMAIN[0][0], prob.DOMAIN[0][1], superp.PLOT_LEN_B[0])
    # plot_sample_y = np.linspace(prob.DOMAIN[1][0], prob.DOMAIN[1][1], superp.PLOT_LEN_B[1])
    # barrier_plot_nn_input = np.meshgrid(plot_sample_x, plot_sample_y)
    barrier_plot_nn_input = gen_plot_data(prob.DOMAIN, superp.PLOT_LEN_B)
    for i in barrier_plot_nn_input:
        if -0.05 < model(i) < 0.05:
            fig.scatter(i[0],i[1],c=color)
        # else:
        #     plt.scatter(i[0], i[1], c='#17becf')
    return fig