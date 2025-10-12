import torch
import torch.nn as nn
import ann
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import itertools
# import plot
import prob
import superp
from veri_util import *
from visualization import *
#def activate_set_tracking()
def enumarate_intersections(x,H1w,H1a,model,M,S_overl,B,limit):
    XS = []
    S_receive = []
    # Compute activated set S of x
    [out_w, out_a, activated] = output_forward_activation(x, H1w, H1a)
    activated = torch.reshape(activated, [len(activated), 1])
    activated_set = activated
    W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, activated_set)
    if any((activated_set == s_i).all() for s_i in S_overl) or limit<=1:
        return S_overl
    S_overl.append(activated_set)
    if limit<=0:
        return S_overl
    # compute polyhedron and update M
    XS.append([B_act, B_inact])
    M.append([B_act, B_inact, [W_overl,r_overl]])

    # compute boundary condition of polyhedron
    W_Bound = torch.Tensor(B_act[0] + B_inact[0])
    r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
    x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
    x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
    res_zero = linprog(c=[0, 0],
                       A_ub=W_Bound, b_ub=-r_Bound,
                       A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b),
                       method='simplex')
    if not res_zero.success:
        return S_overl
    # Solve linear program
    for i in range(len(W_Bound)):
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, activated_set)
        W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        W_cons = torch.cat((torch.reshape(W_Bound[i],[1,len(W_Bound[i])]), W_overl))
        r_cons = torch.cat((torch.reshape(r_Bound[i],[1,len(r_Bound[i])]), r_overl))
        res = linprog(c=[0, 0],
                    A_ub=W_Bound, b_ub=-r_Bound,
                    A_eq=W_cons, b_eq=-r_cons,
                    bounds=(x1b, x2b), method='simplex')
        x_prime = torch.Tensor(res.x)
        [H1w_prime, H1a_prime, activated_prime] = output_forward_activation(x_prime, H1w, H1a)

        if res.success: ######################################################################################
            print(x_prime, "x_prime")##################################
            print(activated_prime, " activated_prime")###############################
            if len(B) == 0:
                B.append(x_prime)

            limit -= 1
            S_receive = enumarate_intersections(x_prime, H1w, H1a, model, M, S_overl, B, limit)
            if S_receive != None:
                for i in S_overl:
                    if not ((i == s_i).all() for s_i in S_overl):
                        S_overl.append(i)

                if not any((x_prime == b_i).all() for b_i in B):
                    B.append(x_prime)
    return S_overl, B

# Find activated set such that b(x)=0
def find_zerolevelset(model):
    S = []
    l = [False, True]
    List = [list(i) for i in itertools.product(l, repeat=5)]
    for i in List:
        act = torch.reshape(torch.Tensor(i), [len(i), 1])
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)
        W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
        x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
        res_zero = linprog(c=[1, 1],
                           A_ub=W_Bound, b_ub=-r_Bound,
                           A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b),
                           method='simplex')
        if res_zero.success == True:
            S.append(act)
            print(res_zero)
    return S

def identify_zerolevelset(model,act):
    # act = torch.Tensor([[False], [True], [True], [True], [True]])
    W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)
    W_Bound = torch.Tensor(B_act[0] + B_inact[0])
    r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
    x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
    x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
    # res_zero = linprog(c=[1, 1], A_ub=W_Bound, b_ub=r_Bound, A_eq=W_overl, b_eq=r_overl,
    #                    bounds=(x1b, x2b), method='simplex')
    res_zero = linprog(c=[1, 1], A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b), method='simplex')
    print(res_zero.success)
    if res_zero.success == True:
        print(res_zero)

# Define main test function
def HeuristicSearch():
    # Init
    M = []
    S_overl = []
    S_overl_veri = []
    B = []

    # Load pre-trained model
    model = ann.gen_nn()
    model.load_state_dict(torch.load('pre-trained.pt'), strict=True)

    # Get layers' output
    Layers = get_layers(model)

    # Prajna 07 example
    H1w = Layers[1] # Hidden layer pre-act = Layer 1
    H1a = Layers[2] # Hidden layer active = Layer 2

    # Get output and activated set when given an input
    x = torch.Tensor([-2,  0])
    [out_w, out_a, activated] = output_forward_activation(x,H1w,H1a)
    activated = torch.reshape(activated,[len(activated),1])
    activated_set = activated

    S_overl_veri = find_zerolevelset(model)
    print('S_overl_veri',S_overl_veri)
    limit = 20
    S_overl, B = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)
    print('S_overl',S_overl)
    print('B', B)

    ############# Visualization #############
    # plt1 = visualize_activated_set(model, S_overl_veri, color='#17becf')
    # # plt1 = visualize_zero(model, plt1)
    # plt1.xlim([prob.DOMAIN[0][0], prob.DOMAIN[0][1]])
    # plt1.ylim([prob.DOMAIN[1][0], prob.DOMAIN[1][1]])
    # plt1.show()
    #
    # plt2 = visualize_activated_set(model, S_overl, color='blue')
    # # plt2 = visualize_zero(model,plt2)
    # plt2.xlim([prob.DOMAIN[0][0], prob.DOMAIN[0][1]])
    # plt2.ylim([prob.DOMAIN[1][0], prob.DOMAIN[1][1]])
    # plt2.show()
    # # print(S_overl==S_overl_veri)
    ############# -------------- #############



