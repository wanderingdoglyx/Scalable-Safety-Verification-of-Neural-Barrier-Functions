import matlab.engine
import matplotlib.pyplot
import numpy as np
import torch as torch
# from SumOfSquares import *
import plot3d
from heuristic_search import *
from scipy.optimize import minimize
from visualization import *
from plot import *
import time

def JointBound(S_overl):
    joint_pair = []
    index_buffer = []
    for i in S_overl:
        for j in S_overl:
            if (i == j).all():
                continue
            else:
                counter = 0
                buffer = -1
                for ind in range(len(i)):
                    if i[ind] != j[ind]:
                        counter += 1
                        buffer = ind
                if counter == 1 and buffer not in index_buffer:
                    joint_pair.append([i,j,buffer])
                    index_buffer.append(buffer)
    return joint_pair

def boundary_verifier(S_overl):
    print('Verify Boundaries of Activated Sets')
    model = ann.gen_nn()
    model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)
    joint_pairs = JointBound(S_overl)
    eng = matlab.engine.start_matlab()
    eng.cd(r'C:\Users\droid\Documents\Research\HSCC22_CaseStudy\Psatz_verify_Obs')
    flags = []
    FC = []
    for pair in joint_pairs:
        act0 = torch.Tensor(pair[0])
        act1 = torch.Tensor(pair[1])
        index = pair[2]
        W_overl0, r_overl0, B_act0, B_inact0 = activated_weight_bias(model, act0)
        W_Bound0 = torch.Tensor(-B_act0[0] + B_inact0[0])[index]
        r_Bound0 = torch.Tensor(-B_act0[1] - B_inact0[1])[index]
        W_overl1, r_overl1, B_act1, B_inact1 = activated_weight_bias(model, act1)
        print(W_Bound0)
        # print(W_Bound1)
        W_overl0 = matlab.double(W_overl0.tolist())
        r_overl0 = matlab.double(r_overl0.tolist())
        W_overl1 = matlab.double(W_overl1.tolist())
        r_overl1 = matlab.double(r_overl1.tolist())
        W_Bound0 = matlab.double(W_Bound0.tolist())
        r_Bound0 = matlab.double(r_Bound0.tolist())
        UNSAT = eng.nonlprog_boundary(W_overl0, r_overl0,
                                      W_overl1, r_overl1,
                                      W_Bound0, r_Bound0)
        if UNSAT == True:
            FC.append([pair[0],pair[1],pair[2]])
        flags.append(UNSAT)
    print(flags)
    return flags

def PsatzVerifier(S_overl):
    print('Verify inner part of Activated Sets')
    eng = matlab.engine.start_matlab()
    flags = []
    FC = []
    SOSflag = False
    # start = time.time()
    for i in S_overl:
        act = torch.Tensor(i)
        model = ann.gen_nn()
        model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)

        W_in_overl = matlab.double(W_overl.tolist())
        r_in_overl = matlab.double(r_overl.tolist())
        B_act_w = matlab.double(B_act[0].tolist())
        B_act_r = matlab.double(B_act[1].tolist())
        B_inact_w = matlab.double(B_inact[0].tolist())
        B_inact_r = matlab.double(B_inact[1].tolist())

        eng.cd(r'C:\Users\droid\Documents\Research\HSCC22_CaseStudy\Psatz_verify_Obs')
        if SOSflag:
            UNSAT = eng.psatz_prajna_verifier(W_in_overl, r_in_overl, B_act_w, B_act_r,
                                              B_inact_w, B_inact_r)
            if UNSAT == True:
                FC.append(act)
                UNSAT = eng.nonlinear_prog_checking(W_in_overl, r_in_overl, B_act_w, B_act_r,
                                              B_inact_w, B_inact_r)
        else:
            UNSAT = eng.nonlinear_prog_checking(W_in_overl, r_in_overl, B_act_w, B_act_r,
                                                B_inact_w, B_inact_r)
        flags.append(UNSAT)
    if flags==False:
        print('Success')
    return flags
    # end = time.time()
    # print(end-start)

def bfcnverifier(S_overl,model,unsafe_c,unsafe_r):
    print('Verify Barrier Function')
    # todo check if inside
    # x_rand_in = 0.5*unsafe_r*torch.rand(2)+unsafe_c
    # b_out = -model.forward(x_rand_in)
    # if b_out >= 0:
    #     UNSAT = True
    #     return UNSAT
    # check intersections
    eng = matlab.engine.start_matlab()
    flags = []
    FC = []
    unsafe_in_c = matlab.double(unsafe_c.tolist())
    unsafe_in_r = matlab.double(unsafe_r.tolist())
    for i in S_overl:
        act = torch.Tensor(i)
        model = ann.gen_nn()
        model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)

        W_in_overl = matlab.double(W_overl.tolist())
        r_in_overl = matlab.double(r_overl.tolist())
        B_act_w = matlab.double(B_act[0].tolist())
        B_act_r = matlab.double(B_act[1].tolist())
        B_inact_w = matlab.double(B_inact[0].tolist())
        B_inact_r = matlab.double(B_inact[1].tolist())

        eng.cd(r'C:\Users\droid\Documents\Research\HSCC22_CaseStudy\Psatz_verify_Obs')
        UNSAT = eng.barrierverifier(unsafe_in_c, unsafe_in_r, W_in_overl, r_in_overl, B_act_w, B_act_r,
                                            B_inact_w, B_inact_r)
        flags.append(UNSAT)
    print(flags)
    return flags

def main():
    # Init
    start = time.time()
    M = []
    S_overl = []
    S_overl_veri = []
    S = []
    B = []

    # Load pre-trained model
    model = ann.gen_nn()
    model.load_state_dict(torch.load('obstacle_50_1_lr01.pt'), strict=True)

    # Get layers' output
    Layers = get_layers(model)
    # Heuristic Search
    H1w = Layers[1]  # Hidden layer pre-act = Layer 1
    H1a = Layers[2]  # Hidden layer active = Layer 2
    x_init = torch.Tensor([[0, -1.85, 0]])
    x = find_one_zero_point_autograd(x_init, model)
    # Enumerate
    limit = 20
    ####################################
    S_overl = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)
    print(S_overl)
    for i in S_overl:
        S.append(i.tolist())

    # Psatz Verification
    unsafe_c = torch.Tensor([-1,0])
    unsafe_r = torch.Tensor([0.4])
    # bv_flags = bfcnverifier(S_overl, model, unsafe_c, unsafe_r)
    # b_flags = boundary_verifier(S_overl)
    # i_flags = PsatzVerifier(S_overl)
    end = time.time()
    print(len(S_overl))
    print(len(S_overl_veri))
    # print(bv_flags)
    # print(b_flags)
    # print(i_flags)
    print(end - start)
    plot_input = plot3d.gen_act_data(model, S_overl)
    print(len(plot_input))
    plot3d.plot_act_3d(S_overl)

    # plt2 = visualize_activated_set(model, S_overl, color='blue')
    # # plt2 = visualize_zero(model,plt2)
    # plt2.xlim([prob.DOMAIN[0][0], prob.DOMAIN[0][1]])
    # plt2.ylim([prob.DOMAIN[1][0], prob.DOMAIN[1][1]])
    # plt2.show()
    # # print(S_overl==S_overl_veri)

if __name__ == "__main__":
    main()

