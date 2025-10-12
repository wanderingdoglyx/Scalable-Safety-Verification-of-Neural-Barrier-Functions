import matlab.engine
import numpy as np
import torch as torch
# from SumOfSquares import *
from heuristic_search import *
from scipy.optimize import minimize
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    model.load_state_dict(torch.load('pre-trained.pt'), strict=True)
    joint_pairs = JointBound(S_overl)
    eng = matlab.engine.start_matlab()
    eng.cd(r'E:\wustl\0.5\HSCC\Prajna_NBF_Psatz_verify')
    flags = []
    FC = []
    for pair in joint_pairs:
        act0 = torch.Tensor(pair[0])
        act1 = torch.Tensor(pair[1])
        index = pair[2]
        W_overl0, r_overl0, B_act0, B_inact0 = activated_weight_bias(model, act0)
        W_Bound0 = torch.Tensor(B_act0[0] + B_inact0[0])[index]
        r_Bound0 = torch.Tensor(-B_act0[1] - B_inact0[1])[index]
        W_overl1, r_overl1, B_act1, B_inact1 = activated_weight_bias(model, act1)
        print(W_Bound0)
        W_Bound = torch.Tensor(B_act0[0] + B_inact0[0])
        r_Bound = torch.Tensor(-B_act0[1] - B_inact0[1])
        x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
        x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
        res_zero = linprog(c=[1, 1],
                           A_ub=W_Bound, b_ub=-r_Bound,
                           A_eq=W_overl0, b_eq=-r_overl0, bounds=(x1b, x2b),
                           method='simplex')
        # print(W_Bound1)
        W_overl0 = matlab.double(W_overl0.tolist())
        r_overl0 = matlab.double(r_overl0.tolist())
        W_overl1 = matlab.double(W_overl1.tolist())
        r_overl1 = matlab.double(r_overl1.tolist())
        W_Bound0 = matlab.double(W_Bound0.tolist())
        r_Bound0 = matlab.double(r_Bound0.tolist())
        x_init = matlab.double(res_zero.x.tolist())
        UNSAT = eng.nonlprog_boundary(x_init,W_overl0, r_overl0,
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
        model.load_state_dict(torch.load('pre-trained.pt'), strict=True)
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)
        W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
        x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
        res_zero = linprog(c=[1, 1],
                           A_ub=W_Bound, b_ub=-r_Bound,
                           A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b),
                           method='simplex')
        W_in_overl = matlab.double(W_overl.tolist())
        r_in_overl = matlab.double(r_overl.tolist())
        B_act_w = matlab.double(B_act[0].tolist())
        B_act_r = matlab.double(B_act[1].tolist())
        B_inact_w = matlab.double(B_inact[0].tolist())
        B_inact_r = matlab.double(B_inact[1].tolist())
        x_init = matlab.double(res_zero.x.tolist())
        eng.cd(r'E:\wustl\0.5\HSCC\Prajna_NBF_Psatz_verify')
        if SOSflag:
            UNSAT = eng.psatz_prajna_verifier(x_init,W_in_overl, r_in_overl, B_act_w, B_act_r,
                                              B_inact_w, B_inact_r)
            if UNSAT == True:
                FC.append(act)
                UNSAT = eng.nonlinear_prog_checking(x_init,W_in_overl, r_in_overl, B_act_w, B_act_r,
                                              B_inact_w, B_inact_r)
        else:
            UNSAT = eng.nonlinear_prog_checking(x_init,W_in_overl, r_in_overl, B_act_w, B_act_r,
                                                B_inact_w, B_inact_r)
        flags.append(UNSAT)
    if flags==False:
        print('Success')
    return flags
    # end = time.time()
    # print(end-start)

def bfcnverifier(S_overl,model,unsafe_c,unsafe_r):
    print('Verify Barrier Function')
    # check if inside
    x_rand_in = 0.5*unsafe_r*torch.rand(2)+unsafe_c
    b_out = -model(x_rand_in)
    if b_out >= 0:
        UNSAT = True
        return UNSAT
    # check intersections
    eng = matlab.engine.start_matlab()
    flags = []
    FC = []
    for i in S_overl:
        act = torch.Tensor(i)
        model = ann.gen_nn()
        model.load_state_dict(torch.load('pre-trained.pt'), strict=True)
        W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, act)
        # W_Bound = torch.Tensor(B_act[0] + B_inact[0])
        # r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
        # x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
        # x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
        # res_zero = linprog(c=[1, 1],
        #                    A_ub=W_Bound, b_ub=-r_Bound,
        #                    A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b),
        #                    method='simplex')

        unsafe_in_c = matlab.double(unsafe_c.tolist())
        unsafe_in_r = matlab.double(unsafe_r.tolist())
        W_in_overl = matlab.double(W_overl.tolist())
        r_in_overl = matlab.double(r_overl.tolist())
        B_act_w = matlab.double(B_act[0].tolist())
        B_act_r = matlab.double(B_act[1].tolist())
        B_inact_w = matlab.double(B_inact[0].tolist())
        B_inact_r = matlab.double(B_inact[1].tolist())
        # x_init = matlab.double(res_zero.x.tolist())

        eng.cd(r'E:\wustl\0.5\HSCC\Prajna_NBF_Psatz_verify')
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
    model.load_state_dict(torch.load('pre-trained.pt'), strict=True)

    # Get layers' output
    Layers = get_layers(model)
    # Heuristic Search
    H1w = Layers[1]  # Hidden layer pre-act = Layer 1
    H1a = Layers[2]  # Hidden layer active = Layer 2
    x_init = torch.Tensor([[-2, 0]])
    x = find_one_zero_point_autograd(x_init, model)
    # Enumerate
    limit = 20
    S_overl, B = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)
    print(S_overl,"")
    # for i in S_overl:
    #     S.append(i.tolist())
    #
    # # Psatz Verification
    # unsafe_c = torch.Tensor([-1,-1])
    # unsafe_r = torch.Tensor([0.4])
    # bv_flags = bfcnverifier(S_overl, model, unsafe_c, unsafe_r)
    # b_flags = boundary_verifier(S_overl)
    # i_flags = PsatzVerifier(S_overl)
    # end = time.time()
    # print(bv_flags)
    # print(b_flags)
    # print(i_flags)
    # print(end - start)

if __name__ == "__main__":
    main()

model = ann.gen_nn()
model.load_state_dict(torch.load('pre-trained.pt'), strict=True)

Layers = get_layers(model)
