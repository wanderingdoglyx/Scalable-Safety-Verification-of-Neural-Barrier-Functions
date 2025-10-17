from Psatz_inner_bound_matlab import *
M = []
S_overl = []
S_overl_veri = []
S = []
B = []

model = ann.gen_nn()
model.load_state_dict(torch.load('darboux_1_20_lr01.pt'), strict=True)

# Get layers' output
Layers = get_layers(model)
# Heuristic Search
H1w = Layers[1]  # Hidden layer pre-act = Layer 1
H1a = Layers[2]  # Hidden layer active = Layer 2
x_init = torch.Tensor([[1, 1]])
x = find_one_zero_point_autograd(x_init, model)
# Enumerate
limit = 20
# S_overl, B = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)

counter = 0
FC = []

# x = find_one_zero_point_autograd(x_init, model)
# x = torch.Tensor([1.9, 0])
# S_overl, B = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)
x = torch.Tensor([-0.1271, 1.8])
S_overl, B = enumarate_intersections(x, H1w, H1a, model, M, S_overl, B, limit)

for i in S_overl:
    W_overl, r_overl, B_act, B_inact = activated_weight_bias(model, i)

    W_Bound = torch.Tensor(-B_act[0] + B_inact[0])
    r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
    x1b = (prob.DOMAIN[0][0], prob.DOMAIN[0][1])
    x2b = (prob.DOMAIN[1][0], prob.DOMAIN[1][1])
    res_zero = linprog(c=[1, 1],
                       A_ub=W_Bound, b_ub=-r_Bound,
                       A_eq=W_overl, b_eq=-r_overl, bounds=(x1b, x2b),
                       method='simplex')
    if res_zero.success:
        print(res_zero.x)
        counter += 1
        FC.append(i)
print('numb: ',len(S_overl))
print(counter)