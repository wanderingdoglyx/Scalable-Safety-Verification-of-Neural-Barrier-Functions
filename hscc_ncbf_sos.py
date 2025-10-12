import numpy as np
import sympy as sp
from SumOfSquares import *

x1, x2 = sp.symbols('x1 x2')
x = [x1, x2]

# load data
Set_Act_set = [[[False],[True],[True],[True],[False]],[[False],[False],[True],[False],[False]]]
w = [-0.508,-0.5813]
r = -0.5700

prog = SOSProblem()

# Define Lie derivatives (db/dx)g(x) and (db/dx)f(x)
# dbdxg = None
dbdxf = (1/3)*x1**3 - w[0]*x1 + (w[0]-w[1])*x2
# Define b(x) candidate
b = w[0]*x1 + w[1]*x2 - r

alpha = poly_variable(name='alpha',variables=[x1,x2],deg=4)
alpha0 = poly_variable(name='alpha0',variables=[x1,x2],deg=4)
beta = poly_variable(name='beta',variables=[x1,x2],deg=4)

eta = poly_variable(name='eta',variables=[x1,x2],deg=4)
theta = poly_variable(name='theta',variables=[x1,x2],deg=4)
omega = poly_variable(name='omega',variables=[x1,x2],deg=4)

# Define SOS constraints
cons_alpha = alpha
cons_alpha0 = alpha0
cons_beta = beta

# cons1 = eta*b + theta*dbdxg + alpha*dbdxf - dbdxf**2

cons1 = eta*b + alpha*dbdxf - dbdxf**2 + alpha0
# cons1 = eta*b + alpha*dbdxf - dbdxf**2

# cons1 = prog.get_constraint(cons1)
# cons2 = sosineq(prog,omega*h - 1 - beta*b);

# prog = poly_cert_prob(x,poly=0,eqs=[cons1,0])
# We do not set object function to check if the polynomial is SOS
check_alpha = prog.add_sos_constraint(alpha,x)
check_beta = prog.add_sos_constraint(beta,x)
# prog.add_sos_constraint(cons1,[x1,x2])
prog.add_constraint(cons1==0)
prog.solve()
print(prog.last_solution.lastStatus)
print(prog.value)

# eta_solution = sosgetsol(prog,eta);
# theta_solution = sosgetsol(prog,theta);
# alpha_solution = sosgetsol(prog,alpha);
# beta_solution = sosgetsol(prog,beta);
# omega_solution = sosgetsol(prog,omega);