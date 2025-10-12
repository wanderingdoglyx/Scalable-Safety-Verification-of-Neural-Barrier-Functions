import sympy as sp
from SumOfSquares import *

x1, x2 = sp.symbols('x1 x2')
x = [x1, x2]

prog = SOSProblem()

# Define Lie derivatives (db/dx)g(x) and (db/dx)f(x)
dbdxg = -2*x1*x2
dbdxf = -x2**2*(1-x2**2)
b0 = 1 # b0 = 2 works. Also returns infeasible if b0 = 0.9. Works for b0 = 1
# Define b(x) candidate
b = b0 - x1**2 - x2**2

alpha = poly_variable(name='alpha',variables=[x1, x2],deg=4)
beta = poly_variable(name='beta',variables=[x1, x2],deg=4)

eta = poly_variable(name='eta',variables=[x1, x2],deg=4)
theta = poly_variable(name='theta',variables=[x1, x2],deg=4)
omega = poly_variable(name='omega',variables=[x1, x2],deg=4)

# Define SOS constraints
cons_alpha = alpha
cons_beta = beta
cons1 = eta*b + theta*dbdxg + alpha*dbdxf - dbdxf**2
# cons2 = sosineq(prog,omega*h - 1 - beta*b);

# We do not set object function to check if the polynomial is SOS
check_alpha = prog.add_sos_constraint(cons_alpha,[x1, x2])
check_beta = prog.add_sos_constraint(cons_beta,[x1, x2])
prog.add_sos_constraint(cons1,[x1, x2])
prog.solve()
print(prog.last_solution.lastStatus)
print(prog.value)

# eta_solution = sosgetsol(prog,eta);
# theta_solution = sosgetsol(prog,theta);
# alpha_solution = sosgetsol(prog,alpha);
# beta_solution = sosgetsol(prog,beta);
# omega_solution = sosgetsol(prog,omega);