import numpy as np
from sequence_jacobian import simple, solved
from sequence_jacobian.steady_state.classes import SteadyStateDict
from sequence_jacobian.jacobian.classes import FactoredJacobianDict


@simple
def myblock(u, i):
    res = 0.5 * i - u
    return res 


@solved(unknowns={'u': (-10.0, 10.0)}, targets=['res'], solver='brentq')
def myblock_solved(u, i):
    res = 0.5 * i - u
    return res 


ss = SteadyStateDict({'u': 5, 'i': 10, 'res': 0.0})

# Compute jacobian of myblock_solved from scratch
Ja = myblock_solved.jacobian(ss, exogenous=['i'], T=5)

# Compute jacobian of SolvedBlock using a pre-computed FactoredJacobian
J_d = myblock.jacobian(ss, exogenous=['u'], T=5)  # square jac of underlying simple block 
J_factored = FactoredJacobianDict(J_d, T=5)       
J_c = myblock.jacobian(ss, exogenous=['i'], T=5)  # jac of underlying simple block wrt inputs that are NOT unknowns 
Jb = J_factored.compose(J_c, T=5)                 # obtain jac of unknown wrt to non-unknown inputs using factored jac

assert np.allclose(Ja['u']['i'], Jb['u']['i'])