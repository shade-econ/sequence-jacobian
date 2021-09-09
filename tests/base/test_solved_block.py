import numpy as np
from sequence_jacobian import simple, solved
from sequence_jacobian.classes.steady_state_dict import SteadyStateDict
from sequence_jacobian.classes.jacobian_dict import FactoredJacobianDict


@simple
def myblock(u, i):
    res = 0.5 * i(1) - u**2 - u(1)
    return res 


@solved(unknowns={'u': (-10.0, 10.0)}, targets=['res'], solver='brentq')
def myblock_solved(u, i):
    res = 0.5 * i(1) - u**2 - u(1)
    return res 

def test_solved_block():
    ss = SteadyStateDict({'u': 5, 'i': 10, 'res': 0.0})

    # Compute jacobian of myblock_solved from scratch
    J1 = myblock_solved.jacobian(ss, inputs=['i'], T=20)

    # Compute jacobian of SolvedBlock using a pre-computed FactoredJacobian
    J_u = myblock.jacobian(ss, inputs=['u'], T=20)  # square jac of underlying simple block 
    J_factored = FactoredJacobianDict(J_u, T=20)       
    J_i = myblock.jacobian(ss, inputs=['i'], T=20)  # jac of underlying simple block wrt inputs that are NOT unknowns 
    J2 = J_factored.compose(J_i)              # obtain jac of unknown wrt to non-unknown inputs using factored jac

    assert np.allclose(J1['u']['i'], J2['u']['i'])

