from ..blocks.simple_block import simple
from ..blocks.combined_block import create_model


'''Part 1: Blocks'''

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y


@simple
def household(K, L, w, eis, frisch, vphi, delta):
    C = (w / vphi / L ** (1 / frisch)) ** eis
    I = K - (1 - delta) * K(-1)
    return C, I


@simple
def mkt_clearing(r, C, Y, I, K, L, w, eis, beta):
    goods_mkt = Y - C - I
    euler = C ** (-1 / eis) - beta * (1 + r(+1)) * C(+1) ** (-1 / eis)
    walras = C + K - (1 + r) * K(-1) - w * L
    return goods_mkt, euler, walras


'''Part 2: Assembling the model'''

def dag():
    # Combine blocks
    blocks = [household, firm, mkt_clearing]
    rbc_model = create_model(blocks, name="RBC")

    # Steady state
    calibration = {'eis': 1., 'frisch': 1., 'delta': 0.025, 'alpha': 0.11, 'L': 1.}
    unknowns_ss = {'vphi': 0.92, 'beta': 1 / (1 + 0.01), 'K': 2., 'Z': 1.}
    targets_ss = {'goods_mkt': 0., 'r': 0.01, 'euler': 0., 'Y': 1.}
    ss = rbc_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver='hybr')

    # Transitional dynamics
    unknowns = ['K', 'L']
    targets = ['goods_mkt', 'euler']
    exogenous = ['Z']

    return rbc_model, ss, unknowns, targets, exogenous
