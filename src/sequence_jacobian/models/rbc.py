import numpy as np

from ..blocks.simple_block import simple

'''Part 1: Simple blocks'''


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


@simple
def steady_state_solution(Y, L, r, eis, delta, alpha, frisch):
    # 1. Solve for beta to hit r
    beta = 1 / (1 + r)

    # 2. Solve for K to hit goods_mkt
    K = alpha * Y / (r + delta)
    w = (1 - alpha) * Y / L
    C = w * L + (1 + r) * K(-1) - K
    I = delta * K
    goods_mkt = Y - C - I

    # 3. Solve for Z to hit Y
    Z = Y * K ** (-alpha) * L ** (alpha - 1)

    # 4. Solve for vphi to hit L
    vphi = w * C ** (-1 / eis) * L ** (-1 / frisch)

    # 5. Have to return euler because it's a target
    euler = C ** (-1 / eis) - beta * (1 + r(+1)) * C(+1) ** (-1 / eis)

    return beta, K, w, C, I, goods_mkt, Z, vphi, euler


'''Part 2: Steady state'''


def rbc_ss(r=0.01, eis=1, frisch=1, delta=0.025, alpha=0.11):
    """Solve steady state of simple RBC model.

    Parameters
    ----------
    r      : scalar, real interest rate
    eis    : scalar, elasticity of intertemporal substitution (1/sigma)
    frisch : scalar, Frisch elasticity (1/nu)
    delta  : scalar, depreciation rate
    alpha  : scalar, capital share

    Returns
    -------
    ss : dict, steady state values
    """
    # solve for aggregates analytically
    rk = r + delta
    Z = (rk / alpha) ** alpha  # normalize so that Y=1
    K = (alpha * Z / rk) ** (1 / (1 - alpha))
    Y = Z * K ** alpha
    w = (1 - alpha) * Z * K ** alpha
    I = delta * K
    C = Y - I

    # preference params
    beta = 1 / (1 + r)
    vphi = w * C ** (-1 / eis)

    # check Walras's law, goods market clearing, and the euler equation
    walras = C - r * K - w
    goods_mkt = Y - C - I
    euler = C ** (-1 / eis) - beta * (1 + r) * C ** (-1 / eis)
    assert np.abs(walras) < 1E-12

    return {'beta': beta, 'eis': eis, 'frisch': frisch, 'vphi': vphi, 'delta': delta, 'alpha': alpha,
            'Z': Z, 'K': K, 'I': I, 'Y': Y, 'L': 1, 'C': C, 'w': w, 'r': r, 'walras': walras, 'euler': euler,
            'goods_mkt': goods_mkt}
