"""This model is to test DisContBlock"""

import numpy as np

from sequence_jacobian import simple, create_model, solved
import single_endo as single
import couple_endo as couple


'''Rest of the model'''


@simple
def firm(Z, K, L, mc, tax, alpha, delta0, delta1, psi):
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    w = (1 - alpha) * mc * Y / L
    u = (alpha / delta0 / delta1 * mc * Y / K(-1)) ** (1 / delta1)
    delta = delta0 * u ** delta1
    Q = 1 + psi * (K / K(-1) - 1)
    I = K - (1 - delta) * K(-1) + psi / 2 * (K / K(-1) - 1) ** 2 * K(-1)
    transfer = Y - w * L - I
    atw = (1 - tax) * w
    return Y, w, u, delta, Q, I, transfer, atw


@simple
def monetary(pi, rstar, phi_pi):
    # rpost = (1 + rstar(-1) + phi_pi * pi(-1)) / (1 + pi) - 1
    rpost = rstar
    return rpost


@simple
def nkpc(pi, mc, eps, Y, rpost, kappa):
    nkpc_res = kappa * (mc - (eps - 1) / eps) + Y(+1) / Y * np.log(1 + pi(+1)) / (1 + rpost(+1)) - np.log(1 + pi)
    return nkpc_res


@simple
def valuation(rpost, mc, Y, K, Q, delta, psi, alpha):
    val = alpha * mc(+1) * Y(+1) / K - (K(+1) / K - (1 - delta) + psi / 2 * (K(+1) / K - 1) ** 2) + \
          K(+1) / K * Q(+1) - (1 + rpost(+1)) * Q
    return val


@solved(unknowns={'B': [0.0, 10.0]}, targets=['budget'], solver='brentq')
def fiscal(B, tax, w, rpost, G, Ze, Ui):
    budget = (1 + rpost) * B + G + (1 - tax) * w * Ui - tax * w * Ze - B
    # tax_rule = tax - tax.ss - phi * (B(-1) - B.ss) / Y.ss
    tax_rule = B - B.ss
    return budget, tax_rule


@simple
def mkt_clearing(A, B, Y, C, I, G, L, Ze):
    asset_mkt = A - B
    goods_mkt = Y - C - I - G
    labor_mkt = L - Ze
    return asset_mkt, goods_mkt, labor_mkt


@simple
def dividends(transfer, pop_sm, pop_sw, pop_mc, illiq_sm, illiq_sw, illiq_mc):
    transfer_sm = illiq_sm * transfer / (pop_sm * illiq_sm + pop_sw * illiq_sw + pop_mc * illiq_mc)
    transfer_sw = illiq_sw * transfer / (pop_sm * illiq_sm + pop_sw * illiq_sw + pop_mc * illiq_mc)
    transfer_mc = illiq_mc * transfer / (pop_sm * illiq_sm + pop_sw * illiq_sw + pop_mc * illiq_mc)
    return transfer_sm, transfer_sw, transfer_mc


@simple
def aggregate(A_sm, C_sm, Ze_sm, Ui_sm, pop_sm, A_sw, C_sw, Ze_sw, Ui_sw, pop_sw, A_mc, C_mc, Ze_mc, Ui_mc, pop_mc):
    A = pop_sm * A_sm + pop_sw * A_sw + pop_mc * A_mc
    C = pop_sm * C_sm + pop_sw * C_sw + pop_mc * C_mc
    Ze = pop_sm * Ze_sm + pop_sw * Ze_sw + pop_mc * Ze_mc
    Ui = pop_sm * Ui_sm + pop_sw * Ui_sw + pop_mc * Ui_mc
    return A, C, Ze, Ui


'''Steady-state helpers'''


@simple
def firm_ss(eps, rpost, Y, L, alpha, delta0):
    # uses u=1
    mc = (eps - 1) / eps
    K = mc * Y * alpha / (rpost + delta0)
    delta1 = alpha / delta0 * mc * Y / K
    Z = Y / (K ** alpha * L ** (1 - alpha))
    w = (1 - alpha) * mc * Y / L
    return mc, K, delta1, Z, w


@simple
def fiscal_ss(A, tax, w, Ze, rpost, Ui):
    # after hh block
    B = A
    G = tax * w * Ze - rpost * B - (1 - tax) * w * Ui
    return B, G


'''Calibration'''

cali_sm = {'beta': 0.9833, 'vphi': 0.7625, 'chi': 0.5585, 'fU': 0.2499, 'fN': 0.1148, 's': 0.0203,
           'mean_z': 1.0, 'rho_z': 0.98, 'sd_z': 0.943*0.82,
           'fU_eps': 10.69, 'fN_eps': 5.57, 's_eps': -11.17,
           'illiq': 0.25, 'pop': 0.29}

cali_sw = {'beta': 0.9830, 'vphi': 0.9940, 'chi': 0.4958, 'fU': 0.2207, 'fN': 0.1098, 's': 0.0132,
           'mean_z': 0.8, 'rho_z': 0.98, 'sd_z': 0.86*0.82,
           'fU_eps': 8.72, 'fN_eps': 3.55, 's_eps': -6.55,
           'illiq': 0.15, 'pop': 0.29}

cali_mc = {'beta': 0.9882, 'rho': 0.042,
           'vphi_m': 0.1545, 'chi_m': 0.2119, 'fU_m': 0.3013, 'fN_m': 0.1840, 's_m': 0.0107,
           'vphi_f': 0.2605, 'chi_f': 0.2477, 'fU_f': 0.2519, 'fN_f': 0.1691, 's_f': 0.0103,
           'mean_m': 1.0, 'rho_m': 0.98, 'sd_m': 0.943*0.82,
           'mean_f': 0.8, 'rho_f': 0.98, 'sd_f': 0.86*0.82,
           'fU_eps_m': 10.37, 'fN_eps_m': 9.74, 's_eps_m': -13.60,
           'fU_eps_f': 8.40, 'fN_eps_f': 2.30, 's_eps_f': -7.87,
           'illiq': 0.6, 'pop': 0.42}

'''Remap'''

# variables to remap
to_map_single = ['beta', 'vphi', 'chi', 'fU', 'fN', 's', 'mean_z', 'rho_z', 'sd_z', 'transfer',
                 'fU_eps', 'fN_eps', 's_eps', *single.hh.outputs]
to_map_couple = ['beta', 'transfer', 'rho',
                 'vphi_m', 'chi_m', 'fU_m', 'fN_m', 's_m', 'mean_m', 'rho_m', 'sd_m', 'fU_eps_m', 'fN_eps_m', 's_eps_m',
                 'vphi_f', 'chi_f', 'fU_f', 'fN_f', 's_f', 'mean_f', 'rho_f', 'sd_f', 'fU_eps_f', 'fN_eps_f', 's_eps_f',
                 *couple.hh.outputs]

# Single men
hh_sm = create_model([single.income_state_vars, single.employment_state_vars, single.asset_state_vars, single.flows,
                      single.household.rename('single_men')], name='SingleMen')
hh_sm = hh_sm.remap({k: k + '_sm' for k in to_map_single})

# Single women
hh_sw = create_model([single.income_state_vars, single.employment_state_vars, single.asset_state_vars, single.flows,
                      single.household.rename('single_women')], name='SingleWomen')
hh_sw = hh_sw.remap({k: k + '_sw' for k in to_map_single})

# Married couples
hh_mc = create_model([couple.income_state_vars, couple.employment_state_vars, couple.asset_state_vars,
                      couple.flows_m, couple.flows_f, couple.household.rename('couples')], name='Couples')
hh_mc = hh_mc.remap({k: k + '_mc' for k in to_map_couple})


'''Solve ss'''


hank = create_model([hh_sm, hh_sw, hh_mc, aggregate, dividends, firm, monetary, valuation, nkpc, fiscal, mkt_clearing],
                    name='HANK')

# remap calibration
cali_sm = {k + '_sm': v for k, v in cali_sm.items()}
cali_sw = {k + '_sw': v for k, v in cali_sw.items()}
cali_mc = {k + '_mc': v for k, v in cali_mc.items()}

calibration = {**cali_sm, **cali_sw, **cali_mc,
               'eis': 1.0, 'uicap': 0.66, 'uirate': 0.5, 'expiry': 1/6, 'eps': 10.0, 'tax': 0.3,
               'amin': 0.0, 'amax': 500, 'nA': 100, 'lamM': 0.01, 'lamB': 0.01, 'lamL': 0.04, 'nZ': 7,
               'kappa': 0.03, 'phi_pi': 1.25, 'rstar': 0.002, 'pi': 0.0, 'alpha': 0.2, 'psi': 30, 'delta0': 0.0083}

ss = hank.solve_steady_state(calibration, solver='toms748', dissolve=[fiscal], ttol=1E-6,
                             unknowns={'L': (1.06, 1.12),
                                       'Z': 0.63, 'mc': 0.9, 'G': 0.2, 'delta1': 1.2, 'B': 3.0, 'K': 17.0},
                             targets={'labor_mkt': 0.0,
                                      'Y': 1.0, 'nkpc_res': 0.0, 'budget': 0.0, 'asset_mkt': 0.0, 'val': 0.0, 'u': 1.0},
                             helper_blocks=[firm_ss, fiscal_ss],
                             helper_targets=['asset_mkt', 'budget', 'val', 'nkpc_res', 'Y', 'u'])


# jacobians
# J = {}
# J['sm'] = hh_sm.jacobian(ss, exogenous=['atw', 'rpost', 'beta_sm', 'Y', 'transfer_sm'], T=500)

# td_nonlin = hank.solve_impulse_nonlinear(ss, {'Z': 0.001*0.9**np.arange(300)},
#                                          unknowns=['K'], targets=['asset_mkt'])

