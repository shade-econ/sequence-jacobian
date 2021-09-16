import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import simple, solved, combine, create_model, markov_rouwenhorst, agrid 
from sequence_jacobian.classes.impulse_dict import ImpulseDict
from sequence_jacobian.examples.hetblocks import household_sim as hh


'''Part 1: Household block'''

@simple
def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, e_dist, Pi = markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = agrid(amin=amin, amax=amax, n=nA)
    return e_grid, e_dist, Pi, a_grid


def income(atw, N, e_grid, transfer):
    y = atw * N * e_grid + transfer
    return y


def get_mpcs(c, a, a_grid, r):
    mpcs_ = np.empty_like(c)
    post_return = (1 + r) * a_grid
    mpcs_[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (post_return[2:] - post_return[:-2])
    mpcs_[:, 0] = (c[:, 1] - c[:, 0]) / (post_return[1] - post_return[0])
    mpcs_[:, -1] = (c[:, -1] - c[:, -2]) / (post_return[-1] - post_return[-2])
    mpcs_[a == a_grid[0]] = 1
    return mpcs_


def mpcs(c, a, a_grid, r):
    mpc = get_mpcs(c, a, a_grid, r)
    return mpc


def weighted_uc(c, e_grid, eis):
    uce = c ** (-1 / eis) * e_grid[:, np.newaxis]
    return uce


'''Part 2: rest of the model'''

@solved(unknowns={'C': 1.0, 'A': 1.0}, targets=['euler', 'budget_constraint'], solver='broyden_custom')
def household_ra(C, A, r, atw, N, transfer, beta, eis):
    euler = beta * (1 + r(1)) * C(1) ** (-1 / eis) - C ** (-1 / eis)
    budget_constraint = (1 + r) * A(-1) + atw * N + transfer - C - A
    UCE = C ** (-1 / eis)
    return euler, budget_constraint, UCE


@simple
def firm(N, Z):
    Y = Z * N
    w = Z
    return Y, w


@simple
def union(UCE, tau, w, N, pi, muw, kappaw, nu, vphi, beta):
    wnkpc = kappaw * N * (vphi * N ** nu - (1 - tau) * w * UCE / muw) + \
        beta * (1 + pi(+1)).apply(np.log) - (1 + pi).apply(np.log)
    return wnkpc


@solved(unknowns={'B': (0.0, 10.0)}, targets=['B_rule'], solver='brentq')
def fiscal(B, G, rb, w, N, transfer, rho_B):
    B_rule = B.ss + rho_B * (B(-1) - B.ss + G - G.ss) - B
    rev = (1 + rb) * B(-1) + G + transfer - B  # revenue to be raised
    tau = rev / (w * N)
    atw = (1 - tau) * w
    return B_rule, rev, tau, atw


# Use this to test zero impulse once we have it
@simple
def real_bonds(r):
    rb = r(-1)
    return rb


@simple
def mkt_clearing(A, B, C, G, Y):
    asset_mkt = A - B
    goods_mkt = C + G - Y
    return asset_mkt, goods_mkt


'''Part 3: Helper blocks'''

@simple
def household_ra_ss(r, B, tau, w, N, transfer, eis):
    beta = 1 / (1 + r)
    A = B
    C = r * A + (1 - tau) * w * N + transfer
    UCE = C ** (-1 / eis)
    return beta, A, C, UCE


@simple
def union_ss(atw, UCE, muw, N, nu, kappaw, beta, pi):
    vphi = atw * UCE / (muw * N ** nu)
    wnkpc = kappaw * N * (vphi * N ** nu - atw * UCE / muw) + \
        beta * (1 + pi(+1)).apply(np.log) - (1 + pi).apply(np.log)
    return wnkpc, vphi



'''Tests'''

def test_all():
    # Assemble HA block (want to test nesting)
    household_ha = hh.household.add_hetinputs([income])
    household_ha = household_ha.add_hetoutputs([mpcs, weighted_uc]).rename('household_ha')
    household_ha = combine([household_ha, make_grids], name='HH')

    # Assemble DAG (for transition dynamics)
    dag = {}
    common_blocks = [firm, union, fiscal, real_bonds, mkt_clearing]
    dag['ha'] = create_model([household_ha] + common_blocks, name='HANK')
    dag['ra'] = create_model([household_ra] + common_blocks, name='RANK')
    unknowns = ['N', 'pi']
    targets = ['asset_mkt', 'wnkpc']

    # Solve steady state
    calibration = {'N': 1.0, 'Z': 1.0, 'r': 0.005, 'pi': 0.0, 'eis': 0.5, 'nu': 0.5,
                   'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3, 'amin': 0.0, 'amax': 200,
                   'nA': 100, 'kappaw': 0.1, 'muw': 1.2, 'transfer': 0.143, 'rho_B': 0.9}
    
    ss = {}
    ss['ha'] = dag['ha'].solve_steady_state(calibration, dissolve=['fiscal'], solver='hybr',
            unknowns={'beta': 0.96, 'B': 3.0, 'vphi': 1.0, 'G': 0.2},
            targets={'asset_mkt': 0.0, 'MPC': 0.25, 'wnkpc': 0.0, 'tau': 0.334},
            helper_blocks=[union_ss], helper_targets=['wnkpc'])
    
    # TODO: why doesn't this work?
    # ss['ra'] = dag['ra'].solve_steady_state(ss['ha'],
    #         unknowns={'beta': 0.96, 'vphi': 1.0}, targets={'asset_mkt': 0.0, 'wnkpc': 0.0},
    #         helper_blocks=[household_ra_ss, union_ss], helper_targets=['asset_mkt', 'wnkpc'],
    #         dissolve=['fiscal', 'household_ra'], solver='solved')

    # Constructing ss-dag manually works just fine
    dag_ss = create_model([household_ra_ss, union_ss, firm, fiscal,
                           real_bonds, mkt_clearing])
    ss['ra'] = dag_ss.steady_state(ss['ha'], dissolve=['fiscal'])
    assert np.isclose(ss['ra']['goods_mkt'], 0.0)
    assert np.isclose(ss['ra']['asset_mkt'], 0.0)
    assert np.isclose(ss['ra']['wnkpc'], 0.0)

    # Precompute HA Jacobian
    Js = {'ra': {}, 'ha': {}}
    Js['ha']['household_ha'] = household_ha.jacobian(ss['ha'],
        inputs=['N', 'atw', 'r', 'transfer'], outputs=['C', 'A', 'UCE'], T=300) 

    # Linear impulse responses from Jacobian vs directly
    shock = ImpulseDict({'G': 1E-4 * 0.9 ** np.arange(300)})
    G, td_lin1, td_lin2 = dict(), dict(), dict()
    for k in ['ra', 'ha']:
        G[k] = dag[k].solve_jacobian(ss[k], unknowns, targets, inputs=['G'], T=300, Js=Js[k],
                                     outputs=['Y', 'C', 'asset_mkt', 'goods_mkt'])
        td_lin1[k] = G[k] @ shock
        td_lin2[k] = dag[k].solve_impulse_linear(ss[k], unknowns, targets, shock, Js=Js[k],
                                                 outputs=['Y', 'C', 'asset_mkt', 'goods_mkt'])

        assert all(np.allclose(td_lin1[k][i], td_lin2[k][i]) for i in td_lin1[k])

    # Nonlinear vs linear impulses
    td_nonlin = dag['ha'].solve_impulse_nonlinear(ss['ha'], unknowns, targets,
        inputs=shock, outputs=['Y', 'C', 'A', 'MPC', 'asset_mkt', 'goods_mkt'],
        Js=Js, internals=['household_ha'])
    assert np.max(np.abs(td_nonlin['goods_mkt'])) < 1E-8
    assert all(np.allclose(td_lin1['ha'][k], td_nonlin[k], atol=1E-6, rtol=1E-6) for k in td_lin1['ha'] if k != 'MPC')

    # See if D change matches up with aggregate assets
    assert np.allclose(np.sum(td_nonlin.internals['household_ha']['D']*td_nonlin.internals['household_ha']['c'], axis=(1, 2)), td_nonlin['C'])
