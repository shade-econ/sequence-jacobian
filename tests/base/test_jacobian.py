"""Test all models' Jacobian calculations"""

import numpy as np

def test_ks_jac(krusell_smith_dag):
    _, ss, ks_model, unknowns, targets, exogenous = krusell_smith_dag
    household, firm = ks_model['household'], ks_model['firm']
    T = 10

    # Automatically calculate the general equilibrium Jacobian
    G2 = ks_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

    # Manually calculate the general equilibrium Jacobian
    J_firm = firm.jacobian(ss, inputs=['K', 'Z'])
    J_ha = household.jacobian(ss, T=T, inputs=['r', 'w'])
    J_curlyK_K = J_ha['A']['r'] @ J_firm['r']['K'] + J_ha['A']['w'] @ J_firm['w']['K']
    J_curlyK_Z = J_ha['A']['r'] @ J_firm['r']['Z'] + J_ha['A']['w'] @ J_firm['w']['Z']
    J_curlyK = {'curlyK': {'K': J_curlyK_K, 'Z': J_curlyK_Z}}

    H_K = J_curlyK['curlyK']['K'] - np.eye(T)
    H_Z = J_curlyK['curlyK']['Z']

    G = {'K': -np.linalg.solve(H_K, H_Z)}  # H_K^(-1)H_Z
    G['r'] = J_firm['r']['Z'] + J_firm['r']['K'] @ G['K']
    G['w'] = J_firm['w']['Z'] + J_firm['w']['K'] @ G['K']
    G['Y'] = J_firm['Y']['Z'] + J_firm['Y']['K'] @ G['K']
    G['C'] = J_ha['C']['r'] @ G['r'] + J_ha['C']['w'] @ G['w']

    for o in G:
        assert np.allclose(G2[o]['Z'], G[o])


# TODO: decide whether to get rid of this or revise it with manual solve_jacobian stuff
# def test_hank_jac(one_asset_hank_dag):
#     hank_model, exogenous, unknowns, targets, ss = one_asset_hank_dag
#     T = 10

#     # Automatically calculate the general equilibrium Jacobian
#     G2 = hank_model.solve_jacobian(ss, unknowns, targets, exogenous, T=T)

#     # Manually calculate the general equilibrium Jacobian
#     curlyJs, required = curlyJ_sorted(hank_model.blocks, unknowns + exogenous, ss, T)
#     J_curlyH_U = forward_accumulate(curlyJs, unknowns, targets, required)
#     J_curlyH_Z = forward_accumulate(curlyJs, exogenous, targets, required)
#     H_U = J_curlyH_U[targets, unknowns].pack(T)
#     H_Z = J_curlyH_Z[targets, exogenous].pack(T)
#     G_U = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, exogenous, T)
#     curlyJs = [G_U] + curlyJs
#     outputs = set().union(*(curlyJ.outputs for curlyJ in curlyJs)) - set(targets)
#     G = forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))

#     for o in G:
#         for i in G[o]:
#             assert np.allclose(G[o][i], G2[o][i])



def test_fake_news_v_direct_method(one_asset_hank_dag):
    hank_model, ss, *_ = one_asset_hank_dag

    household = hank_model['household']
    T = 40
    exogenous = ['r']
    output_list = household.non_backward_outputs
    h = 1E-4

    Js = household.jacobian(ss, exogenous, T=T)
    Js_direct = {o.upper(): {i: np.empty((T, T)) for i in exogenous} for o in output_list}

    # run td once without any shocks to get paths to subtract against
    # (better than subtracting by ss since ss not exact)
    # monotonic=True lets us know there is monotonicity of policy rule, makes TD run faster
    # .impulse_nonlinear requires at least one input 'shock', so we put in steady-state w
    td_noshock = household.impulse_nonlinear(ss, {'w': np.zeros(T)})

    for i in exogenous:
        # simulate with respect to a shock at each date up to T
        for t in range(T):
            td_out = household.impulse_nonlinear(ss, {i: h * (np.arange(T) == t)})

            # store results as column t of J[o][i] for each outcome o
            for o in output_list:
                Js_direct[o.upper()][i][:, t] = (td_out[o.upper()] - td_noshock[o.upper()]) / h

    assert np.linalg.norm(Js['C']['r'] - Js_direct['C']['r'], np.inf) < 3e-4
