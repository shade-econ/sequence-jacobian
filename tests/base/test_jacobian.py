"""Test all models' Jacobian calculations"""

import numpy as np

from sequence_jacobian.jacobian.drivers import get_G, forward_accumulate, curlyJ_sorted
from sequence_jacobian.jacobian.classes import JacobianDict


def test_ks_jac(krusell_smith_model):
    blocks, exogenous, unknowns, targets, ss = krusell_smith_model
    household, firm, mkt_clearing, _, _, _ = blocks
    T = 10

    # Automatically calculate the general equilibrium Jacobian
    G2 = get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
               targets=targets, T=T, ss=ss)

    # Manually calculate the general equilibrium Jacobian
    J_firm = firm.jac(ss, shock_list=['K', 'Z'])
    J_ha = household.jac(ss, T=T, shock_list=['r', 'w'])
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


def test_hank_jac(one_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = one_asset_hank_model
    T = 10

    # Automatically calculate the general equilibrium Jacobian
    G2 = get_G(block_list=blocks, exogenous=exogenous, unknowns=unknowns,
               targets=targets, T=T, ss=ss)

    # Manually calculate the general equilibrium Jacobian
    curlyJs, required = curlyJ_sorted(blocks, unknowns+exogenous, ss, T)
    J_curlyH_U = forward_accumulate(curlyJs, unknowns, targets, required)
    J_curlyH_Z = forward_accumulate(curlyJs, exogenous, targets, required)
    H_U = J_curlyH_U[targets, unknowns].pack(T)
    H_Z = J_curlyH_Z[targets, exogenous].pack(T)
    G_U = JacobianDict.unpack(-np.linalg.solve(H_U, H_Z), unknowns, exogenous, T)
    curlyJs = [G_U] + curlyJs
    outputs = set().union(*(curlyJ.outputs for curlyJ in curlyJs)) - set(targets)
    G = forward_accumulate(curlyJs, exogenous, outputs, required | set(unknowns))

    for o in G:
        for i in G[o]:
            assert np.allclose(G[o][i], G2[o][i])


def test_fake_news_v_actual(one_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = one_asset_hank_model

    household = blocks[0]
    T = 40
    shock_list=['w', 'r', 'Div', 'Tax']
    Js = household.jac(ss, shock_list, T)
    output_list = household.non_back_iter_outputs

    # Preliminary processing of the steady state
    (ssin_dict, Pi, ssout_list, ss_for_hetinput, sspol_i, sspol_pi, sspol_space) = household.jac_prelim(ss)

    # Step 1 of fake news algorithm: backward iteration
    h = 1E-4
    curlyYs, curlyDs = {}, {}
    for i in shock_list:
        curlyYs[i], curlyDs[i] = household.backward_iteration_fakenews(i, output_list, ssin_dict,
                                                                       ssout_list, ss['D'], Pi.T.copy(), sspol_i,
                                                                       sspol_pi, sspol_space, T, h, ss_for_hetinput)

    asset_effects = np.sum(curlyDs['r'] * ss['a_grid'], axis=(1, 2))
    assert np.linalg.norm(asset_effects - curlyYs["r"]["a"], np.inf) < 1e-15

    # Step 2 of fake news algorithm: (transpose) forward iteration
    curlyPs = {}
    for o in output_list:
        curlyPs[o] = household.forward_iteration_fakenews(ss[o], Pi, sspol_i, sspol_pi, T-1)

    persistent_asset = np.array([np.vdot(curlyDs['r'][0, ...],
                                         curlyPs['a'][u, ...]) for u in range(30)])

    assert np.linalg.norm(persistent_asset - Js["A"]["r"][1:31, 0], np.inf) < 3e-15

    # Step 3 of fake news algorithm: combine everything to make the fake news matrix for each output-input pair
    Fs = {o.capitalize(): {} for o in output_list}
    for o in output_list:
        for i in shock_list:
            F = np.empty((T,T))
            F[0, ...] = curlyYs[i][o]
            F[1:, ...] = curlyPs[o].reshape(T-1, -1) @ curlyDs[i].reshape(T, -1).T
            Fs[o.capitalize()][i] = F


    impulse = Fs['C']['w'][:10, 1].copy()  # start with fake news impulse
    impulse[1:10] += Js['C']['w'][:9, 0]   # add unanticipated impulse, shifted by 1

    assert np.linalg.norm(impulse - Js["C"]["w"][:10, 1], np.inf) == 0.0

    # Step 4 of fake news algorithm: recursively convert fake news matrices to actual Jacobian matrices
    Js_original = Js
    Js = {o.capitalize(): {} for o in output_list}
    for o in output_list:
        for i in shock_list:
            # implement recursion (30): start with J=F and accumulate terms along diagonal
            J = Fs[o.capitalize()][i].copy()
            for t in range(1, J.shape[1]):
                J[1:, t] += J[:-1, t-1]
            Js[o.capitalize()][i] = J

    for o in output_list:
        for i in shock_list:
            assert np.array_equal(Js[o.capitalize()][i], Js_original[o.capitalize()][i])


def test_fake_news_v_direct_method(one_asset_hank_model):
    blocks, exogenous, unknowns, targets, ss = one_asset_hank_model

    household = blocks[0]
    T = 40
    shock_list = ('r')
    output_list = household.non_back_iter_outputs
    h = 1E-4

    Js = household.jac(ss, shock_list, T)
    Js_direct = {o.capitalize(): {i: np.empty((T, T)) for i in shock_list} for o in output_list}

    # run td once without any shocks to get paths to subtract against
    # (better than subtracting by ss since ss not exact)
    # monotonic=True lets us know there is monotonicity of policy rule, makes TD run faster
    # .td requires at least one input 'shock', so we put in steady-state w
    td_noshock = household.td(ss, exogenous={"w": np.zeros(T)}, monotonic=True)

    for i in shock_list:
        # simulate with respect to a shock at each date up to T
        for t in range(T):
            td_out = household.td(ss, exogenous={i: h * (np.arange(T) == t)})

            # store results as column t of J[o][i] for each outcome o
            for o in output_list:
                Js_direct[o.capitalize()][i][:, t] = (td_out[o.capitalize()] - td_noshock[o.capitalize()]) / h

    assert np.linalg.norm(Js["C"]["r"] - Js_direct["C"]["r"], np.inf) < 3e-4
