import numpy as np
import inspect
import re
import utils


'''Part 0: Set up the stage for general HA jacobian code.'''


def extract_info(back_step_fun, ss):
    """Process source code of a backward iteration function.

    Parameters
    ----------
    back_step_fun : function
        backward iteration function
    ss : dict
        steady state dictionary
    Returns
    ----------
    ssinput_dict : dict
      {name: ss value} for all inputs to back_step_fun
    ssy_list : list
      steady state value of outputs of back_step_fun in same order
    outcome_list : list
      names of variables returned by back_step_fun other than V
    V_name : str
      name of backward variable
    """
    V_name, *outcome_list = re.findall('return (.*?)\n',
                                       inspect.getsource(back_step_fun))[-1].replace(' ', '').split(',')

    #ssy_list = [ss[k] for k in [V_name] + outcome_list]

    input_names = inspect.getfullargspec(back_step_fun).args
    ssinput_dict = {}
    for k in input_names:
        if k.endswith('_p'):
            ssinput_dict[k] = ss[k[:-2]]
        else:
            ssinput_dict[k] = ss[k]

    # want numerical differentiation to subtract against this for greater accuracy,
    # since steady state is not *exactly* a fixed point of back_step_fun
    ssy_list = back_step_fun(**ssinput_dict)

    return ssinput_dict, ssy_list, outcome_list, V_name


'''
Part 1: get curlyYs and curlyDs

That is, dynamic effects that propagate through changes in policy holding distribution constant.    
'''


def backward_step(dinput_dict, back_step_fun, ssinput_dict, ssy_list, outcome_list, D, Pi, a_pol_i, a_space, h=1E-4):
    # shock perturbs policies
    curlyV, da, *dy_list = utils.numerical_diff(back_step_fun, ssinput_dict, dinput_dict, h, ssy_list)

    # which affects the distribution tomorrow
    da_pol_pi = -da / a_space
    curlyD = utils.forward_step_policy_shock(D, Pi.T, a_pol_i, da_pol_pi)

    # and the aggregate outcomes today
    curlyY = {name: np.vdot(D, dy) for name, dy in zip(outcome_list, [da] + dy_list)}

    return curlyV, curlyD, curlyY


def backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list, V_name, D, Pi, a_pol_i, a_space, T):
    """Iterate policy steps backward T times for a single shock."""
    # contemporaneous response to unit scalar shock
    curlyV, curlyD, curlyY = backward_step(shock, back_step_fun, ssinput_dict,
                                           ssy_list, outcome_list, D, Pi, a_pol_i, a_space)

    # infer dimensions from this and initialize empty arrays
    curlyDs = np.empty((T,) + curlyD.shape)
    curlyYs = {k: np.empty(T) for k in curlyY.keys()}

    # fill in current effect of shock
    curlyDs[0, ...] = curlyD
    for k in curlyY.keys():
        curlyYs[k][0] = curlyY[k]

    # fill in anticipation effects
    for t in range(1, T):
        curlyV, curlyDs[t, ...], curlyY = backward_step({V_name + '_p': curlyV}, back_step_fun, ssinput_dict,
                                                        ssy_list, outcome_list, D, Pi, a_pol_i, a_space)
        for k in curlyY.keys():
            curlyYs[k][t] = curlyY[k]

    return curlyYs, curlyDs


'''
Part 2: get curlyPs, aka prediction vectors

That is dynamic effects that propagate through the distribution's law of motion holding policy constant.    
'''


def forward_iteration_transpose(y_ss, Pi, a_pol_i, a_pol_pi, T):
    """Iterate transpose forward T steps to get full set of prediction vectors for a given outcome."""
    curlyPs = np.empty((T,) + y_ss.shape)
    curlyPs[0, ...] = y_ss
    for t in range(1, T):
        curlyPs[t, ...] = utils.forward_step_transpose(curlyPs[t-1, ...], Pi, a_pol_i, a_pol_pi)
    return curlyPs


'''
Part 3: construct fake news matrix, and then Jacobian.
'''


def build_F(curlyYs, curlyDs, curlyPs):
    T = curlyDs.shape[0]
    F = np.empty((T, T))
    F[0, :] = curlyYs
    F[1:, :] = curlyPs[:T - 1, ...].reshape((T - 1, -1)) @ curlyDs.reshape((T, -1)).T
    return F


def J_from_F(F):
    J = F.copy()
    for t in range(1, J.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


'''Part 4: Putting it all together'''


def all_Js(back_step_fun, ss, T, shock_dict):
    # preliminary a: process back_step_funtion
    ssinput_dict, ssy_list, outcome_list, V_name = extract_info(back_step_fun, ss)

    # preliminary b: get sparse representation of asset policy rule, then distance between neighboring policy gridpoints
    a_pol_i, a_pol_pi = utils.interpolate_coord(ss['a_grid'], ss['a'])
    a_space = ss['a_grid'][a_pol_i + 1] - ss['a_grid'][a_pol_i]

    # step 1: compute curlyY and curlyD (backward iteration) for each input i
    curlyYs, curlyDs = dict(), dict()
    for i, shock in shock_dict.items():
        curlyYs[i], curlyDs[i] = backward_iteration(shock, back_step_fun, ssinput_dict, ssy_list, outcome_list,
                                                    V_name, ss['D'], ss['Pi'], a_pol_i, a_space, T)

    # step 2: compute prediction vectors curlyP (forward iteration) for each outcome o
    curlyPs = dict()
    for o, ssy in zip(outcome_list, ssy_list[1:]):
        curlyPs[o] = forward_iteration_transpose(ssy, ss['Pi'], a_pol_i, a_pol_pi, T)

    # step 3: make fake news matrix and Jacobian for each outcome-input pair
    J = {o: {} for o in outcome_list}
    for o in outcome_list:
        for i in shock_dict:
            F = build_F(curlyYs[i][o], curlyDs[i], curlyPs[o])
            J[o][i] = J_from_F(F)

    # remap outcomes to capital letters to avoid conflicts
    for k in list(J.keys()):
        K = k.upper()
        J[K] = J.pop(k)

    # report Jacobians
    return J
