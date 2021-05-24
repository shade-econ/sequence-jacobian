"""Numerical differentiation"""

from .misc import make_tuple


def numerical_diff(func, ssinputs_dict, shock_dict, h=1E-4, y_ss_list=None):
    """Differentiate function numerically via forward difference, i.e. calculate

    f'(xss)*shock = (f(xss + h*shock) - f(xss))/h

    for small h. (Variable names inspired by application of differentiating around ss.)

    Parameters
    ----------
    func            : function, 'f' to be differentiated
    ssinputs_dict   : dict, values in 'xss' around which to differentiate
    shock_dict      : dict, values in 'shock' for which we're taking derivative
                        (keys in shock_dict are weak subset of keys in ssinputs_dict)
    h               : [optional] scalar, scaling of forward difference 'h'
    y_ss_list       : [optional] list, value of y=f(xss) if we already have it

    Returns
    ----------
    dy_list : list, output f'(xss)*shock of numerical differentiation
    """
    # compute ss output if not supplied
    if y_ss_list is None:
        y_ss_list = make_tuple(func(**ssinputs_dict))

    # response to small shock
    shocked_inputs = {**ssinputs_dict, **{k: ssinputs_dict[k] + h * shock for k, shock in shock_dict.items()}}
    y_list = make_tuple(func(**shocked_inputs))

    # scale responses back up, dividing by h
    dy_list = [(y - y_ss) / h for y, y_ss in zip(y_list, y_ss_list)]

    return dy_list


def numerical_diff_symmetric(func, ssinputs_dict, shock_dict, h=1E-4):
    """Same as numerical_diff, but differentiate numerically using central (symmetric) difference, i.e.

    f'(xss)*shock = (f(xss + h*shock) - f(xss - h*shock))/(2*h)
    """

    # response to small shock in each direction
    shocked_inputs_up = {**ssinputs_dict, **{k: ssinputs_dict[k] + h * shock for k, shock in shock_dict.items()}}
    y_up_list = make_tuple(func(**shocked_inputs_up))

    shocked_inputs_down = {**ssinputs_dict, **{k: ssinputs_dict[k] - h * shock for k, shock in shock_dict.items()}}
    y_down_list = make_tuple(func(**shocked_inputs_down))

    # scale responses back up, dividing by h
    dy_list = [(y_up - y_down) / (2*h) for y_up, y_down in zip(y_up_list, y_down_list)]

    return dy_list
