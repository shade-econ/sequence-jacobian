"""Assorted other utilities"""

import numpy as np
import scipy.linalg
import re
import inspect


def make_tuple(x):
    """If not tuple or list, make into tuple with one element.

    Wrapping with this allows user to write, e.g.:
    "return r" rather than "return (r,)"
    "policy='a'" rather than "policy=('a',)"
    """
    return (x,) if not (isinstance(x, tuple) or isinstance(x, list)) else x


def input_list(f):
    """Return list of function inputs (both positional and keyword arguments)"""
    return list(inspect.signature(f).parameters)


def input_arg_list(f):
    """Return list of function positional arguments *only*"""
    arg_list = []
    for p in inspect.signature(f).parameters.values():
        if p.default == p.empty:
            arg_list.append(p.name)
    return arg_list


def input_kwarg_list(f):
    """Return list of function keyword arguments *only*"""
    kwarg_list = []
    for p in inspect.signature(f).parameters.values():
        if p.default != p.empty:
            kwarg_list.append(p.name)
    return kwarg_list


def output_list(f):
    """Scans source code of function to detect statement like

    'return L, Div'

    and reports the list ['L', 'Div'].

    Important to write functions in this way when they will be scanned by output_list, for
    either SimpleBlock or HetBlock.
    """
    return re.findall('return (.*?)\n', inspect.getsource(f))[-1].replace(' ', '').split(',')


def numeric_primitive(instance):
    # If it is already a primitive, just return it
    if type(instance) in {int, float}:
        return instance
    elif isinstance(instance, np.ndarray):
        if np.issubdtype(instance.dtype, np.number):
            return np.array(instance)
        else:
            raise ValueError(f"The tuple/list argument provided to numeric_primitive has dtype: {instance.dtype},"
                             f" which is not a valid numeric type.")
    elif type(instance) in {tuple, list}:
        instance_array = np.asarray(instance)
        if np.issubdtype(instance_array.dtype, np.number):
            return type(instance)(instance_array)
        else:
            raise ValueError(f"The tuple/list argument provided to numeric_primitive has dtype: {instance_array.dtype},"
                             f" which is not a valid numeric type.")
    else:
        return instance.real if np.isscalar(instance) else instance.base


def demean(x):
    return x - x.sum()/x.size


# simpler aliases for LU factorization and solution
def factor(X):
    return scipy.linalg.lu_factor(X)


def factored_solve(Z, y):
    return scipy.linalg.lu_solve(Z, y)


# functions for handling saved Jacobians: extract keys from dicts or key pairs
# from nested dicts, and take subarrays with 'shape' of the values
def extract_dict(savedA, keys, shape):
    return {k: take_subarray(savedA[k], shape) for k in keys}


def extract_nested_dict(savedA, keys1, keys2, shape):
    return {k1: {k2: take_subarray(savedA[k1][k2], shape) for k2 in keys2} for k1 in keys1}


def take_subarray(A, shape):
    # verify leading dimensions of A are >= shape
    if not all(m <= n for m, n in zip(shape, A.shape)):
        raise ValueError(f'Saved has dimensions {A.shape}, want larger {shape} subarray')

    # take subarray along those dimensions: A[:shape, ...]
    return A[tuple(slice(None, x, None) for x in shape) + (Ellipsis,)]


def uncapitalize(s):
    # Similar to s.lower() but only makes the first character lower-case
    return s[0].lower() + s[1:]


# The below functions are used in steady_state
def unprime(s):
    """Given a variable's name as a `str`, check if the variable is a prime, i.e. has "_p" at the end.
    If so, return the unprimed version, if not return itself."""
    if s[-2:] == "_p":
        return s[:-2]
    else:
        return s


def dict_diff(d1, d2):
    """Returns the dictionary that is the "set difference" between d1 and d2 (based on keys, not key-value pairs)
    E.g. d1 = {"a": 1, "b": 2}, d2 = {"b": 5}, then dict_diff(d1, d2) = {"a": 1}
    """
    o_dict = {}
    for k in set(d1.keys()).difference(set(d2.keys())):
        o_dict[k] = d1[k]

    return o_dict


def smart_zip(keys, values):
    """For handling the case where keys and values may be scalars"""
    if isinstance(values, float):
        return zip(keys, [values])
    else:
        return zip(keys, values)


def smart_zeros(n):
    """Return either the float 0. or a np.ndarray of length 0 depending on whether n > 1"""
    if n > 1:
        return np.zeros(n)
    else:
        return 0.
