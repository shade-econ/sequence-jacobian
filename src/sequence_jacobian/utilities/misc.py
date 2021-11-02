"""Assorted other utilities"""

import numpy as np
import scipy.linalg


def make_tuple(x):
    """If not tuple or list, make into tuple with one element.

    Wrapping with this allows user to write, e.g.:
    "return r" rather than "return (r,)"
    "policy='a'" rather than "policy=('a',)"
    """
    return (x,) if not (isinstance(x, tuple) or isinstance(x, list)) else x


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


# The below functions are used in steady_state
def unprime(s):
    """Given a variable's name as a `str`, check if the variable is a prime, i.e. has "_p" at the end.
    If so, return the unprimed version, if not return itself."""
    if s[-2:] == "_p":
        return s[:-2]
    else:
        return s


def uncapitalize(s):
    return s[0].lower() + s[1:]


def list_diff(l1, l2):
    """Returns the list that is the "set difference" between l1 and l2 (based on element values)"""
    o_list = []
    for k in set(l1) - set(l2):
        o_list.append(k)
    return o_list


def dict_diff(d1, d2):
    """Returns the dictionary that is the "set difference" between d1 and d2 (based on keys, not key-value pairs)
    E.g. d1 = {"a": 1, "b": 2}, d2 = {"b": 5}, then dict_diff(d1, d2) = {"a": 1}
    """
    o_dict = {}
    for k in set(d1.keys()) - set(d2.keys()):
        o_dict[k] = d1[k]
    return o_dict


def smart_set(data):
    # We want set to construct a single-element set for strings, i.e. ignoring the .iter method of strings
    if isinstance(data, str):
        return {data}
    else:
        return set(data)


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


'''Tools for taste shocks used in discrete choice problems'''

def logit(V, i, scale):
    """Logit choice probability of choosing along 0th axis"""
    Vnorm = V - V[0, ...]
    P = np.exp(Vnorm / scale) / np.sum(np.exp(Vnorm / scale), axis=i)
    return P


def logsum(V, i, scale):
    """Logsum formula along ith dimension"""
    const = V[0, ...]
    Vnorm = V - const
    EV = const + scale * np.log(np.exp(Vnorm / scale).sum(axis=i))
    return EV 
