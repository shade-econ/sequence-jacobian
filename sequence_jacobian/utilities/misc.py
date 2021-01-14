"""Assorted other utilities"""

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
    """Return list of function inputs"""
    return inspect.getfullargspec(f).args


def output_list(f):
    """Scans source code of function to detect statement like

    'return L, Div'

    and reports the list ['L', 'Div'].

    Important to write functions in this way when they will be scanned by output_list, for
    either SimpleBlock or HetBlock.
    """
    return re.findall('return (.*?)\n', inspect.getsource(f))[-1].replace(' ', '').split(',')


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
