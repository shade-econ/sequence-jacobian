import numpy as np
import inspect
import re


class RecursiveSparse:
    """Simple representation of the Jacobians of recursive equations, which are sparse, banded, and Toeplitz around
    the steady state (e.g. the derivative of w(n) wrt k(n-1) is the same no matter what n is).

    Soon, will do matrix multiplication and other operations directly on these sparse representations,
    which is more efficient. For now, just convert to matrices before doing operations.
    """

    def __init__(self, derivs):
        # derivs is dict mapping i -> x, where 'x' is the derivative
        # with respect to i around the steady state
        self.derivs = derivs

    def matrix(self, T):
        """converts sparse to standard matrix form, with size T*T"""
        # be smart, DON'T do a bunch of diags calls and add them up
        M = np.zeros(T*T)
        for index, der in self.derivs.items():
            if index < 0:
                M[T*(-index)::T+1] = der
            else:
                M[index::T+1] = der
        return M.reshape((T, T))


class Ignore(float):
    """This class ignores time displacements of a scalar. Needed for SS eval of RecursiveBlocks"""
    def __call__(self, index):
        return self


class Displace(np.ndarray):
    """This class makes time displacements of a time path, given the steady-state value.
    Needed for TD eval of RecursiveBlocks"""
    def __new__(cls, x, ss=None, name='UNKNOWN'):
        obj = np.asarray(x).view(cls)
        obj.ss = ss
        obj.name = name
        return obj

    def __call__(self, index):
        if index != 0:
            if self.ss is None:
                raise KeyError(f'Trying to call {self.name}({index}), but steady-state {self.name} not given!')
            newx = np.empty_like(self)
            if index > 0:
                newx[:-index] = self[index:]
                newx[-index:] = self.ss
            else:
                newx[-index:] = self[:index]
                newx[:-index] = self.ss
            return newx
        else:
            return self


class RecursiveBlock:
    """Generated from recursive block written in Dynare-ish style and decorated with @recursive"""
    def __init__(self, f):
        self.f = f
        self.input_list = inspect.getfullargspec(f).args
        self.output_list = re.findall('return (.*?)\n', inspect.getsource(f))[-1].replace(' ', '').split(',')

    def __repr__(self):
        return f"<RecursiveBlock '{self.f.__name__}'>"
    
    def ss(self, *args, **kwargs):
        args = [Ignore(x) for x in args]
        kwargs = {k: Ignore(v) for k,v in kwargs.items()}
        return self.f(*args, **kwargs)

    def td(self, ss, **kwargs):
        kwargs_new = {}
        for k, v in kwargs.items():
            if np.isscalar(v):
                raise ValueError(f'Keyword argument {k}={v} is scalar, should be time path.')
            kwargs_new[k] = Displace(v, ss=ss.get(k, None), name=k)
        
        for k in self.input_list:
            if k not in kwargs_new:
                kwargs_new[k] = Ignore(ss[k])

        return self.f(**kwargs_new)

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def recursive(f):
    return RecursiveBlock(f)


class Reporter(float):
    """This class adds to a shared set to tell us what x[i] are accessed. Needed for differentiation
    of RecursiveBlocks"""
    def __init__(self, value):
        self.myset = set()

    def __call__(self, index):
        self.myset.add(index)
        return self


class Perturb(float):
    """This class uses the shared set to perturb each x[i] separately, starting at steady-state values,
    for differentiation of RecursiveBlocks."""
    def __new__(cls, value, h, index):
        if index == 0:
            return float.__new__(cls, value + h)
        else:
            return float.__new__(cls, value)
    
    def __init__(self, value, h, index):
        self.h = h
        self.index = index

    def __call__(self, index):
        if self.index == 0:
            if index == 0:
                return self
            else:
                return self - self.h
        else:
            if self.index == index:
                return self + self.h
            else:
                return self


def convert_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    else:
        return x


def all_derivatives(f, x_ss, output_name_list, inputs_to_diff_name_list, h=1E-5):
    """Takes in function f with steady-state input dict x_ss whose outputs have the
    names in output_name_list, and symmetrically numerically differentiates with
    respect to each of the input names in inputs_to_diff_name_list, producing
    a three-level dict where derivatives[o][i] gives the derivative of output o
    wrt input i, itself expressed as a dict from indices to outputs (in the future,
    this will be a RecursiveSparse object instead). zero derivatives are omitted
    by convention.
    """

    derivatives = {o: {} for o in output_name_list}
    x_ss_new = {k: Ignore(v) for k, v in x_ss.items()}

    # loop over all inputs to differentiate
    for i in inputs_to_diff_name_list:
        # detect all indices with which i appears
        reporter = Reporter(x_ss[i])
        x_ss_new[i] = reporter
        f(**x_ss_new)
        relevant_indices = reporter.myset
        relevant_indices.add(0)

        # evaluate derivative with respect to each and store in dict
        for index in relevant_indices:
            x_ss_new[i] = Perturb(x_ss[i], h, index)
            y_up_all = convert_tuple(f(**x_ss_new))

            x_ss_new[i] = Perturb(x_ss[i], -h, index)
            y_down_all = convert_tuple(f(**x_ss_new))
            for y_up, y_down, o in zip(y_up_all, y_down_all, output_name_list):
                if y_up != y_down:
                    sparsederiv = derivatives[o].setdefault(i, {})
                    sparsederiv[index] = (y_up - y_down)/(2*h)
        x_ss_new[i] = Ignore(x_ss[i])

    return derivatives


def all_Js(f, ss, T, shock_list=None):
    """
    Assemble Jacobians as full matrices.

    Parameters
    ----------
    f : function,
        simple model block
    ss : dict,
        steady state values
    T : int,
        time dimension
    shock_list : list of str, optional
        names of input variables to differentiate wrt

    Returns
    -------
    J : dict,
        Jacobians as nested dict of (T*T) matrices
    """
    # process function
    if shock_list is None:
        shock_list = f.input_list

    # sparse jacobian
    fd_sparse = all_derivatives(f, {k: ss[k] for k in f.input_list}, f.output_list, shock_list)

    # full jacobian
    J = {o: {} for o in f.output_list}
    for o in f.output_list:
        for i in fd_sparse[o].keys():
            J[o][i] = RecursiveSparse(fd_sparse[o][i]).matrix(T)

    return J
