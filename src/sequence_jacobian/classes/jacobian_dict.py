import copy
import warnings
import numpy as np

from ..utilities.misc import factor, factored_solve
from ..utilities.ordered_set import OrderedSet
from ..utilities.bijection import Bijection
from .impulse_dict import ImpulseDict
from .sparse_jacobians import IdentityMatrix, SimpleSparse, make_matrix
from typing import Any, Dict, Union

Array = Any

Jacobian = Union[np.ndarray, IdentityMatrix, SimpleSparse]

class NestedDict:
    def __init__(self, nesteddict, outputs: OrderedSet=None, inputs: OrderedSet=None, name: str=None):
        if isinstance(nesteddict, NestedDict):
            self.nesteddict = nesteddict.nesteddict
            self.outputs: OrderedSet = nesteddict.outputs
            self.inputs: OrderedSet = nesteddict.inputs
            self.name: str = nesteddict.name
        else:
            self.nesteddict = nesteddict
            if outputs is None:
                outputs = OrderedSet(nesteddict.keys())
            if inputs is None:
                inputs = OrderedSet([])
                for v in nesteddict.values():
                    inputs |= v

            if not outputs or not inputs:
                outputs = OrderedSet([])
                inputs = OrderedSet([])

            self.outputs = OrderedSet(outputs)
            self.inputs = OrderedSet(inputs)
            if name is None:
                # TODO: Figure out better default naming scheme for NestedDicts
                self.name = "NestedDict"
            else:
                self.name = name

    def __repr__(self):
        return f'<{type(self).__name__} outputs={self.outputs}, inputs={self.inputs}>'

    def __iter__(self):
        return iter(self.outputs)

    def __or__(self, other):
        # non-in-place merge: make a copy, then update
        merged = type(self)(self.nesteddict, self.outputs, self.inputs)
        merged.update(other)
        return merged

    def __getitem__(self, x):
        if isinstance(x, str):
            # case 1: just a single output, give subdict
            return self.nesteddict[x]
        elif isinstance(x, tuple):
            # case 2: tuple, referring to output and input
            o, i = x
            o = self.outputs if o == slice(None, None, None) else o
            i = self.inputs if i == slice(None, None, None) else i
            if isinstance(o, str):
                if isinstance(i, str):
                    # case 2a: one output, one input, return single Jacobian
                    return self.nesteddict[o][i]
                else:
                    # case 2b: one output, multiple inputs, return dict
                    return subdict(self.nesteddict[o], i)
            else:
                # case 2c: multiple outputs, one or more inputs, return NestedDict with outputs o and inputs i
                i = (i,) if isinstance(i, str) else i
                return type(self)({oo: subdict(self.nesteddict[oo], i) for oo in o}, o, i)
        elif isinstance(x, OrderedSet) or isinstance(x, list) or isinstance(x, set):
            # case 3: assume that list or set refers just to outputs, get all of those
            return type(self)({oo: self.nesteddict[oo] for oo in x}, x, self.inputs)
        else:
            raise ValueError(f'Tried to get impermissible item {x}')

    def get(self, *args, **kwargs):
        # this is for compatibility, not a huge fan
        return self.nesteddict.get(*args, **kwargs)

    def update(self, J):
        if not J.outputs or not J.inputs:
            return
        if set(self.inputs) != set(J.inputs):
            raise ValueError \
                (f'Cannot merge {type(self).__name__}s with non-overlapping inputs {set(self.inputs) ^ set(J.inputs)}')
        if not set(self.outputs).isdisjoint(J.outputs):
            raise ValueError \
                (f'Cannot merge {type(self).__name__}s with overlapping outputs {set(self.outputs) & set(J.outputs)}')
        self.outputs = self.outputs | J.outputs
        self.nesteddict = {**self.nesteddict, **J.nesteddict}

    # Ensure that every output in self has either a Jacobian or filler value for each input,
    # s.t. all inputs map to all outputs
    def complete(self, filler):
        nesteddict = {}
        for o in self.outputs:
            nesteddict[o] = dict(self.nesteddict[o])
            for i in self.inputs:
                if i not in nesteddict[o]:
                    nesteddict[o][i] = filler
        return type(self)(nesteddict, self.outputs, self.inputs)


def deduplicate(mylist):
    """Remove duplicates while otherwise maintaining order"""
    return list(dict.fromkeys(mylist))


def subdict(d, ks):
    """Return subdict of d with only keys in ks (if some ks are not in d, ignore them)"""
    return {k: d[k] for k in ks if k in d}


class JacobianDict(NestedDict):
    def __init__(self, nesteddict, outputs=None, inputs=None, name=None, T=None, check=False):
        if check:
            ensure_valid_jacobiandict(nesteddict)
        super().__init__(nesteddict, outputs=outputs, inputs=inputs, name=name)
        self.T = T

    @staticmethod
    def identity(ks):
        return JacobianDict({k: {k: IdentityMatrix()} for k in ks}, ks, ks)

    def addinputs(self):
        """Add any inputs that were not already in output list as outputs, with the identity"""
        inputs = [x for x in self.inputs if x not in self.outputs]
        return self | JacobianDict.identity(inputs)

    def __matmul__(self, x):
        if isinstance(x, JacobianDict):
            return self.compose(x)
        elif isinstance(x, Bijection):
            return self.remap(x)
        else:
            return self.apply(x)

    def __rmatmul__(self, x):
        if isinstance(x, Bijection):
            return self.remap(x)

    def remap(self, x: Bijection):
        if not x:
            return self
        nesteddict = x @ self.nesteddict
        for o in nesteddict.keys():
            nesteddict[o] = x @ nesteddict[o]
        return JacobianDict(nesteddict, inputs=x @ self.inputs, outputs=x @ self.outputs)

    def __bool__(self):
        return bool(self.outputs) and bool(self.inputs)

    def compose(self, J):
        """Returns self @ J"""
        if self.T is not None and J.T is not None and self.T != J.T:
            raise ValueError(f'Trying to multiply JacobianDicts with inconsistent dimensions {self.T} and {J.T}')

        o_list = self.outputs
        m_list = tuple(set(self.inputs) & set(J.outputs))
        i_list = J.inputs

        J_om = self.nesteddict
        J_mi = J.nesteddict
        J_oi = {}

        for o in o_list:
            J_oi[o] = {}
            for i in i_list:
                Jout = None
                for m in m_list:
                    if m in J_om[o] and i in J_mi[m]:
                        if Jout is None:
                            Jout = J_om[o][m] @ J_mi[m][i]
                        else:
                            Jout += J_om[o][m] @ J_mi[m][i]
                if Jout is not None:
                    J_oi[o][i] = Jout

        return JacobianDict(J_oi, o_list, i_list)

    def apply(self, x: Union[ImpulseDict, Dict[str, Array]]):
        """Returns J @ x"""
        x = ImpulseDict(x)

        inputs = x.keys() & set(self.inputs)
        J_oi = self.nesteddict
        y = {}

        for o in self.outputs:
            y[o] = np.zeros(x.T)
            J_i = J_oi[o]
            for i in inputs:
                if i in J_i:
                    y[o] += J_i[i] @ x[i]

        return ImpulseDict(y, T=x.T)

    def pack(self, T=None):
        if T is None:
            if self.T is not None:
                T = self.T
            else:
                raise ValueError('Trying to pack {self} into matrix, but do not know {T}')
        else:
            if self.T is not None and T != self.T:
                raise ValueError('{self} has dimension {self.T}, but trying to pack it with alternate dimension {T}')

        J = np.empty((len(self.outputs) * T, len(self.inputs) * T))
        for iO, O in enumerate(self.outputs):
            for iI, I in enumerate(self.inputs):
                J_OI = self[O].get(I)
                if J_OI is not None:
                    J[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = make_matrix(J_OI, T)
                else:
                    J[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))] = 0
        return J

    @staticmethod
    def unpack(bigjac, outputs, inputs, T):
        """If we have an (nO*T)*(nI*T) jacobian and provide names of nO outputs and nI inputs, output nested dictionary"""
        jacdict = {}
        for iO, O in enumerate(outputs):
            jacdict[O] = {}
            for iI, I in enumerate(inputs):
                jacdict[O][I] = bigjac[(T * iO):(T * (iO + 1)), (T * iI):(T * (iI + 1))]
        return JacobianDict(jacdict, outputs, inputs, T=T)

class FactoredJacobianDict:
    def __init__(self, jacobian_dict: JacobianDict, T=None):
        if jacobian_dict.T is None:
            if T is None:
                raise ValueError(f'Trying to factor (solve) {jacobian_dict} but do not know T')
            self.T = T
        else:
            self.T = jacobian_dict.T

        H_U = jacobian_dict.pack(T)
        self.targets = jacobian_dict.outputs
        self.unknowns = jacobian_dict.inputs
        if len(self.targets) != len(self.unknowns):
            raise ValueError('Trying to factor JacobianDict unequal number of inputs (unknowns)'
                            f' {self.unknowns} and outputs (targets) {self.targets}')
        self.H_U_factored = factor(H_U)

    def __repr__(self):
        return f'<{type(self).__name__} unknowns={self.unknowns}, targets={self.targets}>'

    # TODO: test this
    def to_jacobian_dict(self):
        return JacobianDict.unpack(-factored_solve(self.H_U_factored, np.eye(self.T*len(self.unknowns))),
                                    self.unknowns, self.targets, self.T)

    def __matmul__(self, x):
        if isinstance(x, JacobianDict):
            return self.compose(x)
        elif isinstance(x, Bijection):
            return self.remap(x)
        else:
            return self.apply(x)

    def __rmatmul__(self, x):
        if isinstance(x, Bijection):
            return self.remap(x)

    def remap(self, x: Bijection):
        if not x:
            return self
        newself = copy.copy(self)
        newself.unknowns = x @ self.unknowns
        newself.targets = x @ self.targets
        return newself

    def compose(self, J: JacobianDict):
        """Returns = -H_U^{-1} @ J"""
        Jsub = J[[o for o in self.targets if o in J.outputs]].pack(self.T)
        out = -factored_solve(self.H_U_factored, Jsub) 
        return JacobianDict.unpack(out, self.unknowns, J.inputs, self.T)

    def apply(self, x: Union[ImpulseDict, Dict[str, Array]]):
        """Returns -H_U^{-1} @ x"""
        xsub = ImpulseDict(x)[self.targets].pack()
        out = -factored_solve(self.H_U_factored, xsub)
        return ImpulseDict.unpack(out, self.unknowns, self.T)


def ensure_valid_jacobiandict(d):
    """The valid structure of `d` is a Dict[str, Dict[str, Jacobian]], where calling `d[o][i]` yields a
    Jacobian of type Jacobian mapping sequences of `i` to sequences of `o`. The null type for `d` is assumed
    to be {}, which is permitted the empty version of a valid nested dict."""

    if d and not isinstance(d, JacobianDict):
        # Assume it's sufficient to just check one of the keys
        if not isinstance(next(iter(d.keys())), str):
            raise ValueError(f"The dict argument {d} must have keys with type `str` to indicate `output` names.")

        jac_o_dict = next(iter(d.values()))
        if isinstance(jac_o_dict, dict):
            if jac_o_dict:
                if not isinstance(next(iter(jac_o_dict.keys())), str):
                    raise ValueError(f"The values of the dict argument {d} must be dicts with keys of type `str` to indicate"
                                    f" `input` names.")
                jac_o_i = next(iter(jac_o_dict.values()))
                if not isinstance(jac_o_i, Jacobian):
                    raise ValueError(f"The dict argument {d}'s values must be dicts with values of type `Jacobian`.")
                else:
                    if isinstance(jac_o_i, np.ndarray) and np.shape(jac_o_i)[0] != np.shape(jac_o_i)[1]:
                        raise ValueError(f"The Jacobians in {d} must be square matrices of type `Jacobian`.")
        else:
            raise ValueError(f"The argument {d} must be of type `dict`, with keys of type `str` and"
                             f" values of type `Jacobian`.")


def verify_saved_jacobian(block_name, Js, outputs, inputs, T):
    """Verify that pre-computed Jacobian has all the right outputs, inputs, and length."""
    if block_name not in Js.keys():
        # don't throw warning, this will happen often for simple blocks
        return False
    J = Js[block_name]

    if not isinstance(J, JacobianDict):
        warnings.warn(f'Js[{block_name}] is not a JacobianDict.')
        return False

    if not set(outputs).issubset(set(J.outputs)):
        missing = set(outputs).difference(set(J.outputs))
        warnings.warn(f'Js[{block_name}] misses required outputs {missing}.')
        return False

    if not set(inputs).issubset(set(J.inputs)):
        missing = set(inputs).difference(set(J.inputs))
        warnings.warn(f'Js[{block_name}] misses required inputs {missing}.')
        return False

    # Jacobian of simple blocks may have a sparse representation
    if T is not None:
        Tsaved = J[J.outputs[0]][J.inputs[0]].shape[-1]
        if T != Tsaved:
            warnings.warn(f'Js[{block_name} has length {Tsaved}, but you asked for {T}')
            return False

    return True
