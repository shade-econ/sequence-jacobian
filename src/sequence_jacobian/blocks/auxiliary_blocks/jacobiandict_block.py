"""A simple wrapper for JacobianDicts to be embedded in DAGs"""

from ...primitives import Block, Array
from ...jacobian.classes import JacobianDict
from ..support.impulse import ImpulseDict


class JacobianDictBlock(JacobianDict, Block):
    """A wrapper for nested dicts/JacobianDicts passed directly into DAGs to ensure method compatibility"""
    def __init__(self, nesteddict, outputs=None, inputs=None, name=None):
        super().__init__(nesteddict, outputs=outputs, inputs=inputs, name=name)

    def __repr__(self):
        return f"<JacobianDictBlock outputs={self.outputs}, inputs={self.inputs}>"

    def _impulse_linear(self, ss, inputs, outputs, Js):
        return ImpulseDict(self.jacobian(ss, list(inputs.keys()), outputs, inputs.T, Js).apply(inputs))

    def _jacobian(self, ss, inputs, outputs, T):
        if not inputs <= self.inputs:
            raise KeyError(f'Asking JacobianDictBlock for {inputs - self.inputs}, which are among its inputs {self.inputs}')
        if not outputs <= self.outputs:
            raise KeyError(f'Asking JacobianDictBlock for {outputs - self.outputs}, which are among its outputs {self.outputs}')
        return self[outputs, inputs]
