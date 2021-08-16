"""A simple wrapper for JacobianDicts to be embedded in DAGs"""

from numbers import Real
from typing import Dict, Union, List

from ...primitives import Block, Array
from ...jacobian.classes import JacobianDict


class JacobianDictBlock(JacobianDict, Block):
    """A wrapper for nested dicts/JacobianDicts passed directly into DAGs to ensure method compatibility"""
    def __init__(self, nesteddict, outputs=None, inputs=None, name=None):
        super().__init__(nesteddict, outputs=outputs, inputs=inputs, name=name)

    def __repr__(self):
        return f"<JacobianDictBlock outputs={self.outputs}, inputs={self.inputs}>"

    def impulse_linear(self, ss: Dict[str, Union[Real, Array]],
                       exogenous: Dict[str, Array], **kwargs) -> Dict[str, Array]:
        return self.jacobian(list(exogenous.keys())).apply(exogenous)

    def _jacobian(self, ss, inputs, outputs, T) -> JacobianDict:
        # TODO: T should be an attribute of JacobianDict
        if not inputs <= self.inputs:
            raise KeyError(f'Asking JacobianDictBlock for {inputs - self.inputs}, which are among its inputs {self.inputs}')
        if not outputs <= self.outputs:
            raise KeyError(f'Asking JacobianDictBlock for {outputs - self.outputs}, which are among its outputs {self.outputs}')
        return self[outputs, inputs]
