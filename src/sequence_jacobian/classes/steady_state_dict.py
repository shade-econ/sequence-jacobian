from copy import deepcopy

from .result_dict import ResultDict
from ..utilities.misc import dict_diff
from ..utilities.ordered_set import OrderedSet
from ..utilities.bijection import Bijection

import numpy as np

from numbers import Real
from typing import Any, Dict, Union
Array = Any

class SteadyStateDict(ResultDict):
    def difference(self, data_to_remove):
        return SteadyStateDict(dict_diff(self.toplevel, data_to_remove), deepcopy(self.internals))

    def _vector_valued(self):
        return OrderedSet([k for k, v in self.toplevel.items() if np.size(v) > 1])

UserProvidedSS = Dict[str, Union[Real, Array]]
