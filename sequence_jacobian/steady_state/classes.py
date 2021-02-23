"""Various classes to support the computation of steady states"""

from copy import deepcopy
import numpy as np


class SteadyStateDict:
    def __init__(self, raw_dict, blocks=None):
        # TODO: Will need to re-think flat storage of data if/when we move to implementing remapping
        self.data = deepcopy(raw_dict)

        if blocks is not None:
            self.block_map = {b.name: b.outputs for b in blocks}
            self.block_names = list(self.block_map.keys())

            # Record the values in raw_dict not output by any of the blocks as part of the `calibration`
            self.block_map["calibration"] = set(raw_dict.keys()) - set().union(*self.block_map.values())
        else:
            self.block_map = {"calibration": set(raw_dict.keys())}
            self.block_names = ["calibration"]

    def __repr__(self, raw=False):
        if set(self.block_names) == {"calibration"} or raw:
            return self.data.__repr__()
        else:
            return f"<{type(self).__name__} blocks={self.block_names}"

    def __getitem__(self, key):
        if key in self.block_names:
            return {k: v for k, v in self.data.items() if k in self.block_map[key]}
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        if key not in self.data:
            block_name_to_assign = "calibration"
            self.block_map[block_name_to_assign] = self.block_map[block_name_to_assign] | {key}
        self.data[key] = value

    def aggregates(self):
        return {k: v for k, v in self.data.items() if np.isscalar(v)}
