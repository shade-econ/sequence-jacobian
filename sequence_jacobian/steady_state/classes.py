"""Various classes to support the computation of steady states"""

from copy import deepcopy

from ..utilities.misc import dict_diff, smart_set


def construct_internal_namespace(data, block):
    # Only supporting internal namespaces for HetBlocks currently
    if hasattr(block, "back_step_fun"):
        return {block.name: {k: v for k, v in deepcopy(data).items() if k in
                             smart_set(block.back_step_outputs) | smart_set(block.exogenous) | {"D"} |
                             smart_set(block.hetinput_outputs) | smart_set(block.hetoutput_outputs)}}
    else:
        return {}


class SteadyStateDict:
    def __init__(self, data, internal=None):
        if isinstance(data, SteadyStateDict):
            self.internal = deepcopy(data.internal)
            self.toplevel = deepcopy(data.toplevel)
        else:
            if internal is not None:
                # Either we can construct the internal namespace for you (if it's a Block) otherwise,
                # you can provide the nested dict representing the internal namespace directly
                if hasattr(internal, "inputs") and hasattr(internal, "outputs"):
                    self.internal = construct_internal_namespace(data, internal)
                else:
                    self.internal = internal
            else:
                self.internal = {}
            if self.internal:
                toplevel = deepcopy(data)
                for internal_dict in self.internal.values():
                    toplevel = dict_diff(toplevel, internal_dict)
                self.toplevel = toplevel
            else:
                self.toplevel = deepcopy(data)

    def __repr__(self):
        if self.internal:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}, internal={list(self.internal.keys())}>"
        else:
            return f"<{type(self).__name__}: {list(self.toplevel.keys())}>"

    def __iter__(self):
        return iter(self.toplevel)

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.toplevel[k]
        else:
            try:
                return {ki: self.toplevel[ki] for ki in k}
            except TypeError:
                raise TypeError(f'Key {k} needs to be a string or an iterable (list, set, etc) of strings')

    def __setitem__(self, k, v):
        self.toplevel[k] = v

    def keys(self):
        return self.toplevel.keys()

    def values(self):
        return self.toplevel.values()

    def items(self):
        return self.toplevel.items()

    def update(self, new_data):
        if isinstance(new_data, SteadyStateDict):
            self.toplevel.update(new_data.toplevel)
            self.internal.update(new_data.internal)
        else:
            # TODO: This is assuming new_data only contains aggregates. Upgrade in later commit to handle the case of
            #   vector-valued variables/collection into internal namespaces
            self.toplevel.update(new_data)

    def difference(self, data_to_remove):
        return SteadyStateDict(dict_diff(self.toplevel, data_to_remove), internal=deepcopy(self.internal))
