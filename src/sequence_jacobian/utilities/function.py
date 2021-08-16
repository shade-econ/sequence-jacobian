from sequence_jacobian.utilities.ordered_set import OrderedSet
import re
import inspect

# TODO: fix this, have it twice (main version in misc) due to circular import problem
# let's make everything point to here for input_list, etc. so that this is unnecessary
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


def metadata(f):
    name = f.__name__
    inputs = OrderedSet(input_list(f))
    outputs = OrderedSet(output_list(f))
    return name, inputs, outputs


class ExtendedFunction:
    """Wrapped function that knows its inputs and outputs. Evaluates on dict containing necessary
    inputs, returns dict containing outputs by name"""

    def __init__(self, f):
        self.f = f
        self.name, self.inputs, self.outputs = metadata(f)

    def __call__(self, input_dict):
        # take subdict of d contained in inputs
        # this allows for d not to include all inputs (if there are optional inputs)
        input_dict = {k: v for k, v in input_dict.items() if k in self.inputs}
        return self.outputs.dict_from(make_tuple(self.f(**input_dict)))

    def wrapped_call(self, input_dict, preprocess=None, postprocess=None):
        if preprocess is not None:
            input_dict = {k: preprocess(v) for k, v in input_dict.items() if k in self.inputs}
        else:
            input_dict = {k: v for k, v in input_dict.items() if k in self.inputs}

        output_dict = self.outputs.dict_from(make_tuple(self.f(**input_dict)))
        if postprocess is not None:
            output_dict = {k: postprocess(v) for k, v in output_dict.items()}
        
        return output_dict
