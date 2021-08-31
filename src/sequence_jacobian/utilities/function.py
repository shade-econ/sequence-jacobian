from sequence_jacobian.utilities.ordered_set import OrderedSet
import re
import inspect
import numpy as np

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
        if isinstance(f, ExtendedFunction):
            self.f, self.name, self.inputs, self.outputs = f.f, f.name, f.inputs, f.outputs
        else:
            self.f = f
            self.name, self.inputs, self.outputs = metadata(f)

    def __call__(self, input_dict):
        # take subdict of d contained in inputs
        # this allows for d not to include all inputs (if there are optional inputs)
        input_dict = {k: v for k, v in input_dict.items() if k in self.inputs}
        return self.outputs.dict_from(make_tuple(self.f(**input_dict)))

    def __repr__(self):
        return f'<{type(self).__name__}({self.name}): {self.inputs} -> {self.outputs}>'

    def wrapped_call(self, input_dict, preprocess=None, postprocess=None):
        if preprocess is not None:
            input_dict = {k: preprocess(v) for k, v in input_dict.items() if k in self.inputs}
        else:
            input_dict = {k: v for k, v in input_dict.items() if k in self.inputs}

        output_dict = self.outputs.dict_from(make_tuple(self.f(**input_dict)))
        if postprocess is not None:
            output_dict = {k: postprocess(v) for k, v in output_dict.items()}
        
        return output_dict

    def differentiable(self, input_dict, h=1E-6, h2=1E-4):
        return DifferentiableExtendedFunction(self.f, self.name, self.inputs, self.outputs, input_dict, h, h2)


class DifferentiableExtendedFunction(ExtendedFunction):
    def __init__(self, f, name, inputs, outputs, input_dict, h=1E-6, h2=1E-4):
        self.f, self.name, self.inputs, self.outputs = f, name, inputs, outputs
        self.input_dict = input_dict
        self.output_dict = None # lazy evaluation of outputs for one-sided diff
        self.h = h
        self.h2 = h2


    def diff(self, shock_dict, h=None, hide_zeros=False):
        if h is None:
            h = self.h

        if self.output_dict is None:
            self.output_dict = self(self.input_dict)

        shocked_input_dict = {**self.input_dict,
            **{k: self.input_dict[k] + h * shock for k, shock in shock_dict.items() if k in self.input_dict}}

        shocked_output_dict = self(shocked_input_dict)

        derivative_dict = {k: (shocked_output_dict[k] - self.output_dict[k])/h for k in self.output_dict}

        if hide_zeros:
            derivative_dict = hide_zero_values(derivative_dict)

        return derivative_dict

    def diff2(self, shock_dict, h=None, hide_zeros=False):
        if h is None:
            h = self.h2

        shocked_input_dict_up = {**self.input_dict,
            **{k: self.input_dict[k] + h * shock for k, shock in shock_dict.items() if k in self.input_dict}}
        shocked_input_dict_dn = {**self.input_dict,
            **{k: self.input_dict[k] - h * shock for k, shock in shock_dict.items() if k in self.input_dict}}

        shocked_output_dict_up = self(shocked_input_dict_up)
        shocked_output_dict_dn = self(shocked_input_dict_dn)

        derivative_dict = {k: (shocked_output_dict_up[k] - shocked_output_dict_dn[k])/(2*h) for k in shocked_output_dict_dn}

        if hide_zeros:
            derivative_dict = hide_zero_values(derivative_dict)

        return derivative_dict


def hide_zero_values(d):
    return {k: v for k, v in d.items() if not np.allclose(v, 0)}


class ExtendedParallelFunction(ExtendedFunction):
    def __init__(self, fs, name=None):
        inputs = OrderedSet([])
        outputs = OrderedSet([])
        functions = {}
        for f in fs:
            ext_f = ExtendedFunction(f)
            if not outputs.isdisjoint(ext_f.outputs):
                raise ValueError(f'Overlap in outputs of ParallelFunction: {ext_f.name} and others both have {outputs & ext_f.outputs}')
            inputs |= ext_f.inputs
            outputs |= ext_f.outputs

            if ext_f.name in functions:
                raise ValueError(f'Overlap in function names of ParallelFunction: {ext_f.name} listed twice')
            functions[ext_f.name] = ext_f

        self.inputs = inputs
        self.outputs = outputs
        self.functions = functions

        if name is None:
            names = list(functions)
            if len(names) == 1:
                self.name = names[0]
            else:
                self.name = f'{names[0]}_{names[-1]}'
        else:
            self.name = name

    def __call__(self, input_dict, outputs=None):
        results = {}
        for f in self.functions.values():
            if outputs is None or not f.outputs.isdisjoint(outputs): 
                results.update(f(input_dict))
        if outputs is not None:
            results = {k: results[k] for k in outputs}
        return results

    def call_on_deviations(self, ss, dev_dict, outputs=None):
        results = {}
        input_dict = {**ss, **dev_dict}
        for f in self.functions.values():
            if not f.inputs.isdisjoint(dev_dict):
                if outputs is None or not f.outputs.isdisjoint(outputs):
                    results.update(f(input_dict))
        if outputs is not None:
            results = {k: results[k] for k in outputs if k in results}
        return results
    
    def wrapped_call(self, input_dict, preprocess=None, postprocess=None):
        raise NotImplementedError

    def add(self, f):
        if isinstance(f, function) or isinstance(f, ExtendedFunction):
            return ExtendedParallelFunction(list(self.functions.values()) + [f])
        else:
            # otherwise assume f is iterable
            return ExtendedParallelFunction(list(self.functions.values()) + list(f))
        
    def remove(self, name):
        if isinstance(name, str):
            return ExtendedParallelFunction([v for k, v in self.functions.items() if k != name])
        else:
            # otherwise assume name is iterable
            return ExtendedParallelFunction([v for k, v in self.functions.items() if k not in name])

    def children(self):
        return OrderedSet(self.functions)

    def differentiable(self, input_dict, h=1E-6, h2=1E-4):
        return DifferentiableExtendedParallelFunction(self.functions, self.name, self.inputs, self.outputs, input_dict, h, h2)        


class DifferentiableExtendedParallelFunction(ExtendedParallelFunction, DifferentiableExtendedFunction):
    def __init__(self, functions, name, inputs, outputs, input_dict, h=1E-6, h2=1E-4):
        self.name, self.inputs, self.outputs = name, inputs, outputs
        diff_functions = {}
        for k, f in functions.items():
            diff_functions[k] = f.differentiable(input_dict, h, h2)
        self.diff_functions = diff_functions
    
    def diff(self, shock_dict, h=None, outputs=None, hide_zeros=False):
        results = {}
        for f in self.diff_functions.values():
            if not f.inputs.isdisjoint(shock_dict):
                if outputs is None or not f.outputs.isdisjoint(outputs):
                    results.update(f.diff(shock_dict, h, hide_zeros))
        if outputs is not None:
            results = {k: results[k] for k in outputs if k in results}
        return results

    def diff2(self, shock_dict, h=None, outputs=None, hide_zeros=False):
        results = {}
        for f in self.diff_functions.values():
            if not f.inputs.isdisjoint(shock_dict):
                if outputs is None or not f.outputs.isdisjoint(outputs):
                    results.update(f.diff2(shock_dict, h, hide_zeros))
        if outputs is not None:
            results = {k: results[k] for k in outputs if k in results}
        return results

