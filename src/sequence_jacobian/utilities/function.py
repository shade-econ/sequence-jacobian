import re
import inspect
import numpy as np

from .ordered_set import OrderedSet
from . import graph

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
    return OrderedSet(inspect.signature(f).parameters)


def input_defaults(f):
    defaults = {}
    for p in inspect.signature(f).parameters.values():
        if p.default != p.empty:
            defaults[p.name] = p.default
    return defaults


def output_list(f):
    """Scans source code of function to detect statement like

    'return L, Div'

    and reports the list ['L', 'Div'].

    Important to write functions in this way when they will be scanned by output_list, for
    either SimpleBlock or HetBlock.
    """
    return OrderedSet(re.findall('return (.*?)\n', inspect.getsource(f))[-1].replace(' ', '').split(','))


def metadata(f):
    name = f.__name__
    inputs = input_list(f)
    outputs = output_list(f)
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

    def differentiable(self, input_dict, h=1E-5, twosided=False):
        return DifferentiableExtendedFunction(self.f, self.name, self.inputs, self.outputs, input_dict, h, twosided)


class DifferentiableExtendedFunction(ExtendedFunction):
    def __init__(self, f, name, inputs, outputs, input_dict, h=1E-5, twosided=False):
        self.f, self.name, self.inputs, self.outputs = f, name, inputs, outputs
        self.input_dict = input_dict
        self.output_dict = None # lazy evaluation of outputs for one-sided diff
        self.h = h
        self.default_twosided = twosided

    def diff(self, shock_dict, h=None, hide_zeros=False, twosided=None):
        if twosided is None:
            twosided = self.default_twosided

        if not twosided:
            return self.diff1(shock_dict, h, hide_zeros)
        else:
            return self.diff2(shock_dict, h, hide_zeros)

    def diff1(self, shock_dict, h=None, hide_zeros=False):
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
            h = self.h

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


class CombinedExtendedFunction(ExtendedFunction):
    def __init__(self, fs, name=None):
        self.dag = graph.DAG([ExtendedFunction(f) for f in fs])
        self.inputs = self.dag.inputs
        self.outputs = self.dag.outputs
        self.functions = {b.name: b for b in self.dag.blocks}

        if name is None:
            names = list(self.functions)
            if len(names) == 1:
                self.name = names[0]
            else:
                self.name = f'{names[0]}_{names[-1]}'
        else:
            self.name = name
    
    def __call__(self, input_dict, outputs=None):
        functions_to_visit = list(self.functions.values())
        if outputs is not None:
            functions_to_visit = [functions_to_visit[i] for i in self.dag.visit_from_outputs(outputs)]
        
        results = input_dict.copy()
        for f in functions_to_visit:
            results.update(f(results))

        if outputs is not None:
            return {k: results[k] for k in outputs}
        else:
            return results

    def call_on_deviations(self, ss, dev_dict, outputs=None):
        functions_to_visit = self.filter(list(self.functions.values()), dev_dict, outputs)

        results = {}
        input_dict = {**ss, **dev_dict}
        for f in functions_to_visit:
            out = f(input_dict)
            results.update(out)
            input_dict.update(out)

        if outputs is not None:
            return {k: v for k, v in results.items() if k in outputs}
        else:
            return results

    def filter(self, function_list, inputs, outputs=None):
        nums_to_visit = self.dag.visit_from_inputs(inputs)
        if outputs is not None:
            nums_to_visit &= self.dag.visit_from_outputs(outputs)
        return [function_list[n] for n in nums_to_visit]

    def wrapped_call(self, input_dict, preprocess=None, postprocess=None):
        raise NotImplementedError

    def add(self, f):
        if inspect.isfunction(f) or isinstance(f, ExtendedFunction):
            return CombinedExtendedFunction(list(self.functions.values()) + [f])
        else:
            # otherwise assume f is iterable
            return CombinedExtendedFunction(list(self.functions.values()) + list(f))
        
    def remove(self, name):
        if isinstance(name, str):
            return CombinedExtendedFunction([v for k, v in self.functions.items() if k != name])
        else:
            # otherwise assume name is iterable
            return CombinedExtendedFunction([v for k, v in self.functions.items() if k not in name])

    def children(self):
        return OrderedSet(self.functions)

    def differentiable(self, input_dict, h=1E-5, twosided=False):
        return DifferentiableCombinedExtendedFunction(self.functions, self.dag, self.name, self.inputs, self.outputs, input_dict, h, twosided)        


class DifferentiableCombinedExtendedFunction(CombinedExtendedFunction, DifferentiableExtendedFunction):
    def __init__(self, functions, dag, name, inputs, outputs, input_dict, h=1E-5, twosided=False):
        self.dag, self.name, self.inputs, self.outputs = dag, name, inputs, outputs
        diff_functions = {}
        for k, f in functions.items():
            diff_functions[k] = f.differentiable(input_dict, h)
        self.diff_functions = diff_functions
        self.default_twosided = twosided
    
    def diff(self, shock_dict, h=None, outputs=None, hide_zeros=False, twosided=False):
        if twosided is None:
            twosided = self.default_twosided

        if not twosided:
            return self.diff1(shock_dict, h, outputs, hide_zeros)
        else:
            return self.diff2(shock_dict, h, outputs, hide_zeros)

    def diff1(self, shock_dict, h=None, outputs=None, hide_zeros=False):
        functions_to_visit = self.filter(list(self.diff_functions.values()), shock_dict, outputs)

        shock_dict = shock_dict.copy()
        results = {}
        for f in functions_to_visit:
            out = f.diff1(shock_dict, h, hide_zeros)
            results.update(out)
            shock_dict.update(out)
        
        if outputs is not None:
            return {k: v for k, v in results.items() if k in outputs}
        else:
            return results

    def diff2(self, shock_dict, h=None, outputs=None, hide_zeros=False):
        functions_to_visit = self.filter(list(self.diff_functions.values()), shock_dict, outputs)
        
        shock_dict = shock_dict.copy()
        results = {}
        for f in functions_to_visit:
            out = f.diff2(shock_dict, h, hide_zeros)
            results.update(out)
            shock_dict.update(out)
        
        if outputs is not None:
            return {k: v for k, v in results.items() if k in outputs}
        else:
            return results

