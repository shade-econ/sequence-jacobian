from sequence_jacobian.utilities.ordered_set import OrderedSet
from sequence_jacobian.utilities.function import DifferentiableExtendedFunction, ExtendedFunction, ExtendedParallelFunction, metadata
import numpy as np

def f1(a, b, c):
    k = a + 1
    l = b - c
    return k, l

def f2(b):
    k = b + 4
    return k


def test_metadata():
    assert metadata(f1) == ('f1', OrderedSet(['a', 'b', 'c']), OrderedSet(['k', 'l']))
    assert metadata(f2) == ('f2', OrderedSet(['b']), OrderedSet(['k']))


def test_extended_function():
    inputs = {'a': 1, 'b': 2, 'c': 3}
    assert ExtendedFunction(f1)(inputs) == {'k': 2, 'l': -1}
    assert ExtendedFunction(f2)(inputs) == {'k': 6}


def f3(a, b):
    c = a*b - 3*a
    d = 3*b**2
    return c, d


def test_differentiable_extended_function():
    extf3 = ExtendedFunction(f3)

    ss1 = {'a': 1, 'b': 2}
    inputs1 = {'a': 0.5}

    diff = extf3.differentiable(ss1).diff(inputs1)
    assert np.isclose(diff['c'], -0.5)
    assert np.isclose(diff['d'], 0)


def f4(a, e):
    f = a / e
    return f


def test_differentiable_extended_parallel_function():
    fs = ExtendedParallelFunction([f3, f4])

    ss1 = {'a': 1, 'b': 2, 'e': 4}
    inputs1 = {'a': 0.5, 'e': 1}

    diff = fs.differentiable(ss1).diff(inputs1)
    assert np.isclose(diff['c'], -0.5)
    assert np.isclose(diff['d'], 0)
    assert np.isclose(diff['f'], 1/16)

    inputs2 = {'e': -2}
    diff = fs.differentiable(ss1).diff2(inputs2)
    assert list(diff) == ['f']
    assert np.isclose(diff['f'], 1/8)
