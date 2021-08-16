from sequence_jacobian.utilities.ordered_set import OrderedSet
from sequence_jacobian.utilities.function import ExtendedFunction, metadata


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

