"""Test displacement handler classes: Ignore, IgnoreVector, Displace, Perturb, Reporter"""

import numpy as np

from sequence_jacobian.blocks.support.simple_displacement import (
    Ignore, IgnoreVector, Displace, AccumulatedDerivative, numeric_primitive
)

# Define useful helper functions for testing
# Assumes "op" is an actual well-defined arithmetic operator. If necessary, implement more stringent checks
# on the "op" being passed in so nonsense doesn't come out.
# i.e. reverse_op("__round__") doesn't return reverse_op("__ound__")
def reverse_op(op):
    if op[2] == "r":
        return op[0:2] + op[3:]
    else:
        return op[0:2] + "r" + op[2:]


def apply_unary_op(op, a):
    return getattr(a, op)()


def apply_binary_op(op, a1, a2):
    if getattr(a1, op)(a2) is not NotImplemented:
        return getattr(a1, op)(a2)
    elif getattr(a2, reverse_op(op))(a1) is not NotImplemented:
        return getattr(a2, reverse_op(op))(a1)
    else:
        raise NotImplementedError(f"{op} cannot be performed between {a1} and {a2} directly, and no"
                                  f" valid reverse operation exists either.")


def apply_op(op, *args):
    if len(args) == 1:
        return apply_unary_op(op, *args)
    elif len(args) == 2:
        return apply_binary_op(op, *args)
    else:
        raise ValueError(f"apply_op only supports unary or binary operators currently. {len(args)} is an invalid"
                         f" number of arguments to provide.")


def test_ignore():
    # Test unary operations
    arg_singles = [Ignore(1), Ignore(1)(-1)]
    for t1 in arg_singles:
        for op in ["__neg__", "__pos__"]:
            assert type(apply_op(op, t1)) == Ignore
            assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(Ignore(1), 1), (1, Ignore(1)), (Ignore(1), Ignore(2)),
                 (Ignore(1)(-1), 1), (1, Ignore(1)(-1)), (Ignore(1)(-1), Ignore(2)), (Ignore(1), Ignore(2)(-1))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            assert type(apply_op(op, t1, t2)) == Ignore
            assert np.all(numeric_primitive(apply_op(op, t1, t2)) == apply_op(op, numeric_primitive(t1),
                                                                              numeric_primitive(t2)))

    # Test call
    for t1 in arg_singles:
        assert numeric_primitive(t1) == numeric_primitive(t1(+1))


def test_ignore_vector():
    # Test unary operations
    arg_singles = [IgnoreVector(np.array([1, 2, 3])), IgnoreVector(np.array([1, 2, 3]))(-1)]
    for t1 in arg_singles:
        for op in ["__neg__", "__pos__"]:
            assert type(apply_op(op, t1)) == IgnoreVector
            assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(IgnoreVector(np.array([1, 2, 3])), 1),
                 (IgnoreVector(np.array([1, 2, 3])), Ignore(1)),
                 (IgnoreVector(np.array([1, 2, 3])), IgnoreVector(np.array([2, 3, 4]))),
                 (1, IgnoreVector(np.array([1, 2, 3]))),
                 (Ignore(1), IgnoreVector(np.array([1, 2, 3]))),

                 (IgnoreVector(np.array([1, 2, 3]))(-1), 1),
                 (IgnoreVector(np.array([1, 2, 3]))(-1), Ignore(1)),
                 (IgnoreVector(np.array([1, 2, 3]))(-1), IgnoreVector(np.array([2, 3, 4]))),
                 (IgnoreVector(np.array([1, 2, 3])), IgnoreVector(np.array([2, 3, 4]))(-1)),
                 (1, IgnoreVector(np.array([1, 2, 3]))(-1)),
                 (Ignore(1), IgnoreVector(np.array([1, 2, 3]))(-1))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            assert type(apply_op(op, t1, t2)) == IgnoreVector
            assert np.all(numeric_primitive(apply_op(op, t1, t2)) == apply_op(op, numeric_primitive(t1),
                                                                              numeric_primitive(t2)))

    # Test call
    for t1 in arg_singles:
        assert np.all(numeric_primitive(t1) == numeric_primitive(t1(+1)))


def test_displace():
    # Test unary operations
    arg_singles = [Displace(np.array([1, 2, 3]), ss=2), Displace(np.array([1, 2, 3]), ss=2)(-1)]
    for t1 in arg_singles:
        for op in ["__neg__", "__pos__"]:
            assert type(apply_op(op, t1)) == Displace
            assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(Displace(np.array([1, 2, 3]), ss=2), 1),
                 (Displace(np.array([1, 2, 3]), ss=2), Ignore(1)),
                 (Displace(np.array([1, 2, 3]), ss=2), Displace(np.array([2, 3, 4]), ss=3)),
                 (1, Displace(np.array([1, 2, 3]), ss=2)),
                 (Ignore(1), Displace(np.array([1, 2, 3]), ss=2)),

                 (Displace(np.array([1, 2, 3]), ss=2)(-1), 1),
                 (Displace(np.array([1, 2, 3]), ss=2)(-1), Ignore(1)),
                 (Displace(np.array([1, 2, 3]), ss=2)(-1), Displace(np.array([2, 3, 4]), ss=3)),
                 (Displace(np.array([1, 2, 3]), ss=2), Displace(np.array([2, 3, 4]), ss=3)(-1)),
                 (1, Displace(np.array([1, 2, 3]), ss=2)(-1)),
                 (Ignore(1), Displace(np.array([1, 2, 3]), ss=2)(-1))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            assert type(apply_op(op, t1, t2)) == Displace
            assert np.all(numeric_primitive(apply_op(op, t1, t2)) == apply_op(op, numeric_primitive(t1),
                                                                              numeric_primitive(t2)))
            assert np.all(numeric_primitive(apply_op(op, t1, t2).ss) ==\
                   apply_op(op, t1.ss if isinstance(t1, Displace) else numeric_primitive(t1),
                            t2.ss if isinstance(t2, Displace) else numeric_primitive(t2)))

    # Test call
    for t1 in arg_singles:
        t1_manual_displace = np.zeros(len(t1))
        t1_manual_displace[:-1] = numeric_primitive(t1)[1:]
        t1_manual_displace[-1:] = t1.ss
        assert np.all(numeric_primitive(t1(1)) == t1_manual_displace)


def test_accumulated_derivative():
    # Test unary operations
    arg_singles = [AccumulatedDerivative(), AccumulatedDerivative()(-1), AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)]
    for t1 in arg_singles:
        for op in ["__neg__", "__pos__"]:
            assert type(apply_op(op, t1)) == AccumulatedDerivative
            assert np.all(np.fromiter(apply_op(op, t1).elements.values(), dtype=float) ==
                          np.array([apply_op(op, v) for v in t1.elements.values()]))

    # TODO: Only test against scalars as of now. Will need to revisit this to test against vectors
    #   e.g. IgnoreVector, once hetinput/hetoutput functionality is enhanced
    # Test binary operations
    arg_pairs = [(AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.), 3),
                 (AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.), Ignore(3)),
                 (3, AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)),
                 (Ignore(3), AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)),
                 (AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.),
                  AccumulatedDerivative(elements={(1, 1): 4.}, f_value=5.)),
                 # TODO: Implement test for elements not in the same (i, m)
                 # (AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.),
                 #  AccumulatedDerivative(elements={(1, 0): 4.}, f_value=5.)),

                 (AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)(-1), 3),
                 (AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)(-1), Ignore(3)),
                 (3, AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)(-1)),
                 (Ignore(3), AccumulatedDerivative(elements={(1, 1): 2.}, f_value=2.)(-1))]

    def get_fp_value(x):
        return x._fp_values[0]

    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:

            assert type(apply_op(op, t1, t2)) == AccumulatedDerivative

            result = get_fp_value(apply_op(op, t1, t2))

            if isinstance(t1, AccumulatedDerivative) and isinstance(t2, AccumulatedDerivative):
                assert apply_op(op, t1, t2).f_value == apply_op(op, t1.f_value, t2.f_value)

                if op in ["__add__", "__radd__", "__sub__", "__rsub__"]:
                    assert result == apply_op(op, get_fp_value(t1), get_fp_value(t2))
                elif op in ["__mul__", "__rmul__"]:
                    assert result == get_fp_value(t1) * t2.f_value + t1.f_value * get_fp_value(t2)
                elif op == "__truediv__":
                    assert result == (t2.f_value * get_fp_value(t1) - t1.f_value * get_fp_value(t2))/t2.f_value**2
                elif op == "__rtruediv__":
                    assert result == (t1.f_value * get_fp_value(t2) - t2.f_value * get_fp_value(t1))/t1.f_value**2
                elif op == "__pow__":
                    assert result == (t1.f_value ** (t2.f_value - 1)) * (t2.f_value * get_fp_value(t1) +\
                                                               t1.f_value * np.log(t1.f_value) * get_fp_value(t2))
                else:  # op == "__rpow__":
                    assert result == (t2.f_value ** (t1.f_value - 1)) * (t1.f_value * get_fp_value(t2) +\
                                                               t2.f_value * np.log(t2.f_value) * get_fp_value(t1))
            else:
                assert apply_op(op, t1, t2).f_value == apply_op(op, t1.f_value, numeric_primitive(t2))\
                    if isinstance(t1, AccumulatedDerivative) else apply_op(op, numeric_primitive(t1), t2.f_value)

                if op in ["__add__", "__radd__", "__sub__"]:
                    assert result == get_fp_value(t1) if\
                        isinstance(t1, AccumulatedDerivative) else get_fp_value(t2)
                elif op == "__rsub__":
                    assert result == -get_fp_value(t1) if \
                        isinstance(t1, AccumulatedDerivative) else -get_fp_value(t2)
                elif op in ["__mul__", "__rmul__"]:
                    assert result == numeric_primitive(t2) * get_fp_value(t1)\
                        if isinstance(t1, AccumulatedDerivative) else numeric_primitive(t1) * get_fp_value(t2)
                elif op == "__truediv__":
                    assert result == get_fp_value(t1)/numeric_primitive(t2) if isinstance(t1, AccumulatedDerivative)\
                        else -numeric_primitive(t1)/t2.f_value**2 * get_fp_value(t2)
                elif op == "__rtruediv__":
                    assert result == -numeric_primitive(t2)/t1.f_value**2 * get_fp_value(t1)\
                        if isinstance(t1, AccumulatedDerivative) else get_fp_value(t2)/numeric_primitive(t1)
                elif op == "__pow__":
                    assert result == numeric_primitive(t2) * t1.f_value ** (numeric_primitive(t2) - 1) * get_fp_value(t1)\
                        if isinstance(t1, AccumulatedDerivative) else\
                        np.log(numeric_primitive(t1)) * numeric_primitive(t1) ** t2.f_value * get_fp_value(t2)
                else:  # op == "__rpow__"
                    assert result == np.log(numeric_primitive(t2)) * numeric_primitive(t2) ** t1.f_value * get_fp_value(t1)\
                        if isinstance(t1, AccumulatedDerivative) else\
                        numeric_primitive(t1) * t2.f_value ** (numeric_primitive(t1) - 1) * get_fp_value(t2)