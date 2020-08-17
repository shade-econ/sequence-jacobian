"""Test displacement handler classes: Ignore, IgnoreVector, Displace, Perturb, Reporter"""
import numpy as np

from sequence_jacobian.blocks.simple_block import (
    Ignore, IgnoreVector, Displace, numeric_primitive
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
