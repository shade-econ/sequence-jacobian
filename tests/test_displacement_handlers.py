"""Test displacement handler classes: Ignore, IgnoreVector, Displace, Perturb, Reporter"""
import numpy as np

from sequence_jacobian.blocks.simple_block import (
    Ignore, IgnoreVector, Displace, Reporter, Perturb, apply_op, numeric_primitive
)


def test_ignore():
    # Test unary operations
    t1 = Ignore(1)
    for op in ["__neg__", "__pos__"]:
        assert type(apply_op(op, t1)) == Ignore
        assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(Ignore(1), 1), (1, Ignore(1)), (Ignore(1), Ignore(2))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            assert type(apply_op(op, t1, t2)) == Ignore
            assert np.all(numeric_primitive(apply_op(op, t1, t2)) == apply_op(op, numeric_primitive(t1),
                                                                              numeric_primitive(t2)))


def test_ignore_vector():
    # Test unary operations
    t1 = IgnoreVector(np.array([1, 2, 3]))
    for op in ["__neg__", "__pos__"]:
        assert type(apply_op(op, t1)) == IgnoreVector
        assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(IgnoreVector(np.array([1, 2, 3])), 1),
                 (IgnoreVector(np.array([1, 2, 3])), Ignore(1)),
                 (IgnoreVector(np.array([1, 2, 3])), IgnoreVector(np.array([2, 3, 4]))),
                 (1, IgnoreVector(np.array([1, 2, 3]))),
                 (Ignore(1), IgnoreVector(np.array([1, 2, 3])))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            assert type(apply_op(op, t1, t2)) == IgnoreVector
            assert np.all(numeric_primitive(apply_op(op, t1, t2)) == apply_op(op, numeric_primitive(t1),
                                                                              numeric_primitive(t2)))


def test_displace():
    # Test unary operations
    t1 = Displace(np.array([1, 2, 3]), ss=2)
    for op in ["__neg__", "__pos__"]:
        assert type(apply_op(op, t1)) == Displace
        assert np.all(numeric_primitive(apply_op(op, t1)) == apply_op(op, numeric_primitive(t1)))

    # Test binary operations
    arg_pairs = [(Displace(np.array([1, 2, 3]), ss=2), 1),
                 (Displace(np.array([1, 2, 3]), ss=2), Ignore(1)),
                 (Displace(np.array([1, 2, 3]), ss=2), IgnoreVector(np.array([2, 3, 4]))),
                 (Displace(np.array([1, 2, 3]), ss=2), Displace(np.array([2, 3, 4]), ss=3)),
                 (1, Displace(np.array([1, 2, 3]), ss=2)),
                 (Ignore(1), Displace(np.array([1, 2, 3]), ss=2)),
                 (IgnoreVector(np.array([2, 3, 4])), Displace(np.array([1, 2, 3]), ss=2))]
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
