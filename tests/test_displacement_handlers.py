"""Test displacement handler classes: Ignore, IgnoreVector, Displace, Perturb, Reporter"""
import numpy as np

from sequence_jacobian.blocks.simple_block import (
    Ignore, IgnoreVector, Displace, Reporter, Perturb, apply_op
)


def test_ignore():
    # Test unary operations
    t1 = Ignore(1)
    for op in ["__neg__", "__pos__"]:
        # TODO: Check that the primitive arithmetic is correct also
        assert type(apply_op(op, t1)) == Ignore

    # Test binary operations
    arg_pairs = [(Ignore(1), 1), (1, Ignore(1)), (Ignore(1), Ignore(2))]
    for pair in arg_pairs:
        t1, t2 = pair
        for op in ["__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                   "__truediv__", "__rtruediv__", "__pow__", "__rpow__"]:
            # TODO: Check that the primitive arithmetic is correct also
            assert type(apply_op(op, t1, t2)) == Ignore


def test_ignore_vector():
    # Test unary operations
    t1 = IgnoreVector(np.array([1, 2, 3]))
    for op in ["__neg__", "__pos__"]:
        # TODO: Check that the primitive arithmetic is correct also
        assert type(apply_op(op, t1)) == IgnoreVector

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
            # TODO: Check that the primitive arithmetic is correct also
            assert type(apply_op(op, t1, t2)) == IgnoreVector


def test_displace():
    # Test unary operations
    t1 = Displace(np.array([1, 2, 3]), ss=2)
    for op in ["__neg__", "__pos__"]:
        # TODO: Check that the primitive arithmetic is correct also
        assert type(apply_op(op, t1)) == Displace

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
            # TODO: Check that the primitive arithmetic is correct also
            assert type(apply_op(op, t1, t2)) == Displace
