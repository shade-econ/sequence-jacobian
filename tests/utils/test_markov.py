from sequence_jacobian.utilities.discretize import big_outer
import numpy as np

def test_2d():
    a = np.random.rand(10)
    b = np.random.rand(12)
    assert np.allclose(np.outer(a,b), big_outer([a,b]))

def test_3d():
    a = np.array([1., 2])
    b = np.array([1., 7])
    small = np.outer(a, b)

    c = np.array([2., 4])
    product = np.empty((2,2,2))
    product[..., 0] = 2*small
    product[..., 1] = 4*small

    assert np.array_equal(product, big_outer([a,b,c]))
