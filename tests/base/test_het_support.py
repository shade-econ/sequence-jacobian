import numpy as np
from sequence_jacobian.blocks.support.het_support import (Transition,
    PolicyLottery1D, PolicyLottery2D, Markov, CombinedTransition,
    lottery_1d, lottery_2d)
from sequence_jacobian.utilities.multidim import batch_multiply_ith_dimension


def test_combined_markov():
    shape = (5, 6, 7)
    np.random.seed(12345)

    for _ in range(10):
        D = np.random.rand(*shape)
        Pis = [np.random.rand(s, s) for s in shape[:2]]
        markovs = [Markov(Pi, i) for i, Pi in enumerate(Pis)]
        combined = CombinedTransition(markovs)

        Dout = combined.expectation(D)
        Dout_forward = combined.forward(D)

        D_kron = D.reshape((-1, D.shape[2]))
        Pi_kron = np.kron(Pis[0], Pis[1])
        Dout2 = (Pi_kron @ D_kron).reshape(Dout.shape)
        Dout2_forward = (Pi_kron.T @ D_kron).reshape(Dout.shape)

        assert np.allclose(Dout, Dout2)
        assert np.allclose(Dout_forward, Dout2_forward)


def test_many_markov_shock():
    shape = (5, 6, 7)
    np.random.seed(12345)

    for _ in range(10):
        D = np.random.rand(*shape)
        Pis = [np.random.rand(s, s) for s in shape[:2]]
        dPis = [np.random.rand(s, s) for s in shape[:2]]
        
        h = 1E-4
        Dout_up = CombinedTransition([Markov(Pi + h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]).forward(D)
        Dout_dn = CombinedTransition([Markov(Pi - h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]).forward(D)
        Dder = (Dout_up - Dout_dn) / (2*h)

        Dder2 = CombinedTransition([Markov(Pi, i) for i, Pi in enumerate(Pis)]).forward_shockable(D).forward_shock(dPis)

        assert np.allclose(Dder, Dder2)


def test_policy_shock():
    shape = (3, 4, 30)
    grid = np.geomspace(0.5, 10, shape[-1])
    np.random.seed(98765)

    a = (np.full(shape[0], 0.01)[:, np.newaxis, np.newaxis] 
        + np.linspace(0, 1, shape[1])[:, np.newaxis]
        + 0.001*grid**2 + 0.9*grid + 0.5)

    for _ in range(10):
        D = np.random.rand(*shape)

        da = np.random.rand(*shape)
        h = 1E-5
        Dout_up = lottery_1d(a + h*da, grid).forward(D)
        Dout_dn = lottery_1d(a - h*da, grid).forward(D)
        Dder = (Dout_up - Dout_dn) / (2*h)

        Dder2 = lottery_1d(a, grid).forward_shockable(D).forward_shock(da)

        assert np.allclose(Dder, Dder2, atol=1E-4)
    

def test_law_of_motion_shock():
    # shock everything in the law of motion, and see if it works!
    shape = (3, 4, 30)
    grid = np.geomspace(0.5, 10, shape[-1])
    np.random.seed(98765)

    a = (np.full(shape[0], 0.01)[:, np.newaxis, np.newaxis] 
        + np.linspace(0, 1, shape[1])[:, np.newaxis]
        + 0.001*grid**2 + 0.9*grid + 0.5)

    for _ in range(10):
        D = np.random.rand(*shape)
        Pis = [np.random.rand(s, s) for s in shape[:2]]

        da = np.random.rand(*shape)
        dPis = [np.random.rand(s, s) for s in shape[:2]]

        h = 1E-5
        policy_up = lottery_1d(a + h*da, grid)
        policy_dn = lottery_1d(a - h*da, grid)
        markovs_up = [Markov(Pi + h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]
        markovs_dn =[Markov(Pi - h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]
        Dout_up = CombinedTransition([policy_up, *markovs_up]).forward(D)
        Dout_dn = CombinedTransition([policy_dn, *markovs_dn]).forward(D)
        Dder = (Dout_up - Dout_dn) / (2*h)

        markovs = [Markov(Pi, i) for i, Pi, in enumerate(Pis)]
        Dder2 = CombinedTransition([lottery_1d(a, grid), *markovs]).forward_shockable(D).forward_shock([da, *dPis])

        assert np.allclose(Dder, Dder2, atol=1E-4)


def test_2d_policy_shock():
    shape = (3, 4, 20, 30)
    a_grid = np.geomspace(0.5, 10, shape[-2])
    b_grid = np.geomspace(0.2, 8, shape[-1])
    np.random.seed(98765)

    a = (0.001*a_grid**2 + 0.9*a_grid + 0.5)[:, np.newaxis]
    b = (-0.001*b_grid**2 + 0.9*b_grid + 0.5)

    a = np.broadcast_to(a, shape)
    b = np.broadcast_to(b, shape)

    for _ in range(10):
        D = np.random.rand(*shape)
        Pis = [np.random.rand(s, s) for s in shape[:2]]

        da = np.random.rand(*shape)
        db = np.random.rand(*shape)
        dPis = [np.random.rand(s, s) for s in shape[:2]]

        h = 1E-5

        policy_up = lottery_2d(a + h*da, b + h*db, a_grid, b_grid)
        policy_dn = lottery_2d(a - h*da, b - h*db, a_grid, b_grid)
        markovs_up = [Markov(Pi + h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]
        markovs_dn = [Markov(Pi - h*dPi, i) for i, (Pi, dPi) in enumerate(zip(Pis, dPis))]
        Dout_up = CombinedTransition([policy_up, *markovs_up]).forward(D)
        Dout_dn = CombinedTransition([policy_dn, *markovs_dn]).forward(D)
        Dder = (Dout_up - Dout_dn) / (2*h)

        policy = lottery_2d(a, b, a_grid, b_grid)

        markovs = [Markov(Pi, i) for i, Pi, in enumerate(Pis)]
        Dder2 = CombinedTransition([policy, *markovs]).forward_shockable(D).forward_shock([[da, db], *dPis])

        assert np.allclose(Dder, Dder2, atol=1E-4)


def test_forward_expectations_symmetry():
    # given a random law of motion, should be identical to iterate forward on distribution,
    # then aggregate, or take expectations backward on outcome, then aggregate
    shape = (3, 4, 30)
    grid = np.geomspace(0.5, 10, shape[-1])
    np.random.seed(1423)

    a = (np.full(shape[0], 0.01)[:, np.newaxis, np.newaxis] 
        + np.linspace(0, 1, shape[1])[:, np.newaxis]
        + 0.001*grid**2 + 0.9*grid + 0.5)

    for _ in range(10):
        D = np.random.rand(*shape)
        X = np.random.rand(*shape)
        Pis = [np.random.rand(s, s) for s in shape[:2]]

        markovs = [Markov(Pi, i) for i, Pi, in enumerate(Pis)]
        lom = CombinedTransition([lottery_1d(a, grid), *markovs])

        Dforward = D
        for _ in range(30):
            Dforward = lom.forward(Dforward)
        outcome = np.vdot(Dforward, X)

        Xbackward = X
        for _ in range(30):
            Xbackward = lom.expectation(Xbackward)
        outcome2 = np.vdot(D, Xbackward)

        assert np.isclose(outcome, outcome2)


def test_einsum():
    D = np.random.rand(2, 5, 10)
    P = np.random.rand(3, 2, 5, 10)
    Dnew = np.einsum('xij,dxij->dij', D, P)
    assert Dnew[0, 1, 1] == np.sum(P[0, :, 1, 1] * D[:, 1, 1])

    # can I generalize this? reshape and then einsum
    Dnew2 = batch_multiply_ith_dimension(P, 0, D)

    assert (Dnew == Dnew2).all()