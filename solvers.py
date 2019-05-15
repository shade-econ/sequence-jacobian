import numpy as np


"""Part 1: supporting functions for solvers"""


def obtain_J(f, x, y, h=1E-5):
    """finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((nx, ny))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx) - y) / h
    return J


def broyden_update(J, dx, dy):
    """Returns Broyden update to approximate Jacobian J given that last
    change in inputs to function was dx and led to output change of dy"""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)


def printit(it, x, y, **kwargs):
    """Convenience printing function for noisy iterations"""
    print(f'On iteration {it}')
    print(('x = %.3f' + ',%.3f' * (len(x) - 1)) % tuple(x))
    print(('y = %.3f' + ',%.3f' * (len(y) - 1)) % tuple(y))
    for kw, val in kwargs.items():
        print(f'{kw} = {val:.3f}')
    print('\n')


"""Part 2: define simple Newton, Broyden, and Levenberg-Marquardt solvers
In each case, catch ValueError in calls to x and assume step too large"""


def newton_solver(f, x, y=None, tol=1E-9, maxcount=100, backtrack_c=0.5, noisy=True):
    """Simple line search solver in Newton direction, backtracks if input
    invalid or if improvement is not at least half the predicted improvement"""
    if y is None:
        y = f(x)

    for count in range(maxcount):
        if noisy:
            printit(count, x, y)

        if np.max(np.abs(y)) < tol:
            return x, y

        J = obtain_J(f, x, y)
        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                predicted_improvement = -np.sum((J @ dx) * y) * ((1 - 1 / 2 ** bcount) + 1) / 2
                actual_improvement = (np.sum(y ** 2) - np.sum(ynew ** 2)) / 2
                if actual_improvement < predicted_improvement / 2:
                    if noisy:
                        print('backtracking\n')
                    dx *= backtrack_c
                else:
                    y = ynew
                    x += dx
                    break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')


def broyden_solver(f, x, y=None, tol=1E-9, maxcount=100, backtrack_c=0.5, noisy=True):
    """Simple line search solver in approximate Newton direction, obtaining approximate J from Broyden updating."""
    if y is None:
        y = f(x)

    # initialize J with Newton!
    J = obtain_J(f, x, y)
    for count in range(maxcount):
        if noisy:
            printit(count, x, y)

        if np.max(np.abs(y)) < tol:
            return x, y

        dx = np.linalg.solve(J, -y)

        # backtrack at most 29 times
        for bcount in range(30):
            # note: can't test for improvement with Broyden because maybe
            # the function doesn't improve locally in this direction, since
            # J isn't the exact Jacobian
            try:
                ynew = f(x + dx)
            except ValueError:
                if noisy:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                J = broyden_update(J, dx, ynew - y)
                y = ynew
                x += dx
                break
        else:
            raise ValueError('Too many backtracks, maybe bad initial guess?')
    else:
        raise ValueError(f'No convergence after {maxcount} iterations')
