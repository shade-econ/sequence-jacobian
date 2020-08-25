"""Simple nonlinear solvers"""

import numpy as np


def newton_solver(f, x0, y0=None, tol=1E-9, maxcount=100, backtrack_c=0.5, verbose=True):
    """Simple line search solver for root x satisfying f(x)=0 using Newton direction.

    Backtracks if input invalid or improvement is not at least half the predicted improvement.

    Parameters
    ----------
    f               : function, to solve for f(x)=0, input and output are arrays of same length
    x0              : array (n), initial guess for x
    y0              : [optional] array (n), y0=f(x0), if already known
    tol             : [optional] scalar, solver exits successfully when |f(x)| < tol
    maxcount        : [optional] int, maximum number of Newton steps
    backtrack_c     : [optional] scalar, fraction to backtrack if step unsuccessful, i.e.
                        if we tried step from x to x+dx, now try x+backtrack_c*dx

    Returns
    ----------
    x       : array (n), (approximate) root of f(x)=0
    y       : array (n), y=f(x), satisfies |y| < tol
    """

    x, y = x0, y0
    if y is None:
        y = f(x)

    for count in range(maxcount):
        if verbose:
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
                if verbose:
                    print('backtracking\n')
                dx *= backtrack_c
            else:
                predicted_improvement = -np.sum((J @ dx) * y) * ((1 - 1 / 2 ** bcount) + 1) / 2
                actual_improvement = (np.sum(y ** 2) - np.sum(ynew ** 2)) / 2
                if actual_improvement < predicted_improvement / 2:
                    if verbose:
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


def broyden_solver(f, x0, y0=None, tol=1E-9, maxcount=100, backtrack_c=0.5, verbose=True):
    """Similar to newton_solver, but solves f(x)=0 using approximate rather than exact Newton direction,
    obtaining approximate Jacobian J=f'(x) from Broyden updating (starting from exact Newton at f'(x0)).

    Backtracks only if error raised by evaluation of f, since improvement criterion no longer guaranteed
    to work for any amount of backtracking if Jacobian not exact.
    """

    x, y = x0, y0
    if y is None:
        y = f(x)

    # initialize J with Newton!
    J = obtain_J(f, x, y)
    for count in range(maxcount):
        if verbose:
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
                if verbose:
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


def obtain_J(f, x, y, h=1E-5):
    """Finds Jacobian f'(x) around y=f(x)"""
    nx = x.shape[0]
    ny = y.shape[0]
    J = np.empty((nx, ny))

    for i in range(nx):
        dx = h * (np.arange(nx) == i)
        J[:, i] = (f(x + dx) - y) / h
    return J


def broyden_update(J, dx, dy):
    """Returns Broyden update to approximate Jacobian J, given that last change in inputs to function
    was dx and led to output change of dy."""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)


def printit(it, x, y, **kwargs):
    """Convenience printing function for verbose iterations"""
    print(f'On iteration {it}')
    print(('x = %.3f' + ',%.3f' * (len(x) - 1)) % tuple(x))
    print(('y = %.3f' + ',%.3f' * (len(y) - 1)) % tuple(y))
    for kw, val in kwargs.items():
        print(f'{kw} = {val:.3f}')
    print('\n')

