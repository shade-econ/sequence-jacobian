"""Tools for deprecating older SSJ code conventions in favor of newer conventions"""

import warnings

# The code in this module is meant to assist with users who have used past versions of sequence-jacobian, by temporarily
# providing support for old conventions via deprecated methods, providing time to allow for a seamless upgrade
# to newer versions sequence-jacobian.

# TODO: There are also the .ss, .td, and .jac methods that are deprecated within the various Block class definitions
#   themselves.


# For impulse_nonlinear, td_solve, and td_map
def deprecated_shock_input_convention(shocked_paths, kwargs):
    if kwargs:
        warnings.warn("Passing shock paths through kwargs is deprecated. Please explicitly specify"
                      " the dict of shocks in the keyword argument `shocked_paths`.", DeprecationWarning)
        if shocked_paths is None:
            return kwargs
        else:
            # If for whatever reason kwargs and shocked_paths is non-empty?
            shocked_paths.update(kwargs)
    else:
        if shocked_paths is None:
            return {}
        else:
            return shocked_paths
