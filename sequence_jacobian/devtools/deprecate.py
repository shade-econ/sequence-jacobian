"""Tools for deprecating older SSJ code conventions in favor of newer conventions"""

import warnings

# The code in this module is meant to assist with users who have used past versions of sequence-jacobian, by temporarily
# providing support for old conventions via deprecated methods, providing time to allow for a seamless upgrade
# to newer versions sequence-jacobian.

# TODO: There are also the .ss, .td, and .jac methods that are deprecated within the various Block class definitions
#   themselves.
