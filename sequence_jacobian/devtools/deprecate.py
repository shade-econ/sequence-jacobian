"""Tools for deprecating older SSJ code conventions in favor of newer conventions"""

import warnings

# The code in this module is meant to assist with users who have used past versions of sequence-jacobian, by temporarily
# providing support for old conventions via deprecated methods, providing time to allow for a seamless upgrade
# to newer versions sequence-jacobian.

# TODO: There are also the .ss, .td, and .jac methods that are deprecated within the various Block class definitions
#   themselves.


def rename_output_list_to_outputs(outputs=None, output_list=None):
    if output_list is not None:
        warnings.warn("The output_list kwarg has been deprecated and replaced with the outputs kwarg.",
                      DeprecationWarning)
        if outputs is None:
            return output_list
        else:
            return list(set(outputs) | set(output_list))
    else:
        return outputs
