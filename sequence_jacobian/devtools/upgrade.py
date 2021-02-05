"""Tools for upgrading from older SSJ code conventions"""

# The code in this module is meant to assist with users who have used past versions of sequence-jacobian, and who
# may want additional support/tools for ensuring that their attempts to upgrade to use newer versions of
# sequence-jacobian has been successfully.

import numpy as np


def compare_steady_states(ss_ref, ss_comp, name_map=None, verbose=True):
    """
    This code is meant to provide a quick comparison of `ss_ref` the reference steady state dict from old code, and
    `ss_comp` the steady state computed from the newer code.
    """
    if name_map is None:
        name_map = {}

    # Compare the steady state values present in both ss_ref and ss_comp
    for key_ref in ss_ref.keys():
        if key_ref in ss_comp.keys():
            key_comp = key_ref
        elif key_ref in name_map:
            key_comp = name_map[key_ref]
        else:
            continue
        if verbose:
            if np.isscalar(ss_ref[key_ref]):
                print(f"{key_ref} resid: {abs(ss_ref[key_ref] - ss_comp[key_comp])}")
            else:
                print(f"{key_ref} resid: {np.linalg.norm(ss_ref[key_ref] - ss_comp[key_comp], np.inf)}")
        else:
            assert np.isclose(ss_ref[key_ref], ss_comp[key_comp])

    # Show the steady state values present in only one of ss_ref or ss_comp
    ss_ref_incl_mapped = set(ss_ref.keys()) - set(name_map.keys())
    ss_comp_incl_mapped = set(ss_comp.keys()) - set(name_map.values())
    diff_keys = ss_ref_incl_mapped.symmetric_difference(ss_comp_incl_mapped)
    if diff_keys:
        print(f"The keys present in only one of the two steady state dicts are {diff_keys}")
