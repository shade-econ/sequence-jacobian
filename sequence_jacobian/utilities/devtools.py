"""Useful functions for debugging SSJ code"""

import numpy as np


# Tools for upgrading from older SSJ code conventions
def compare_steady_states(ss_ref, ss_comp, name_map=None, verbose=True):
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
    print(f"The keys present in only one of the two steady state dicts are"
          f" {ss_ref_incl_mapped.symmetric_difference(ss_comp_incl_mapped)}")

