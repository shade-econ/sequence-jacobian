"""Test JacobianDictBlock functionality"""

import numpy as np

from sequence_jacobian import combine
from sequence_jacobian.models import rbc
from sequence_jacobian.blocks.auxiliary_blocks.jacobiandict_block import JacobianDictBlock


def test_jacobian_dict_block_impulses(rbc_dag):
    rbc_model, exogenous, unknowns, _, ss = rbc_dag

    T = 10
    J_pe = rbc_model.jacobian(ss, exogenous=unknowns + exogenous, T=10)
    J_block = JacobianDictBlock(J_pe)

    J_block_Z = J_block.jacobian(["Z"])
    for o in J_block_Z.outputs:
        assert np.all(J_block[o]["Z"] == J_block_Z[o]["Z"])

    dZ = 0.8 ** np.arange(T)

    dO1 = J_block @ {"Z": dZ}
    dO2 = J_block_Z @ {"Z": dZ}

    for k in J_block:
        assert np.all(dO1[k] == dO2[k])


def test_jacobian_dict_block_combine(rbc_dag):
    _, exogenous, _, _, ss = rbc_dag

    J_firm = rbc.firm.jacobian(ss, exogenous=exogenous)
    blocks_w_jdict = [rbc.household, J_firm, rbc.mkt_clearing]
    cblock_w_jdict = combine(blocks_w_jdict)

    # Using `combine` converts JacobianDicts to JacobianDictBlocks
    assert isinstance(cblock_w_jdict._blocks_unsorted[1], JacobianDictBlock)
