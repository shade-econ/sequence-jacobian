import numpy as np

def test_jacobian_h(krusell_smith_dag):
    dag, *_, ss = krusell_smith_dag
    hh = dag['household']

    lowacc = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=0.05)
    midacc = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=1E-3)
    usual = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, h=1E-4)
    nooption = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10)

    assert np.array_equal(usual['C','r'], nooption['C','r'])
    assert np.linalg.norm(usual['C','r'] - midacc['C','r']) < np.linalg.norm(usual['C','r'] - lowacc['C','r'])

    midacc_alt = hh.jacobian(ss, inputs=['r'], outputs=['C'], T=10, options={'household': {'h': 1E-3}})
    assert np.array_equal(midacc['C', 'r'], midacc_alt['C', 'r'])

    lowacc = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 0.05}})
    midacc = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 1E-3}})
    usual = dag.jacobian(ss, inputs=['K'], outputs=['C'], T=10, options={'household': {'h': 1E-4}})

    assert np.linalg.norm(usual['C','K'] - midacc['C','K']) < np.linalg.norm(usual['C','K'] - lowacc['C','K'])
