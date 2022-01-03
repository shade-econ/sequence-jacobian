from sequence_jacobian.utilities.graph import DAG
from sequence_jacobian.utilities.ordered_set import OrderedSet
from sequence_jacobian import simple, combine
import pytest

class Block:
    def __init__(self, inputs, outputs):
        self.inputs = OrderedSet(inputs)
        self.outputs = OrderedSet(outputs)


test_dag = DAG([Block(inputs=['a', 'b', 'z'], outputs=['c', 'd']),
                Block(inputs=['a', 'e'], outputs=['b']),
                Block(inputs = ['d'], outputs=['f'])])


def test_dag_constructor():
    # the blocks should be ordered 1, 0, 2
    assert list(test_dag.blocks[0].inputs) == ['a', 'e']
    assert list(test_dag.blocks[1].inputs) == ['a', 'b', 'z']
    assert list(test_dag.blocks[2].inputs) == ['d']

    assert set(test_dag.inmap['a']) == {0, 1}
    assert set(test_dag.inmap['b']) == {1}

    assert test_dag.outmap['c'] == 1
    assert test_dag.outmap['f'] == 2
    assert test_dag.outmap['d'] == 1

    assert set(test_dag.adj[0]) == {1}
    assert set(test_dag.adj[1]) == {2}
    assert set(test_dag.revadj[2]) == {1}
    assert set(test_dag.revadj[1]) == {0}


def test_visited():
    test_dag.visit_from_outputs(['f']) == OrderedSet([0, 1, 2])
    test_dag.visit_from_outputs(['b']) == OrderedSet([0])
    test_dag.visit_from_outputs(['d']) == OrderedSet([0, 1])

    test_dag.visit_from_inputs(['e']) == OrderedSet([0, 1, 2])
    test_dag.visit_from_inputs(['z']) == OrderedSet([1, 2])


def test_find_cycle():
    @simple
    def f1(x):
        y = x
        return y

    @simple
    def f2(y, theta):
        z = y
        return z

    @simple
    def f3(x, z):
        w = x * z
        return w

    @simple
    def f4(z):
        x = z
        return x

    with pytest.raises(Exception) as exception:
        a = combine([f1, f2, f3, f4])
    assert "Topological sort failed: cyclic dependency f1 -> f4 -> f2 -> f1" in str(exception.value)
