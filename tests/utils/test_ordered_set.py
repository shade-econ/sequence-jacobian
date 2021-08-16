from sequence_jacobian.utilities.ordered_set import OrderedSet

def test_ordered_set():
    # order matters
    assert OrderedSet([1,2,3]) != OrderedSet([3,2,1]) 

    # first insertion determines order
    assert OrderedSet([5,1,6,5]) == OrderedSet([5,1,6])

    # union preserves first and second order
    assert (OrderedSet([6,1,3]) | OrderedSet([3,1,7,9])) == OrderedSet([6,1,3,7,9])

    # intersection preserves first order
    assert (OrderedSet([6,1,3]) & OrderedSet([3,1,7])) == OrderedSet([1,3])

    # difference works
    assert (OrderedSet([6,1,3,2]) - OrderedSet([3,1,7])) == OrderedSet([6,2])

    # symmetric difference: first then second
    assert (OrderedSet([6,1,3,8]) ^ OrderedSet([3,1,7,9])) == OrderedSet([6,8,7,9])

    # in-place versions of these
    s = OrderedSet([6,1,3])
    s2 = s
    s2 |= OrderedSet([3,1,7,9])
    assert s == OrderedSet([6,1,3,7,9])

    s = OrderedSet([6,1,3])
    s2 = s
    s2 &= OrderedSet([3,1,7])
    assert s == OrderedSet([1,3])

    s = OrderedSet([6,1,3,2])
    s2 = s
    s2 -= OrderedSet([3,1,7])
    assert s == OrderedSet([6,2])

    s = OrderedSet([6,1,3,8])
    s2 = s
    s2 ^= OrderedSet([3,1,7,9])
    assert s == OrderedSet([6,8,7,9])

    # comparisons (order not used for these)
    assert OrderedSet([4,3,2,1]) <= OrderedSet([1,2,3,4])
    assert not (OrderedSet([4,3,2,1]) < OrderedSet([1,2,3,4]))
    assert OrderedSet([3,2,1]) < OrderedSet([1,2,3,4])

    # allow second argument (but ONLY second argument) to be any iterable, not just ordered set
    # we use the order from the iterable...
    assert (OrderedSet([6,1,3]) | [3,1,7,9]) == OrderedSet([6,1,3,7,9])
    assert (OrderedSet([6,1,3]) & [3,1,7]) == OrderedSet([1,3])
    assert (OrderedSet([6,1,3,2]) - [3,1,7]) == OrderedSet([6,2])
    assert (OrderedSet([6,1,3,8]) ^ [3,1,7,9]) == OrderedSet([6,8,7,9])


def test_ordered_set_dict_from():
    assert OrderedSet(['a','b','c']).dict_from([1, 2, 3]) == {'a': 1, 'b': 2, 'c': 3}