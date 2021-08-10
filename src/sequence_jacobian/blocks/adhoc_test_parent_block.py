from parent import Parent
from sequence_jacobian.blocks.support.bijection import Bijection

class DummyBlock:
    def __init__(self, name):
        self.name = name

def test():
    grand1 = DummyBlock('grandkid1')
    grand2 = DummyBlock('grandkid2')
    grand2.unknowns = {'thing1': 3, 'thing2': 5}
    grand2.M = Bijection({'thing1': 'othername1'})

    kid2 = Parent([grand1, grand2], name='kid2')
    kid2.M = Bijection({'thing2': 'othername2', 'othername1': 'othername3'})

    b = Parent([DummyBlock('kid1'),
                kid2,
                DummyBlock('kid3')], name='me')

    b1 = b['grandkid1']
    assert isinstance(b1, DummyBlock) and b1.name == 'grandkid1'
    assert b.path('grandkid2') == ['me', 'kid2', 'grandkid2']

    assert b.get_attribute('grandkid2', 'unknowns') == {'othername3': 3, 'othername2': 5}

test()
