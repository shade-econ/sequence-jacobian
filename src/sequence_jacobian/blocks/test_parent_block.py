from parent_block import ParentBlock

class DummyBlock:
    def __init__(self, name):
        self.name = name

def test():
    b = ParentBlock([DummyBlock('kid1'),
                    ParentBlock([DummyBlock('grandkid1'),
                                DummyBlock('grandkid2')], name='kid2'),
                    DummyBlock('kid3')], name='me')

    b1 = b['grandkid1']
    assert isinstance(b1, DummyBlock) and b1.name == 'grandkid1'

    print(b.path('grandkid2'))
    assert b.path('grandkid2') == ['me', 'kid2', 'grandkid2']
