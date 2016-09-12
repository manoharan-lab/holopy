from holopy.core.tests.common import assert_pickle_roundtrip

from holopy.scattering.theory import Mie

def assert_method_roundtrip(o):
    #assert_method_equal(o, pickle.loads(pickle.dumps(o)), 'pickled method')
    assert_method_equal(o, cPickle.loads(cPickle.dumps(o)), 'pickled method')

def test_pickle_mie_object():
    m = Mie()
    assert_pickle_roundtrip(m)
