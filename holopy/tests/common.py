import numpy as np
from numpy.testing import assert_allclose, assert_equal

import scatterpy

    
def assert_parameters_allclose(actual, desired, rtol=1e-3):
    if isinstance(actual, scatterpy.scatterer.Scatterer):
        actual = actual.parameters
    if isinstance(actual, dict):
        actual = np.array([p[1] for p in actual.iteritems()])
    if isinstance(desired, scatterpy.scatterer.Scatterer):
        desired = desired.parameters
    if isinstance(desired, dict):
        desired = np.array([p[1] for p in desired.iteritems()])
    assert_allclose(actual, desired, rtol=rtol)

def assert_obj_close(actual, desired, rtol=1e-3):
    if isinstance(actual, (scatterpy.scatterer.Scatterer, dict)):
        assert_parameters_allclose(actual, desired, rtol)
    elif hasattr(actual, '__dict__'):
        for key, val in actual.__dict__.iteritems():
            assert_obj_close(getattr(actual, key), getattr(desired, key), rtol)
    elif actual is not None and not np.isscalar(actual):
        for i, item in enumerate(actual):
            assert_obj_close(actual[i], desired[i], rtol)
    else:
        assert_equal(actual, desired)
        
