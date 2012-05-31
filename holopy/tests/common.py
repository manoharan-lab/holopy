import numpy as np
from numpy.testing import assert_allclose

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
