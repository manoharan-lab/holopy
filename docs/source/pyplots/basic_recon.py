import numpy as np
import holopy as hp
from holopy.propagation import propagate
from holopy.core.tests.common import get_example_data
from holopy.core import load

holo = get_example_data('image0001.yaml')
rec_vol = propagate(holo, np.linspace(4e-6, 10e-6, 7))
hp.show(rec_vol)
