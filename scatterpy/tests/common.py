import holopy
import os
import numpy
import re
from scatterpy.theory.scatteringtheory import ElectricField

from numpy.testing import assert_array_almost_equal, assert_almost_equal


wavelen = 658e-9
ypolarization = [0., 1.0] # y-polarized
xpolarization = [1.0, 0.] # x-polarized
divergence = 0
pixel_scale = [.1151e-6, .1151e-6]
index = 1.33

yoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                               pixel_scale=pixel_scale,
                               polarization=ypolarization,
                               divergence=divergence)
    
xoptics = holopy.optics.Optics(wavelen=wavelen, index=index,
                               pixel_scale=pixel_scale,
                               polarization=xpolarization,
                               divergence=divergence)

optics=yoptics

class DataNotPresent(Exception):
    def __init__(self, name):
        self.name = name
    def message(self, context):
        return "External data not present, skipping correctness testing for \
{0}".format(context)
    def __str__(self):
        return "External data file {0} not found".format(self.name)

    
def get_data(name):
    name = name + '.npy'
    try:
        return numpy.load(os.path.join('ext_data',name))
    except IOError:
        raise DataNotPresent(name)


def compare_to_data(calc, name):
    try:
        gold = get_data(name)
    except DataNotPresent as e:
        print e
#        return

    if isinstance(calc, ElectricField):
        assert_array_almost_equal(calc._array(), gold)
    else:
        assert_array_almost_equal(calc, gold)

