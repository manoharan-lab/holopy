# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
# W. Perry, Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.

import scatterpy
import holopy as hp
from scatterpy.tests.common import assert_obj_close
import tempfile
from nose.plugins.attrib import attr
import os



def assert_write_read(o):
    tf = tempfile.TemporaryFile()
    hp.save(tf, o)
    tf.flush()
    tf.seek(0)
    assert_obj_close(hp.load(tf), o)

@attr('fast')
def test_yaml_io():
    s = scatterpy.scatterer.Sphere(1.59, .5, (1, 3, 4))
    assert_write_read(s)

    o = hp.Optics(wavelen=.66, index=1.33, pixel_scale=.1)
    assert_write_read(o)
    
    t = scatterpy.theory.Mie(o, 100)
    assert_write_read(o)

    mod = hp.analyze.fit.Model(s, t)

    assert_write_read(mod)

@attr('fast')
def test_hologram_io():
    o = hp.Optics(wavelen=.66, index=1.33, pixel_scale=.1)

    path = os.path.abspath(hp.__file__)
    path = os.path.join(os.path.split(path)[0],'tests', 'exampledata')
    holo = hp.process.normalize(hp.load(os.path.join(path, 'image0001.npy'),
                                        optics=o))
    
    assert_write_read(holo)

@attr('fast')
def test_load_optics():
    optics_yaml = """wavelen: 785e-9
polarization: [1.0, 0]
divergence: 0
pixel_size: [6.8e-6, 6.8e-6]
pixel_scale: [3.3e-7, 3.3e-7]"""
    t = tempfile.TemporaryFile()
    t.write(optics_yaml)
    t.seek(0)

    o = hp.Optics(**hp.load(t))

    assert_obj_close(o, hp.Optics(wavelen=7.85e-07, polarization=[1.0, 0.0], divergence=0, pixel_size=[6.8e-06, 6.8e-06], pixel_scale=[3.3e-07, 3.3e-07]))
