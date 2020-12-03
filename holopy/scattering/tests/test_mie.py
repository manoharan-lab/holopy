# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
'''
Test fortran-based Mie calculations and python interface.

.. moduleauthor:: Vinothan N. Manoharan <vnm@seas.harvard.edu>
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
'''

import os
import yaml

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_raises, assert_equal, assert_allclose)
from nose.plugins.attrib import attr

from holopy.scattering.scatterer import (
    Sphere, Spheres, Ellipsoid, LayeredSphere)
from holopy.scattering.theory import Mie
from holopy.scattering.imageformation import ImageFormation
from holopy.scattering.errors import TheoryNotCompatibleError, InvalidScatterer
from holopy.core.metadata import (
    detector_grid, detector_points, to_vector, update_metadata)
from holopy.core.process import subimage
from holopy.scattering.tests.common import (
    sphere, xschema, scaling_alpha, yschema, xpolarization, ypolarization,
    x, y, z, n, radius, wavelen, index)
from holopy.core.tests.common import assert_obj_close, verify
from holopy.scattering.interface import (
    calc_field, calc_holo, calc_intensity, calc_scat_matrix,
    calc_cross_sections)



@attr('medium')
def test_single_sphere():
    # single sphere hologram (only tests that functions return)
    thry = Mie(False)
    holo = calc_holo(xschema, sphere, index, wavelen, xpolarization, theory=thry, scaling=scaling_alpha)
    field = calc_field(xschema, sphere, index, wavelen, xpolarization, theory=thry)

    intensity = calc_intensity(xschema, sphere, medium_index=index, illum_wavelen=wavelen, illum_polarization=xpolarization, theory=thry)

    verify(holo, 'single_holo')
    verify(field, 'single_field')

    # now test some invalid scatterers and confirm that it rejects calculating
    # for them

    # large radius (calculation not attempted because it would take forever
    assert_raises(InvalidScatterer, calc_holo, xschema, Sphere(r=1, n = 1.59, center = (5,5,5)), medium_index=index, illum_wavelen=wavelen)

@attr('medium')
def test_farfield_holo():
    # Tests that a far field calculation gives a hologram that is
    # different from a full radial dependence calculation, but not too different
    holo_full = calc_holo(xschema, sphere, index, wavelen, xpolarization, scaling=scaling_alpha)
    holo_far = calc_holo(xschema, sphere, index, wavelen, xpolarization, scaling=scaling_alpha, theory=Mie(False, False))


    # the two arrays should not be equal
    try:
        assert_array_almost_equal(holo_full, holo_far)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for near and far field "
                             "are too similar.")


    # but their max and min values should be close
    assert_obj_close(holo_full.max(), holo_far.max(), .1,
                        context="Near and Far field holograms too different")
    assert_obj_close(holo_full.min(), holo_far.min(), .1,
                        context="Near and Far field holograms too different")


@attr('medium')
def test_subimaged():
    # make a dummy image so that we can pretend we are working with
    # data we want to subimage
    im = xschema
    h = calc_holo(im, sphere, index, wavelen, xpolarization)
    sub = (60, 70), 30
    hs = calc_holo(subimage(im, *sub), sphere, index, wavelen, xpolarization)

    assert_obj_close(subimage(h, *sub), hs)


@attr('medium')
def test_Mie_multiple():
    s1 = Sphere(n = 1.59, r = 5e-7, center = (1e-6, -1e-6, 10e-6))
    s2 = Sphere(n = 1.59, r = 1e-6, center=[8e-6,5e-6,5e-6])
    s3 = Sphere(n = 1.59+0.0001j, r = 5e-7, center=[5e-6,10e-6,3e-6])
    sc = Spheres(scatterers=[s1, s2, s3])
    thry = Mie(False)

    schema = yschema
    fields = calc_field(schema, sc, index, wavelen, ypolarization, thry)

    verify(fields, 'mie_multiple_fields')
    calc_intensity(schema, sc, index, wavelen, ypolarization, thry)

    holo = calc_holo(schema, sc, index, wavelen, theory=thry)
    verify(holo, 'mie_multiple_holo')
    # should throw exception when fed a ellipsoid
    el = Ellipsoid(n = 1.59, r = (1e-6, 2e-6, 3e-6), center=[8e-6,5e-6,5e-6])
    with assert_raises(TheoryNotCompatibleError) as cm:
        calc_field(schema, el, index, wavelen, theory=Mie)
    assert_equal(str(cm.exception), "Mie scattering theory can't handle "
                 "scatterers of type Ellipsoid")
    assert_raises(TheoryNotCompatibleError, calc_field, schema, el, index, wavelen, xpolarization, Mie)
    assert_raises(TheoryNotCompatibleError, calc_intensity,
                  schema, el, index, wavelen, xpolarization, Mie)
    assert_raises(TheoryNotCompatibleError, calc_holo, schema, el, index, wavelen, xpolarization, Mie)

@attr('medium')
def test_mie_polarization():

    # test holograms for orthogonal polarizations; make sure they're
    # not the same, nor too different from one another.
    thry = Mie(False)
    xholo = calc_holo(xschema, sphere, index, wavelen, illum_polarization=xpolarization, scaling=scaling_alpha)
    yholo = calc_holo(yschema, sphere, index, wavelen, illum_polarization=ypolarization, scaling=scaling_alpha)

    # the two arrays should not be equal
    try:
        assert_array_almost_equal(xholo, yholo)
    except AssertionError:
        pass
    else:
        raise AssertionError("Holograms computed for both x- and y-polarized "
                             "light are too similar.")

    # but their max and min values should be close
    assert_obj_close(xholo.max(), yholo.max())
    assert_obj_close(xholo.min(), yholo.min())
    return xholo, yholo


@attr('medium')
def test_linearity():
    # look at superposition of scattering from two point particles;
    # make sure that this is sum of holograms from individual point
    # particles (scattered intensity should be negligible for this
    # case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0
    r = 1e-2*wavelen    # something much smaller than wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = Spheres(scatterers = [sphere1, sphere2])

    holo_1 = calc_holo(xschema, sphere1, index, wavelen, xpolarization, scaling=scaling_alpha)
    holo_2 = calc_holo(xschema, sphere2, index, wavelen, xpolarization, scaling=scaling_alpha)
    holo_super = calc_holo(xschema, sc, index, wavelen, xpolarization, theory=Mie, scaling=scaling_alpha)

    # make sure we're not just looking at uniform arrays (could
    # happen if the size is set too small)
    try:
        assert_array_almost_equal(holo_1, holo_2, decimal=12)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Hologram computed for point particle" +
                             " looks suspiciously close to having" +
                             " no fringes")

    # Test linearity by subtracting off individual holograms.
    # This should recover the other hologram
    assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    assert_array_almost_equal(holo_super - holo_2 + 1, holo_1)

    # uncomment to debug
    #return holo_1, holo_2, holo_super

@attr('medium')
def test_nonlinearity():
    # look at superposition of scattering from two large particles;
    # make sure that this is *not equal* to sum of holograms from
    # individual scatterers (scattered intensity should be
    # non-negligible for this case)

    x2 = x*2
    y2 = y*2
    z2 = z*2
    scaling_alpha = 1.0
    r = wavelen    # order of wavelength

    sphere1 = Sphere(n=n, r=r, center = (x, y, z))
    sphere2 = Sphere(n=n, r=r, center = (x2, y2, z2))

    sc = Spheres(scatterers = [sphere1, sphere2])

    holo_1 = calc_holo(xschema, sphere1, index, wavelen, illum_polarization=xpolarization, scaling=scaling_alpha)
    holo_2 = calc_holo(xschema, sphere2, index, wavelen, illum_polarization=xpolarization, scaling=scaling_alpha)
    holo_super = calc_holo(xschema, sc, index, wavelen, xpolarization, scaling=scaling_alpha, theory=Mie)

    # test nonlinearity by subtracting off individual holograms
    try:
        assert_array_almost_equal(holo_super - holo_1 + 1, holo_2)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms computed for "
                             "wavelength-scale scatterers should "
                             "not superpose linearly")

    # uncomment to debug
    #return holo_1, holo_2, holo_super


@attr('fast')
def test_radiometric():
    cross_sects = calc_cross_sections(sphere, index, wavelen, illum_polarization=xpolarization)
    # turn cross sections into efficiencies
    cross_sects[0:3] = cross_sects[0:3] / (np.pi * radius**2)

    # create a dict from the results
    result = {}
    result_keys = ['qscat', 'qabs', 'qext', 'costheta']
    for key, val in zip(result_keys, cross_sects):
        result[key] = val

    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold',
                             'gold_mie_radiometric')
    with open(gold_name + '.yaml') as gold_file:
        gold = yaml.safe_load(gold_file)
    for key, val in gold.items():
        assert_almost_equal(gold[key], val, decimal = 5)

@attr('fast')
def test_farfield_matr():
    schema = detector_points(theta = np.linspace(0, np.pi/2), phi = np.linspace(0, 1))
    sphere = Sphere(r = .5, n = 1.59+0.1j)

    matr = calc_scat_matrix(schema, sphere, index, .66)
    verify(matr, 'farfield_matricies', rtol = 1e-6)

@attr('medium')
def test_radialEscat():
    thry_1 = Mie()
    thry_2 = Mie(False)

    sphere = Sphere(r = 1e-6, n = 1.4 + 0.01j, center = [10e-6, 10e-6,
                                                         1.2e-6])
    h1 = calc_holo(xschema, sphere, index, wavelen, illum_polarization=xpolarization)
    h2 = calc_holo(xschema, sphere, index, wavelen, illum_polarization=xpolarization, theory=thry_2)

    try:
        assert_array_almost_equal(h1, h2, decimal=12)
    except AssertionError:
        pass    # no way to do "assert array not equal" in numpy.testing
    else:
        raise AssertionError("Holograms w/ and w/o full radial fields" +
                             " are exactly equal")


@attr("medium")
def test_layered():
    l = LayeredSphere(n = (1, 2), t = (1, 1), center = (2, 2, 2))
    s = Sphere(n = (1,2), r = (1, 2), center = (2, 2, 2))
    sch = detector_grid((10, 10), .2)
    wavelen = .66
    hl = calc_holo(sch, l, index, wavelen, illum_polarization=xpolarization)
    hs = calc_holo(sch, s, index, wavelen, illum_polarization=xpolarization)
    assert_obj_close(hl, hs, rtol=0)

@attr("fast")
def test_large_sphere():
    large_sphere_gold=[[[0.96371831],[1.04338683]],[[1.04240049],[0.99605225]]]
    s=Sphere(n=1.5, r=5, center=(10,10,10))
    sch=detector_grid(10,.2)
    hl=calc_holo(sch, s, illum_wavelen=.66, medium_index=1, illum_polarization=(1,0))
    assert_obj_close(np.array(hl[0:2,0:2]),large_sphere_gold)

@attr('fast')
def test_calc_scat_coeffs():
    sp = Sphere(r=.5, n=1.6, center=(10, 10, 5))
    wavevec = 2* np.pi / (.66/1.33)
    scat_coeffs = Mie()._scat_coeffs(sp, wavevec, 1.33)
    assert_allclose(scat_coeffs, [[(0.893537889855249-0.308428158974303j),
  (0.8518237942576172-0.35527456677079167j),
  (0.8514945265371544-0.3556003343845751j),
  (0.6716114989265135-0.4696269726455193j),
  (0.4463235347943387-0.49711048780228473j),
  (0.10807327505985087-0.31047293324489444j),
  (0.007047039370772889-0.08365033536621158j),
  (0.00023637042768157927-0.01537252603518683j),
  (4.947915829486452e-06-0.002224385611267839j),
  (6.65551498173517e-08-0.00025798283932805245j),
  (5.916757117384697e-10-2.4324385118403086e-05j),
  (3.5939293107529156e-12-1.895766154023222e-06j),
  (1.5398821099306434e-14-1.2409198644274415e-07j),
  (4.7871541500938646e-17-6.918926325734264e-09j),
  (1.1064408835358364e-19-3.3263206152381594e-10j),
  (1.941747305677948e-22-1.3934659327295906e-11j)],
 [(0.9165672213503293-0.2765352601323488j),
  (0.8925153551366475-0.3097284229481555j),
  (0.724406068807489-0.44681306637381196j),
  (0.79999539554102-0.40000345330282716j),
  (0.5815720097871232-0.4933011323920608j),
  (0.059989429815427465-0.23746725695524293j),
  (0.0016746678595338474-0.040888425588350694j),
  (3.157538369132347e-05-0.005619109065187133j),
  (4.0598276866269854e-07-0.0006371676418656946j),
  (3.5327216259785343e-09-5.943670257928523e-05j),
  (2.1212298756767418e-11-4.605681139236351e-06j),
  (9.027481439768403e-14-3.004576748856249e-07j),
  (2.7948125784531043e-16-1.6717692958219755e-08j),
  (6.442879235052296e-19-8.026754783255993e-10j),
  (1.128893090815587e-21-3.359900431286003e-11j),
  (1.5306616534558257e-24-1.2371991163332706e-12j)]])

@attr("fast")
def test_raw_fields():
    sp = Sphere(r=.5, n=1.6, center=(10, 10, 5))
    wavelen = .66
    index = 1.33
    pol = to_vector((0, 1))
    sch = detector_grid(3, .1)
    wavevec=2*np.pi/(wavelen/index)
    imageformer = ImageFormation(Mie())
    pos = imageformer._transform_to_desired_coordinates(
        sch, (10, 10, 5), wavevec=wavevec)
    rf = Mie().raw_fields(
        pos, sp, medium_wavevec=wavevec, medium_index=index,
        illum_polarization=pol)
    assert_allclose(rf, [[(0.0015606995428858754-0.0019143174710834162j),
  (-0.0003949071974815011-0.0024154494284017187j),
  (-0.002044525390662322-0.001302770747742109j),
  (-0.0003949071974815009-0.002415449428401719j),
  (-0.002055824337886397-0.0012853546864338861j),
  (-0.00230285180386436+0.000678693819245102j),
  (-0.0020445253906623225-0.0013027707477421095j),
  (-0.0023028518038643603+0.0006786938192451026j),
  (-0.0010011090105680883+0.0021552249454706712j)],
 [(-0.0010507058414478587+0.0036584360153097306j),
  (0.0020621595919700776+0.003210547679920805j),
  (0.0037794246074692407+0.000585690417403587j),
  (0.0020542215584045407+0.0031619947065620246j),
  (0.0037426710578253295+0.000527040269055415j),
  (0.002871631795307833-0.002470099566862354j),
  (0.0036968090916832948+0.0005330478443315597j),
  (0.002824872178181336-0.0024563186266035124j),
  (2.261564613123139e-06-0.003751168280253104j)],
 [(0.0010724312167657794+0.0039152445632936j),
  (0.003651474601303447+0.0017688083711547462j),
  (0.003740131549224567-0.001566271371618957j),
  (0.0036883581831347947+0.0017866751223785315j),
  (0.0037648739662344477-0.001614943488355339j),
  (0.0012643679510138835-0.003894481935619062j),
  (0.003816460764514863-0.0015982360934887314j),
  (0.0012772696647997395-0.0039342215472070105j),
  (-0.0021320123934202356-0.0035427449839031066j)]])


@attr('medium')
def test_j0_roots():
    # Checks for misbehavior when j_0(x) = 0
    eps = 1e-12
    d = detector_grid(shape=8, spacing=1)
    d = update_metadata(d,illum_wavelen=1, medium_index=1,
                                illum_polarization=(1,0))

    s_exact = Sphere(r=1,n=1.1,center=(4,4,5)) # causes kr = 0 near center
    s_close = Sphere(r=1,n=1.1,center=(4,4,5-eps))

    h_exact = calc_field(d,s_exact)
    h_close = calc_field(d,s_close)

    np.testing.assert_allclose(h_exact, h_close)
