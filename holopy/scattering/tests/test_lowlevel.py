# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
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
Test low-level physics and mathematical primitives that are part of 
scattering calculations.  

Most of these tests will check Fortran extensions.

These tests are intended to evaluate well-established low-level 
quantities (such as scattering coefficients or matrices calculated 
by independent codebases) or mathematical identities (such as 
coordinate transformations).  While the tests of physically 
measurable quantities (such as holograms) in test_mie.py and
test_multisphere.py are important, it is hoped that should any
of those fail, failures in these low-level tests will help pin
down the problem.


.. moduleauthor:: Jerome Fung <fung@physics.harvard.edu>
'''

from __future__ import division

import os
import yaml
from nose.tools import assert_raises
from numpy.testing import assert_allclose
import numpy as np
from numpy import sqrt, dot, pi, conj, real, imag, exp
from nose.tools import with_setup
from nose.plugins.attrib import attr

from ..theory.mie_f import mieangfuncs, miescatlib, multilayer_sphere_lib, \
    scsmfo_min

from scipy.special import sph_jn, sph_yn

# basic defs
kr = 10.
kr_asym = 1.9e4
theta = pi/4.
phi = -pi/4.

@attr('fast')
def test_spherical_vector_to_cartesian():
    '''
    Test conversions between complex vectors in spherical components
    and cartesian.

    Tests mieangfuncs.fieldstocart and mieangfuncs.radial_vect_to_cart.
    '''
    # acts on a column spherical vector from left
    conversion_mat = np.array([[1/2., 1/2., 1./sqrt(2)],
                               [-1/2., -1/2., 1./sqrt(2)],
                               [1./sqrt(2), -1./sqrt(2), 0.]])

    # test conversion of a vector with r, theta, and phi components
    test_vect = np.array([0.2, 1. + 1.j, -1.])
    fortran_conversion = mieangfuncs.radial_vect_to_cart(test_vect[0],
                                                         theta, phi)
    fortran_conversion += mieangfuncs.fieldstocart(test_vect[1:], 
                                                   theta, phi)

    assert_allclose(fortran_conversion, dot(conversion_mat, test_vect))


@attr('fast')
def test_polarization_to_scatt_coords():
    '''
    Test conversion of an incident polarization (specified as a
    Cartesian vector in the lab frame) to an incident field
    in scattering spherical coordinates.

    For convention, see Bohren & Huffman ([Bohren1983]_) pp. 61-62. 
    '''

    conversion_mat = 1./sqrt(2) * np.array([[1., -1.],
                                            [-1., -1.]])
    
    test_vect = np.array([-1., 3.])
    fortran_result = mieangfuncs.incfield(test_vect[0], test_vect[1], phi)
    assert_allclose(fortran_result, dot(conversion_mat, test_vect))


@attr('medium')
def test_mie_amplitude_scattering_matrices():
    '''
    Test calculation of Mie amplitude scattering matrix elements.
    We will check the following:
        far-field matrix elements (direct comparison with [Bohren1983]_)
        near-field matrix for kr ~ 10 differs from far-field result
        near-field matrix for kr ~ 10^4 is close to far-field result

    While radiometric quantities (such as cross sections) implicitly test
    the Mie scattering coefficients, they do not involve any angular 
    quantities.
    '''

    # scattering units
    m = 1.55
    x = 2. * pi * 0.525 / 0.6328
    
    asbs = miescatlib.scatcoeffs(m, x, miescatlib.nstop(x))
    amp_scat_mat = mieangfuncs.asm_mie_far(asbs, theta)
    amp_scat_mat_asym = mieangfuncs.asm_mie_fullradial(asbs, np.array([kr_asym,
                                                                       theta,
                                                                       phi]))
    amp_scat_mat_near = mieangfuncs.asm_mie_fullradial(asbs, np.array([kr, 
                                                                      theta,
                                                                      phi]))

    # gold results directly from B/H p.482.
    location = os.path.split(os.path.abspath(__file__))[0]
    gold_name = os.path.join(location, 'gold',
                             'gold_mie_scat_matrix')
    gold_dict = yaml.load(file(gold_name + '.yaml'))
    gold = np.array([gold_dict['S11'], gold_dict['pol'], 
                     gold_dict['S33'], gold_dict['S34']])


    # B/H gives real scattering matrix elements, which are related
    # to the amplitude scatering elements.  See p. 65.
    def massage_into_bh_form(asm):
        S2, S3, S4, S1 = np.ravel(asm)
        S11 = 0.5 * (abs(asm)**2).sum()
        S12 = 0.5 * (abs(S2)**2 - abs(S1)**2)
        S33 = real(S1 * conj(S2)) 
        S34 = imag(S2 * conj(S1))
        deg_of_pol = -S12/S11
        # normalization factors: see comment lines 40-44 on p. 479
        asm_fwd = mieangfuncs.asm_mie_far(asbs, 0.)
        S11_fwd = 0.5 * (abs(asm_fwd)**2).sum()
        results = np.array([S11/S11_fwd, deg_of_pol, S33 / S11, S34 / S11])
        return results

    # off-diagonal elements should be zero
    assert_allclose(np.ravel(amp_scat_mat)[1:3], np.zeros(2))

    # check far-field computation
    assert_allclose(massage_into_bh_form(amp_scat_mat), gold, 
                    rtol = 1e-5)
 
    # check asymptotic behavior of near field matrix
    asym = massage_into_bh_form(amp_scat_mat_asym)
    assert_allclose(asym, gold, rtol = 1e-4, atol = 5e-5)
 
    # check that the near field is different
    try:
        assert_allclose(amp_scat_mat, amp_scat_mat_near)
    except AssertionError:
        pass
    else:
        raise AssertionError("Near-field amplitude scattering matrix " +
                             "suspiciously close to far-field result.")

@attr('fast')
def test_scattered_field_from_asm():
    '''
    Test the calculation of the scattered field, given the amplitude
    scattering matrix.  We will here use a fictitious (and probably
    unphysical) amplitude scattering matrix.
    '''
    asm = np.array([[1., -1.j],
                   [2., -0.1]])
    fortran_test = mieangfuncs.calc_scat_field(kr, phi, asm, np.array([1., 0.]))
    gold = (1./sqrt(2)) * 0.1j * exp(10.j) * np.array([1. + 1.j, -2.1])
    assert_allclose(fortran_test, gold)


@attr('medium')
def test_mie_internal_coeffs():
    m = 1.5 + 0.1j
    x = 50.
    n_stop = miescatlib.nstop(x)
    al, bl = miescatlib.scatcoeffs(m, x, n_stop)
    cl, dl = miescatlib.internal_coeffs(m, x, n_stop)
    jlx = sph_jn(n_stop, x)[0][1:]
    jlmx = sph_jn(n_stop, m * x)[0][1:]
    hlx = jlx + 1.j * sph_yn(n_stop, x)[0][1:]
    
    assert_allclose(cl, (jlx - hlx * bl) / jlmx, rtol = 1e-6, atol = 1e-6)
    assert_allclose(dl, (jlx - hlx * al)/ (m * jlmx), rtol = 1e-6, atol = 1e-6)
    
@attr('fast')
def test_mie_bndy_conds():
    '''
    Check that appropriate boundary conditions are satisfied:
    m^2 E_radial continuous (bound charge Gaussian pillbox)
    E_parallel continuous (Amperian loop)

    Checks to do (all on E_x):
    theta = 0, phi = 0 (theta component)
    theta = 90, phi = 0 (radial component)
    theta = 90, phi = 90 (phi component)
    '''
    m = 1.2 + 0.01j
    x = 10.
    pol = np.array([1., 0.]) # assume x polarization
    n_stop = miescatlib.nstop(x)
    asbs = miescatlib.scatcoeffs(m, x, n_stop)
    csds = miescatlib.internal_coeffs(m, x, n_stop)

    # define field points
    eps = 1e-6 # get just inside/outside boundary
    kr_ext = np.ones(3) * (x + eps)
    kr_int = np.ones(3) * (x - eps)
    thetas = np.array([0., pi/2., pi/2.])
    phis = np.array([0., 0., pi/2.])
    points_int = np.vstack((kr_int, thetas, phis))
    points_ext = np.vstack((kr_ext, thetas, phis))

    # calc escat
    es_x, es_y, es_z = mieangfuncs.mie_fields(points_ext, asbs, pol, 1)
    # calc eint
    eint_x, eint_y, eint_z = mieangfuncs.mie_internal_fields(points_int, m,
                                                             csds, pol)
    # theta check
    assert_allclose(eint_x[0], es_x[0] + exp(1.j * (x + eps)), rtol = 5e-6)
    # r check
    assert_allclose(m**2 * eint_x[1], es_x[1] + 1., rtol = 5e-6)
    # phi check
    assert_allclose(eint_x[2], es_x[2] + 1., rtol = 5e-6)


# TODO: another check on the near-field result: calculate the scattered
# power by 4pi integration of E_scat^2 over 4pi. The result should be
# independent of kr and close to the analytical result.


@attr('medium')
def test_mie_multisphere_singlesph():
    '''
    Check that fields from mie_fields and tmatrix_fields are consistent
    at several points. This includes a check on the radial component of E_scat.
    '''
    # sphere params
    x = 5.
    m = 1.2+0.1j
    pol = np.array([1., 0.]) # assume x polarization
    
    # points to check
    # at last two points: E_s
    kr = np.ones(4) * 6.
    thetas = np.array([0., pi/3., pi/2., pi/2.])
    phis = np.array([0., pi/6., 0., pi/2.])
    field_pts = np.vstack((kr, thetas, phis))

    # calculate fields with Mie
    n_stop_mie = miescatlib.nstop(x)
    asbs = miescatlib.scatcoeffs(m, x, n_stop_mie)
    emie_x, emie_y, emie_z = mieangfuncs.mie_fields(field_pts, asbs, pol, 1)

    # calculate fields with Multisphere
    _, lmax, amn0, conv = scsmfo_min.amncalc(1, 0., 0., 0., m.real, m.imag,
                                             x, 100, 1e-6, 1e-8, 1e-8, 1, 
                                             (0., 0.))
    # increase qeps1 from usual here
    limit = lmax**2 + 2 * lmax
    amn = amn0[:, 0:limit, :]
    etm_x, etm_y, etm_z = mieangfuncs.tmatrix_fields(field_pts, amn, lmax,
                                                     0., pol, 1)

    assert_allclose(etm_x, emie_x, rtol = 1e-6, atol = 1e-6)
    assert_allclose(etm_y, emie_y, rtol = 1e-6, atol = 1e-6)
    assert_allclose(etm_z, emie_z, rtol = 1e-6, atol = 1e-6)
