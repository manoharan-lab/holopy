import unittest

import numpy as np
from scipy.special import jn_zeros
from nose.plugins.attrib import attr

from ..theory import mielensfunctions

TOLS = {'atol': 1e-10, 'rtol': 1e-10}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}
SOFTTOLS = {'atol': 1e-3, 'rtol': 1e-3}


class TestMieLensCalculator(unittest.TestCase):
    @attr("fast")
    def test_fields_nonzero(self):
        field1_x, field1_y = evaluate_scattered_field_in_lens()
        should_be_nonzero = [np.linalg.norm(f) for f in [field1_x, field1_y]]
        should_be_false = [np.isclose(v, 0, **TOLS) for v in should_be_nonzero]
        self.assertFalse(any(should_be_false))

    @attr("fast")
    def test_scatteredfield_linear_at_low_contrast(self):
        dn1 = 1e-3
        dn2 = 5e-4
        size_parameter = 0.1  # shouldn't need to be small as long as dn*ka is
        field1_x, field1_y = evaluate_scattered_field_in_lens(
            delta_index=dn1, size_parameter=size_parameter)
        field2_x, field2_y = evaluate_scattered_field_in_lens(
            delta_index=dn2, size_parameter=size_parameter)
        x_ratio = np.linalg.norm(field1_x) / np.linalg.norm(field2_x)
        y_ratio = np.linalg.norm(field1_y) / np.linalg.norm(field2_y)
        expected_ratio = dn1 / dn2
        self.assertTrue(np.isclose(x_ratio, expected_ratio, **SOFTTOLS))
        self.assertTrue(np.isclose(y_ratio, expected_ratio, **SOFTTOLS))

    @attr("fast")
    def test_fields_are_correct_values(self):
        # Tests fields are correct for a sphere 5 um above the focus

        # 1. Setup
        k = 2 * np.pi / 0.66  # 660 nm red light
        kwargs = {'particle_kz': 5.0 * k,
                  'index_ratio': 1.1,
                  'size_parameter': 0.5 * k,  # 1 um sphere = 0.5 um radius
                  'lens_angle': 0.8}

        calculator = mielensfunctions.MieLensCalculator(**kwargs)
        # 2. Calculate
        # We calculate at 3 angles: phi=0, pi/4, pi/2:
        rho = np.linspace(0, 10., 15)
        phi_0 = np.full(rho.size, 0)
        phi_pi4 = np.full(rho.size, np.pi / 4)
        phi_pi2 = np.full(rho.size, np.pi / 2)

        field_0 = calculator.calculate_scattered_field(k * rho, phi_0)
        field_pi4 = calculator.calculate_scattered_field(k * rho, phi_pi4)
        field_pi2 = calculator.calculate_scattered_field(k * rho, phi_pi2)

        truefield_0 = (
            np.array([-3.27079860e-01-4.70379875e-02j,
                      -2.80959213e-01+6.80780916e-02j,
                      2.54955299e-02+2.40707196e-01j,
                      1.39935878e-01-1.26875397e-01j,
                      -1.14236425e-01+8.06731118e-02j,
                      3.92558914e-02-7.48109016e-02j,
                      2.14646096e-02+4.28868140e-02j,
                      -2.71995120e-02+1.40341926e-04j,
                      1.00685307e-03-1.51698424e-02j,
                      9.39750739e-03+2.60677841e-03j,
                      7.11607699e-05+6.38939223e-03j,
                      -4.18131588e-03-7.93259837e-04j,
                      -1.01836410e-03-3.62553570e-03j,
                      2.07861434e-03-2.15136052e-04j,
                      1.28224948e-03+2.32453972e-03j]),
            np.array([0.+0.j, -0.+0.j,  0.+0.j,  0.+0.j, -0.+0.j,  0.-0.j,
                      0.+0.j, -0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j, -0.+0.j,
                      0.-0.j,  0.+0.j, 0.+0.j]))

        truefield_pi4 = (
            np.array([-0.32707986-4.70379875e-02j,
                      -0.27979525+6.75418547e-02j,
                      0.02352869+2.38883129e-01j,
                      0.13800729-1.20878576e-01j,
                      -0.11251044+7.30985316e-02j,
                      0.04114558-6.95304359e-02j,
                      0.01798897+4.22325799e-02j,
                      -0.02589335-1.55992775e-03j,
                      0.00198359-1.44364821e-02j,
                      0.00876035+3.11081706e-03j,
                      -0.00041122+6.12421341e-03j,
                      -0.00392536-1.04334576e-03j,
                      -0.00067622-3.54685029e-03j,
                      0.00200258-6.15323313e-05j,
                      0.00102324+2.32013460e-03j]),
            np.array([0.00000000e+00+0.00000000e+00j,
                      -1.16396716e-03+5.36236896e-04j,
                      1.96683587e-03+1.82406720e-03j,
                      1.92859274e-03-5.99682108e-03j,
                      -1.72598409e-03+7.57458020e-03j,
                      -1.88968753e-03-5.28046568e-03j,
                      3.47564367e-03+6.54234157e-04j,
                      -1.30616666e-03+1.70026968e-03j,
                      -9.76736864e-04-7.33360292e-04j,
                      6.37155633e-04-5.04038648e-04j,
                      4.82378596e-04+2.65178826e-04j,
                      -2.55960799e-04+2.50085923e-04j,
                      -3.42146032e-04-7.86854048e-05j,
                      7.60326047e-05-1.53603720e-04j,
                      2.59010725e-04+4.40511853e-06j]))

        truefield_pi2 = (
            np.array([-0.32707986-4.70379875e-02j,
                      -0.27863128+6.70056178e-02j,
                      0.02156186+2.37059062e-01j,
                      0.13607869-1.14881755e-01j,
                      -0.11078446+6.55239514e-02j,
                      0.04303527-6.42499702e-02j,
                      0.01451332+4.15783457e-02j,
                      -0.02458718-3.26019743e-03j,
                      0.00296033-1.37031218e-02j,
                      0.0081232 +3.61485571e-03j,
                      -0.0008936 +5.85903458e-03j,
                      -0.00366939-1.29343168e-03j,
                      -0.00033407-3.46816489e-03j,
                      0.00192655+9.20713891e-05j,
                      0.00076423+2.31572949e-03j]),
            np.array([0.00000000e+00+0.00000000e+00j,
                      -1.42544865e-19+6.56700798e-20j,
                      2.40867925e-19+2.23383806e-19j,
                      2.36184492e-19-7.34398774e-19j,
                      -2.11372089e-19+9.27618539e-19j,
                      -2.31419979e-19-6.46670539e-19j,
                      4.25643589e-19+8.01205766e-20j,
                      -1.59959282e-19+2.08222982e-19j,
                      -1.19615767e-19-8.98107334e-20j,
                      7.80290606e-20-6.17269316e-20j,
                      5.90743403e-20+3.24750400e-20j,
                      -3.13461573e-20+3.06266925e-20j,
                      -4.19008043e-20-9.63618292e-21j,
                      9.31130860e-21-1.88110305e-20j,
                      3.17196656e-20+5.39471431e-22j]))

        # compare; medtols b/c fields are only copied to 9 digits
        self.assertTrue(np.allclose(field_0, truefield_0, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi2, truefield_pi2, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi4, truefield_pi4, **MEDTOLS))

    # other possible tests:
    # 1. E(x, y) = E(-x, -y)
    # 2. E_x = 0 at phi = pi/2, E_y = 0 at phi = 0


class TestFarfieldMieInterpolator(unittest.TestCase):
    @attr("fast")
    def test_interpolator_1_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.FarfieldMieInterpolator(i=1)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_interpolator_2_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.FarfieldMieInterpolator(i=2)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_interpolator_maxl_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator_low_l = mielensfunctions.FarfieldMieInterpolator(i=1)

        higher_l = np.ceil(interpolator_low_l.size_parameter * 8).astype('int')
        interpolator_higher_l = mielensfunctions.FarfieldMieInterpolator(
            i=1, max_l=higher_l)

        exact = interpolator_low_l._eval(theta)
        approx = interpolator_higher_l._eval(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)


class TestGaussQuad(unittest.TestCase):
    @attr("fast")
    def test_constant_integrand(self):
        f = lambda x: np.ones_like(x, dtype='float')
        should_be_one = integrate_like_mielens(f, [3, 4])
        self.assertTrue(np.isclose(should_be_one, 1.0, **TOLS))

    @attr("fast")
    def test_linear_integrand(self):
        f = lambda x: x
        should_be_onehalf = integrate_like_mielens(f, [0, 1])
        self.assertTrue(np.isclose(should_be_onehalf, 0.5, **TOLS))

    @attr("fast")
    def test_transcendental_integrand(self):
        should_be_two = integrate_like_mielens(np.sin, [0, np.pi])
        self.assertTrue(np.isclose(should_be_two, 2.0, **TOLS))


class TestJ2(unittest.TestCase):
    @attr("fast")
    def test_fastj2_zeros(self):
        j2_zeros = jn_zeros(2, 50)
        should_be_zero = mielensfunctions.j2(j2_zeros)
        self.assertTrue(np.allclose(should_be_zero, 0, atol=1e-13))

    @attr("fast")
    def test_fastj2_notzero(self):
        # We pick some values that should not be zero or close to it
        j0_zeros = jn_zeros(0, 50)  # some conjecture -- bessel functions
        # only share the zero at z=0, which is not for j0
        should_not_be_zero = mielensfunctions.j2(j0_zeros)
        self.assertFalse(np.isclose(should_not_be_zero, 0, atol=1e-10).any())


def evaluate_scattered_field_in_lens(delta_index=0.1, size_parameter=0.1):
    miecalculator = mielensfunctions.MieLensCalculator(
        particle_kz=10, index_ratio=1.0 + delta_index,
        size_parameter=size_parameter, lens_angle=0.9)
    krho = np.linspace(0, 30, 300)
    # for phi, we pick an off-axis value where both polarizations are nonzero
    kphi = np.full_like(krho, 0.25 * np.pi)
    return miecalculator.calculate_scattered_field(krho, kphi)  # x,y component


def integrate_like_mielens(function, bounds):
    pts, wts = mielensfunctions.gauss_legendre_pts_wts(bounds[0], bounds[1])
    return (function(pts) * wts).sum()

if __name__ == '__main__':
    unittest.main()
