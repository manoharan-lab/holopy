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
            np.array([8.17699650e-02+1.17594969e-02j,
                      7.02398032e-02-1.70195229e-02j,
                      -6.37388248e-03-6.01767990e-02j,
                      -3.49839695e-02+3.17188493e-02j,
                      2.85591063e-02-2.01682780e-02j,
                      -9.81397285e-03+1.87027254e-02j,
                      -5.36615240e-03-1.07217035e-02j,
                      6.79987800e-03-3.50854815e-05j,
                      -2.51713267e-04+3.79246060e-03j,
                      -2.34937685e-03-6.51694602e-04j,
                      -1.77901925e-05-1.59734806e-03j,
                      1.04532897e-03+1.98314959e-04j,
                      2.54591025e-04+9.06383925e-04j,
                      -5.19653585e-04+5.37840130e-05j,
                      -3.20562370e-04-5.81134930e-04j]),
            np.array([0.+0.j, -0.+0.j,  0.+0.j,  0.+0.j, -0.+0.j,  0.-0.j,
                      0.+0.j, -0.+0.j,  0.-0.j,  0.+0.j,  0.+0.j, -0.+0.j,
                      0.-0.j,  0.+0.j, 0.+0.j]))

        truefield_pi4 = (
            np.array([0.08176997+1.17594969e-02j,
                      0.06994881-1.68854637e-02j,
                      -0.00588217-5.97207822e-02j,
                      -0.03450182+3.02196440e-02j,
                      0.02812761-1.82746329e-02j,
                      -0.0102864 +1.73826090e-02j,
                      -0.00449724-1.05581450e-02j,
                      0.00647334+3.89981938e-04j,
                      -0.0004959 +3.60912053e-03j,
                      -0.00219009-7.77704265e-04j,
                      0.00010281-1.53105335e-03j,
                      0.00098134+2.60836440e-04j,
                      0.00016905+8.86712572e-04j,
                      -0.00050064+1.53830828e-05j,
                      -0.00025581-5.80033650e-04j]),
            np.array([0.00000000e+00+0.00000000e+00j,
                      -2.90991790e-04+1.34059224e-04j,
                      4.91708968e-04+4.56016800e-04j,
                      4.82148185e-04-1.49920527e-03j,
                      -4.31496022e-04+1.89364505e-03j,
                      -4.72421882e-04-1.32011642e-03j,
                      8.68910917e-04+1.63558539e-04j,
                      -3.26541665e-04+4.25067420e-04j,
                      -2.44184216e-04-1.83340073e-04j,
                      1.59288908e-04-1.26009662e-04j,
                      1.20594649e-04+6.62947065e-05j,
                      -6.39901998e-05+6.25214807e-05j,
                      -8.55365080e-05-1.96713512e-05j,
                      1.90081512e-05-3.84009300e-05j,
                      6.47526812e-05+1.10127963e-06j]))

        truefield_pi2 = (
            np.array([8.17699649e-02+1.17594969e-02j,
                      6.96578195e-02-1.67514045e-02j,
                      -5.39046454e-03-5.92647654e-02j,
                      -3.40196732e-02+2.87204387e-02j,
                      2.76961142e-02-1.63809878e-02j,
                      -1.07588166e-02+1.60624926e-02j,
                      -3.62833057e-03-1.03945864e-02j,
                      6.14679468e-03+8.15049357e-04j,
                      -7.40081700e-04+3.42578045e-03j,
                      -2.03079903e-03-9.03713927e-04j,
                      2.23399105e-04-1.46475865e-03j,
                      9.17348569e-04+3.23357921e-04j,
                      8.35180100e-05+8.67041222e-04j,
                      -4.81637283e-04-2.30178473e-05j,
                      -1.91057008e-04-5.78932371e-04j]),
            np.array([0.00000000e+00+0.00000000e+00j,
                      -3.56362163e-20+1.64175199e-20j,
                      6.02169813e-20+5.58459515e-20j,
                      5.90461230e-20-1.83599694e-19j,
                      -5.28430223e-20+2.31904635e-19j,
                      -5.78549947e-20-1.61667635e-19j,
                      1.06410897e-19+2.00301441e-20j,
                      -3.99898205e-20+5.20557455e-20j,
                      -2.99039417e-20-2.24526834e-20j,
                      1.95072651e-20-1.54317329e-20j,
                      1.47685851e-20+8.11876000e-21j,
                      -7.83653933e-21+7.65667312e-21j,
                      -1.04752011e-20-2.40904573e-21j,
                      2.32782715e-21-4.70275763e-21j,
                      7.92991640e-21+1.34867858e-22j]))
        # compare; medtols b/c fields are only copied to 9 digits
        self.assertTrue(np.allclose(field_0, truefield_0, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi2, truefield_pi2, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi4, truefield_pi4, **MEDTOLS))

    @attr("fast")
    def test_fields_go_to_zero_at_large_distances(self):
        rho = np.logspace(2.4, 5, 80)
        calculator = mielensfunctions.MieLensCalculator(
            size_parameter=10, lens_angle=1.0, particle_kz=10., index_ratio=1.1)
        field_x, field_y = calculator.calculate_scattered_field(rho, 0*rho)
        fields_dont_explode = np.all(np.abs(field_x < 2e-4))
        self.assertTrue(fields_dont_explode)

    # other possible tests:
    # 1. E(x, y) = E(-x, -y)
    # 2. E_x = 0 at phi = pi/2, E_y = 0 at phi = 0


class TestFarfieldMieEvaluator(unittest.TestCase):
    @attr("fast")
    def test_interpolator_1_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.FarfieldMieEvaluator(i=1)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_interpolator_2_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.FarfieldMieEvaluator(i=2)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_interpolator_maxl_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator_low_l = mielensfunctions.FarfieldMieEvaluator(i=1)

        higher_l = np.ceil(interpolator_low_l.size_parameter * 8).astype('int')
        interpolator_higher_l = mielensfunctions.FarfieldMieEvaluator(
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
