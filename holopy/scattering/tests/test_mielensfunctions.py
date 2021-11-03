import warnings
import unittest
import itertools

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import jn_zeros
from nose.plugins.attrib import attr

from holopy.scattering.theory import mielensfunctions
from holopy.scattering.errors import MissingParameter


TOLS = {'atol': 1e-12, 'rtol': 1e-12}
MEDTOLS = {"atol": 1e-6, "rtol": 1e-6}
SOFTTOLS = {'atol': 1e-3, 'rtol': 1e-3}


class TestMieLensCalculator(unittest.TestCase):
    @attr("fast")
    def test_raises_error_when_no_params_are_specified(self):
        self.assertRaises(MissingParameter, mielensfunctions.MieLensCalculator)

    @attr("fast")
    def test_raises_error_when_any_params_isnt_specified(self):
        kwargs = {'particle_kz': 10.0,
                  'index_ratio': 1.3,
                  'size_parameter': 5.0,
                  'lens_angle': 0.8,
                  }

        def create_calculator(**kwargs):
            return mielensfunctions.MieLensCalculator(**kwargs)

        for key in kwargs.keys():
            value = kwargs[key]  # popping it out
            kwargs[key] = None
            self.assertRaises(MissingParameter, create_calculator, **kwargs)
            kwargs[key] = value  # putting it back

    @attr("fast")
    def test_raises_error_when_inputs_mismatched_size(self):
        miecalculator = mielensfunctions.MieLensCalculator(
            particle_kz=10, index_ratio=1.2, size_parameter=10.0,
            lens_angle=0.9)
        krho = np.linspace(0, 30, 300)
        phi = np.full(krho.size - 4, 0.25 * np.pi)
        self.assertRaises(
            ValueError, miecalculator.calculate_scattered_field,
            krho, phi)

    @attr("fast")
    def test_raises_error_integral_is_not_i0_or_i2(self):
        miecalculator = mielensfunctions.MieLensCalculator(
            particle_kz=10, index_ratio=1.2, size_parameter=10.0,
            lens_angle=0.9)
        krho = np.linspace(0, 30, 300)
        self.assertRaises(
            ValueError, miecalculator._eval_mielens_i_n, krho, n=1)

    @attr("fast")
    def test_fields_nonzero(self):
        field1_x, field1_y = evaluate_scattered_field_in_lens()
        should_be_nonzero = [np.linalg.norm(f) for f in [field1_x, field1_y]]
        should_be_false = [np.isclose(v, 0, **TOLS) for v in should_be_nonzero]
        self.assertFalse(any(should_be_false))

    @attr("fast")
    def test_calculation_doesnt_crash_for_large_rho(self):
        miecalculator = mielensfunctions.MieLensCalculator(
            particle_kz=10, index_ratio=1.2, size_parameter=10.,
            lens_angle=0.9)
        krho = np.linspace(1000, 1100, 10)
        kphi = np.full_like(krho, 0.25 * np.pi)
        fields = miecalculator.calculate_scattered_field(krho, kphi)
        self.assertTrue(fields is not None)

    @attr("medium")
    def test_central_lobe_is_bright_when_particle_is_above_focus(self):
        zs = np.linspace(2, 10, 11)
        k = 2 * np.pi / 0.66
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        central_lobes = np.squeeze(
            [evaluate_intensity_at_rho_zero(k * z, index_ratio=1.05)
             for z in zs])
        self.assertTrue(np.all(central_lobes > 1))

    @attr("medium")
    def test_central_lobe_is_dark_when_particle_is_below_focus(self):
        zs = np.linspace(-2, -10, 11)
        k = 2 * np.pi / 0.66
        # This test only works at low index contrast, when the scattered
        # beam is everywhere weaker than the unscattered:
        central_lobes = np.squeeze(
            [evaluate_intensity_at_rho_zero(k * z, index_ratio=1.05)
             for z in zs])
        self.assertTrue(np.all(central_lobes < 1))

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
    def test_xpolarization_is_larger_along_phi_equals_0(self):
        fieldcalc = mielensfunctions.MieLensCalculator(
            particle_kz=0.0, index_ratio=1.19, size_parameter=0.1,
            lens_angle=0.8)
        krho = np.linspace(0.1, 2, 30)
        phi_0 = np.zeros(krho.shape)
        phi_pi4 = phi_0 + np.pi / 4.

        fx_0, fy_0 = fieldcalc.calculate_scattered_field(krho, phi_0)
        fx_pi4, fy_pi4 = fieldcalc.calculate_scattered_field(krho, phi_pi4)

        is_ok = fx_0.real > fx_pi4.real
        self.assertTrue(np.all(is_ok))

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
            np.array([-1.32809510e-01+0.08146271j,
                      -9.06794903e-02+0.1172825j ,
                      8.21893968e-02+0.08204799j,
                      7.56371204e-03-0.08694946j,
                      -6.16821976e-03+0.05753421j,
                      -1.28205169e-02-0.02865722j,
                      1.52658533e-02+0.00194537j,
                      -4.02116676e-03+0.00734771j,
                      -3.67872355e-03-0.00180371j,
                      1.60209148e-03-0.00231238j,
                      1.70980783e-03+0.0002438j ,
                      -5.04523640e-04+0.00112576j,
                      -1.11374256e-03+0.00025535j,
                      4.17810243e-05-0.00059301j,
                      7.81002777e-04-0.00039032j]),
            np.array([0.+0.j,  0.+0.j,  0.-0.j, -0.+0.j,  0.+0.j,  0.+0.j,
                      0.-0.j, 0.+0.j,  0.+0.j, -0.+0.j,  0.-0.j,  0.+0.j,
                      0.+0.j, -0.+0.j, 0.-0.j]))

        truefield_pi4 = (
            np.array([-1.32809510e-01+8.14627106e-02j,
                      -9.10453076e-02+1.17600194e-01j,
                      8.35512121e-02+8.30232340e-02j,
                      8.14492728e-03-8.96475613e-02j,
                      -7.20628427e-03+6.07260692e-02j,
                      -1.33209135e-02-3.11720004e-02j,
                      1.68662078e-02+2.68433934e-03j,
                      -4.88607511e-03+7.98665504e-03j,
                      -4.02303040e-03-2.38353337e-03j,
                      2.01334697e-03-2.43175325e-03j,
                      1.86014012e-03+5.10936483e-04j,
                      -6.88469988e-04+1.18558752e-03j,
                      -1.23001081e-03+1.16072274e-04j,
                      1.22899446e-04-6.46969142e-04j,
                      8.78058364e-04-3.11103837e-04j]),
            np.array([ 0.00000000e+00+0.00000000e+00j,
                      3.65817305e-04-3.17693628e-04j,
                      -1.36181529e-03-9.75243907e-04j,
                      -5.81215242e-04+2.69810296e-03j,
                      1.03806451e-03-3.19185972e-03j,
                      5.00396567e-04+2.51477939e-03j,
                      -1.60035452e-03-7.38974073e-04j,
                      8.64908349e-04-6.38943288e-04j,
                      3.44306845e-04+5.79818489e-04j,
                      -4.11255487e-04+1.19369202e-04j,
                      -1.50332293e-04-2.67135310e-04j,
                      1.83946348e-04-5.98313492e-05j,
                      1.16268243e-04+1.39275966e-04j,
                      -8.11184216e-05+5.39575376e-05j,
                      -9.70555866e-05-7.92120153e-05j]))

        truefield_pi2 = (
            np.array([-0.13280951+8.14627106e-02j,
                      -0.09141112+1.17917888e-01j,
                      0.08491303+8.39984779e-02j,
                      0.00872614-9.23456643e-02j,
                      -0.00824435+6.39179289e-02j,
                      -0.01382131-3.36867798e-02j,
                      0.01846656+3.42331341e-03j,
                      -0.00575098+8.62559833e-03j,
                      -0.00436734-2.96335185e-03j,
                      0.0024246 -2.55112245e-03j,
                      0.00201047+7.78071792e-04j,
                      -0.00087242+1.24541887e-03j,
                      -0.00134628-2.32036918e-05j,
                      0.00020402-7.00926680e-04j,
                      0.00097511-2.31891822e-04j]),
            np.array([ 0.00000000e+00+0.00000000e+00j,
                      4.47996992e-20-3.89062485e-20j,
                      -1.66774273e-19-1.19432933e-19j,
                      -7.11783386e-20+3.30422315e-19j,
                      1.27126238e-19-3.90890079e-19j,
                      6.12809053e-20+3.07971653e-19j,
                      -1.95986904e-19-9.04982233e-20j,
                      1.05920724e-19-7.82479853e-20j,
                      4.21654276e-20+7.10072856e-20j,
                      -5.03642716e-20+1.46185111e-20j,
                      -1.84103962e-20-3.27146402e-20j,
                      2.25269306e-20-7.32722703e-21j,
                      1.42387532e-20+1.70563866e-20j,
                      -9.93414154e-21+6.60789257e-21j,
                      -1.18858814e-20-9.70067410e-21j]))


        # compare; medtols b/c fields are only copied to 9 digits
        self.assertTrue(np.allclose(field_0, truefield_0, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi2, truefield_pi2, **MEDTOLS))
        self.assertTrue(np.allclose(field_pi4, truefield_pi4, **MEDTOLS))

    @attr("fast")
    def test_fields_go_to_zero_at_large_distances(self):
        rho = np.logspace(2.4, 5, 80)
        calculator = mielensfunctions.MieLensCalculator(
            size_parameter=10, lens_angle=1.0, particle_kz=10.,
            index_ratio=1.1)
        field_x, field_y = calculator.calculate_scattered_field(rho, 0*rho)
        fields_dont_explode = np.all(np.abs(field_x < 2e-4))
        self.assertTrue(fields_dont_explode)

    @attr("fast")
    def test_calculate_phase(self):
        np.random.seed(119)
        kz = 20.0 * np.random.rand()
        calculator = mielensfunctions.MieLensCalculator(
            size_parameter=10, lens_angle=1.0, particle_kz=kz,
            index_ratio=1.1)

        correct_phase = kz * (1 - calculator._quad_pts)
        phase = calculator._calculate_phase()
        self.assertTrue(np.allclose(correct_phase, phase, **TOLS))

    @attr("slow")
    def test_interpolate_is_same_as_direct_computation(self):
        k = 2 * np.pi / 0.66

        rho = np.linspace(0, 15, 151)
        phi = np.linspace(0, 8 * np.pi, rho.size)

        # Test over a few particle parameters:
        radii = [0.5, 1.5, 6.5]
        zs = [5.0, 10.0, -10.]
        index_ratios = [1.1, 1.2, 1.3]

        for rad, z, index_ratio in itertools.product(radii, zs, index_ratios):
            kz = k * z
            ka = k * rad
            kwargs = {'particle_kz': kz,
                      'size_parameter': ka,
                      'index_ratio': index_ratio,
                      'lens_angle': 1.0,
                      }
            direct_calculator = mielensfunctions.MieLensCalculator(
                interpolate_integrals=False, **kwargs)
            interpolating_calculator = mielensfunctions.MieLensCalculator(
                interpolate_integrals=True, **kwargs)

            fdx, fdy = direct_calculator.calculate_scattered_field(
                k * rho, phi)
            fix, fiy = interpolating_calculator.calculate_scattered_field(
                k * rho, phi)

            with self.subTest(rad=rad, z=z, index_ratio=index_ratio):
                close_enough_x = np.allclose(fdx, fix, **TOLS)
                close_enough_y = np.allclose(fdy, fiy, **TOLS)
                self.assertTrue(close_enough_x)
                self.assertTrue(close_enough_y)

    @attr("medium")
    def test_energy_is_conserved(self):

        def get_excess_power(max_x, npts):
            """Calculates (total power) - (power at detector)

            This should be _negative_, slightly, as some of the scattered
            beam does not enter the lens"""
            t = np.linspace(-max_x, max_x, npts)
            x = t.reshape(-1, 1)
            y = t.reshape(1, -1)
            dt = t[1] - t[0]
            dA = dt**2
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            k = 2 * np.pi / 0.66 * 1.33

            kwargs = {'particle_kz': 5 * k,
                      'index_ratio': 1.59 / 1.33,
                      'size_parameter': k * 0.5,
                      'lens_angle': 0.8,
                      }
            calc = mielensfunctions.MieLensCalculator(**kwargs)
            hologram = calc.calculate_total_intensity(k*rho, phi)
            excess_power = dA * (hologram - 1).sum()
            return excess_power

        excess_power = get_excess_power(10, 101)
        # Energy is conserved, but some exits the lens. So this should
        # be slightly _negative_. Unfortunately I don't know exactly how
        # much... so we just check that it is consistent with what I have
        # now.
        ok = (excess_power < 0) and (excess_power > -0.15)
        self.assertTrue(ok)

    # other possible tests:
    # 1. E(x, y) = E(-x, -y)
    # 2. E_x = 0 at phi = pi/2, E_y = 0 at phi = 0


class TestAberratedMieLensCalculator(unittest.TestCase):
    @attr("fast")
    def test_raises_error_when_no_params_are_specified(self):
        self.assertRaises(
            MissingParameter, mielensfunctions.AberratedMieLensCalculator)

    @attr("fast")
    def test_raises_error_when_any_params_isnt_specified(self):
        kwargs = {'particle_kz': 10.0,
                  'index_ratio': 1.3,
                  'size_parameter': 5.0,
                  'lens_angle': 0.8,
                  'spherical_aberration': 1.0,
                  }

        def create_calculator(**kwargs):
            return mielensfunctions.AberratedMieLensCalculator(**kwargs)

        for key in kwargs.keys():
            value = kwargs[key]  # popping it out
            kwargs[key] = None
            self.assertRaises(MissingParameter, create_calculator, **kwargs)
            kwargs[key] = value  # putting it back

    @attr("fast")
    def test_calculate_aberration_form(self):
        calculator = make_calculator_with_aberration_of(0.0)
        aberration = calculator._pupil_x_squared

        theta = np.arccos(calculator._quad_pts)
        correct_aberration = (np.cos(theta) - 1.0)
        self.assertTrue(np.allclose(aberration, correct_aberration, **TOLS))

    @attr("fast")
    def test_calculate_phase(self):
        np.random.seed(143)
        spherical_aberration = np.random.randn()
        calculator = make_calculator_with_aberration_of(spherical_aberration)
        kz = calculator.particle_kz

        phase = calculator._calculate_phase()
        correct_phase = (
            kz * (1 - calculator._quad_pts) +
            spherical_aberration * calculator._pupil_x_squared**2)
        self.assertTrue(np.allclose(correct_phase, phase, **TOLS))

    @attr("fast")
    def test_gives_correct_values(self):
        calculator = make_calculator_with_aberration_of(42.1)
        krho = np.linspace(0, 100, 10)
        fx_calc, fy_calc = calculator.calculate_scattered_field(krho, 0 * krho)

        fx_true = np.array([
            1.62200970e+00 + 2.11177687e+00j,
            1.97957403e-01 + 2.91793510e-01j,
            -1.23716044e-02 + 5.20378583e-02j,
            -7.72814013e-03 - 8.83646666e-03j,
            -9.92495711e-03 + 1.45672004e-02j,
            1.27054494e-02 - 7.32949748e-04j,
            -6.93675387e-03 - 4.27264277e-03j,
            3.06621454e-03 + 4.21161773e-03j,
            -1.33377823e-03 - 3.21410269e-03j,
            6.26057275e-04 + 2.38456278e-03j])
        fy_true = 0
        self.assertTrue(np.allclose(fx_calc, fx_true, atol=1e-9))
        self.assertTrue(np.allclose(fy_calc, fy_true, **TOLS))

    @attr("fast")
    def test_higher_order_aberrations_differ_from_3rd_order(self):
        np.random.seed(354)
        high_order = 10 * np.random.randn(10)  # 10th-order aberrations!
        calc_high = make_calculator_with_aberration_of(high_order)
        calc_low = make_calculator_with_aberration_of(high_order[:1])

        krho = np.linspace(0, 100, 10)
        fields_high = calc_high.calculate_scattered_field(krho, 0*krho)
        fields_low = calc_low.calculate_scattered_field(krho, 0*krho)
        self.assertFalse(np.allclose(fields_high, fields_low, **TOLS))

    @attr("fast")
    def test_higher_order_aberrations_zero_same_as_3rd_order(self):
        np.random.seed(354)
        high_order = np.zeros(7)
        high_order[0] = 10 * np.random.randn()
        calc_high = make_calculator_with_aberration_of(high_order)
        calc_low = make_calculator_with_aberration_of(high_order[:1])

        krho = np.linspace(0, 100, 10)
        fields_high = calc_high.calculate_scattered_field(krho, 0*krho)
        fields_low = calc_low.calculate_scattered_field(krho, 0*krho)
        self.assertTrue(np.allclose(fields_high, fields_low, **TOLS))


class TestMieScatteringMatrix(unittest.TestCase):
    default_kwargs = {'index_ratio': 1.1, 'size_parameter': 10.0}

    @attr("fast")
    def test_raises_error_on_nans(self):
        theta = np.array([np.nan])
        interpolator = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', **self.default_kwargs)
        self.assertRaises(RuntimeError, interpolator._eval, theta)

    @attr('fast')
    def test_clips_high_max_l_to_avoid_nans(self):
        theta = np.array([0.3])

        msm_highl = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', max_l=1000,
            **self.default_kwargs)
        should_be_warned = 'invalid value encountered in cdouble_scalars'
        with self.assertWarnsRegex(Warning, should_be_warned):
            warnings.simplefilter('always')
            s_theta = msm_highl._eval(theta)

        self.assertFalse(np.isnan(s_theta))

    @attr("fast")
    def test_perpendicular_interpolator_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', **self.default_kwargs)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_parallel_interpolator_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='parallel', **self.default_kwargs)

        exact = interpolator._eval(theta)
        approx = interpolator(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_interpolator_maxl_accuracy(self):
        theta = np.linspace(0, 1.5, 1000)
        interpolator_low_l = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', **self.default_kwargs)

        higher_l = np.ceil(interpolator_low_l.size_parameter * 8).astype('int')
        interpolator_higher_l = mielensfunctions.MieScatteringMatrix(
            parallel_or_perpendicular='perpendicular', max_l=higher_l,
            **self.default_kwargs)

        exact = interpolator_low_l._eval(theta)
        approx = interpolator_higher_l._eval(theta)
        rescale = np.abs(exact).max()

        is_ok = np.allclose(exact / rescale, approx / rescale, **MEDTOLS)
        self.assertTrue(is_ok)

    @attr('medium')
    def test_works_when_large_size_parameter(self):
        theta = np.linspace(0, 1.5, 11)
        interpolator_low_l = mielensfunctions.MieScatteringMatrix(
            index_ratio=1.1,
            size_parameter=1000.0,  # roughly 80 um sphere
            parallel_or_perpendicular='perpendicular')

        out = interpolator_low_l._eval(theta)
        self.assertFalse(np.any(np.isnan(out)))


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


class TestMiscMath(unittest.TestCase):
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

    @attr("fast")
    def test_guass_legendre_pts_wts_n10_uses_10_points(self):
        npts = 10
        p10, w10 = mielensfunctions.gauss_legendre_pts_wts(0, 1, npts=npts)
        self.assertEqual(p10.size, npts)
        self.assertEqual(w10.size, npts)

    @attr("fast")
    def test_guass_legendre_pts_wts_n10_gives_correct_value(self):
        p10, w10 = mielensfunctions.gauss_legendre_pts_wts(0, 1, npts=10)
        quad_10 = np.sum(np.cos(p10) * w10)
        truth = np.sin(1) - np.sin(0)
        self.assertTrue(np.isclose(quad_10, truth, **TOLS))

    @attr("fast")
    def test_guass_legendre_pts_wts_ndefault(self):
        p100, w100 = mielensfunctions.gauss_legendre_pts_wts(0, 1)
        quad_100 = np.sum(np.cos(p100) * w100)
        truth = np.sin(1) - np.sin(0)
        self.assertTrue(np.isclose(quad_100, truth, **TOLS))

    @attr("fast")
    def test_spherical_hankel_1(self):
        # uses tests for the exact forms, taken from
        # mathworld.wolfram.com/SphericalHankelFunctionoftheFirstKind.html
        np.random.seed(10)
        z = np.random.rand(10) * 10
        calculated = mielensfunctions.spherical_h1n(1, z, derivative=False)
        truth = -np.exp(1j*z) * (z + 1j) / z**2  # from wolfram
        self.assertTrue(np.allclose(truth, calculated, **TOLS))

    @attr("fast")
    def test_spherical_hankel_2(self):
        # uses tests for the exact forms, taken from
        # mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html
        np.random.seed(10)
        z = np.random.rand(10) * 10
        calculated = mielensfunctions.spherical_h2n(1, z, derivative=False)
        truth = -np.exp(-1j*z) * (z - 1j) / z**2  # from wolfram
        self.assertTrue(np.allclose(truth, calculated, **TOLS))


class TestCalculation(unittest.TestCase):
    # We use a weak tolerance
    _lowna_tols = {'atol': 2e-2, 'rtol': 0}
    _highna_tols = {'atol': 3e-3, 'rtol': 0}

    @attr("fast")
    def test_energy_is_conserved_at_low_na_pointparticle(self):
        ratio = get_ratio_of_scattered_powerin_to_scattered_powerout(
            lens_angle=0.1, index_ratio=1.5, particle_kz=0, size_parameter=0.1)
        self.assertTrue(np.isclose(ratio, 1.0, **self._lowna_tols))

    @attr("medium")
    def test_energy_is_conserved_at_low_na_largeparticle(self):
        ratio = get_ratio_of_scattered_powerin_to_scattered_powerout(
            lens_angle=0.1, index_ratio=1.5, particle_kz=0, size_parameter=40.)
        self.assertTrue(np.isclose(ratio, 1.0, **self._lowna_tols))

    @attr("fast")
    def test_energy_is_conserved_at_high_na_pointparticle(self):
        ratio = get_ratio_of_scattered_powerin_to_scattered_powerout(
            lens_angle=0.9, index_ratio=1.5, particle_kz=0, size_parameter=0.1)
        self.assertTrue(np.isclose(ratio, 1.0, **self._highna_tols))

    @attr("medium")
    def test_energy_is_conserved_at_high_na_largeparticle(self):
        ratio = get_ratio_of_scattered_powerin_to_scattered_powerout(
            lens_angle=0.9, index_ratio=1.5, particle_kz=0, size_parameter=40.)
        self.assertTrue(np.isclose(ratio, 1.0, **self._highna_tols))

    @attr("medium")
    def test_energy_is_conserved_at_high_na_largeparticle_defocus(self):
        ratio = get_ratio_of_scattered_powerin_to_scattered_powerout(
            lens_angle=0.9, index_ratio=1.5, particle_kz=200.,
            size_parameter=40.)
        self.assertTrue(np.isclose(ratio, 1.0, **self._highna_tols))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                           Interpolation Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class TestPiecewiseChebyshevApproximant(unittest.TestCase):
    @attr("fast")
    def test_mask_window(self):
        window = (0, 1)
        x = np.linspace(0, 2, 101)
        mask = mielensfunctions.PiecewiseChebyshevApproximant._mask_window(
            x, window)
        self.assertLess(x[mask].max(), 1.0)
        self.assertGreaterEqual(x[~mask].min(), 1.0)

    @attr("fast")
    def test_setup_windows_splits_into_n_windows(self):
        nwindows = 5
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, degree=10,
            window_breakpoints=np.linspace(0, 1, nwindows + 1))
        windows = piecewisecheb._setup_windows()
        self.assertEqual(len(windows), nwindows)

    @attr("fast")
    def test_setup_windows_partitions_window(self):
        nwindows = 5
        window = (0, 1)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, degree=10,
            window_breakpoints=np.linspace(*window, nwindows + 1))
        windows = piecewisecheb._setup_windows()

        np.random.seed(72)
        x = np.random.rand(101) * np.ptp(window) + window[0]
        masks = [piecewisecheb._mask_window(x, w) for w in windows]
        number_of_masks_contained = np.sum(masks, axis=0)
        self.assertTrue(np.all(number_of_masks_contained == 1))

    @attr("fast")
    def test_setup_approximants_generates_approximants(self):
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, degree=10, window_breakpoints=np.linspace(0, 1, 6))
        approximants = piecewisecheb._setup_approximants()
        for approximant in approximants:
            self.assertTrue(isinstance(approximant, Chebyshev))

    @attr("fast")
    def test_dtype_on_float(self):
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, 10, window_breakpoints=np.linspace(0, 1, 6))
        self.assertEqual(piecewisecheb._dtype.name, 'float64')

    @attr("fast")
    def test_dtype_on_complex(self):
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            lambda x: np.exp(1j * x), 10,
            window_breakpoints=np.linspace(0, 1, 6))
        self.assertEqual(piecewisecheb._dtype.name, 'complex128')

    @attr("fast")
    def test_call_raises_error_when_x_less_than_window(self):
        window = (0, 10)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, 10, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[0] - 1)

    @attr("fast")
    def test_call_raises_error_when_x_greater_than_window(self):
        window = (0, 10)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, 10, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[1] + 1)

    @attr("fast")
    def test_call_raises_error_when_x_equal_to_max_window(self):
        window = (0, 10)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, 10, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[1])

    @attr("fast")
    def test_call_returns_correct_shape(self):
        window = (0, 20)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, 10, window_breakpoints=np.linspace(*window, 6))
        x = np.linspace(window[0], window[1] - 0.1, 101)
        true = np.sin(x)
        approx = piecewisecheb(x)
        self.assertEqual(true.shape, approx.shape)

    @attr("fast")
    def test_call_accurately_approximates(self):
        window = (0, 20)
        piecewisecheb = mielensfunctions.PiecewiseChebyshevApproximant(
            np.sin, degree=12,
            window_breakpoints=np.linspace(*window, 21))
        x = np.linspace(window[0], window[1] - 0.1, 101)
        true = np.sin(x)
        approx = piecewisecheb(x)
        self.assertTrue(np.allclose(true, approx, **TOLS))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                           Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate_scattered_field_in_lens(delta_index=0.1, size_parameter=0.1):
    miecalculator = mielensfunctions.MieLensCalculator(
        particle_kz=10, index_ratio=1.0 + delta_index,
        size_parameter=size_parameter, lens_angle=0.9)
    krho = np.linspace(0, 30, 300)
    # for phi, we pick an off-axis value where both polarizations are nonzero
    kphi = np.full_like(krho, 0.25 * np.pi)
    return miecalculator.calculate_scattered_field(krho, kphi)  # x,y component


def evaluate_intensity_at_rho_zero(kz, index_ratio=1.05):
    miecalculator = mielensfunctions.MieLensCalculator(
        particle_kz=kz, index_ratio=index_ratio, size_parameter=10,
        lens_angle=0.8)
    krho = np.array([0])
    return miecalculator.calculate_total_intensity(krho, 0 * krho)


def integrate_like_mielens(function, bounds):
    pts, wts = mielensfunctions.gauss_legendre_pts_wts(bounds[0], bounds[1])
    return (function(pts) * wts).sum()


class CheckEnergyIsConserved(object):
    """Tools for checking that the energy of the scattered beam which
    enters the entrance pupil is the same as the power incident on the
    detector.

    For the scattered beam, conservation of energy is equivalent to
    the statement that the integral
        .. math::

            \\int_0^\beta
                \\left[ |S_\\parallel|^2 + |S_\\perp|^2 \\right]
                \\sin \\theta \\, d\\theta

    is equal to the integral
        .. math::

            \\frac 1 2 \\int_0^\\inf
                \\left[ |I_0(k\\rho)|^2 + |I_2(k\\rho)|^2 \\right]
                (k\\rho) \\, d(k\\rho)

    """
    _npts = 1000

    def __init__(self, mielenscalculator):
        self.mielenscalculator = mielenscalculator

    def evaluate_scattered_power_incident_on_pupil(self):
        parallel = self.mielenscalculator._scat_prll_values.squeeze()
        perpendicular = self.mielenscalculator._scat_perp_values.squeeze()
        cos_theta = self.mielenscalculator._quad_pts.squeeze()
        wts = self.mielenscalculator._quad_wts.squeeze()
        integrand = np.abs(parallel)**2 + np.abs(perpendicular)**2
        # mielenscalculator uses x = cos(theta), so sin(theta) dtheta = dx
        return (integrand * wts).sum()

    def evaluate_scattered_power_incident_on_detector(self):
        # We use knowledge from high-order leggauss that quadratically
        # spaced points near boundaries are good to sample the integrand:
        t = np.linspace(0, 1, self._npts)
        # krhomax is the largest stable value, which we use as "infinity":
        krhomax = 3.9 * self.mielenscalculator.quad_npts
        krho = krhomax * (2 * t**2 / (1 + t))

        i0 = self.mielenscalculator._eval_mielens_i_n(krho, n=0)
        i2 = self.mielenscalculator._eval_mielens_i_n(krho, n=2)

        integrand = (np.abs(i0)**2 + np.abs(i2)**2) * krho
        return 0.5 * np.trapz(integrand, krho)

    def check_if_energy_is_conserved(self):
        ratio = self.get_ratio_of_powerin_to_powerout()
        return np.isclose(ratio, 1, atol=1e-3, rtol=0)

    def get_ratio_of_scattered_powerin_to_scattered_powerout(self):
        power_in = self.evaluate_scattered_power_incident_on_pupil()
        power_out = self.evaluate_scattered_power_incident_on_detector()
        return power_in / power_out


def get_ratio_of_scattered_powerin_to_scattered_powerout(**kwargs):
    mielenscalc = mielensfunctions.MieLensCalculator(**kwargs)
    checker = CheckEnergyIsConserved(mielenscalc)
    power_in = checker.evaluate_scattered_power_incident_on_pupil()
    power_out = checker.evaluate_scattered_power_incident_on_detector()
    return power_in / power_out


def make_calculator_with_aberration_of(spherical_aberration):
    calculator = mielensfunctions.AberratedMieLensCalculator(
        size_parameter=10,
        lens_angle=1.0,
        particle_kz=10.0,
        index_ratio=1.1,
        spherical_aberration=spherical_aberration)
    return calculator


if __name__ == '__main__':
    unittest.main()

