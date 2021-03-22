import unittest

import numpy as np
import xarray as xr
from nose.plugins.attrib import attr

from holopy.core import detector_grid, detector_points
from holopy.core.metadata import flat
from holopy.scattering.imageformation import ImageFormation
from holopy.scattering.theory import Mie
from holopy.scattering.scatterer import Sphere, Spheres, Ellipsoid
from holopy.scattering.errors import TheoryNotCompatibleError
from holopy.scattering.interface import prep_schema
from holopy.scattering.tests.common import xschema as XSCHEMA
from holopy.scattering.tests.test_scatteringtheory import (
    MockTheory, MockScatteringMatrixBasedTheory)


SPHERE = Sphere(n=1.5, r=1.0, center=(0, 0, 2))
SPHERES = Spheres([
    Sphere(n=1.5, r=1.0, center=(-1, -1, 2)),
    Sphere(n=1.5, r=1.0, center=(+1, +1, 2)),
    ])
ELLIPSOID = Ellipsoid(n=1.5, r=(0.5, 0.5, 1.2), center=(0, 0, 2))
TOLS = {'atol': 1e-14, 'rtol': 1e-14}
MEDTOLS = {'atol': 1e-7, 'rtol': 1e-7}
SCAT_SCHEMA = prep_schema(
    detector_grid(shape=(5, 5), spacing=.1),
    medium_index=1.33, illum_wavelen=0.66, illum_polarization=False)


class TestImageFormation(unittest.TestCase):
    @attr("fast")
    def test_calc_field_returns_xarray_of_correct_shape(self):
        imageformer = make_imageformer()
        fields = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        correct_shape = (XSCHEMA.values.size, 3)
        self.assertTrue(type(fields) is xr.DataArray)
        self.assertTrue(fields.shape == correct_shape)

    @attr("fast")
    def test_calc_field_keeps_same_attrs_as_input_schema(self):
        imageformer = make_imageformer()
        fields = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        old_attrs = XSCHEMA.attrs
        new_attrs = fields.attrs
        self.assertTrue(old_attrs == new_attrs)
        self.assertFalse(old_attrs is new_attrs)

    @attr("fast")
    def test_calc_field_keeps_same_coords_as_flattened_input_schema(self):
        imageformer = make_imageformer()
        fields = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        flat_schema = flat(XSCHEMA)
        self.assertTrue(np.all(flat_schema.x.shape == fields.x.shape))
        self.assertTrue(np.all(flat_schema.y.shape == fields.y.shape))
        self.assertTrue(np.all(flat_schema.z.shape == fields.z.shape))

        self.assertTrue(np.all(flat_schema.x.values == fields.x.values))
        self.assertTrue(np.all(flat_schema.y.values == fields.y.values))
        self.assertTrue(np.all(flat_schema.z.values == fields.z.values))

    @attr("medium")  # FIXME why is this slow?
    def test_calc_field_keeps_same_coords_as_flattened_input_for_spheres(self):
        imageformer = make_imageformer()
        fields = imageformer.calculate_scattered_field(SPHERES, XSCHEMA)
        flat_schema = flat(XSCHEMA)
        self.assertTrue(np.all(flat_schema.x.shape == fields.x.shape))
        self.assertTrue(np.all(flat_schema.y.shape == fields.y.shape))
        self.assertTrue(np.all(flat_schema.z.shape == fields.z.shape))

        self.assertTrue(np.all(flat_schema.x.values == fields.x.values))
        self.assertTrue(np.all(flat_schema.y.values == fields.y.values))
        self.assertTrue(np.all(flat_schema.z.values == fields.z.values))

    @attr("fast")
    def test_calc_field_has_correct_dims(self):
        imageformer = make_imageformer()
        fields = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        self.assertTrue(fields.dims == ('flat', 'vector'))

    @attr("fast")
    def test_calc_field_equals_calc_singlecolor_for_single_color(self):
        imageformer = make_imageformer()
        from_calc_scat = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        from_calc_single = imageformer._calculate_single_color_scattered_field(
            SPHERE, XSCHEMA)
        is_ok = np.allclose(
            from_calc_scat.values, from_calc_single.values, **TOLS)
        self.assertTrue(is_ok)

    @attr("medium")  # FIXME why is this slow?
    def test_calc_singlecolor_equals_get_field_from_for_sphere(self):
        imageformer = make_imageformer()
        from_calc_single = imageformer._calculate_single_color_scattered_field(
            SPHERE, XSCHEMA)
        from_get_field = imageformer._get_field_from(SPHERE, XSCHEMA)
        is_ok = np.allclose(from_get_field, from_calc_single, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_calc_singlecolor_raises_error_for_cant_handle(self):
        imageformer = make_imageformer()
        assert not imageformer.scattering_theory.can_handle(ELLIPSOID)
        self.assertRaises(
            TheoryNotCompatibleError,
            imageformer._calculate_single_color_scattered_field,
            ELLIPSOID, XSCHEMA)

    @attr("fast")
    def test_calc_singlecolor_adds_get_field_from_for_spheres(self):
        imageformer = make_imageformer()
        from_calc_single = imageformer._calculate_single_color_scattered_field(
            SPHERES, XSCHEMA)
        components = SPHERES.get_component_list()
        from_get_field = sum([
            imageformer._get_field_from(c, XSCHEMA) for c in components])

        is_ok = np.allclose(
            from_get_field, from_calc_single.values, **TOLS)
        self.assertTrue(is_ok)

    @attr("fast")
    def test_pack_field_into_xarray_returns_correct_dims(self):
        imageformer = make_imageformer()
        scattered_field = imageformer._get_field_from(SPHERE, XSCHEMA)
        packed_field = imageformer._pack_field_into_xarray(
            scattered_field, XSCHEMA)

        has_flat_or_point = packed_field.dims[0] in {'flat', 'point'}
        second_dimension_is_vector = packed_field.dims[1] == 'vector'

        self.assertTrue(has_flat_or_point and second_dimension_is_vector)

    @attr("fast")
    def test_pack_field_into_xarray_returns_correct_coords(self):
        imageformer = make_imageformer()
        scattered_field = imageformer._get_field_from(SPHERE, XSCHEMA)
        packed_field = imageformer._pack_field_into_xarray(
            scattered_field, XSCHEMA)

        x_ok = np.all(
            np.unique(packed_field.x.values) ==
            np.unique(XSCHEMA.x.values))
        y_ok = np.all(
            np.unique(packed_field.y.values) ==
            np.unique(XSCHEMA.y.values))
        z_ok = np.all(
            np.unique(packed_field.z.values) ==
            np.unique(XSCHEMA.z.values))
        self.assertTrue(x_ok and y_ok and z_ok)

    @attr("fast")
    def test_is_detector_view_point_or_flat_when_neither(self):
        imageformer = make_imageformer()
        self.assertRaises(
            ValueError,
            imageformer._is_detector_view_point_or_flat,
            XSCHEMA)

    @attr("fast")
    def test_is_detector_view_point_or_flat_when_flat(self):
        imageformer = make_imageformer()
        flattened = flat(XSCHEMA)
        point_or_flat = imageformer._is_detector_view_point_or_flat(flattened)
        self.assertTrue(point_or_flat == 'flat')

    @attr("fast")
    def test_is_detector_view_point_or_flat_when_point(self):
        imageformer = make_imageformer()
        point_view = detector_points(
            theta=3 * np.pi / 4,
            phi=np.arange(4) * np.pi / 2,
            r=np.sqrt(2))
        point_or_flat = imageformer._is_detector_view_point_or_flat(point_view)
        self.assertTrue(point_or_flat == 'point')

    @attr("fast")
    def test_calculate_scattering_matrix_has_correct_dims(self):
        imageformer = ImageFormation(Mie())
        scat_matrs = imageformer.calculate_scattering_matrix(
            SPHERE, SCAT_SCHEMA)
        self.assertTrue(scat_matrs.dims == ('flat', 'E_out', 'E_in'))

    @attr("fast")
    def test_calculate_scattering_matrix_has_correct_coords(self):
        imageformer = ImageFormation(Mie())
        scat_matrs = imageformer.calculate_scattering_matrix(
            SPHERE, SCAT_SCHEMA)
        expected = np.array(['parallel', 'perpendicular'])
        for dim in ['E_in', 'E_out']:
            with self.subTest(dim=dim):
                coords = scat_matrs.coords[dim].values
                self.assertTrue(np.all(coords == expected))

    @attr("fast")
    def test_calculate_scattering_matrix_has_correct_spherical_coords(self):
        imageformer = ImageFormation(Mie())
        scat_matrs = imageformer.calculate_scattering_matrix(
            SPHERE, SCAT_SCHEMA)
        true_r = np.array([
            2.        , 2.00249844, 2.00997512, 2.02237484, 2.03960781,
            2.00249844, 2.00499377, 2.01246118, 2.02484567, 2.04205779,
            2.00997512, 2.01246118, 2.01990099, 2.03224014, 2.04939015,
            2.02237484, 2.02484567, 2.03224014, 2.04450483, 2.06155281,
            2.03960781, 2.04205779, 2.04939015, 2.06155281, 2.07846097])
        true_theta = np.array([
            0.        , 0.0499584 , 0.09966865, 0.14888995, 0.19739556,
            0.0499584 , 0.07059318, 0.11134101, 0.15681569, 0.20330703,
            0.09966865, 0.11134101, 0.1404897 , 0.17836178, 0.21998798,
            0.14888995, 0.15681569, 0.17836178, 0.2090333 , 0.24497866,
            0.19739556, 0.20330703, 0.21998798, 0.24497866, 0.2756428 ])
        true_phi = np.array([
            0.        , 1.57079633, 1.57079633, 1.57079633, 1.57079633,
            0.        , 0.78539816, 1.10714872, 1.24904577, 1.32581766,
            0.        , 0.46364761, 0.78539816, 0.98279372, 1.10714872,
            0.        , 0.32175055, 0.5880026 , 0.78539816, 0.92729522,
            0.        , 0.24497866, 0.46364761, 0.64350111, 0.78539816])
        packed_r = scat_matrs.coords['r'].values
        packed_theta = scat_matrs.coords['theta'].values
        packed_phi = scat_matrs.coords['phi'].values
        self.assertTrue(np.allclose(true_r, packed_r, **MEDTOLS))
        self.assertTrue(np.allclose(true_theta, packed_theta, **MEDTOLS))
        self.assertTrue(np.allclose(true_phi, packed_phi, **MEDTOLS))

    @attr("fast")
    def test_calculate_scattering_matrix_has_correct_cartesian_coords(self):
        imageformer = ImageFormation(Mie())
        scat_matrs = imageformer.calculate_scattering_matrix(
            SPHERE, SCAT_SCHEMA)
        true_x = np.array([
            0. , 0. , 0. , 0. , 0. , 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2,
            0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4])
        true_y = np.array([
            0. , 0.1, 0.2, 0.3, 0.4, 0. , 0.1, 0.2, 0.3, 0.4, 0. , 0.1,
            0.2, 0.3, 0.4, 0. , 0.1, 0.2, 0.3, 0.4, 0. , 0.1, 0.2, 0.3, 0.4])
        true_z = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0])
        packed_x = scat_matrs.coords['x'].values
        packed_y = scat_matrs.coords['y'].values
        packed_z = scat_matrs.coords['z'].values
        self.assertTrue(np.allclose(true_x, packed_x, **MEDTOLS))
        self.assertTrue(np.allclose(true_y, packed_y, **MEDTOLS))
        self.assertTrue(np.allclose(true_z, packed_z, **MEDTOLS))

    @attr("medium")  # FIXME why is this slow?
    def test_scattering_matrix_pathway_returns_correct_type(self):
        imageformer = ImageFormation(MockScatteringMatrixBasedTheory())
        fields = imageformer.calculate_scattered_field(SPHERE, XSCHEMA)
        correct_shape = (XSCHEMA.values.size, 3)
        self.assertTrue(type(fields) is xr.DataArray)
        self.assertTrue(fields.shape == correct_shape)

    @attr("medium")  # FIXME why is this slow?
    def test_scattering_matrix_pathway_returns_linear_in_scatmatrs(self):
        imageformer = ImageFormation(MockScatteringMatrixBasedTheory())
        sphere01 = Sphere(n=1.5, r=1.0, center=(0, 0, 2))
        sphere02 = Sphere(n=1.5, r=2.0, center=(0, 0, 2))
        fields01 = imageformer.calculate_scattered_field(sphere01, XSCHEMA)
        fields02 = imageformer.calculate_scattered_field(sphere02, XSCHEMA)
        self.assertTrue(
            np.allclose(fields02.values, 2 * fields01.values, **TOLS))


class TestTransformToDesiredCoords(unittest.TestCase):
    @attr("fast")
    def test_transform_to_desired_coordinates(self):
        detector = detector_grid(shape=(2, 2), spacing=0.1)
        imageformer = ImageFormation(MockTheory())
        pos = imageformer._transform_to_desired_coordinates(
            detector, origin=(0, 0, 1), wavevec=2*np.pi*1.33/.66)
        true_pos = np.transpose([
            [12.66157039,   0.        ,   0.        ],
            [12.72472076,   0.09966865,   1.57079633],
            [12.72472076,   0.09966865,   0.        ],
            [12.78755927,   0.1404897 ,   0.78539816]])
        self.assertTrue(np.allclose(pos, true_pos))


def make_imageformer():
    return ImageFormation(MockTheory())


if __name__ == '__main__':
    unittest.main()
