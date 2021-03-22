import numpy as np
import xarray as xr

from holopy.core.math import find_transformation_function
from holopy.core.holopy_object import HoloPyObject
from holopy.scattering.scatterer import Scatterers
from holopy.scattering.errors import TheoryNotCompatibleError, MissingParameter
from holopy.core.metadata import (
    vector, illumination, flat, update_metadata, clean_concat)
from holopy.core.utils import ensure_array


class ImageFormation(HoloPyObject):
    """
    Calculates fields, holograms, intensities, etc.
    """
    def __init__(self, scattering_theory):
        self.scattering_theory = scattering_theory

    def calculate_scattered_field(self, scatterer, schema):
        """
        Parameters
        ----------
        scatterer : :mod:`.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        e_field : :mod:`.VectorGrid`
            scattered electric field
        """
        if scatterer.center is None:
            raise MissingParameter("center")
        is_multicolor_hologram = len(ensure_array(schema.illum_wavelen)) > 1
        field = (
            self._calculate_multiple_color_scattered_field(scatterer, schema)
            if is_multicolor_hologram else
            self._calculate_single_color_scattered_field(scatterer, schema))
        return field

    def calculate_cross_sections(
            self, scatterer, medium_wavevec, medium_index, illum_polarization):
        raw_sections = self.scattering_theory.raw_cross_sections(
            scatterer=scatterer, medium_wavevec=medium_wavevec,
            medium_index=medium_index, illum_polarization=illum_polarization)
        coords = {'cross_section': ['scattering', 'absorbtion',
                                    'extinction', 'assymetry']}
        dims = ['cross_section']
        return xr.DataArray(raw_sections, dims=dims, coords=coords)

    def calculate_scattering_matrix(self, scatterer, schema):
        """
        Compute scattering matrices for scatterer

        Parameters
        ----------
        scatterer : :mod:`holopy.scattering.scatterer` object
            (possibly composite) scatterer for which to compute scattering

        Returns
        -------
        scat_matr : :mod:`.Marray`
            Scattering matrices at specified positions
        """
        positions = self._transform_to_desired_coordinates(
            schema, scatterer.center)
        scat_matrs = self.scattering_theory.raw_scat_matrs(
            scatterer, positions, medium_wavevec=get_wavevec_from(schema),
            medium_index=schema.medium_index)
        return self._pack_scattering_matrix_into_xarray(
            scat_matrs, positions, schema)

    def _calculate_multiple_color_scattered_field(self, scatterer, schema):
        field = []
        for illum in schema.illum_wavelen.illumination.values:
            this_schema = update_metadata(
                schema,
                illum_wavelen=ensure_array(
                    schema.illum_wavelen.sel(illumination=illum).values)[0],
                illum_polarization=ensure_array(
                    schema.illum_polarization.sel(illumination=illum).values))
            this_scatterer = select_scatterer_by_illumination(scatterer, illum)
            this_field = self._calculate_single_color_scattered_field(
                this_scatterer, this_schema)
            field.append(this_field)
        field = clean_concat(field, dim=schema.illum_wavelen.illumination)
        return field

    def _calculate_scattered_field_from_superposition(
            self, scatterers, schema):
        field = self._calculate_single_color_scattered_field(
            scatterers[0], schema)
        for s in scatterers[1:]:
            field += self._calculate_single_color_scattered_field(s, schema)
        return field

    def _calculate_single_color_scattered_field(self, scatterer, schema):
        if self.scattering_theory.can_handle(scatterer):
            field = self._get_field_from(scatterer, schema)
        elif isinstance(scatterer, Scatterers):
            field = self._calculate_scattered_field_from_superposition(
                scatterer.get_component_list(), schema)
        else:
            raise TheoryNotCompatibleError(self.scattering_theory, scatterer)
        return self._pack_field_into_xarray(field, schema)

    def _get_field_from(self, scatterer, schema):
        """
        Parameters
        ----------
        scatterer
        schema : xarray
            (it's always passed in as an xarray)

        Returns
        -------
        raveled fields, shape (npoints = nx*ny = schema.shape.prod(), 3)
        """
        wavevector = get_wavevec_from(schema)
        positions = self._transform_to_desired_coordinates(
            schema, scatterer.center, wavevec=wavevector)
        scattered_field = np.transpose(
            self.scattering_theory.raw_fields(
                positions,
                scatterer,
                medium_wavevec=wavevector,
                medium_index=schema.medium_index,
                illum_polarization=schema.illum_polarization)
            )
        phase = np.exp(-1j * wavevector * scatterer.center[2])
        scattered_field *= phase
        return scattered_field

    def _pack_field_into_xarray(self, scattered_field, schema):
        """Packs the numpy.ndarray, shape (N, 3) ``scattered_field`` into
        an xr.DataArray, shape (N, 3). This function needs to pack the
        fields [flat or point, vector], with the coordinates the
        same as that of the schema."""
        flattened_schema = flat(schema)  # now either point or flat
        point_or_flat = self._is_detector_view_point_or_flat(flattened_schema)
        coords = {
            key: (point_or_flat, val.values)
            for key, val in flattened_schema[point_or_flat].coords.items()}

        coords.update(
            {point_or_flat: flattened_schema[point_or_flat],
             vector: ['x', 'y', 'z']})
        scattered_field = xr.DataArray(
            scattered_field, dims=[point_or_flat, vector], coords=coords,
            attrs=schema.attrs)
        return scattered_field

    def _pack_scattering_matrix_into_xarray(
            self, scat_matrs, r_theta_phi, schema):
        flattened_schema = flat(schema)
        point_or_flat = self._is_detector_view_point_or_flat(flattened_schema)
        dims = [point_or_flat, 'E_out', 'E_in']

        coords = {point_or_flat: flattened_schema.coords[point_or_flat]}
        coords.update({
            'r': (point_or_flat, r_theta_phi[0]),
            'theta': (point_or_flat, r_theta_phi[1]),
            'phi': (point_or_flat, r_theta_phi[2]),
            'E_out': ['parallel', 'perpendicular'],
            'E_in': ['parallel', 'perpendicular'],
            })
        # Bohren and Huffman (1998) define the following:
        # S1: E_out perpendicular, E_in perpendicular
        # S2: E_out parallel,      E_in parallel
        # S3: E_out paralell,      E_in perpendicular
        # S4: E_out perpendicular, E_in parallel
        packed = xr.DataArray(
            scat_matrs, dims=dims, coords=coords, attrs=schema.attrs)
        return packed

    @classmethod
    def _is_detector_view_point_or_flat(cls, detector_view):
        detector_dims = detector_view.dims
        if 'flat' in detector_dims:
            point_or_flat = 'flat'
        elif 'point' in detector_dims:
            point_or_flat = 'point'
        else:
            msg = ("xarray `detector_view` is not in the form of a 1D list " +
                   "of coordinates. Call ``flat`` first.")
            raise ValueError(msg)
        return point_or_flat

    def _transform_to_desired_coordinates(self, detector, origin, wavevec=1):
        if hasattr(detector, 'theta') and hasattr(detector, 'phi'):
            original_coordinate_system = 'spherical'
            original_coordinate_values = [
                (detector.r.values * wavevec if hasattr(detector, 'r')
                    else np.full(detector.theta.values.shape, np.inf)),
                detector.theta.values,
                detector.phi.values,
                ]
        else:
            original_coordinate_system = 'cartesian'
            f = flat(detector)  # 1.6 ms
            original_coordinate_values = [
                wavevec * (f.x.values - origin[0]),
                wavevec * (f.y.values - origin[1]),
                wavevec * (origin[2] - f.z.values),
                # z is defined opposite light propagation, so we invert
                ]
        method = find_transformation_function(
            original_coordinate_system,
            self.scattering_theory.desired_coordinate_system)
        return method(original_coordinate_values)


def select_scatterer_by_illumination(scatterer, illum):
    select_parameters = {}
    for key, val in scatterer.parameters.items():
        selected_val = val
        if isinstance(val, dict) and illum in val.keys():
            selected_val = val[illum]
        elif isinstance(val, xr.DataArray):
            try:
                selected_val = val.sel(illumination=illum).values
            except (KeyError, ValueError):
                pass
        select_parameters[key] = selected_val
    return scatterer.from_parameters(select_parameters)


def get_wavevec_from(schema):
    return 2 * np.pi / (schema.illum_wavelen / schema.medium_index)
