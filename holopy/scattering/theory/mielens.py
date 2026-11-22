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
"""

.. moduleauthor:: Brian D. Leahy <bleahy@seas.harvard.edu>
.. moduleauthor:: Ron Alexander <ralexander@g.harvard.edu>
"""

import numpy as np

from holopy.scattering.scatterer import Sphere
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
from holopy.scattering.theory.mielensfunctions import (
    MieLensCalculator, AberratedMieLensCalculator)


class MieLens(ScatteringTheory):
    """
    Exact scattering from a sphere imaged through a perfect lens.

    Calculates holograms of spheres using an analytical solution of the
    Mie scattered field imaged by a perfect lens (see [Leahy2020]_). Can
    use superposition to calculate scattering from multiple spheres.

    See Also
    --------
    mielensfunctions.MieLensCalculator
    """

    desired_coordinate_system = 'cylindrical'
    parameter_names = ('lens_angle',)

    def __init__(self, lens_angle=1.0, calculator_accuracy_kwargs={}):
        """
        Parameters
        ----------
        lens_angle : float
            Lens acceptance angle in radians.
        calculator_accuracy_kwargs : dict
            While the analytical calculation is exact, the numerical
            implementation is approximate. Pass in any parameters to
            increase or decrease the accuracy via
            `calculator_accuracy_kwargs.` Valid keys are {`quad_npts`,
            `interpolate_integrals`, `interpolator_window_size`,
            `interpolator_degree`}, as explained in
            mielensfunctions.MieLensCalculator. The default number of
            quadrature points is chosen from the defocus and detector size;
            specify quad_npts to override the default.
        """
        super(MieLens, self).__init__()
        self.lens_angle = lens_angle
        self.calculator_accuracy_kwargs = calculator_accuracy_kwargs

    def can_handle(self, scatterer):
        return isinstance(scatterer, Sphere)

    def raw_fields(self, positions, scatterer, medium_wavevec, medium_index,
                    illum_polarization):
        """
        Parameters
        ----------
        positions : (3, N) numpy.ndarray
            The (k * rho, phi, z) coordinates, relative to the sphere,
            of the points to calculate the fields. Note that the radial
            coordinate is rescaled by the wavevector.
        scatterer : ``scatterer.Sphere`` object
        medium_wavevec : float
        medium_index : float
        illum_polarization : `xr.DataArray`
            The polarization vector of the incident field (z component, if any,
            is ignored).
        """
        index_ratio = scatterer.n / medium_index
        size_parameter = medium_wavevec * scatterer.r

        # phi_lab is the azimuthal coordinate in the lab frame
        rho, phi_lab, z = positions
        pol_angle = np.arctan2(
            illum_polarization.values[1], illum_polarization.values[0])
        # Because MieLensCalculator assumes the incident polarization is along
        # x, we compute each azimuthal angle relative to the polarization
        # (phi_pol) before doing the calculation
        phi_pol = (phi_lab - pol_angle) % (2 * np.pi)

        # FIXME mielens assumes that the detector points are at a fixed z!
        # right now I'm picking one z:
        particle_kz = np.mean(z)
        if np.ptp(z) > 1e-13 * (1 + np.abs(particle_kz)):
            msg = ("mielens currently assumes the detector is a fixed " +
                   "z from the particle")
            raise ValueError(msg)

        # Scale the number of quadrature points with the defocus and detector
        # size, since the integrand over the pupil oscillates with phase
        # kz (1 - cos theta) + krho sin theta. Expected relative error is
        # 1e-10 at one point per 3 radians.
        max_phase = (np.abs(particle_kz) * (1 - np.cos(self.lens_angle)) +
                     np.max(rho))
        quad_npts = max(100, int(np.ceil(max_phase / 3)))

        field_calculator = self._create_calculator(
            particle_kz=particle_kz,
            index_ratio=index_ratio,
            size_parameter=size_parameter,
            quad_npts=quad_npts,
            )

        fields_pll, fields_prp = field_calculator.calculate_scattered_field(
            rho, phi_pol)  # parallel and perp to the polarization

        # Now convert back to lab frame
        # Transfer from (parallel to, perpendicular to) polarization
        # to (x, y)
        parallel = np.array([np.cos(pol_angle), np.sin(pol_angle)])
        perpendicular = np.array([-np.sin(pol_angle), np.cos(pol_angle)])
        field_xyz = np.zeros([3, fields_pll.size], dtype='complex')
        for i in range(2):
            field_xyz[i, :] += fields_pll * parallel[i]
            field_xyz[i, :] += fields_prp * perpendicular[i]

        # Then we need to do 2 separate modifications to the fields.
        # First, in a lens, the incident field is Gouy phase shifted
        # to be E0 * -1, whereas in holopy the field is considered as
        # imaged without a phase shift. So we need to re-scale the
        # scattered field by 1 / the incident field.
        # Second, holopy considers the reference wave to have phase
        # e^{ikz}. For numerical reasons, in the mielens calculation I
        # consider the incident wave to have phase 1. So we need to fix
        # this by multiplying by e^{ikz}.
        # Combined, we multiply by e^{ikz} / incident_field[x-component]:
        incident_field_x, _ = field_calculator._calculate_incident_field()
        field_xyz *= np.exp(1j * particle_kz) / incident_field_x
        return field_xyz

    def _create_calculator(
            self, particle_kz=None, index_ratio=None, size_parameter=None,
            quad_npts=100):
        field_calculator = MieLensCalculator(
            particle_kz=particle_kz,
            index_ratio=index_ratio,
            size_parameter=size_parameter,
            lens_angle=self.lens_angle,
            # user-specified quad_npts will override auto-calculated one
            **{'quad_npts': quad_npts, **self.calculator_accuracy_kwargs})
        return field_calculator


class AberratedMieLens(MieLens):
    parameter_names = ('lens_angle', 'spherical_aberration')

    def __init__(self, spherical_aberration=0.0, lens_angle=1.0,
                 calculator_accuracy_kwargs={}):
        """
        Parameters
        ----------
        spherical_aberration : float or array-like of floats
            The spherical aberration, up to arbitrary order. If a float,
            just the coefficient of the 3rd-order aberration (4th-order
            in wavefront). When an array, the coefficients of
            aberrations in ascending order (3rd, 5th, 7th, etc), where
            the wavefront distortion for the nth-order aberration is of
            the form (cos(theta) - 1)^(n+1), where n = 3, 5, 7, etc
            Default is None, which raises an error.
        lens_angle : float
            Lens acceptance angle in radians.
        calculator_accuracy_kwargs : dict
            While the analytical calculation is exact, the numerical
            implementation is approximate. Pass in any parameters to
            increase or decrease the accuracy via
            `calculator_accuracy_kwargs.` Valid keys are {`quad_npts`,
            `interpolate_integrals`, `interpolator_window_size`,
            `interpolator_degree`}, as explained in
            mielensfunctions.MieLensCalculator. The default number of
            quadrature points is chosen from the defocus and detector size;
            specify quad_npts to override the default.
        """
        super(AberratedMieLens, self).__init__(
            lens_angle=lens_angle,
            calculator_accuracy_kwargs=calculator_accuracy_kwargs)
        self.spherical_aberration = spherical_aberration

    def _create_calculator(
            self, particle_kz=None, index_ratio=None, size_parameter=None,
            quad_npts=100):
        field_calculator = AberratedMieLensCalculator(
            particle_kz=particle_kz,
            index_ratio=index_ratio,
            size_parameter=size_parameter,
            lens_angle=self.lens_angle,
            spherical_aberration=self.spherical_aberration,
            # user-specified quad_npts will override auto-calculated one
            **{'quad_npts': quad_npts, **self.calculator_accuracy_kwargs})
        return field_calculator
