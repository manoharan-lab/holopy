! Copyright 2013, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
! W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
!
! This file is part of HoloPy.
!
! HoloPy is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! HoloPy is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
!
! mieangfuncs.f90
!
! Author: Jerome Fung
!
! Description:
! Library of routines for calculating model holograms of single spheres and clusters.
!
! At compile this must be linked with uts_scsmfo.for
! Functions are designed to be compiled with f2py and called from Python.

      subroutine calc_scat_field(kr, phi, ascatm, einc, escat_sph)
        ! Do the matrix multiplication to calculate scattered field
        ! from incident polarization and amplitude scattering matrix.
        ! Parameters
        ! ----------
        ! kr: real
        !     dimensionless distance from scatterer to field point
        ! phi: real
        !     azimuthal spherical coordinate to field point
        ! ascatm: complex array (2, 2)
        !     Amplitude scattering matrix
        ! einc: real array(2)
        !     Incident polarization vector
        !
        ! Returns
        ! -------
        ! escat_sph: complex array (2)
        !     Theta and phi spherical components of scattered field
        implicit none
        real (kind = 8), intent(in) :: kr, phi
        real (kind = 8), intent(in), dimension(2) :: einc
        complex (kind = 8), intent(in), dimension(2, 2) :: ascatm
        complex (kind = 8), intent(out), dimension(2) :: escat_sph
        real (kind = 8), dimension(2) :: signarr, einc_sph
        complex (kind = 8) :: prefactor, ci
        data ci/(0.d0, 1.d0)/

        ! convert polarization to spherical coords
        call incfield(einc(1), einc(2), phi, einc_sph)
        prefactor = ci / kr * exp(ci * kr) ! Bohren & Huffman formalism
        signarr = (/ 1.0, -1.0 /) ! accounts for escatperp = -escatphi
        escat_sph = prefactor * matmul(ascatm, einc_sph) * signarr

        return
        end


      subroutine mie_fields(n_pts, calc_points, asbs, nstop, einc, rad, &
           es_x, es_y, es_z)
        ! Calculate fields scattered by a sphere in the Lorenz-Mie solution,
        ! at a list of selected points.  Use for hologram calculations or
        ! general scattering.
        !
        ! Function calls from Python may need to transpose calc_points.
        !
        ! Parameters
        ! ----------
        ! calc_points: array (3 x n_pts)
        !     Array of points over which scattered field is calculated. Points
        !     should be in spherical coordinates relative to scatterer:
        !     non-dimensional radial coordinate (kr), theta and phi.
        ! asbs: complex array (2, nstop)
        !     Mie coefficients from miescatlib.scatcoeffs(m_p, x_p, nstop)
        ! nstop: int
        !     Expansion order (from miescatlib.nstop(x_p))
        ! einc: real array (2)
        !     polarization (from optics.polarization)
        ! rad: logical
        !     If .true., calculate radial component of the scattered field.
        !     Neglected in most scattering calculations b/c radial component
        !     falls off faster than 1/r.
        !
        ! Returns
        ! -------
        ! es_x, es_y, es_z: complex array (n_pts)
        !     The three electric field components at points in calc_points
        implicit none
        integer, intent(in) :: n_pts, nstop
        real (kind = 8), intent(in), dimension(3, n_pts) :: calc_points
        logical, intent(in) :: rad
        complex (kind = 8), intent(out), dimension(n_pts) :: es_x, &
             es_y, es_z
        complex (kind = 8), intent(in), dimension(2, nstop) :: asbs
        real (kind = 8), intent(in), dimension(2) :: einc ! polarization
        real (kind = 8) :: kr, theta, phi
        real (kind = 8), dimension(2) :: einc_sph
        complex (kind = 8), dimension(2,2) :: asm_scat
        complex (kind = 8), dimension(2) :: escat_sph
        complex (kind = 8), dimension(3) :: escat_rect, erad_cart
        complex (kind = 8) :: escat_rad
        integer :: i

        ! Main loop over field points.
        do i = 1, n_pts, 1
           kr = calc_points(1, i)
           theta = calc_points(2, i)
           phi = calc_points(3, i)

           ! calculate the amplitude scattering matrix
           call asm_mie_fullradial(nstop, asbs, calc_points(:, i), asm_scat)

           ! calculate scattered fields in spherical coordinates
           call calc_scat_field(kr, phi, asm_scat, einc, escat_sph)

           ! convert to rectangular
           call fieldstocart(escat_sph, theta, phi, escat_rect)

           ! calculate radial components of scattered field
           if (rad) then
               call incfield(einc(1), einc(2), phi, einc_sph)
               call radial_field_mie(nstop, asbs(1, :), kr, theta, escat_rad)
               escat_rad = escat_rad * einc_sph(1)
               call radial_vect_to_cart(escat_rad, theta, phi, erad_cart)
               escat_rect = escat_rect + erad_cart
           endif

           es_x(i) = escat_rect(1)
           es_y(i) = escat_rect(2)
           es_z(i) = escat_rect(3)

        end do

        return
        end


      subroutine mie_internal_fields(n_pts, calc_points, m, csds, nstop, &
           einc, eint_x, eint_y, eint_z)
        ! Calculate internal fields inside a sphere in the Lorenz-Mie solution,
        ! at a list of selected points.  
        !
        ! Function calls from Python may need to transpose calc_points.
        !
        ! Parameters
        ! ----------
        ! calc_points: array (3 x n_pts)
        !     Array of points over which internal field is calculated. Points
        !     should be in spherical coordinates relative to scatterer:
        !     non-dimensional radial coordinate (kr), theta and phi.
        !     They should also truly be inside the particle (no check here).
        ! m: complex
        !     Relative index of particle (n_particle/n_medium)
        ! csds: complex array (2, nstop)
        !     Mie internal coefficients from 
        !     miescatlib.internal_coeffs(m_p, x_p, nstop)
        ! nstop: int
        !     Expansion order (from miescatlib.nstop(x_p))
        ! einc: real array (2)
        !     polarization (from optics.polarization)
        !
        ! Returns
        ! -------
        ! eint_x, eint_y, eint_z: complex array (n_pts)
        !     The three electric field components at points in calc_points
        implicit none
        integer, intent(in) :: n_pts, nstop
        complex (kind = 8), intent(in) :: m
        real (kind = 8), intent(in), dimension(3, n_pts) :: calc_points
        complex (kind = 8), intent(in), dimension(2, nstop) :: csds
        real (kind = 8), intent(in), dimension(2) :: einc ! polarization
        real (kind = 8) :: theta, phi
        complex (kind = 8) :: mkr
        real (kind = 8), dimension(2) :: einc_sph
        complex (kind = 8), intent(out), dimension(n_pts) :: eint_x, &
             eint_y, eint_z
        integer :: i
        complex (kind = 8), dimension(3) :: eint_sph, eint_cart1, eint_cart2
        
        ! Loop over field points
        do i = 1, n_pts, 1
           mkr = m * calc_points(1, i)
           theta = calc_points(2, i)
           phi = calc_points(3, i)

           ! get incident field in spherical
           call incfield(einc(1), einc(2), phi, einc_sph)

           ! calculate the field amplitudes
           call mie_int_point(nstop, csds, mkr, theta, eint_sph)
           eint_sph(1) = eint_sph(1) * einc_sph(1)
           eint_sph(2) = eint_sph(2) * einc_sph(1)
           eint_sph(3) = eint_sph(3) * einc_sph(2)

           ! convert to rectangular
           call fieldstocart(eint_sph(2:3), theta, phi, eint_cart1)
           call radial_vect_to_cart(eint_sph(1), theta, phi, eint_cart2)
           eint_x(i) = eint_cart1(1) + eint_cart2(1)
           eint_y(i) = eint_cart1(2) + eint_cart2(2)
           eint_z(i) = eint_cart1(3) + eint_cart2(3)
        end do
      
        return
        end


      subroutine tmatrix_fields(n_pts, calc_points, amn, lmax, euler_gamma, &
           inc_pol, rad, es_x, es_y, es_z)
        ! Calculate fields scattered by a cluster of spheres using
        ! D. Mackowski's code SCSMFO.
        !
        ! Function calls from Python may need to transpose calc_points.
        !
        ! Parameters
        ! ----------
        ! calc_points: array (3 x n_pts)
        !     Array of points over which scattered field is calculated. Points
        !     should be in spherical coordinates relative to scatterer COM:
        !     non-dimensional radial coordinate (kr), theta and phi.
        ! amn: complex array (2, lmax * (lmax+2), 2) complex
        !     Scattered field expansion coefficients calculated by
        !     scsmfo_min.amncalc(), stripped
        ! lmax: int
        !     Maximum order of scattered field expansion
        ! euler_gamma: real
        !     Euler angle gamma to rotate cluster frame into laboratory frame.
        ! inc_pol: real array (2)
        !     polarization (from optics.polarization)
        ! rad: logical
        !     If .true., calculate radial component of the scattered field.
        !     Neglected in most scattering calculations b/c radial component
        !     falls off faster than 1/r.
        !
        ! Returns
        ! -------
        ! es_x, es_y, es_z: complex array (n_pts)
        !     The three electric field components at points in calc_points

        implicit none
        integer, intent(in) :: n_pts, lmax
        real (kind = 8), intent(in), dimension(3, n_pts) :: calc_points
        complex (kind = 8), intent(in), dimension(2,lmax*(lmax+2),2) :: amn
        real (kind = 8), intent(in) :: euler_gamma
        real (kind = 8), intent(in), dimension(2) :: inc_pol
        logical, intent(in) :: rad
        complex (kind = 8), intent(out), dimension(n_pts) :: es_x, &
             es_y, es_z
        complex (kind = 8), dimension(4) :: ascatmat
        complex (kind = 8), dimension(2,2) :: asreshape
        real (kind = 8) :: kr, theta, phi
        real (kind = 8), dimension(2) :: einc_sph
        complex (kind = 8), dimension(3) :: escat_rect, erad_cart
        complex (kind = 8), dimension(2) :: escat_sph, rad_amplitude
        complex (kind = 8) :: escat_rad
        integer :: i

        ! Main loop over hologram points
        do i = 1, n_pts, 1
           kr = calc_points(1, i)
           theta = calc_points(2, i)
           phi = calc_points(3, i)

           ! calculate amplitude scattering matrix from amn coefficients
           ! subroutine asmfr is in uts_scsmfo.for
           call asmfr(amn, lmax, theta, phi + euler_gamma, kr, ascatmat)
           ! fudge factor of -0.5 for agreement with single sphere case
           asreshape = reshape(cshift(ascatmat, shift = 1), (/ 2, 2 /), &
                order = (/ 2, 1 /)) * (-0.5)

           ! calculate scattered fields in spherical coordinates
           call calc_scat_field(kr, phi, asreshape, inc_pol, escat_sph)

           ! convert to rectangular
           call fieldstocart(escat_sph, theta, phi, escat_rect)

           ! calculate radial components of scattered field
           if (rad) then
               call incfield(inc_pol(1), inc_pol(2), phi, einc_sph)
               call ms_radial_fields(amn, lmax, theta, phi + euler_gamma, &
                    kr, rad_amplitude)
               ! order in dot product matters b/c of complex conjugate
               ! again, fudge factor of -0.5 for single sphere agreement
               escat_rad = dot_product(einc_sph, rad_amplitude) * (-0.5)
               call radial_vect_to_cart(escat_rad, theta, phi, erad_cart)
               escat_rect = escat_rect + erad_cart
           endif

           es_x(i) = escat_rect(1)
           es_y(i) = escat_rect(2)
           es_z(i) = escat_rect(3)
        end do

        return
        end


      subroutine mie_int_point(nstop, csds, mkr, theta, esph_out)
        ! calculate summations for internal field (analogous to per-point asm)
        ! multiply output by E_par,i or E_perp,i to get actual field.
        implicit none
        integer, intent(in) :: nstop
        real (kind = 8), intent(in) :: theta
        complex (kind = 8), intent(in) :: mkr
        complex (kind = 8), dimension(2, nstop), intent(in) :: csds
        complex (kind = 8), dimension(3), intent(out) :: esph_out
        complex (kind = 8) :: ci, derj
        complex (kind = 8), dimension(0:nstop) :: jl, djl, yl, dyl
        real (kind = 8), dimension(nstop) :: pi_l, tau_l
        real (kind = 8) :: st, pref_up, pref_dn
        integer :: n, nmx_csphjy

        ! initialize
        data ci/(0.d0, 1.d0)/
        esph_out = (/ 0., 0., 0. /)
        
        ! special function calls
        call pisandtaus(nstop, theta, pi_l, tau_l)
        call csphjy(nstop, mkr, nmx_csphjy, jl, djl, yl, dyl)
        st = dsin(theta)

        do n = 1, nstop, 1
           pref_up = 2. * n + 1.
           pref_dn = n * (n + 1.)
           derj = jl(n) / mkr + djl(n) 
           ! radial
           esph_out(1) = esph_out(1) + ci**(n-1) * pref_up * csds(2, n) * & 
                st * pi_l(n) * jl(n) / mkr
           ! theta
           esph_out(2) = esph_out(2) + ci**n * pref_up / pref_dn * &
                (-ci*csds(2,n)*tau_l(n)*derj + csds(1,n)*pi_l(n)*jl(n))
           ! phi
           esph_out(3) = esph_out(3) + ci**n * pref_up / pref_dn * &
                (ci*csds(2,n)*pi_l(n)*derj - csds(1,n)*tau_l(n)*jl(n))
        end do
        return
        end

      subroutine asm_mie_fullradial(nstop, asbs, sphcoords, asm_out)
        ! perform summations to calculate amplitude scattering matrix
        ! with full radial dependence, compatible with B/H formalism
        implicit none
        integer, intent(in) :: nstop
        real (kind = 8), intent(in), dimension(3) :: sphcoords
        complex (kind = 8), dimension(2, nstop), intent(in) :: asbs
        complex (kind = 8), dimension(2, 2), intent(out) :: asm_out
        complex (kind = 8) :: ci, hl, dhl, inv_pref
        data ci/(0.d0, 1.d0)/
        complex (kind = 8), dimension(4) :: asm
        integer :: n, ifail
        real (kind = 8) :: prefactor, kr, theta
        real (kind = 8), dimension(nstop) :: pi_n, tau_n
        real (kind = 8), dimension(0:nstop) :: jn, djn, yn, dyn

        ! initialize
        asm = (/0., 0., 0., 0. /)
        kr = sphcoords(1)
        theta = sphcoords(2)
        inv_pref = cdexp(-1*ci*kr)*kr

        ! compute special functions (angular and spherical bessel)
        call pisandtaus(nstop, theta, pi_n, tau_n)
        call sbesjy(kr, nstop, jn, yn, djn, dyn, ifail)

        ! main loop
        do n = 1, nstop, 1
           prefactor = (2.*n + 1.) / (n * (n + 1.))
           hl = jn(n) + ci*yn(n) ! spherical hankel
           dhl = hl/kr + djn(n) + ci*dyn(n)
           asm(1) = asm(1) + prefactor * ci**n * ( &
                asbs(1,n)*pi_n(n)*dhl + ci*asbs(2,n)*tau_n(n)*hl)
           asm(2) = asm(2) + prefactor * ci**n * ( &
                asbs(1,n)*tau_n(n)*dhl + ci*asbs(2,n)*pi_n(n)*hl)
        end do

        ! apply inverse prefactor so B/H far field formalism can be used
        asm = asm * inv_pref

        asm_out = reshape(cshift(asm, shift = 1), (/ 2, 2 /), &
             order = (/ 2, 1 /))

        return
        end


      subroutine asm_mie_far(nstop, asbs, theta, asm_out)
        ! Calculate amplitude scattering matrix for a spherically
        ! symmetric scatterer, from scattering coefficients calculated
        ! from the Mie solution or its extension to layered spheres,
        ! in the far field limit.
        !
        ! Inputs:
        ! =======
        ! nstop (int):
        !     Maximum order of vector spherical harmonic expansion
        ! asbs (complex, (2, nstop)
        !     Scattering coefficients
        ! theta (real)
        !     Spherical coordinate theta (radians).
        !     In the Mie solution the amplitude scattering matrix is
        !     independent of phi.
        !
        ! Outputs:
        ! ========
        ! asm_out (complex, (2,2))
        !     Amplitude scattering matrix in standard (Bohren & Huffman) form
        implicit none
        integer, intent(in) :: nstop
        real (kind = 8), intent(in) :: theta
        complex (kind = 8), dimension(2, nstop), intent(in) :: asbs
        complex (kind = 8), dimension(2, 2), intent(out) :: asm_out
        complex (kind = 8), dimension(4) :: asm
        integer :: n
        real (kind = 8) :: prefactor
        real (kind = 8), dimension(nstop) :: pi_n, tau_n

        ! initialize
        asm = (/0., 0., 0., 0. /)

        ! compute angular special functions
        call pisandtaus(nstop, theta, pi_n, tau_n)

        ! main loop
        do n = 1, nstop, 1
           prefactor = (2.*n + 1.) / (n * (n + 1.))
           asm(1) = asm(1) + prefactor * ( &
                asbs(1,n) * pi_n(n) + asbs(2,n) * tau_n(n))
           asm(2) = asm(2) + prefactor * ( &
                asbs(1,n) * tau_n(n) + asbs(2,n) * pi_n(n))
        end do

        ! reshape into 2 x 2 form. Only diagonal elts are nonzero.
        asm_out = reshape(cshift(asm, shift = 1), (/ 2, 2 /), &
             order = (/ 2, 1 /))

        return
        end

     
      subroutine radial_field_mie(nstop, as, kr, theta, erad_nd)
        ! Calculate non-dimensional radial component of the scattered 
        ! Lorenz-Mie electric field. Physical E_scat,radial requires
        ! an overall prefactor of E_\parallel,inc (incident field parallel
        ! to scattering plane).
        !
        ! Inputs
        ! ======
        ! nstop (int) :
        !     Maximum order of electric field expansion.
        ! as (complex, (1, nstop) :
        !     Scattering coefficients a_l (B/H convention), to order nstop.
        ! kr (real) :
        !     Dimensionless spherical coordinate r (multiply by wavevector in
        !     medium).
        ! theta (real) :
        !     Spherical coordinate theta (radians).
        !
        ! Outputs
        ! =======
        ! erad_nd (complex) :
        !     Dimensionless radial component of scattered electric field.
        !
        ! Notes
        ! =====
        ! See Bohren & Huffman 94-95. The only VSHs with an r component
        ! are the N_e1l, whose contributions to the scattered E field
        ! are weighted by the a_l Mie coefficients. Note that N_e1l
        ! goes as E_0 cos phi, but this is derived in a formalism assuming
        ! x polarization, and hence the prefactor should physically be
        ! E_\parallel,inc
        !
        ! Code below uses n instead of l for angular momentum quantum
        ! number because it's easier to read.
        !
        ! TODO: for performance could be folded into asm_mie_fullradial,
        ! b/c the special function calls are redundant.
        !
        implicit none
        integer, intent(in) :: nstop
        real (kind = 8), intent(in) :: kr, theta
        complex (kind = 8), dimension(1, nstop), intent(in) :: as
        complex (kind = 8), intent(out) :: erad_nd

        real (kind = 8), dimension(nstop) :: pi_n, tau_n
        real (kind = 8), dimension(0:nstop) :: jn, djn, yn, dyn
        real (kind = 8) :: st
        integer :: n, ifail, prefactor
        complex (kind = 8) :: ci, hl
        data ci/(0.d0, 1.d0)/

        ! initialize output
        erad_nd = 0.

        ! compute special functions (angular and spherical bessel)
        call pisandtaus(nstop, theta, pi_n, tau_n)
        call sbesjy(kr, nstop, jn, yn, djn, dyn, ifail)
        st = dsin(theta)

        ! main loop
        do n = 1, nstop, 1
           prefactor = 2 * n + 1 
           hl = jn(n) + ci * yn(n) ! spherical hankel
           erad_nd = erad_nd + as(1, n) * prefactor * ci**(n + 1) * &
                st * pi_n(n) * hl / kr
        end do

        return
        end


      subroutine fieldstocart(asph, theta, phi, acart)
! Complex routine to convert scattered fields from scat. plane spherical to cartesian.
! asph(1) is theta component, asph(2) is phi component
        real*8 theta, phi, ct, st, cp, sp
        complex*16 asph(2), acart(3)
!f2py   intent(in) asph
!f2py   intent(in) theta
!f2py   intent(in) phi
!f2py   intent(out) acart
        ct = dcos(theta)
        st = dsin(theta)
        cp = dcos(phi)
        sp = dsin(phi)

        acart(1) = ct*cp*asph(1) - sp*asph(2)
        acart(2) = ct*sp*asph(1) + cp*asph(2)
        acart(3) = -1.d0*st*asph(1)

        return
        end


      subroutine radial_vect_to_cart(a_r, theta, phi, acart)
        ! Convert a radial vector component to a Cartesian vector.
        implicit none
        complex (kind = 8), intent(in) :: a_r
        real (kind = 8), intent(in) :: theta, phi
        complex (kind = 8), dimension(3), intent(out) :: acart
        real (kind = 8) :: ct, st, cp, sp

        ct = dcos(theta)
        st = dsin(theta)
        cp = dcos(phi)
        sp = dsin(phi)
        
        acart(1) = st * cp * a_r
        acart(2) = st * sp * a_r
        acart(3) = ct * a_r

        return
        end


! Currently unused. Candidate for deletion?
      subroutine getsphercoords(x, y, z, sph)
! conversion of cartesian to spherical coordinates
! apparently it's ok to have integers as exponents of real numbers
! otherwise the fortran rule of thumb is not to mix types in arithmetic
        real*8 x, y, z, gamma, pi, sph(3)
!f2py intent(in) x
!f2py intent(in) y
!f2py intent(in) z
!f2py intent(out) sph
        pi = 2.d0*dacos(0.d0)

        sph(1) = dsqrt(x**2 + y**2 + z**2)
        sph(2) = datan2(dsqrt(x**2 + y**2), z)
        gamma = datan2(y, x)
        if (gamma.lt.0.d0) then
           gamma = gamma + 2.d0*pi
        end if
        sph(3) = gamma

        return
        end


        subroutine incfield(ex, ey, phi, eincorigin)
! replacement of HoloSimClassMie2.IncField
! Convert rectangular incident field to components relative to scattering plane.
! for now, polarization should be real (linear rather than elliptical)
          real*8 ex, ey, phi, eincorigin(2)
!f2py intent(in) ex
!f2py intent(in) ey
!f2py intent(in) phi
!f2py intent(out) eincorigin
          eincorigin(1) = ex*dcos(phi) + ey*dsin(phi)
          eincorigin(2) = ex*dsin(phi) - ey*dcos(phi)

          return
          end


      subroutine pisandtaus(n, theta, pisout, taus)
! Calculate pi_n and tau angular functions at theta out to order n by up recursion
! returns pis (order 1 to n), taus
! Inputs:
!    n: maximum order
!    theta: angle
! Outputs:
!    pisout: pi_n from n = 1 to n
!    taus: tau_n from n = 1 to n
      integer n, cnt
      real*8 theta, mu
      real*8 pis(0:n), pisout(1:n), taus(1:n)
!f2py intent(in) n
!f2py intent(in) theta
!f2py intent(out) pisout
!f2py intent(out) taus
!f2py depend(n) pis, pisout, taus

      mu = dcos(theta)

      pis(0) = 0.
      pis(1) = 1.
      taus(1) = mu

      do cnt = 2, n, 1
         pis(cnt)=(2.d0*dble(cnt)-1.d0)/(dble(cnt)-1.d0)*mu*pis(cnt-1)-&
              (dble(cnt))/(dble(cnt) - 1.d0)*pis(cnt-2)
         taus(cnt) = dble(cnt)*mu*pis(cnt) - (dble(cnt)+1.d0)*pis(cnt-1)
      end do

      pisout =  pis(1:n)
      return
      end
