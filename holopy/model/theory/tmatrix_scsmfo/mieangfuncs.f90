! Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
! W. Perry, Jerome Fung, and Ryan McGorty
!
! This file is part of Holopy.
!
! Holopy is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! Holopy is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
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

! Common documentation for singleholo, tmdimerholo, and tmdimerholo_nf:
! Functions intended to be used from Python.
!
! Input parameters:
!    
!    kxgrid: k*x coordinates of grid points (1-d array)
!    kygrid: k*y coordinates of grid points
!    kcoords: k*coordinates of cluster or sphere center
!    asbs: array of shape (2, n) of a_n, b_n Mie coefficients (only singleholo)
!    amn: array of amn cluster scattering coefficients (not in singleholo)
!    gamma: Euler angle gamma, degrees (not in singleholo)
!    alpha: scaling factor that should be close to 1
!    einc: polarization vector of incident field
!
! Output parameters:
!    hologram: calculated hologram


      subroutine singleholo(n_rows, n_cols, kxgrid, kygrid, kcoords, asbs, nstop, & 
           alpha, einc, hologram)
        ! Intended fast replacement for FitSingle.CalcHoloSingle
        ! Use Python to calculate scat. coeffs a_n and b_n once per frame
        ! This function performs nested loop over pixels
        implicit none
        integer nstop
        integer n_rows, n_cols, i, j, n, nn, ifail
        real (kind = 8), dimension(n_rows, n_cols) :: hologram
        real*8 kcoords(3), sphcoords(3), alpha, einc(2), kxgrid(n_rows), kygrid(n_cols)
        real*8 pixel, kr, kz, theta, phi, prefactor(nstop), pi_n(nstop), tau_n(nstop)
        ! use single vectors rather than redundant mgrid for x, y values
        complex*16 asbs(2, nstop), ascatmat(4), asreshape(2,2)
        complex*16 hl, dhl, inv_pref, ci
        real*8 jn(0:nstop), djn(0:nstop), yn(0:nstop), dyn(0:nstop) ! radial
        data ci/(0.d0,1.d0)/
! comments for use by f2py
!f2py intent(in) n_rows
!f2py intent(in) n_cols
!f2py intent(in) kxgrid
!f2py intent(in) kygrid
!f2py intent(in) kcoords
!f2py intent(in) asbs
!f2py intent(in) nstop
!f2py intent(in) alpha
!f2py intent(in) einc
!f2py intent(out) hologram


        ! calculate scattering matrix prefactor
        do n = 1, nstop, 1
           prefactor(n) = (2.*n + 1.) / (n * (n+1.))
        end do

        ! the great loop
        do  j = 1, n_cols, 1
           do i = 1, n_rows, 1
              call getsphercoords(kxgrid(i) - kcoords(1), kygrid(j) - kcoords(2), &
                   kcoords(3), sphcoords)
              kr = sphcoords(1)
              kz = kcoords(3)
              theta = sphcoords(2)
              phi = sphcoords(3)

              call pisandtaus(nstop, theta, pi_n, tau_n)

              ! spherical bessels for full radial dependence
              call sbesjy(kr, nstop, jn, yn, djn, dyn, ifail)
                           
              ! calculate S1 and S2 by looping summation
              ascatmat = (/ 0, 0, 0, 0 /)
              ! inverse prefactor allows use of existing radially asymptotic code
              inv_pref = cdexp(-1*ci*kr)*kr

              do nn = 1, nstop, 1
                 hl = jn(nn) + ci*yn(nn)
                 dhl = hl/kr + djn(nn) + ci*dyn(nn)
                 ascatmat(1) = ascatmat(1) + prefactor(nn) * ci**nn * ( &
                      asbs(1,nn)*pi_n(nn)*dhl + ci*asbs(2,nn)*tau_n(nn)*hl)
                 ascatmat(2) = ascatmat(2) + prefactor(nn) * ci**nn * ( &
                      asbs(1,nn)*tau_n(nn)*dhl + ci*asbs(2,nn)*pi_n(nn)*hl)
              end do

              ! apply inverse prefactor
              ascatmat(1) = ascatmat(1) * inv_pref
              ascatmat(2) = ascatmat(2) * inv_pref
   
              asreshape = reshape(cshift(ascatmat, shift = 1), (/ 2, 2 /), &
                   order = (/ 2, 1 /)) 
              call paraxholocl(kr, kz, theta, phi, asreshape, einc, alpha, pixel)
              hologram(i,j) = pixel
           end do
        end do

        return 
        end



      subroutine tmholo_nf(n_rows, n_cols, kxgrid, kygrid, kcoords, amn, &
           nodrt, gamma, alpha, einc, hologram)
      ! Intended faster replacement for TMatDimerHolo doing the loop in Fortran.
      ! Loop over cluster hologram grid, calculated with near-field corrections.
        implicit none
        integer npd, notd, nod
        include 'scfodim.for'
        integer, parameter :: nbc = 4*notd+4
        integer n_rows, n_cols, nodrt, i, j, n
        real (kind = 8), dimension(n_rows, n_cols) :: hologram
        real*8 kcoords(3), sphcoords(3), gamma, einc(2), kxgrid(n_rows)
        real*8 kygrid(n_cols)
        real*8 pixel, kr, kz, theta, phi, alpha
        ! use single vectors rather than redundant mgrid for x, y values
        complex*16 amn(2,nodrt*(nodrt+2),2), ascatmat(4), asreshape(2,2)
        real*8 bcof(0:nbc), fnr(0:2*nbc)
        common/consts2/bcof,fnr
! comments for use by f2py
!f2py intent(in) n_rows
!f2py intent(in) n_cols
!f2py intent(in) kxgrid
!f2py intent(in) kygrid
!f2py intent(in) kcoords
!f2py intent(in) amn
!f2py intent(in) nodrt
!f2py intent(in) gamma
!f2py intent(in) alpha
!f2py intent(in) einc
!f2py intent(out) hologram

        !  added by me to avoid common block, copied from scmsfo1b.for
        do n=1,2*nbc
           fnr(n)=dsqrt(dble(n))
        enddo
        !  make bcof rank 1 since we only need its diagonal elements
        bcof(0)=1.d0
        do n=0,nbc-1
           bcof(n+1)=fnr(n+n+2)*fnr(n+n+1)*bcof(n)/fnr(n+1)/fnr(n+1)
        enddo

        do  j = 1, n_cols, 1
           do i = 1, n_rows, 1
              call getsphercoords(kxgrid(i) - kcoords(1), kygrid(j) - kcoords(2), &
                   kcoords(3), sphcoords)
              kr = sphcoords(1)
              kz = kcoords(3)
              theta = sphcoords(2)
              phi = sphcoords(3)
              
              call asmfr(amn, nodrt, theta, phi+gamma, kr, ascatmat)
              asreshape = reshape(cshift(ascatmat, shift = 1), (/ 2, 2 /), &
                   order = (/ 2, 1 /)) * (-0.5)
              call paraxholocl(kr, kz, theta, phi, asreshape, einc, alpha, pixel)
              hologram(i,j) = pixel
           end do
        end do

        return
        end



      subroutine paraxholocl(kr, kz, theta, phi, ascatm, polvec, alpha, holo)
! Subroutine to calculate hologram at a point given coordinates, S2, S1,
! polarization, and scaling coefficient alpha, using simplified paraxial
! (no Poynting vector) approach a la Grier.
! Right now require polarization vector to be real (e.g. linear polarization)
! Works for a general S matrix.
        real*8 kr, kz, theta, phi, polvec(2), alpha, holo, einc(2)
        real*8 signarr(2)
        complex*16 ascatm(2,2), prefactor, escatsph(2), esrect(3), hcmplx, inc(2)
!f2py intent(in) kr
!f2py intent(in) kz 
!f2py intent(in) phi 
!f2py intent(in) polvec
!f2py intent(in) alpha
!f2py intent(in) ascatm
!f2py intent(out) holo

! First calculate incident field in spherical coordinates at the detector
! do not apply phase here
        call incfield(polvec(1), polvec(2), phi, einc)

! get scattered field in spherical coordinates
! because of no phase applied to incident field, the prefactor is different
! than Bohren & Huffman (This caused JF much grief in figuring out).
        prefactor = (0.0, 1.0) / kr * zexp((0.0, 1.0)*kr)
        signarr = (/ 1.0, -1.0 /) ! needed since escatperp = -escatphi
        escatsph = prefactor*matmul(ascatm, einc)*signarr

! convert spherical E field to cartesian
        call fieldstocart(escatsph, theta, phi, esrect)

! now apply an e-ikz factor at the detector, because it's the CC 
! of the incident field in Cartesian we need
        inc = polvec * zexp((0.0, 1.0)*kz)

! fortran dot product complex conjugates the first argument
        hcmplx = 1. + 2.*alpha*dot_product(inc, esrect(1:2)) + alpha**2 * &
              dot_product(esrect(1:2), esrect(1:2))

! convert to double precision real
        holo = real(hcmplx, 8)
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
        sph(2) = dacos(z / sph(1))
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
