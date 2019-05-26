! Copyright 2013-2016, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca
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
! S.f
!
! Author: Ron Alexander <ralex0@users.noreply.github.com>
!
! Description:
! Calculates the T-Matrix amplitude scattering matrix for list of angles
! with using M Mishchenko's fortran implementation. See permissions.txt
!
! At compile this must be linked with ampld.lp.f and lpd.f
! Functions are designed to be compiled with f2py and called from Python
      subroutine ampld(axi, rat, lam, mrr, mri, eps, np, ndgs, 
     &                      alpha, beta, thet0, thet, phi0, phi, nang,
     &                      s11, s12, s21, s22)
c parameters:
      integer, parameter :: dp = selected_real_kind(15, 307)
c variables:
      integer, intent(in) :: np, ndgs, nang
      integer :: maxi
      real(kind=dp), intent(in) :: lam, mrr, mri, eps
      real(kind=dp), intent(in) :: axi, rat, alpha, beta, thet0, phi0
      real(kind=dp), dimension(nang),intent(in) :: thet, phi
      complex(kind=dp), dimension(nang),intent(out) :: s11,s12,s21,s22

C Call amp_scat_matrix on the first angle to calc the T-matrix
      call amp_scat_matrix (axi,rat,lam,mrr,mri,eps,np,ndgs,alpha,
     &                      beta,thet0,thet(1),phi0,phi(1),
     &                      s11(1),s12(1),s21(1),s22(1),maxi)
C loop over the rest of the angles. T-matrix is a global (common)
      if (nang > 1) then
         do j=2, nang
            call ampl (maxi,lam,thet0,thet(j),phi0,phi(j),alpha,beta,
     &                 s11(j),s12(j),s21(j),s22(j))
         end do
      end if
      return
      end