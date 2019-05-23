      subroutine ampld(axi, rat, lam, mrr, mri, eps, np, ndgs, 
     &                      alpha, beta, thet0, thet, phi0, phi, nang,
     &                      s11, s12, s21, s22)
c parameters:
      integer, parameter :: dp = selected_real_kind(15, 307)
c variables:
      integer, intent(in) :: np, ndgs, nang
      real(kind=dp), intent(in) :: lam, mrr, mri, eps
      real(kind=dp), intent(in) :: axi, rat, alpha, beta, thet0, phi0
      real(kind=dp), dimension(nang),intent(in) :: thet, phi
      complex(kind=dp), dimension(nang),intent(out) :: s11,s12,s21,s22

      do j=1, nang
         call amp_scat_matrix (axi,rat,lam,mrr,mri,eps,np,ndgs,alpha,
     &                         beta,thet0,thet(j),phi0,phi(j),
     &                         s11(j),s12(j),s21(j),s22(j))
      end do
      return
      end