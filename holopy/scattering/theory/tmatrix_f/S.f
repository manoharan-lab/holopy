      subroutine ampld(axi, rat, lam, mrr, mri, eps, np, ndgs, 
     &                      alpha, beta, thet0, thet, phi0, phi,
     &                      s11, s12, s21, s22)
c parameters:
      integer, parameter :: dp = selected_real_kind(15, 307)
c variables:
      integer, intent(in) :: np, ndgs
      real(kind=dp), intent(in) :: lam, mrr, mri, eps
      real(kind=dp), intent(in) :: axi, rat, alpha, beta, thet0, phi0
      real(kind=dp), intent(in) :: thet, phi
      complex(kind=dp), intent(out) :: s11, s12, s21, s22

      call amp_scat_matrix (axi,rat,lam,mrr,mri,eps,np,ndgs,alpha,
     &                      beta,thet0,thet,phi0,phi,s11,s12,s21,s22)
      return
      end