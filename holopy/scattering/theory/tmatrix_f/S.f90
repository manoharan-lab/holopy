      subroutine s(axi, rat, lam, mrr, mri, eps, np, ndgs, alpha, beta,
     &             thet0, thet, phi0, phi, nang, s11, s12, s21, s22)
c parameters:          
      integer maxnang
      parameter(maxnang=100000)
c variables:
      integer, intent(in) :: nang, np, ndgs
      real(kind=8), intent(in) :: lam, mrr, mri, eps
      real(kind=8), intent(in) :: axi, rat, alpha, beta, thet0, phi0
      real(kind=8), dimension(maxnang), intent(in) :: thet, phi
      complex(kind=16), dimension(maxnang), intent(out) :: s11, s12 
      complex(kind=16), dimension(maxnang), intent(out) :: s21, s22

      if(nang.gt.maxnang)stop'***error: nang > maxnang in s'

      do  j=1, nang
         call amp_scat_matrix (axi,rat,lam,mrr,mri,eps,np,ndgs,alpha,
     &                         beta,thet0,thet(j),phi0,phi(j),
     &                         s11(j),s12(j),s21(j),s22(j)) 
      end do
      return
      end