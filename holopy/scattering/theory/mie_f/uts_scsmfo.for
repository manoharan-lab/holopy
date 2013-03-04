c uts_scsmfo.for
c Name derived from "utilities, from scsmfo".

c 2 subroutines from scsmfo1b.for, with modifications by J. Fung
c to not have arrays be declared larger than necessary, and to add 
c f2py comments.
c Subroutines are used to calculate 2x2 scattering matrix S from amn 
c coefficients.
c Subroutines are intended to be compiled with f2py and called from Python.

c All algorithms used here are due to D. Mackwoski, not JF.


c ************************************************************

c  dc is the normalized generalized spherical function
c  dc(k,n*(n+1)+m) = D^k_{mn}
c  where D^k_{mn} is defined in the notes
c  This routine calculates dc(k,n*(n+1)+m) for k=-kmax..kmax and
c  n=0...nmax.
c
c  note that P_n^m = (-1)^m ((n-m)!/(n+m)!) dc(0,n*(n+1)-m)
c

      subroutine rotcoef(cbe,kmax,nmax,dc,ndim)
c Input arguments:
c cbe: cosine of Euler beta
c kmax: max value of k (int)
c nmax: maximum order n (int)
c ndim: max value of m? (int)
c Output:
c dc: D_mn^k(beta)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),notd2=notd+notd,nbc=2*notd2+4)
      implicit real*8(a-h,o-z)
c size declaration changed by me so i don't pass unnecessarily large array
      real*8 dc(-ndim:ndim,0:nmax*(nmax+2)+1)
      real*8 dk0(-notd2:notd2),dk01(-notd2:notd2)
      real*8 bcof(0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/

cf2py intent(in) cbe, kmax, nmax, ndim
cf2py intent(out) dc 

c  added by me to avoid common block, copied from scmsfo1b.for
      do n=1,2*nbc
         fnr(n)=dsqrt(dble(n))
      enddo
c  make bcof rank 1 since we only need its diagonal elements
      bcof(0)=1.d0
      do n=0,nbc-1
         bcof(n+1)=fnr(n+n+2)*fnr(n+n+1)*bcof(n)/fnr(n+1)/fnr(n+1)
      enddo


      sbe=dsqrt((1.d0+cbe)*(1.d0-cbe))
      cbe2=.5d0*(1.d0+cbe)
      sbe2=.5d0*(1.d0-cbe)
      in=1
      dk0(0)=1.d0
      sben=1.d0
      dc(0,0)=1.d0
      dk01(0)=0.
      do n=1,nmax
         knmax=min(n,kmax)
         nn1=n*(n+1)
         in=-in
         sben=sben*sbe/2.d0
         dk0(n)=in*sben*bcof(n)
         dk0(-n)=in*dk0(n)
         dk01(n)=0.
         dk01(-n)=0.
         dc(0,nn1+n)=dk0(n)
         dc(0,nn1-n)=dk0(-n)
         do k=-n+1,n-1
            kn=nn1+k
            dkt=dk01(k)
            dk01(k)=dk0(k)
            dk0(k)=(cbe*dble(n+n-1)*dk01(k)-fnr(n-k-1)*fnr(n+k-1)*dkt)
     1             /(fnr(n+k)*fnr(n-k))
            dc(0,kn)=dk0(k)
         enddo
         im=1
         do m=1,knmax
            im=-im
            fmn=1.d0/fnr(n-m+1)/fnr(n+m)
            m1=m-1
            dkm0=0.
            do k=-n,n
               kn=nn1+k
               dkm1=dkm0
               dkm0=dc(m1,kn)
               if(k.eq.n) then
                  dkn1=0.
               else
                  dkn1=dc(m1,kn+1)
               endif
               dc(m,kn)=(fnr(n+k)*fnr(n-k+1)*cbe2*dkm1
     1           -fnr(n-k)*fnr(n+k+1)*sbe2*dkn1
     1              -dble(k)*sbe*dc(m1,kn))*fmn
               dc(-m,nn1-k)=dc(m,kn)*(-1)**(k)*im
            enddo
         enddo
      enddo
      return
      end


c *****************************************************************************

      subroutine asmfr(amn0,nodrt,theta,phi,kr,sa)
c Calculate amplitude scattering matrix for given cluster as a function of 
c angle.
c Uses exact radial dependence of fields
c Inputs:
c amn0 (array of amn coefficients obtained from amncalc subroutine 
c in scsmfo_min.for)
c nodrt (maximum order of cluster expansion)
c theta 
c phi (detector spherical coordinate angles) 
c kr (dimensionless wavevector*distance to detector point)
c Outputs:
c sa (2x2 complex array)
      include 'scfodim.for'
      implicit real*8(a-h,o-z)
c
      parameter(nbd=nod*(nod+2),nbd2=nbd+nbd,
     1          nbtd=notd*(notd+2),notd2=notd+notd,
     1          nbc=2*notd2+4)
      real*8 drot(-1:1,0:nodrt*(nodrt+2)),tau(2), kr
      complex*16 ci,amn0(2,nodrt*(nodrt+2),2),cin,sa(4), expir,
     1           ephi(-notd-1:notd+1),a,b, radfunc(2, 1:nodrt)
      real*8 jn(0:nodrt), djn(0:nodrt), yn(0:nodrt), dyn(0:nodrt)
      data ci/(0.d0,1.d0)/
cf2py intent(in) amn0, nodrt, theta, phi, kr
cf2py intent(out) sa

      ct = dcos(theta)

      call rotcoef(ct,1,nodrt,drot,1)
      ephi(1)=cdexp(ci*phi)
      ephi(-1)=conjg(ephi(1))
      ephi(0)=1.d0
      do m=2,nodrt+1
         ephi(m)=ephi(1)*ephi(m-1)
         ephi(-m)=conjg(ephi(m))
      enddo
      do i=1,4
         sa(i)=0.
      enddo
     
      expir = cdexp(-ci*kr)

c calculate radial functions to order nodrt
      call sbesjy(kr, nodrt, jn, yn, djn, dyn, ifail)
c     call sphj(nodrt, kr, ndummy, jn, djn)
c     call sphy(nodrt, kr, ndummy, yn, dyn)
      do ncount = 1, nodrt
         radfunc(1, ncount) = -ci * ( (jn(ncount) + ci*yn(ncount))/kr +  
     1       djn(ncount)+ci*dyn(ncount) ) 
         radfunc(2, ncount) = (jn(ncount) + ci * yn(ncount)) 
      enddo

      do n=1,nodrt
         cin=(-ci)**n
         nn1=n*(n+1)
         do m=-n,n
            mn=nn1+m
            mnm=nn1-m
            tau(1)=dsqrt(dble(n+n+1))*(drot(-1,mnm)-drot(1,mnm))
            tau(2)=dsqrt(dble(n+n+1))*(drot(-1,mnm)+drot(1,mnm))
            do ip=1,2
               a=amn0(ip,mn,1)*ephi(m+1)+amn0(ip,mn,2)*ephi(m-1)
               b=ci*(amn0(ip,mn,1)*ephi(m+1)-amn0(ip,mn,2)*ephi(m-1))
c
c  s1,s2,s3,s4: amplitude scattering matrix elements.
c
               sa(1) = sa(1) + ci * kr * expir*radfunc(ip,n)*tau(3-ip)*b
               sa(2) = sa(2) + kr * expir * radfunc(ip,n) * tau(ip) * a
               sa(3) = sa(3) + kr * expir * radfunc(ip,n) * tau(ip) * b
               sa(4) = sa(4) + ci*kr*expir * radfunc(ip,n) * tau(3-ip)*a
            enddo
         enddo
      enddo

      return
      end

c *****************************************************************************

      subroutine ms_radial_fields(amn0, nodrt, theta, phi, kr, as_rad)
c Calculate nondimensional radial components of scattered field for a cluster
c given spherical coordinates. 
c E_s,r = as_rad(1) * Einc, par + as_rad(2) * Einc, perp
c
c Inputs:
c amn0 (array of amn coefficients obtained from amncalc subroutine 
c in scsmfo_min.for)
c nodrt (maximum order of cluster expansion)
c theta 
c phi (detector spherical coordinate angles) 
c kr (dimensionless wavevector*distance to detector point)
c Outputs:
c as_rad (complex array, len(2))
      include 'scfodim.for'
      implicit real*8(a-h,o-z)
c
      parameter(nbd=nod*(nod+2),nbd2=nbd+nbd,
     1          nbtd=notd*(notd+2),notd2=notd+notd,
     1          nbc=2*notd2+4)
      real*8 drot(-1:1,0:nodrt*(nodrt+2)),tau2, kr
      complex*16 ci,amn0(2,nodrt*(nodrt+2),2),cin, as_rad(2), expir,
     1           ephi(-notd-1:notd+1),a,b, radfunc(1:nodrt)
      real*8 jn(0:nodrt), djn(0:nodrt), yn(0:nodrt), dyn(0:nodrt)
      data ci/(0.d0,1.d0)/
cf2py intent(in) amn0, nodrt, theta, phi, kr
cf2py intent(out) as_rad

      ct = dcos(theta)
      st = dsin(theta)

      call rotcoef(ct,1,nodrt,drot,1)
      ephi(1)=cdexp(ci*phi)
      ephi(-1)=conjg(ephi(1))
      ephi(0)=1.d0
      do m=2,nodrt+1
         ephi(m)=ephi(1)*ephi(m-1)
         ephi(-m)=conjg(ephi(m))
      enddo
      do i=1,2
         as_rad(i)=0.
      enddo
     
      expir = cdexp(-ci*kr)

c calculate radial functions to order nodrt
      call sbesjy(kr, nodrt, jn, yn, djn, dyn, ifail)
      do ncount = 1, nodrt
         radfunc(ncount) = (jn(ncount) + ci * yn(ncount)) / kr
      enddo

      do n=1,nodrt
         cin=(-ci)**n
         nn1=n*(n+1)
         do m=-n,n
            if (m .ne. 0) then
              mn=nn1+m
              mnm=nn1-m
              tau2 = dsqrt(dble(n+n+1))*(drot(-1,mnm)+drot(1,mnm))
c scsmfo.ps equation 20, a'_mnp,\par. note phi dependence included
              a=amn0(1,mn,1)*ephi(m+1)+amn0(1,mn,2)*ephi(m-1)
c scsmfo.ps equation 22, a'_mnp,\perp
              b=ci*(amn0(1,mn,1)*ephi(m+1)-amn0(1,mn,2)*ephi(m-1))

              pref = nn1 * (1. / m)
              as_rad(1) = as_rad(1) + pref * st * tau2 * radfunc(n) * a
              as_rad(2) = as_rad(2) + pref * st * tau2 * radfunc(n) * b
            else 
              continue
            endif
         enddo
      enddo

      return
      end

c *****************************************************************************




      subroutine asm(amn0,nodrt,theta,phi,sa)
c Calculate amplitude scattering matrix for given cluster as a function of 
c angle.
c Uses far-field approximation.
c Inputs:
c amn0 (array of amn coefficients obtained from amncalc subroutine 
c in scsmfo_min.for)
c nodrt (maximum order of cluster expansion)
c theta 
c phi (detector spherical coordinates) 
c Outputs:
c sa (2x2 complex array)
      include 'scfodim.for'
      implicit real*8(a-h,o-z)
c
      parameter(nbd=nod*(nod+2),nbd2=nbd+nbd,
     1          nbtd=notd*(notd+2),notd2=notd+notd,
     1          nbc=2*notd2+4)
      real*8 drot(-1:1,0:nodrt*(nodrt+2)),tau(2)
      complex*16 ci,amn0(2,nodrt*(nodrt+2),2),cin,sa(4),
     1           ephi(-notd-1:notd+1),a,b
      data ci/(0.d0,1.d0)/
cf2py intent(in) amn0, nodrt, theta, phi
cf2py intent(out) sa

      ct = dcos(theta)

      call rotcoef(ct,1,nodrt,drot,1)
      ephi(1)=cdexp(ci*phi)
      ephi(-1)=conjg(ephi(1))
      ephi(0)=1.d0
      do m=2,nodrt+1
         ephi(m)=ephi(1)*ephi(m-1)
         ephi(-m)=conjg(ephi(m))
      enddo
      do i=1,4
         sa(i)=0.
      enddo

      do n=1,nodrt
         cin=(-ci)**n
         nn1=n*(n+1)
         do m=-n,n
            mn=nn1+m
            mnm=nn1-m
            tau(1)=dsqrt(dble(n+n+1))*(drot(-1,mnm)-drot(1,mnm))
            tau(2)=dsqrt(dble(n+n+1))*(drot(-1,mnm)+drot(1,mnm))
            do ip=1,2
               a=amn0(ip,mn,1)*ephi(m+1)+amn0(ip,mn,2)*ephi(m-1)
               b=ci*(amn0(ip,mn,1)*ephi(m+1)-amn0(ip,mn,2)*ephi(m-1))
c
c  s1,s2,s3,s4: amplitude scattering matrix elements.
c
               sa(1)=sa(1)+cin*tau(3-ip)*b
               sa(2)=sa(2)-ci*cin*tau(ip)*a
               sa(3)=sa(3)-ci*cin*tau(ip)*b
               sa(4)=sa(4)+cin*tau(3-ip)*a
            enddo
         enddo
      enddo

      return
      end


