c scsmfo_min.for
c Modified by Jerome Fung from scsmfo1b.for by Daniel Mackowski.
c Original code available at ftp://ftp.eng.auburn.edu/pub/dmckwski/scatcodes/
c
c This is a minimal subset of scsmfo1b's subroutines (w/o the main program),
c hence the name scsmfo_min. 
c Note: I think SCSMFO stands for "Scattering, Clusters of Spheres,
c Mackowski, Fixed Orientation"; I don't know what the 1B refers to.

c The code is intended to be compiled with f2py; only the subroutine amncalc 
c is intended to be called from Python. This permits the calculation of 
c amn coefficients for arbitrary sphere clusters.

c amncalc has been modified from original code as follows:
c 1) code to calculate bcof and fnr and avoid common block added
c 2) sizes of necessary arrays determined at run time rather than
c    by allocating way more memory than necessary.

c calculation of cluster T matrix via iteration scheme
c
      subroutine amncalc(inew,npart,xp,yp,zp,sni,ski,xi,nodr,
     1            nodrtmax,niter,eps,qeps1,qeps2,meth,
     1            ea, amn0, status)
c Intended to be called from Python.
c Inputs:
c inew (legacy, for program control -- set to 1)
c xp (array with particle x coords relative to COM, non-dimensionalized by 
c wavevector)
c yp (array with particle y coords, non-dimensionalized)
c zp (array with particle z coords, non-dimensionalized)
c sni (array, real part of relative index)
c ski (array, imaginary part of relative index)
c xi (array, particle size parameters)
c niter (max # of iterations)
c eps (relative error tolerance in sol'n)
c qeps1 (single sphere error tolerance)
c qeps2 (cluster error tolerance)
c meth (set to 1 to use order of scattering)
c ea (array of cluster Euler alpha and beta, degrees)
c Outputs:
c nodr (array of single sphere expansion orders)
c nodrtmax (max order of cluster VSH expansion)
c amn0 (2 x 5040 x 2 array of amn coefficients, listed in a compactified way)
c status (logical, true if iterative solver converges)
c *****************************************************************
c Note: If amn0 is used from Python as an argument to subroutines for
c hologram calculation in mieangfuncs.f90, it is necessary to truncate
c the output ndarray in Python, as follows:
c        amn0 = amn0[:, 0:(nodrtmax**2 + 2 * nodrtmax), :]
c ******************************************************************
      implicit real*8(a-h,o-z)
      include 'scfodim.for'
      parameter(nbd=nod*(nod+2),nbc=4*notd+4,
     1          nbtd=notd*(notd+2),nrd=.5*(npd-1)*(npd-2)+npd-1)
      parameter (nrotd=nod*(2*nod*nod+9*nod+13)/6,
     1           ntrad=nod*(nod*nod+6*nod+5)/6)
      integer nodr(npd),nblk(npd),nodrt(npd),nblkt(npd)
      real*8 xi(npart),sni(npart),ski(npart),rp(npd),qe1(npd),
     1       xp(npart),yp(npart),zp(npart)
      real*8 ea(2),drott(-nod:nod,0:nbd)
      complex*16 ci,cin,a,an1(2,nod,npd),pfac(npd)
      complex*16 amn(2,nbd,npd,2),amn0(2,nbtd,2)
      complex*16 pmn(2,nbd,npd),pp(2,nbd,2),amnlt(2,nod,nbd)
      real*8 drot(nrotd,nrd),dbet(-1:1,0:nbd)
      real*8 max_err
      logical*4 status
      complex*16 ephi,anpt(2,nbtd),amnl(2,ntrad,nrd),
     1           ek(nod,nrd),ealpha(-nod:nod)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/
Cf2py intent(in) inew, npart, xp, yp, zp, sni, ski, xi, niter
Cf2py intent(in) eps, qeps1, qeps2, meth, ea
Cf2py intent(out) nodr, nodrtmax, amn0, status
      
c calculate constants in common block /consts/
      do n=1,2*nbc
         fnr(n)=dsqrt(dble(n))
      enddo
      bcof(0,0)=1.d0
      do n=0,nbc-1
         do l=n+1,nbc
            bcof(n,l)=fnr(n+l)*bcof(n,l-1)/fnr(l)
            bcof(l,n)=bcof(n,l)
         enddo
         bcof(n+1,n+1)=fnr(n+n+2)*fnr(n+n+1)*bcof(n,n)/fnr(n+1)/fnr(n+1)
      enddo


      if(inew.eq.0) goto 15

      pi=4.*datan(1.d0)
      itermax=0
      xv=0.
      xm=0.
      ym=0.
      zm=0.
      nodrmax=0

c
c mie coefficients
c
      do i=1,npart
         call mie1(xi(i),sni(i),ski(i),nodr(i),qeps1,qe1(i),
     1             qs1,an1(1,1,i))
         nblk(i)=nodr(i)*(nodr(i)+2)
         xm=xm+xp(i)
         ym=ym+yp(i)
         zm=zm+zp(i)
         xv=xv+xi(i)*xi(i)*xi(i)
         nodrmax=max(nodr(i),nodrmax)
      enddo


      print*, 'single sphere max. order: ', nodrmax
      if(nodrmax.eq.nod) then
         print*, 'Warning: single--sphere error tolerance may not 
     1            be obtained.'
         print*, 'Decrease qeps1 and/or increase nod.'
      endif

      nblkmax=nodrmax*(nodrmax+2)
      
      xm=xm/dble(npart)
      ym=ym/dble(npart)
      zm=zm/dble(npart)
      xv=xv**(1./3.)
      nodrtmax=0

      do i=1,npart
         x=xm-xp(i)
         y=ym-yp(i)
         z=zm-zp(i)
         rp(i)=sqrt(x*x+y*y+z*z)
         xc=rp(i)+xi(i)
         nodrt(i)=max(nodr(i),nint(xc+4.*xc**(1./3.))+2)
         nodrtmax=max(nodrt(i),nodrtmax)
         nodrt(i)=min(nodrt(i),notd)
         nblkt(i)=nodrt(i)*(nodrt(i)+2)
      enddo
      if(nodrtmax.gt.notd) then
         print*, 'Warning: notd dimension may be too small.'
         print*, 'increase to ', nodrtmax
      endif
      nodrtmax=min(nodrtmax,notd)
      print*,''
      print*, 'Estimated cluster expansion order:', nodrtmax
      nblktmax=nodrtmax*(nodrtmax+2)
c
      do i=1,npart
         print*, 'assembling interaction matrix row: ', i
         do j=i+1,npart
            ij=.5*(j-1)*(j-2)+j-i
            x=xp(i)-xp(j)
            y=yp(i)-yp(j)
            z=zp(i)-zp(j)
            if((x.eq.0.).and.(y.eq.0.)) then
               ephi=1.
            else
               ephi=cdexp(ci*datan2(y,x))
            endif
            nmx=max(nodr(i),nodr(j))
            do m=1,nmx
               ek(m,ij)=ephi**m
            enddo
            r=sqrt(x*x+y*y+z*z)
            ct=z/r
            call rotcoef(ct,nmx,nmx,drott,nod)
            call trancoef(3,r,nmx,nmx,amnlt,nod)
            do n=1,nmx
               nn1=n*(n+1)*(2*n+1)/6
               do m=0,n
                  mn=nn1+m*(m+1)
                  do k=-m,m
                     mnk=mn+k
                     drot(mnk,ij)=drott(k,n*(n+1)+m)
                  enddo
               enddo
            enddo

            do n=1,nmx
               na=(n-1)*n*(n+4)/6
               do l=1,n
                  nla=na+l*(l+1)/2
                  do m=0,l
                     nlma=nla+m
                     amnl(1,nlma,ij)=amnlt(1,n,l*(l+1)+m)
                     amnl(2,nlma,ij)=amnlt(2,n,l*(l+1)+m)
                  enddo
               enddo
            enddo
         enddo
      enddo
      print*, ''

15    do n=1,nblktmax
         do ip=1,2
            do k=1,2
               amn0(ip,n,k)=0.
            enddo
         enddo
      enddo

      alpha=ea(1)*pi/180.
      beta=ea(2)*pi/180.
      cb=cos(beta)
      call rotcoef(cb,1,nodrmax,dbet,1)
      ealpha(1)=cdexp(ci*alpha)
      ealpha(-1)=conjg(ealpha(1))
      ealpha(0)=1.d0
      do n=2,nodrmax
         ealpha(n)=ealpha(1)*ealpha(n-1)
         ealpha(-n)=conjg(ealpha(n))
      enddo
      do n=1,nodrmax
         nn1=n*(n+1)
         cin=ci**(n+1)
         fnm=fnr(n+n+1)/2.
         do m=-n,n
            mn=nn1+m
            im=(-1)**m
            pp(1,mn,1)=-cin*fnm*ealpha(-m)*dbet(-1,mn)*im
            pp(2,mn,1)=cin*fnm*ealpha(-m)*dbet(-1,mn)*im
            pp(1,mn,2)=cin*fnm*ealpha(-m)*dbet(1,mn)*im
            pp(2,mn,2)=cin*fnm*ealpha(-m)*dbet(1,mn)*im
         enddo
      enddo
      do i=1,npart
         if(rp(i).eq.0.) then
            pfac(i)=1.
         else
            xi0=xp(i)-xm
            yi0=yp(i)-ym
            zi0=zp(i)-zm
            if(xi0.eq.0..and.yi0.eq.0.) then
               phii0=0.
            else
               phii0=datan2(yi0,xi0)
            endif
            ci0=zi0/rp(i)
            si0=sqrt((1.-ci0)*(1.+ci0))
            sb=sqrt((1.-cb)*(1.+cb))
            pfac(i)=cdexp(ci*rp(i)*(cb*ci0+sb*si0*cos(phii0-alpha)))
         endif
      enddo



      inew=0

      nodrtmax=0

c Iterative solution for both polarizations begins here
      max_err = 0.

      do k=1,2
         print*, 'Solving for incident state ', k
        
         do i=1,npart
            do n=1,nodr(i)
               nn1=n*(n+1)
               cin=ci**(n+1)
               fnm=fnr(n+n+1)/2.
               do m=-n,n
                  mn=nn1+m
                  do ip=1,2
                     pmn(ip,mn,i)=pfac(i)*an1(ip,n,i)*pp(ip,mn,k)
                     amn(ip,mn,i,k)=pmn(ip,mn,i)
                  enddo
               enddo
            enddo
         enddo

         if(niter.ne.0) then
            call itersoln(npart,nodr,nblk,eps,niter,
     1        meth,itest,ek,drot,amnl,an1,pmn,amn(1,1,1,k),iter,err)
c max_err gets checked at the end for convergence
            max_err = max(max_err, err)
            itermax=max(itermax,iter)
         endif

         nodrt1=0
         do i=1,npart
            qaii=0.
            qeii=0.
            do n=1,nodr(i)
               nn1=n*(n+1)
               do m=-n,n
                  mn=nn1+m
                  anpt(1,mn)=amn(1,mn,i,k)
                  anpt(2,mn)=amn(2,mn,i,k)
               enddo
            enddo


            if(rp(i).ne.0.) then
               x=xm-xp(i)
               y=ym-yp(i)
               z=zm-zp(i)
               ct=z/rp(i)
               if(x.eq.0..and.y.eq.0.) then
                  phi=0.
               else
                  phi=datan2(y,x)
               endif
               call rotvec(phi,ct,nodr(i),nodr(i),anpt,1)
               call tranvec(rp(i),nodr(i),nodrt(i),anpt,qeps2,errt)
               call rotvec(phi,ct,nodrt(i),nodr(i),anpt,2)
               nptrn=nodrt(i)*(nodrt(i)+2)
            else
               nptrn=nblk(i)
            endif
            nodrt1=max(nodrt1,nodrt(i))

            do n=1,nptrn
               do ip=1,2
                  amn0(ip,n,k)=amn0(ip,n,k)+anpt(ip,n)
               enddo
            enddo

         enddo
         nodrtmax=max(nodrtmax,nodrt1)

         call rotvec(alpha,cb,nodrtmax,nodrtmax,amn0(1,1,k),1)

      enddo

      print*, ' Cluster expansion order: ', nodrtmax

c Check convergence: is the maximum error from iteration less than eps?
      status = .false.
      if (max_err.lt.eps) status = .true.

      return
      end
c
c iteration solver
c meth=0: conjugate gradient
c meth=1: order-of-scattering
c Thanks to Piotr Flatau
c
      subroutine itersoln(npart,nodr,nblk,eps,niter,meth,itest,ek,drot,
     1                    amnl,an1,pnp,anp,iter,err)
      implicit real*8(a-h,o-z)
      include 'scfodim.for'
      parameter(nbd=nod*(nod+2),
     1          nbtd=notd*(notd+2),nrd=.5*(npd-1)*(npd-2)+npd-1)
      parameter (nrotd=nod*(2*nod*nod+9*nod+13)/6,
     1           ntrad=nod*(nod*nod+6*nod+5)/6)

      integer nodr(npd),nblk(npd)
      real*8 drot(nrotd,nrd)
      complex*16 anpt(2,nbtd),amnl(2,ntrad,nrd),an1(2,nod,npd),
     1        pnp(2,nbd,npd),ek(nod,nrd),anp(2,nbd,npd)
      complex*16 anptc(2,nbd),
     1        cr(2,nbd,npd),cp(2,nbd,npd),cw(2,nbd,npd),cq(2,nbd,npd),
     1        cap(2,nbd,npd),caw(2,nbd,npd),cak,csk,cbk,csk2
c
      err=0.
      iter=0
      enorm=0.
c
      do i=1,npart
         do n=1,nblk(i)
            enorm=enorm+pnp(1,n,i)*conjg(pnp(1,n,i))
     1                 +pnp(2,n,i)*conjg(pnp(2,n,i))
         enddo
      enddo
c
      if(enorm.eq.0.) return
      if(meth.ne.0) goto 200
      csk=(0.,0.)
c
      do i=1,npart
         do n=1,nblk(i)
            cr(1,n,i)=0.
            cr(2,n,i)=0.
         enddo
         do j=1,npart
            if(j.ne.i) then
               if(i.lt.j) then
                  ij=.5*(j-1)*(j-2)+j-i
                  idir=1
               else
                  ij=.5*(i-1)*(i-2)+i-j
                  idir=2
               endif
               do n=1,nblk(j)
                  anpt(1,n)=anp(1,n,j)
                  anpt(2,n)=anp(2,n,j)
               enddo
               call vctran(anpt,idir,nodr(j),nodr(i),ek(1,ij),
     1              drot(1,ij),amnl(1,1,ij),nod,nod)
               do n=1,nblk(i)
                  cr(1,n,i)=cr(1,n,i)+anpt(1,n)
                  cr(2,n,i)=cr(2,n,i)+anpt(2,n)
               enddo
            endif
         enddo
         do n=1,nodr(i)
            nn1=n*(n+1)
            do m=-n,n
               mn=nn1+m
               do ip=1,2
                  cr(ip,mn,i)=an1(ip,n,i)*cr(ip,mn,i)+anp(ip,mn,i)
               enddo
            enddo
         enddo
         do n=1,nblk(i)
            do ip=1,2
               cr(ip,n,i)=pnp(ip,n,i)-cr(ip,n,i)
               cq(ip,n,i)=conjg(cr(ip,n,i))
               cw(ip,n,i)=cq(ip,n,i)
               cp(ip,n,i)=cr(ip,n,i)
               csk=csk+cr(ip,n,i)*cr(ip,n,i)
            enddo
         enddo
      enddo
      if(cdabs(csk).eq.0.) then
         return
      endif
  40  continue
c     enorm=0.
      cak=(0.,0.)
      do i=1,npart
         do n=1,nblk(i)
            do ip=1,2
               caw(ip,n,i)=0.
               cap(ip,n,i)=0.
            enddo
         enddo
         do j=1,npart
            if(j.ne.i) then
               if(i.lt.j) then
                  ij=.5*(j-1)*(j-2)+j-i
                  idir=1
               else
                  ij=.5*(i-1)*(i-2)+i-j
                  idir=2
               endif
               do n=1,nodr(j)
                  nn1=n*(n+1)
                  do m=-n,n
                     mn=nn1+m
                     do ip=1,2
                        anptc(ip,mn)=an1(ip,n,j)
     1                    *conjg(cw(ip,mn,j))
                        anpt(ip,mn)=cp(ip,mn,j)
                     enddo
                  enddo
               enddo
               call vctran(anpt,idir,nodr(j),nodr(i),ek(1,ij),
     1              drot(1,ij),amnl(1,1,ij),nod,nod)
               call vctran(anptc,5-idir,nodr(j),nodr(i),ek(1,ij),
     1              drot(1,ij),amnl(1,1,ij),nod,nod)
               do n=1,nblk(i)
                  cap(1,n,i)=cap(1,n,i)+anpt(1,n)
                  cap(2,n,i)=cap(2,n,i)+anpt(2,n)
                  caw(1,n,i)=caw(1,n,i)+anptc(1,n)
                  caw(2,n,i)=caw(2,n,i)+anptc(2,n)
               enddo
            endif
         enddo
         do n=1,nodr(i)
            nn1=n*(n+1)
            do m=-n,n
               mn=nn1+m
               do ip=1,2
                  caw(ip,mn,i)=conjg(caw(ip,mn,i))+cw(ip,mn,i)
                  cap(ip,mn,i)=an1(ip,n,i)*cap(ip,mn,i)+cp(ip,mn,i)
               enddo
            enddo
         enddo
         do n=1,nblk(i)
            do ip=1,2
               cak=cak+cap(ip,n,i)*conjg(cw(ip,n,i))
            enddo
         enddo
      enddo
      cak=csk/cak
      csk2=(0.,0.)
      err=0.
      do i=1,npart
         do n=1,nblk(i)
            do ip=1,2
               anp(ip,n,i)=anp(ip,n,i)+cak*cp(ip,n,i)
c              enorm=enorm+anp(ip,n,i)*conjg(anp(ip,n,i))
               cr(ip,n,i)=cr(ip,n,i)-cak*cap(ip,n,i)
               cq(ip,n,i)=cq(ip,n,i)-conjg(cak)*caw(ip,n,i)
               csk2=csk2+cr(ip,n,i)*conjg(cq(ip,n,i))
               err=err+cr(ip,n,i)*conjg(cr(ip,n,i))
            enddo
         enddo
      enddo
      err=err/enorm
      print*, '+iteration: ', iter
      print*, 'error: ', err
      if(err.lt. eps) then
         print*, ''
         return
      endif
      cbk=csk2/csk
      do i=1,npart
         do n=1,nblk(i)
            do ip=1,2
               cp(ip,n,i)=cr(ip,n,i)+cbk*cp(ip,n,i)
               cw(ip,n,i)=cq(ip,n,i)+conjg(cbk)*cw(ip,n,i)
            enddo
         enddo
      enddo
      csk=csk2
      iter=iter+1
      if(iter.le.niter) goto 40
      print*, ''
      return

200   do i=1,npart
         do n=1,nblk(i)
            do ip=1,2
               cq(ip,n,i)=pnp(ip,n,i)
               anp(ip,n,i)=pnp(ip,n,i)
            enddo
         enddo
      enddo
310   err=0.
      do i=1,npart
         do n=1,nblk(i)
            do ip=1,2
               cr(ip,n,i)=0.
            enddo
         enddo
         do j=1,npart
            if(i.ne.j) then
               if(i.lt.j) then
                  ij=.5*(j-1)*(j-2)+j-i
                  idir=1
               else
                  ij=.5*(i-1)*(i-2)+i-j
                  idir=2
               endif
               do n=1,nblk(j)
                  do ip=1,2
                     anpt(ip,n)=cq(ip,n,j)
                  enddo
               enddo
               call vctran(anpt,idir,nodr(j),nodr(i),ek(1,ij),
     1              drot(1,ij),amnl(1,1,ij),nod,nod)
               do n=1,nblk(i)
                  cr(1,n,i)=cr(1,n,i)+anpt(1,n)
                  cr(2,n,i)=cr(2,n,i)+anpt(2,n)
               enddo
            endif
         enddo
      enddo
      do i=1,npart
         do n=1,nodr(i)
            nn1=n*(n+1)
            do m=-n,n
               mn=nn1+m
               do ip=1,2
                  cq(ip,mn,i)=-an1(ip,n,i)*cr(ip,mn,i)
                  err=err+cq(ip,mn,i)*conjg(cq(ip,mn,i))
                  anp(ip,mn,i)=anp(ip,mn,i)+cq(ip,mn,i)
               enddo
            enddo
         enddo
      enddo
      err=err/enorm
      iter=iter+1
      print*, '+iteration: ', iter
      print*, 'error: ', err
      if((err.gt.eps).and.(iter.lt.niter)) goto 310
      print*, ''
      return
      end
c
c
c this performs a vector harmonic expansion coefficient translation
c from one origin to the next.   Uses a rotation-translation-rotation
c formulation.
c
c idir=1: a2=A21 a1
c idir=2: a1=A12 a2
c idir=3: a2 A21=a1
c idir=4: a1 A12=a2
c

      subroutine vctran(anpt,idir,nodrj,nodri,ekt,drott,amnlt,ndd,nda)
      implicit real*8(a-h,o-z)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),nbd=nod*(nod+2))
      parameter (nrotd=nod*(2*nod*nod+9*nod+13)/6,
     1           ntrad=nod*(nod*nod+6*nod+5)/6)
      real*8 drott(*),drot(-nod:nod,nbd)
      complex*16 anpt(2,*),ant(2,notd),amt(2,-notd:notd),a,b,
     1           amnlt(2,*),ekt(*),amnl(2,nod,nbd),ek(-nod:nod)
c
      ek(0)=1.
      nmax=max(nodrj,nodri)
      nmin=min(nodrj,nodri)
      do m=1,nmax
         ek(m)=ekt(m)
         ek(-m)=conjg(ek(m))
      enddo

      do n=1,nmax
         nn1=n*(n+1)
         na=n*(n+1)*(2*n+1)/6
         do m=0,min(n,nodrj)
            mnp=nn1+m
            mnm=nn1-m
            mna=na+m*(m+1)
            kmax=min(m,nodri)
            do k=-kmax,kmax
               mnkap=mna+k
               mnkam=mna-k
               is=(-1)**(m+k)
               drot(k,mnp)=drott(mnkap)
               drot(k,mnm)=is*drott(mnkam)
            enddo
            do k=kmax+1,min(n,nodri)
               knmap=na+k*(k+1)
               is=(-1)**(m+k)
               drot(k,mnp)=is*drott(knmap+m)
               drot(k,mnm)=is*drott(knmap-m)
               drot(-k,mnp)=drott(knmap-m)
               drot(-k,mnm)=drott(knmap+m)
            enddo
         enddo
      enddo

      do n=1,nodri
         nn1=n*(n+1)
         na=(n-1)*n*(n+4)/6
         lmin=min(n,nodrj)
         do l=1,lmin
            ll1=l*(l+1)
            nla=na+ll1/2
            do m=0,l
               mlp=ll1+m
               mlm=ll1-m
               mnp=nn1+m
               mnm=nn1-m
               mnla=nla+m
               amnl(1,n,mlp)=amnlt(1,mnla)
               amnl(1,n,mlm)=amnlt(1,mnla)
               amnl(2,n,mlp)=amnlt(2,mnla)
               amnl(2,n,mlm)=-amnlt(2,mnla)
            enddo
         enddo
         do l=lmin+1,nodrj
            nla=l*(l-1)*(l+4)/6+nn1/2
            isn=(-1)**(n+l)
            ll1=l*(l+1)
            do m=0,min(n,l)
               mlp=ll1+m
               mlm=ll1-m
               mnla=nla+m
               amnl(1,n,mlp)=isn*amnlt(1,mnla)
               amnl(1,n,mlm)=isn*amnlt(1,mnla)
               amnl(2,n,mlp)=isn*amnlt(2,mnla)
               amnl(2,n,mlm)=-isn*amnlt(2,mnla)
            enddo
         enddo
      enddo

      if(idir.eq.3.or.idir.eq.4) then
         do n=1,nodrj
            nn1=n*(n+1)
            im=1
            do m=1,n
               im=-im
               mn=nn1+m
               mnm=nn1-m
               a=anpt(1,mn)
               anpt(1,mn)=im*anpt(1,mnm)
               anpt(1,mnm)=im*a
               a=anpt(2,mn)
               anpt(2,mn)=im*anpt(2,mnm)
               anpt(2,mnm)=im*a
            enddo
         enddo
      endif

      do n=1,nodrj
         nn1=n*(n+1)
         mmax=min(n,nodri)
         do m=-mmax,mmax
            amt(1,m)=0.
            amt(2,m)=0.
         enddo
         do k=-n,n
            knm=nn1-k
            kn=nn1+k
            a=ek(k)*anpt(1,kn)
            b=ek(k)*anpt(2,kn)
            do m=-mmax,mmax
               amt(1,m)=amt(1,m)+a*drot(-m,knm)
               amt(2,m)=amt(2,m)+b*drot(-m,knm)
            enddo
         enddo
         do m=-mmax,mmax
            mn=nn1+m
            anpt(1,mn)=amt(1,m)
            anpt(2,mn)=amt(2,m)
         enddo
      enddo

      mmax=min(nodrj,nodri)
      if(idir.eq.1.or.idir.eq.4) then
         do m=-mmax,mmax
            n1=max(1,abs(m))
            do n=n1,nodrj
               mn=n*(n+1)+m
               do ip=1,2
                  ant(ip,n)=anpt(ip,mn)
               enddo
            enddo
            do n=n1,nodri
               mn=n*(n+1)+m
               a=0.
               b=0.
               do l=n1,nodrj
                  ml=l*(l+1)+m
                  a=a+amnl(1,n,ml)*ant(1,l)
     1               +amnl(2,n,ml)*ant(2,l)
                  b=b+amnl(1,n,ml)*ant(2,l)
     1               +amnl(2,n,ml)*ant(1,l)
               enddo
               anpt(1,mn) = a
               anpt(2,mn) = b
            enddo
         enddo
      else
         do m=-mmax,mmax
            n1=max(1,abs(m))
            do n=n1,nodrj
               mn=n*(n+1)+m
               do ip=1,2
                  ant(ip,n)=anpt(ip,mn)
               enddo
            enddo
            do n=n1,nodri
               mn=n*(n+1)+m
               mnm=n*(n+1)-m
               a=0.
               b=0.
               do l=n1,nodrj
                  a=a+amnl(1,l,mnm)*ant(1,l)
     1               +amnl(2,l,mnm)*ant(2,l)
                  b=b+amnl(1,l,mnm)*ant(2,l)
     1               +amnl(2,l,mnm)*ant(1,l)
               enddo
               anpt(1,mn) = a
               anpt(2,mn) = b
            enddo
         enddo
      endif
c
      in=1
      do n=1,nodri
         in=-in
         nn1=n*(n+1)
         kmax=min(n,nodrj)
         do m=-n,n
            amt(1,m)=0.
            amt(2,m)=0.
         enddo
         ik=-(-1)**kmax
         do k=-kmax,kmax
            ik=-ik
            kn=nn1+k
            knm=nn1-k
            a=ik*anpt(1,kn)
            b=ik*anpt(2,kn)
            do m=-n,n
               amt(1,m)=amt(1,m)+a*drot(-m,knm)
               amt(2,m)=amt(2,m)+b*drot(-m,knm)
            enddo
         enddo
         ik=-in
         do m=-n,n
            ik=-ik
            mn=nn1+m
            anpt(1,mn)=amt(1,m)*ek(-m)*ik
            anpt(2,mn)=amt(2,m)*ek(-m)*ik
         enddo
      enddo
      if(idir.eq.3.or.idir.eq.4) then
         do n=1,nodri
            nn1=n*(n+1)
            im=1
            do m=1,n
               im=-im
               mn=nn1+m
               mnm=nn1-m
               a=anpt(1,mn)
               anpt(1,mn)=im*anpt(1,mnm)
               anpt(1,mnm)=im*a
               a=anpt(2,mn)
               anpt(2,mn)=im*anpt(2,mnm)
               anpt(2,mnm)=im*a
            enddo
         enddo
      endif
      return
      end
c
c  dc is the normalized generalized spherical function
c  dc(k,n*(n+1)+m) = D^k_{mn}
c  where D^k_{mn} is defined in the notes
c  This routine calculates dc(k,n*(n+1)+m) for k=-kmax..kmax and
c  n=0...nmax.
c
c  note that P_n^m = (-1)^m ((n-m)!/(n+m)!) dc(0,n*(n+1)-m)
c

      subroutine rotcoef(cbe,kmax,nmax,dc,ndim)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),notd2=notd+notd,nbc=2*notd2+4)
      implicit real*8(a-h,o-z)
      real*8 dc(-ndim:ndim,0:*)
      real*8 dk0(-notd2:notd2),dk01(-notd2:notd2)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/

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
         dk0(n)=in*sben*bcof(n,n)
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
c
c  rotation of a vector amn through alpha and beta
c  cbe=cos(beta)
c  idir=1:  a_{mnp} = sum_k d^k_{mn}(beta) e^{i k alpha} a_{knp}
c  idir=2:
c        a_{mnp} = e^{-i m alpha} sum_k (-1)^(m+k) d^k_{mn}(beta) a_{knp}
c
      subroutine rotvec(alpha,cbe,nmax,mmax,amn,idir)
      include'scfodim.for'
      parameter(nbtd=notd*(notd+2),nbc=4*notd+4)
      implicit real*8(a-h,o-z)
      real*8 dc(-notd:notd,-notd:notd)
      real*8 dk0(-notd-1:notd+1),dk01(-notd-1:notd+1)
      complex*16 eal(-notd:notd),amn(2,*),amnt(2,-notd:notd),a,b,ci
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/
c
      sbe=sqrt((1.+cbe)*(1.-cbe))
      cbe2=.5*(1.+cbe)
      sbe2=.5*(1.-cbe)
      eal(1)=cdexp(ci*alpha)
      eal(0)=1.
      eal(-1)=conjg(eal(1))
      do k=2,nmax
         eal(k)=eal(k-1)*eal(1)
         eal(-k)=conjg(eal(k))
      enddo
      in=1
      dk0(0)=1.
      sben=1.
      dk01(0)=0.
      do n=1,nmax
         kmax=min(n,mmax)
         nn1=n*(n+1)
         do k=-kmax,kmax
            kn=k+nn1
            if(idir.eq.1) then
               amnt(1,k)=amn(1,kn)*eal(k)
               amnt(2,k)=amn(2,kn)*eal(k)
            else
               amnt(1,-k)=amn(1,kn)
               amnt(2,-k)=amn(2,kn)
            endif
         enddo
         in=-in
         sben=sben*sbe/2.
         dk0(n)=in*sben*bcof(n,n)
         dk0(-n)=in*dk0(n)
         dk01(n)=0.
         dk01(-n)=0.
         dc(0,n)=dk0(n)
         dc(0,-n)=dk0(-n)
         do k=-n+1,n-1
            kn=nn1+k
            dkt=dk01(k)
            dk01(k)=dk0(k)
            dk0(k)=(cbe*(n+n-1)*dk01(k)-fnr(n-k-1)*fnr(n+k-1)*dkt)
     1             /(fnr(n+k)*fnr(n-k))
            dc(0,k)=dk0(k)
         enddo
         im=1
         do m=1,kmax
            im=-im
            fmn=1./fnr(n-m+1)/fnr(n+m)
            m1=m-1
            dkm0=0.
            mn=nn1+m
            mnm=nn1-m
            do k=-n,n
               kn=nn1+k
               dkm1=dkm0
               dkm0=dc(m1,k)
               dc(m,k)=(fnr(n+k)*fnr(n-k+1)*cbe2*dkm1
     1           -fnr(n-k)*fnr(n+k+1)*sbe2*dc(m1,k+1)
     1              -k*sbe*dc(m1,k))*fmn
               dc(-m,-k)=dc(m,k)*(-1)**(k)*im
            enddo
         enddo
         do m=-n,n
            mn=nn1+m
            mnm=nn1-m
            a=0.
            b=0.
            do k=-kmax,kmax
               a=a+dc(k,m)*amnt(1,k)
               b=b+dc(k,m)*amnt(2,k)
            enddo
            if(idir.eq.1) then
               amn(1,mn)=a
               amn(2,mn)=b
            else
               amn(1,mnm)=a*eal(m)
               amn(2,mnm)=b*eal(m)
            endif
         enddo
      enddo
      return
      end

c
c this computes the normalized translation coefficients for an
c axial translation of distance r.  For itype=1 or 3, the translation
c uses the spherical Bessel or Hankel functions as a basis function,
c respectively.    They are related to the coefficients appearing in
c M&M JOSA 96 by
c
c J^{ij}_{mnp mlq} = (E_{ml}/E_{mn})^(1/2) ac(s,n,l*(l+1)+m)
c
c where
c
c   E_{mn} = n(n+1)(n+m)!/((2n+1)(n-m)!)
c   s=mod(p+q,2)+1 (i.e., s=1 for the A coefficient, =2 for the B
c   coefficient)
c
c  ac(2,nd,*) is a complex*16 array.  ac is calculated for n=1,nmax
c  and l=1,lmax.
c
c  The calculation procedure is based on the recent derivation
c  of the addition theorem for vector harmonics, appearing in
c  Fuller and Mackowski, proc. Light Scattering by Nonspherical
c  Particles, NASA/GISS Sept. 1998.
c
      subroutine trancoef(itype,r,nmax,lmax,ac,nd)
      implicit real*8 (a-h,o-z)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),notd2=notd+notd+1,
     1         nfd=notd2*(notd2+2),nbc=4*notd+4)
      real*8 vc1(0:notd2+1),vc2(0:notd2+1),psi(0:notd2)
      complex*16 ci,xi(0:notd2)
      complex*16 a,b,c,ac(2,nd,*)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/
      if(r.eq.0.) then
         lblk=lmax*(lmax+2)
         do n=1,nmax
            do l=1,lblk
               do ip=1,2
                  ac(ip,n,l)=0.
               enddo
            enddo
         enddo
         if(itype.ne.1) return
         do n=1,min(nmax,lmax)
            do l=n*(n+1)-n,n*(n+1)+n
               ac(1,n,l)=1.
            enddo
         enddo
         return
      endif
      if(itype.eq.1) then
         call bessel(nmax+lmax+1,r,nbtot,psi)
         if(nbtot.lt.nmax+lmax+1) then
            do n=nbtot+1,nmax+lmax+1
               psi(n)=0.d0
            enddo
         endif
      else
         call hankel(nmax+lmax+1,r,xi)
      endif
      nlmax=max(nmax,lmax)
      do n=0,nmax+lmax+1
         if(itype.eq.1) then
            xi(n)=psi(n)/r*ci**n
         else
            xi(n)=xi(n)/r*ci**n
         endif
      enddo
      do n=1,nmax
         n21=n+n+1
         do l=1,lmax
            c=fnr(n21)*fnr(l+l+1)*ci**(n-l)
            ll1=l*(l+1)
            call vcfunc(-1,n,1,l,n+l,vc2)
            iwmn=abs(n-l)
            iwmx=n+l
            nlmin=min(l,n)
            do m=-nlmin,nlmin
               a=0.
               b=0.
               call vcfunc(-m,n,m,l,n+l,vc1)
               do iw=iwmn,iwmx
                  alnw=vc1(iw)*vc2(iw)
                  if(mod(n+l+iw,2).eq.0) then
                     a=a+alnw*xi(iw)
                  else
                     b=b+alnw*xi(iw)
                  endif
               enddo
               ac(1,n,ll1+m)=-c*a*(-1)**m
               ac(2,n,ll1+m)=-c*b*(-1)**m
            enddo
         enddo
      enddo
      return
      end

      subroutine tranvec(r,lmax,nmax,amn,eps,err)
      implicit real*8 (a-h,o-z)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),notd2=notd+notd+1,
     1         nfd=notd2*(notd2+2),nbc=4*notd+4)
      real*8 vc1(0:notd2+1),vc2(0:notd2+1),psi(0:notd2)
      complex*16 ci,ant(2,nbtd),amn(2,*)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/
      if(r.eq.0.) return
      call bessel(nmax+lmax+1,r,nbmax,psi)
      nlmax=max(nmax,lmax)
      do n=0,nbmax
         psi(n)=psi(n)/r
      enddo
      lblk=lmax*(lmax+2)
      qs=0.
      do l=1,lblk
         do ip=1,2
            ant(ip,l)=amn(ip,l)
            qs=qs+amn(ip,l)*conjg(amn(ip,l))
         enddo
      enddo
      qst=0.
      do n=1,nmax
         n21=n+n+1
         nn1=n*(n+1)
         mmin=min(n,lmax)
         do m=-mmin,mmin
            mn=nn1+m
            amn(1,mn)=0.
            amn(2,mn)=0.
         enddo
         do l=1,lmax
            c=fnr(n21)*fnr(l+l+1)
            ll1=l*(l+1)
            call vcfunc(-1,n,1,l,n+l,vc2)
            iwmn=abs(n-l)
            iwmx=min(n+l,nbmax)
            nlmin=min(l,n)
            do m=-nlmin,nlmin
               a=0.
               b=0.
               call vcfunc(-m,n,m,l,n+l,vc1)
               do iw=iwmn,iwmx
                  alnw=vc1(iw)*vc2(iw)
                  if(mod(n+l+iw,2).eq.0) then
                     a=a+alnw*psi(iw)*(-1)**(.5*(n-l+iw))
                  else
                     b=b+alnw*psi(iw)*(-1)**(.5*(n-l+iw-1))
                  endif
               enddo
               ml=ll1+m
               mn=nn1+m
               a=-c*a*(-1)**m
               b=-c*b*(-1)**m
               amn(1,mn)=amn(1,mn)+a*ant(1,ml)+ci*b*ant(2,ml)
               amn(2,mn)=amn(2,mn)+a*ant(2,ml)+ci*b*ant(1,ml)
            enddo
         enddo
         do m=-mmin,mmin
            mn=nn1+m
            do ip=1,2
               qst=qst+amn(ip,mn)*conjg(amn(ip,mn))
            enddo
         enddo
         err=abs(1.d0-qst/qs)

         if(err.lt.eps) then
            nmax=n
            goto 20
         endif
      enddo
20    return
      end

c
c vector coupling coefficients vc(iw) = C(m,n|k,l|m+k,iw)
c uses an upwards recurrence
c
      subroutine vcfunc(m,n,k,l,wmax,vcn)
      include 'scfodim.for'
      parameter(nbtd=notd*(notd+2),nbc=4*notd+4)
      implicit real*8(a-h,o-z)
      real*8 vcn(0:*)
      integer w,wmax,w1,w2
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      mk=abs(m+k)
      nl=abs(n-l)
      if(nl.ge.mk) then
         w=nl
         if(n.ge.l) then
            m1=m
            n1=n
            l1=l
            k1=k
         else
            m1=k
            n1=l
            k1=m
            l1=n
         endif
         vc1=(-1)**(k1+l1)*bcof(l1+k1,w-m1-k1)
     1     *bcof(l1-k1,w+m1+k1)/bcof(l1+l1,w+w+1)
      else
         w=mk
         if(m+k.ge.0) then
            vc1=(-1)**(n+m)*bcof(n-l+w,l-k)*bcof(l-n+w,n-m)
     1          /bcof(w+w+1,n+l-w)
         else
            vc1=(-1)**(l+k)*bcof(n-l+w,l+k)*bcof(l-n+w,n+m)
     1          /bcof(w+w+1,n+l-w)
         endif
      endif
      w1=w
      vcn(w)=vc1
      w=w1+1
      mk=m+k
      w2=min(wmax,n+l)
      if(w2.gt.w1) then
         t1=2*w*fnr(w+w+1)*fnr(w+w-1)/(fnr(w+mk)*fnr(w-mk)
     1     *fnr(n-l+w)*fnr(l-n+w)*fnr(n+l-w+1)*fnr(n+l+w+1))
         if(w1.eq.0) then
            t2=.5*dble(m-k)
         else
            t2=dble((m-k)*w*(w-1)-mk*n*(n+1)+mk*l*(l+1))
     1       /dble(2*w*(w-1))
         endif
         vcn(w)=t1*t2*vcn(w1)
      endif
      do w=w1+2,w2
         t1=2*w*fnr(w+w+1)*fnr(w+w-1)/(fnr(w+mk)*fnr(w-mk)
     1     *fnr(n-l+w)*fnr(l-n+w)*fnr(n+l-w+1)*fnr(n+l+w+1))
         t2=dble((m-k)*w*(w-1)-mk*n*(n+1)+mk*l*(l+1))
     1    /dble(2*w*(w-1))
         t3=fnr(w-mk-1)*fnr(w+mk-1)*fnr(l-n+w-1)*fnr(n-l+w-1)
     1     *fnr(n+l-w+2)*fnr(n+l+w)/(dble(2*(w-1))*fnr(2*w-3)
     1     *fnr(2*w-1))
         vcn(w)=t1*(t2*vcn(w-1)-t3*vcn(w-2))
      enddo
      return
      end

      subroutine hankel(n,ds,xi)
      implicit real*8(a-h,o-z)
      complex*16 xi(0:*)
c
c computes riccatti-bessel function xi(0:n)=psi(n)+i*chi(n)
c for real argument ds
c
c check if recurrence is forward or backwards
c
      if(int(ds).ge.n) goto 40
c
c backwards recurrence for log. derivative dn1(n)=psi'(n)/psi(n)
c
      ns=nint(ds+4.*(ds**.3333)+17)
      dns=0.d0
      do 20 i=ns-1,n,-1
         sn=dble(i+1)/ds
         dns=sn-1.d0/(dns+sn)
   20 continue
      xi(n)=dns
      xi(n-1)=dble(n)/ds-1.d0/(dns+dble(n)/ds)
      do 25 i=n-2,1,-1
         sn=dble(i+1)/ds
         xi(i)=sn-1.d0/(xi(i+1)+sn)
   25 continue
c
c forward recurrence: psi(n) computed from dn1(n)
c
      chi0=-dcos(ds)
      psi=dsin(ds)
      chi1=chi0/ds-psi
      xi(0)=dcmplx(psi,chi0)
      do 30 i=1,n
         chi2=dble(i+i+1)/ds*chi1-chi0
         psi=psi/(dble(i)/ds+xi(i))
         xi(i)=dcmplx(psi,chi1)
         chi0=chi1
         chi1=chi2
   30 continue
      return
c
c forward recurrence: applicable only if ds > n
c
   40 chi0=-dcos(ds)
      psi0=dsin(ds)
      chi1=chi0/ds-psi0
      psi1=psi0/ds+chi0
      xi(0)=dcmplx(psi0,chi0)
      xi(1)=dcmplx(psi1,chi1)
      do 50 i=1,n-1
         sn=dble(i+i+1)/ds
         xi(i+1)=sn*xi(i)-xi(i-1)
   50 continue
      return
      end

      subroutine bessel(n,ds,nmax,psi)
      implicit real*8(a-h,o-z)
      real*8 psi(0:*)
c
c computes riccatti-bessel function psi(n)
c for real argument ds
c
c check if recurrence is forward or backwards
c
      if(int(ds).ge.n) goto 40
c
c backwards recurrence for log. derivative dn1(n)=psi'(n)/psi(n)
c
      ns=nint(ds+4.*(ds**.3333)+17)
      dns=0.d0
      do 20 i=ns-1,n,-1
         sn=dble(i+1)/ds
         dns=sn-1.d0/(dns+sn)
   20 continue
      psi(n)=dns
      psi(n-1)=dble(n)/ds-1.d0/(dns+dble(n)/ds)
      do 25 i=n-2,1,-1
         sn=dble(i+1)/ds
         psi(i)=sn-1.d0/(psi(i+1)+sn)
   25 continue
c
c forward recurrence: psi(n) computed from dn1(n)
c
      psit=dsin(ds)
      psi(0)=psit
      ds2=ds*ds
      sum=psit*psit/ds2
      do 30 i=1,n
         psit=psit/(dble(i)/ds+psi(i))
         sum=sum+dble(i+i+1)*psit*psit/ds2
         err=dabs(1.d0-sum)
         psi(i)=psit
         if(err.lt.1.d-14) then
            nmax=i
            return
         endif
   30 continue
      nmax=n
      return
c
c forward recurrence: applicable only if ds > n
c
   40 psi(0)=dsin(ds)
      psi(1)=psi(0)/ds-dcos(ds)
      do 50 i=1,n-1
         sn=dble(i+i+1)/ds
         psi(i+1)=sn*psi(i)-psi(i-1)
   50 continue
      nmax=n
      return
      end
c
c single-sphere lorenz/mie coefficients
c
      subroutine mie1(x,sn,sk,nstop,qeps,qext,qsca,an)
      implicit real*8 (a-h,o-z)
      include 'scfodim.for'
      parameter(nomax2=2*nod)
      complex*16 y,ri,xip,pcp,da,db,pc(0:nomax2),xi(0:nomax2),
     1 an(2,nod),na,nb
c
      ri=cmplx(sn,sk)
      if(qeps.eq.0) then
         nstop=1
         an(1,1)=-dcmplx(0.,1.)*2.*x*x*x*(ri-1.)*(ri+1.)/(ri*ri+2.)/3.
         an(2,1)=1.d-15
         return
      endif
      if(qeps.gt.0.) nstop=nint(x+4.*x**(1./3.))+5.
      nstop=min(nstop,nod)
      y=x*ri
      call cbessel(nstop,y,pc)
      call hankel(nstop,x,xi)
      qsca=0.0
      qext=0.0
      do 300 n=1,nstop
         prn=dble(xi(n))
         pcp=pc(n-1)-n*pc(n)/y
         xip=xi(n-1)-n*xi(n)/x
         prp=dble(xip)
         da=ri*xip*pc(n)-xi(n)*pcp
         db=ri*xi(n)*pcp-xip*pc(n)
         na=ri*prp*pc(n)-prn*pcp
         nb=ri*prn*pcp-prp*pc(n)
         an(1,n)=na/da
         an(2,n)=nb/db
         qsca=qsca+(n+n+1)*(cdabs(an(1,n))*cdabs(an(1,n))
     1        +cdabs(an(2,n))*cdabs(an(2,n)))
         qext1=(n+n+1)*dble(an(1,n)+an(2,n))
         qext=qext+qext1
         err=abs(qext1)/abs(qext)
         if(err.lt.qeps.or.n.eq.nstop) goto 310
  300 continue
  310 nstop=min(n,nod)
      qsca=2./x/x*qsca
      qext=2./x/x*qext
      return
      end
c
c spherical bessel function of a complex argument
c
      subroutine cbessel(n,ds,psi)
      complex*16 ds,psi(0:*),sn,psins
c
      ns=nint(cdabs(ds)+4.*(cdabs(ds)**.3333)+17)
      psins=(0.d0,0.d0)
      do 20 i=ns-1,n,-1
         sn=dble(i+1)/ds
         psins=sn-1.d0/(psins+sn)
   20 continue
      psi(n)=psins
      psi(n-1)=n/ds-1.d0/(psins+n/ds)
      do 25 i=n-2,1,-1
         sn=dble(i+1)/ds
         psi(i)=sn-1.d0/(psi(i+1)+sn)
   25 continue
      psins=cdsin(ds)
      psi(0)=psins
      do 30 i=1,n
         psins=psins/(dble(i)/ds+psi(i))
         psi(i)=psins
   30 continue
      return
      end


c                                                                               c
c  subroutine scatexp(amn0,nodrt,nodrg,gmn) computes the expansion coefficients c
c  for the spherical harmonic expansion of the scattering phase function from   c
c  the scattering coefficients amn0.  For a complete expansion, the max. order  c
c  of the phase function expansion (nodrg) will be 2*nodrt, where nodrt is      c
c  the max. order of the scattered field expansion.   In this code nodrg is     c
c  typically set to 1, so that the subroutine returns the first moments         c
c  of the phase function; gmn(1) and gmn(2).                                    c
c                                                                               c
c  The expansion coefficients are normalized so that gmn(0)=1                   c
c                                                                               c
c  gmn(1)/3 is the asymmetry parameter.                                         c
c                                                                               c

      subroutine scatexp(amn0,nodrt,nodrg,gmn)
      include 'scfodim.for'
      parameter(nbd=notd*(notd+2),notd2=2*notd,ngd=(notd2*(notd2+3))/2,
     1          nbc=2*notd2+4)
      implicit real*8(a-h,o-z)
      complex*16 amn0(2,nbd,2),gmn(0:nodrg*(nodrg+3)/2),a(2,2),c,c2
      real*8 vc1(0:notd2+1),vc2(0:notd2+1)
      integer w,w1,w2,u,uw,ww1
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)

      do w=0,nodrg
         do u=0,w
            uw=(w*(w+1))/2+u
            gmn(uw)=0.
         enddo
      enddo

      do n=1,nodrt
         write(*,'(''+order:'',i5,$)') n
         nn1=n*(n+1)
         l1=max(1,n-nodrg)
         l2=min(nodrt,n+nodrg)
         do l=l1,l2
            ll1=l*(l+1)
            c=fnr(n+n+1)*fnr(l+l+1)*dcmplx(0.d0,1.d0)**(l-n)
            w2=min(n+l,nodrg)
            call vcfunc(-1,l,1,n,w2,vc2)
            do m=-n,n
               mn=nn1+m
               do k=-l,min(l,m)
                  kl=ll1+k
                  ik=(-1)**k
                  c2=ik*c
                  u=m-k
                  do ip=1,2
                     do iq=1,2
                        a(ip,iq)=c2*(amn0(ip,mn,1)*conjg(amn0(iq,kl,1))
     *                  +amn0(ip,mn,2)*conjg(amn0(iq,kl,2)))
                     enddo
                  enddo
                  w1=max(abs(n-l),abs(u))
                  w2=min(n+l,nodrg)
                  call vcfunc(-k,l,m,n,w2,vc1)
                  do w=w1,w2
                     uw=(w*(w+1))/2+u
                     do ip=1,2
                        if(mod(n+l+w,2).eq.0) then
                           iq=ip
                        else
                           iq=3-ip
                        endif
                        gmn(uw)=gmn(uw)-vc1(w)*vc2(w)*a(ip,iq)
                     enddo
                  enddo
               enddo
            enddo
         enddo
      enddo

      g0=dble(gmn(0))
      gmn(0)=1.d0
      do w=1,nodrg
         ww1=(w*(w+1))/2
         gmn(ww1)=dcmplx(dble(gmn(ww1)),0.d0)/g0
         do u=1,w
            uw=ww1+u
            gmn(uw)=(-1)**u*2.d0*gmn(uw)/g0
         enddo
      enddo

      write(*,*)
      return
      end



c *********************** 
c Added by Jerome

c      subroutine xsects(npart, sni, ski, xi, qeps1, amn0)
c npart (integer, number of particles)
c sni (array, real part of relative index)
c ski (array, imaginary part of relative index)
c xi (array, particle size parameters)
c qeps1 (float, single sphere tolerance)
c amn0 (complex array, output of amncalc)
c      include 'scfodim.for'
c      parameter(nbd=nod*(nod+2),nbc=4*notd+4,
c     1          nbtd=notd*(notd+2),nrd=.5*(npd-1)*(npd-2)+npd-1)
c      integer i
c      integer nodr(npart), nblkt(npart)
c      real*8 xi(npart),sni(npart),ski(npart), qe1(npart)
c      real*8 qeps1, qs1, xv, qei(npart, 3)
c      real*8 qet, qetpi2, qetpi4, qat0, qatpi2, qatpi4
c      complex*16 amn0(2,nbtd, 2), an1(2,nod,npd)
c
c      do i=1,npart
c         call mie1(xi(i),sni(i),ski(i),nodr(i),qeps1,qe1(i),
c     1             qs1,an1(1,1,i))
c         xv=xv+xi(i)*xi(i)*xi(i)
c      enddo
c
c      xv=xv**(1./3.)

c
c This computes the total extinction and scattering efficiencies.
c
c      qet0=0.
c      qetpi2=0.
c      qetpi4=0.
c      qat0=0.
c      qatpi2=0.
c      qatpi4=0.
c      do i=1,npart
c         qet0=qet0+qei(i,1)*xi(i)*xi(i)
c         qetpi2=qetpi2+qei(i,2)*xi(i)*xi(i)
c         qetpi4=qetpi4+qei(i,3)*xi(i)*xi(i)
c         qat0=qat0+qai(i,1)*xi(i)*xi(i)
c         qatpi2=qatpi2+qai(i,2)*xi(i)*xi(i)
c         qatpi4=qatpi4+qai(i,3)*xi(i)*xi(i)
c      enddo
c      qet0=qet0/xv/xv
c      qetpi2=qetpi2/xv/xv
c      qetpi4=qetpi4/xv/xv
c      qat0=qat0/xv/xv
c      qatpi2=qatpi2/xv/xv
c      qatpi4=qatpi4/xv/xv

c      call scatexp(amn0,nodrt,1,gmn)
      

c      return
c      end
