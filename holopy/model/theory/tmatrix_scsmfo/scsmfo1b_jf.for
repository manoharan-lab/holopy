      program scsmfo

c                                                                            c
c     Refer to the notes on SCSMFO.FOR for details.                          c
c                                                                            c
c     The code is currently set to be compiled using SUN fortran 77.         c
c                                                                            c
c     Usage: a.out inputfile                                                 c
c                                                                            c
c     where:  a.out: executable of this code                                 c
c             inputfile: file containing the input parameters                c
c                                                                            c
c     if inputfile is blank on command line, code prompts for                c
c     input parameters.                                                      c
c                                                                            c
c     Input parameters:                                                      c
c                                                                            c
c     fpos: ascii file containing radius, xp, yp, zp, Re(m), Im(m)           c
c           of spheres in cluster.  If fpos is blank, the following          c
c           options are taken:                                               c
c             Data read from inputfile:  sphere size, position, and RI       c
c             information appears at end of inputfile (one line for          c
c             each sphere).                                                  c
c             Data read from keyboard: code prompts for sphere data.         c
c                                                                            c
c     npart: number of spheres in cluster.  npart data points are read       c
c            from fpos.   If npart> the number of data points in fpos,       c
c            npart is reset to the number of points in fpos.                 c
c                                                                            c
c     xscale, rscale, rirscale, riiscale: scaling factors for sphere data.   c
c         size parameter of sphere i is xscale * a_i, where a_i is the       c
c         radius read from fpos for the ith sphere.  Position of i is        c
c         (x_i, y_i, z_i)*xscale*rscale.   Refractive index of i is          c
c         (rirscale*Re(m_i), riiscale*Im(m_i))                               c
c                                                                            c
c     fout: output file for efficiency and scattering matrix results.        c
c           Leave blank for no output file                                   c
c                                                                            c
c     famn: output file for scattered field expansion coefficients.          c
c           Leave blank if not needed.                                       c
c                                                                            c
c     itermax, eps, meth, qeps1, qeps2: solution parameters:                 c
c       itermax: max number of iterations in solution method (default: 100)  c
c       eps: relative residual error tolerance in solution (10^(-6))         c
c       meth = 1, uses order-of-scattering                                   c
c            = 0, uses biconjugate gradient method                           c
c       qeps1: error tolerance for determining single-sphere harmonic        c
c              order truncation (10^(-4)).  Set to -n_O to fix number        c
c              of harmonics for spheres.  Set to 0 to use Rayleigh-limit     c
c              formula for a_1.                                              c
c       qeps2: error tolerance for determining cluster expansion             c
c              truncation limit (10^(-9)).                                   c
c                                                                            c
c     norien: number of orientations of the incident wave for which          c
c       solutions are calculated                                             c
c                                                                            c
c     alpha(i), beta(i), i=1, norien:  phi=alpha and theta=beta              c
c       propagation directions for each orientation. Units are degrees       c
c                                                                            c
c     nt, ipltopt: scattering matrix plotting options:                       c
c       ipltopt=1: scattering matrix elements are calculated for the         c
c       forward and backward hemispherical projections:  This will give      c
c       a (2nt+1)*(2nt+1) list of the 16 matrix elements for each            c
c       projection and for each orientation.  On a contour map, the          c
c       forward direction will be the pole, and phi and theta correspond     c
c       to longitude and latitude directions.                                c
c                                                                            c
c       ipltopt=2: this calculates the scattering matrix for nt+1            c
c       theta and nt+1 phi values between 0 and pi, and 0 and 2pi,           c
c       respectively.  The contour map for this option will be               c
c       rectangular, with theta and phi representing the vertical            c
c       and horizontal coordinates.                                          c                           c                                                c
c                                                                            c
c     Output: the code writes to fout, for each orientation,                 c
c      Q_ext,c, Q_abs,c: the extinction and absorption efficiency            c
c        of the cluster for unpolarized incident radiation. The              c
c        efficiencies are based on the volume-mean radius of the cluster.    c
c      Q_exc,c and Q_abs,c for polarization angle gamma=0,pi/2 and pi/4.     c
c        These give the linearly-polarized incident wave efficiencies.       c
c        For an arbitrary gamma, the efficiencies would be obtained from     c
c        Q(gamma) = (Q(0) + Q(pi/2) + cos(2gamma)(Q(0)-Q(pi/2))              c
c                 + sin(2 gamma)(2Q(pi/4)-Q(0)-Q(pi/2)))/2                   c
c      <cos theta>,c: the asymmetry factor of the cluster for unpolarized    c
c        incident radiation.                                                 c
c      Q_ext,i, Q_abs,i: the extinction and absorption efficiency            c
c        of each sphere, based on the sphere radius, for gamma=0,pi/2,pi/4.  c
c      Scattering matrix map                                                 c
c                                                                            c
c      The code writes to famn, for each orientation, the                    c
c      expansion coefficients of the scattered field.  These can             c
c      be used (with auxilliary codes) to re-calculate the                   c
c      scattering matrix map or the azimuth--averaged scattering             c
c      matrix expansion.                                                     c
c                                                                            c
c     Important parameter dimensions:  There is an included file,            c
c     scfodim.for, which contains the parameter dimensions:                  c
c                                                                            c
c           parameter(npd=15,nod=30,notd=85)                                 c
c                                                                            c
c     npd is the maximum number of spheres.                                  c
c     nod is the maximum order of the sphere expansions                      c
c     notd is the maximum order of the cluster expansion                     c
c                                                                            c
c     Set these dimensions according to the size of your problem.            c
c     The memory used by the code scales as npd^2*nod^6 and notd^2.  I       c
c     cannot specify precisely how much memory the code will use: you        c
c     will obviously know if it uses too much.                               c
c                                                                            c
c     Revision history:                                                      c
c                                                                            c
c     August 1999: released to public.                                       c
c     October 1999: Polarized efficiency formulas added to code, and         c
c                   a couple of bugs were fixed (thanks to Nikolai           c
c                   Khlebtsov for debugging).                                c
c                                                                            c
c     I have made all efforts to check the accuracy of the code --           c
c     yet I cannot guarantee that it will work for all cases.                c
c                                                                            c
c     Questions/comments:                                                    c
c                                                                            c
c     Daniel W. Mackowski                                                    c
c     Mechanical Engineering Department                                      c
c     Auburn University, AL 36849, USA                                       c
c     dmckwski@eng.auburn.edu                                                c
c
c     Changes made by J. Fung from scsmfo1b.for (original source):
c         -- printing to screen compile error due to WRITE statements
c         -- amn file: modify format string to print more than 5 sig figs
c     Code is intended to be run as an executable; not set up for compilation
c     with f2py.

      implicit real*8 (a-h,o-z)
      include 'scfodim.for'
c
      parameter(nbd=nod*(nod+2),nbd2=nbd+nbd,
     1          nbtd=notd*(notd+2),nbt1=(notd+1)*(notd+3),
     1          ntd=npd*nbd,notd2=notd+notd,nfd=(notd2*(notd2+3))/2,
     1          nbc=2*notd2+4)
      integer nodr(npd)
      real*8 xi(npd),sni(npd),ski(npd),
     1       xp(npd),yp(npd),zp(npd),qai(npd,3),qei(npd,3)
      real*8 ea(2,50),sm(4,4)
      complex*16 ci,amn0(2,nbtd,2),sa(4),gmn(0:2)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      character fpos*30,fposo*30,fout*30,fdat*30,famn*30
      data ci/(0.d0,1.d0)/
      data itermax,eps,rlx,qeps1,qeps2/100,1.d-6,0,1.d-3,1.d-9/
      data itest/0/


c
c calculation of constants
c

      open(6,form='formatted')

      pi=4.d0*datan(1.d0)
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
c
c data input
c
c the function iargc() returns the number of command-line arguments, and         c
c the subroutine getarg(1,fdat) returns the first argument in the string         c
c fdat.   These are intrinsic on the SUN f77 used to compile this code.          c
c The user should substitute the appropriate functions/subroutines available     c
c on their compiler.                                                             c
c
      na=iargc()
      if(na.eq.1) then
         call getarg(1,fdat)
      else
         fdat='con'
      endif
   10 if(fdat.eq.'con') then
         write(*,'('' particle position file:'',$)')
         read(*,'(a)') fpos
         if(fpos.eq.'/') fpos=fposo
         fposo=fpos
         if(fpos.ne.' ') open(1,file=fpos)
         write(*,'('' npart:'',$)')
         read(*,*) npart
         write(*,'('' xscale, rscale, rirscale, riiscale:'',$)')
         read(*,*) xscale,rscale,rirscale,riiscale
         write(*,'('' output file (return for none):'',$)')
         read(*,'(a)') fout
         if(fout.eq.'/') fout=' '
         write(*,'('' amn output file (return for none):'',$)')
         read(*,'(a)') famn
         if(famn.eq.'/') famn=' '
         write(*,'('' itermax, eps, meth, qeps1, qeps2:'',$)')
         read(*,*) itermax,eps,meth,qeps1,qeps2
         write(*,'('' number of fixed orientations:'',$)')
         read(*,*) norien
         do i=1,norien
            write(*,'('' alpha, beta#'',i2,'':'',$)') i
            read(*,*) ea(1,i),ea(2,i)
         enddo
         write(*,'('' nt, ipltopt:'',$)')
         read(*,*) nt,ipltopt
      else
         open(1,file=fdat)
         read(1,'(a)') fpos
         fpos=fpos(:index(fpos,' '))
         read(1,*) npart
         read(1,*) xscale,rscale,rirscale,riiscale
         read(1,'(a)') fout
         if(fout.eq.'/') fout=' '
         fout=fout(:index(fout,' '))
         read(1,'(a)') famn
         if(famn.eq.'/') famn=' '
         famn=famn(:index(famn,' '))
         read(1,*) itermax,eps,meth,qeps1,qeps2
         read(1,*) norien
         do i=1,norien
            read(1,*) ea(1,i),ea(2,i)
         enddo
         read(1,*) nt,ipltopt
         if(fpos.ne.' ') then
            close(1)
            open(1,file=fpos)
         endif
      endif
      xv=0.
      xm=0.
      ym=0.
      zm=0.
      do i=1,npart
         if(fdat.eq.'con'.and.fpos.eq.' ') then
            write(*,'('' x, xp, yp, zp, Re(m), Im(m)#'',i2,'':'',$)') i
            read(*,*) xii,xpi,ypi,zpi,rir,rii
         else
            read(1,*,end=25,err=25) xii,xpi,ypi,zpi,rir,rii
         endif
         xi(i)=xscale*xii
         xp(i)=xscale*rscale*xpi
         yp(i)=xscale*rscale*ypi
         zp(i)=xscale*rscale*zpi
         sni(i)=rir*rirscale
         ski(i)=rii*riiscale
         xm=xm+xp(i)
         ym=ym+yp(i)
         zm=zm+zp(i)
         xv=xv+xi(i)*xi(i)*xi(i)
         if(qeps1.lt.0.) nodr(i)=-qeps1
      enddo
   25 npart=i-1
      close(1)
c
c xv: volume mean size parameter
c
      xv=xv**(1./3.)
      xm=xm/dble(npart)
      ym=ym/dble(npart)
      zm=zm/dble(npart)

      if(fout.ne.' ') then
         open(1,file=fout)
         write(1,'('' scsmfo results'')')
         if(fdat.eq.'con') then
            write(1,'('' position file:'',a)') fpos
            write(1,'('' number of spheres:'',i6)') npart
            write(1,'('' xscale,rscale,riscales'',4e12.4)') xscale,
     *           rscale,rirscale,riiscale
         else
            write(1,'('' input file:'',a)') fdat
         endif
         write(1,'('' xv:'', e12.4)') xv
         close(1)
      endif

      if(famn.ne.' ') then
         open(2,file=famn)
         close(2,status='delete')
      endif

c
c begin the loop over cluster orientations
c
      inew=1
      do io=1,norien

         write(*,*)
         write(*,'('' solving for alpha,beta:'',2f8.1)')
     1        ea(1,io),ea(2,io)
         niter=itermax
         call amncalc(inew,npart,xp,yp,zp,sni,ski,xi,nodr,
     1            nodrt,niter,eps,qeps1,qeps2,meth,
     1            ea(1,io),qei,qai,amn0)
         nblkt=nodrt*(nodrt+2)
c
c This computes the total extinction and scattering efficiencies.
c
         qet0=0.
         qetpi2=0.
         qetpi4=0.
         qat0=0.
         qatpi2=0.
         qatpi4=0.
         do i=1,npart
            qet0=qet0+qei(i,1)*xi(i)*xi(i)
            qetpi2=qetpi2+qei(i,2)*xi(i)*xi(i)
            qetpi4=qetpi4+qei(i,3)*xi(i)*xi(i)
            qat0=qat0+qai(i,1)*xi(i)*xi(i)
            qatpi2=qatpi2+qai(i,2)*xi(i)*xi(i)
            qatpi4=qatpi4+qai(i,3)*xi(i)*xi(i)
         enddo
         qet0=qet0/xv/xv
         qetpi2=qetpi2/xv/xv
         qetpi4=qetpi4/xv/xv
         qat0=qat0/xv/xv
         qatpi2=qatpi2/xv/xv
         qatpi4=qatpi4/xv/xv

         qetc0=0.
         qetcpi2=0.
         qetcpi4=0.
         qstc0=0.
         qstcpi2=0.
         qstcpi4=0.

         do n=1,nodrt
            nn1=n*(n+1)
            do ip=1,2
               qetc0=qetc0+(-ci)**(n+1)*fnr(n+n+1)
     *            *((amn0(ip,nn1-1,1)+amn0(ip,nn1-1,2))*(-1)**ip
     *            +amn0(ip,nn1+1,1)+amn0(ip,nn1+1,2))
               qetcpi2=qetcpi2+(-ci)**(n+1)*fnr(n+n+1)
     *            *((amn0(ip,nn1-1,1)-amn0(ip,nn1-1,2))*(-1)**ip
     *            -amn0(ip,nn1+1,1)+amn0(ip,nn1+1,2))
               qetcpi4=qetcpi4+(-ci)**(n+1)*fnr(n+n+1)
     *            *((amn0(ip,nn1-1,1)-ci*amn0(ip,nn1-1,2))*(-1)**ip
     *            +ci*(amn0(ip,nn1+1,1)-ci*amn0(ip,nn1+1,2)))
               do m=-n,n
                  mn=nn1+m
                  qstc0=qstc0+cdabs(amn0(ip,mn,1)+amn0(ip,mn,2))**2.
                  qstcpi2=qstcpi2+cdabs(amn0(ip,mn,1)-amn0(ip,mn,2))**2.
                  qstcpi4=qstcpi4
     *              +cdabs(amn0(ip,mn,1)-ci*amn0(ip,mn,2))**2.
               enddo
            enddo
         enddo
         qetc0=-2.*qetc0/xv/xv
         qetcpi2=-2.*qetcpi2/xv/xv
         qetcpi4=-2.*qetcpi4/xv/xv
         qstc0=4.*qstc0/xv/xv
         qstcpi2=4.*qstcpi2/xv/xv
         qstcpi4=4.*qstcpi4/xv/xv
         qatc0=qetc0-qstc0
         qatcpi2=qetcpi2-qstcpi2
         qatcpi4=qetcpi4-qstcpi4

c the asymmetry factor is calculated in scatexp
c
         call scatexp(amn0,nodrt,1,gmn)
c
c The code has calculated extinciton and scattering efficiencies based on       c
c two formulations: 1) by summation of the individual sphere results, and 2)    c
c by analytical integration of the cluster scattered field expansion.  If the   c
c two results do not agree, it indicates that either 1) the interaction         c
c equations did not converge to a solution, or 2) the cluster field expansion   c
c was truncated at an insufficiently large value of nodrt.                      c
c
         qet=(qet0+qetpi2)/2.
         qat=(qat0+qatpi2)/2.
         qst=qet-qat
         qet=(qet0+qetpi2)/2.
         write(*,'(''              qe(0)      qe(pi/2)    qe(pi/4)'',
     *             ''     qa(0)      qa(pi/2)    qa(pi/4)'')') 
         write(*,'('' sphere : '',6e12.4)')
     *    qet0,qetpi2,qetpi4,qat0,qatpi2,qatpi4
         write(*,'('' cluster: '',6e12.4)')
     *    qetc0,qetcpi2,qetcpi4,qatc0,qatcpi2,qatcpi4
         write(*,'('' <cos theta>:'',e12.4)') dble(gmn(1))/3.

         if(fout.ne.' ')  then
            open(1,file=fout,access='append')
            write(1,'('' alpha,beta:'')')
            write(1,'(2f8.1)') ea(1,io),ea(2,io)
            write(1,'('' ave. cluster qext,qabs,qsca,<cos theta>:'')')
            write(1,'(4e13.5)') qet,qat,qst,dble(gmn(1))/3.d0
            write(1,'('' cluster qe(0),qe(pi/2),qe(pi/4),'',
     *             ''qa(0),qa(pi/2),qa(pi/4)'')') 
            write(1,'(6e13.5)') qet0,qetpi2,qetpi4,qat0,qatpi2,qatpi4
            write(1,'('' sphere qext and qabs:'')')
            write(1,'('' sphere   qe(0)       qe(pi/2)     qe(pi/4)'',
     *             ''      qa(0)       qa(pi/2)     qa(pi/4)'')') 
            do i=1,npart
               write(1,'(i5, 6e13.5)') i, qei(i,1),qei(i,2),
     *          qei(i,3),qai(i,1),qai(i,2),qai(i,3)
            enddo
            write(1,'('' scattering map grid size, grid option:'')')
            write(1,*) nt,ipltopt
         endif
c
c  this writes the cluster scattering coefficients to the file famn
c

         if(famn.ne.' ') then
            open(2,file=famn,access='append')
            write(2,'(i5,f10.3,2f8.2)') nodrt,xv,ea(1,io),ea(2,io)
            do ik=1,2
               do n=1,nodrt
                  nn1=n*(n+1)
                  do m=-n,n
                     mn=nn1+m
                     do ip=1,2
                        write(2,'(2e22.14)') amn0(ip,mn,ik)
                     enddo
                  enddo
               enddo
            enddo
            close(2)
         endif

         if(ipltopt.eq.1.and.nt.gt.0) then

            do idir=1,-1,-2
               if(idir.eq.1) then
                  write(*,'('' calculating forward scattering map'')')
                  if(fout.ne.' ') then
                     write(1,'('' forward scattering'')')
                     write(1,'(''  S11         S22         S33'',
     *     ''         S44         S21         S32         S43'',
     *     ''         S31         S42         S41         S12'',
     *     ''         S23         S34         S13         S24'',
     *     ''         S14'')')
                  endif
               else
                  write(*,'('' calculating backward scattering map'')')
                  if(fout.ne.' ') then
                     write(1,'('' backward scattering'')')
                     write(1,'(''  S11         S22         S33'',
     *     ''         S44         S21         S32         S43'',
     *     ''         S31         S42         S41         S12'',
     *     ''         S23         S34         S13         S24'',
     *     ''         S14'')')
                  endif
               endif
               do ix=-nt,nt
                  x=ix/dble(nt)
                  do iy=-nt,nt
                     y=iy/dble(nt)
                     r=sqrt(x*x+y*y)
                     if(r.le.1.) then
                        ct=sqrt((1.-r)*(1.+r))*idir
                        th=dacos(ct)*180./pi
                        if(x.eq.0..and.y.eq.0.) then
                           phi=0.
                        else
                           phi=datan2(x,y)
                        endif
                        call scatfunc(amn0,nodrt,xv,ct,phi,sa,sm)
                        sphi=sin(phi)
                        cphi=cos(phi)
                     else
                        do i=1,4
                           do j=1,4
                              sm(i,j)=0.
                           enddo
                        enddo
                     endif
                     if(fout.ne.' ') write(1,'(16e12.4)')
     *                  (sm(i,i),i=1,4),
     *                  (sm(i+1,i),i=1,3),(sm(i+2,i),i=1,2),sm(4,1),
     *                  (sm(i,i+1),i=1,3),(sm(i,i+2),i=1,2),sm(1,4)
                  enddo
               enddo
            enddo
         elseif(ipltopt.eq.2.and.nt.gt.0) then
            write(*,'('' calculating scattering map'')')
            if(fout.ne.' ') then
               write(1,'('' scattering map'')')
               write(1,'(''  S11         S22         S33'',
     *     ''         S44         S21         S32         S43'',
     *     ''         S31         S42         S41         S12'',
     *     ''         S23         S34         S13         S24'',
     *     ''         S14'')')
                  endif
            do it=0,nt
               th=pi*it/dble(nt)
               ct=cos(th)
               do iphi=0,nt
                  phi=2.*pi*iphi/dble(nt+1)
                  call scatfunc(amn0,nodrt,xv,ct,phi,sa,sm)
                  write(1, '(8e12.4)') sa(1), sa(2), sa(3), sa(4)
                  if(fout.ne.' ') write(1,'(16e12.4)') (sm(i,i),i=1,4),
     *               (sm(i+1,i),i=1,3),(sm(i+2,i),i=1,2),sm(4,1),
     *               (sm(i,i+1),i=1,3),(sm(i,i+2),i=1,2),sm(1,4)
               enddo
            enddo
         endif
         if(fout.ne.' ') close(1)
      enddo

  200 if(fdat.eq.'con') then
         write(*,'('' more (0/1):'',$)')
         read(*,*) more
         if(more.eq.1) goto 10
      endif
      stop
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

      subroutine scatfunc(amn0,nodrt,xv,ct,phi,sa,sm)
      include 'scfodim.for'
      implicit real*8(a-h,o-z)
c
      parameter(nbd=nod*(nod+2),nbd2=nbd+nbd,
     1          nbtd=notd*(notd+2),notd2=notd+notd,
     1          nbc=2*notd2+4)
      real*8 drot(-1:1,0:nbtd),tau(2),sm(4,4)
      complex*16 ci,amn0(2,nbtd,2),cin,sa(4),
     1           sp(4,4),ephi(-notd-1:notd+1),a,b
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/

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
            tau(1)=fnr(n+n+1)*(drot(-1,mnm)-drot(1,mnm))
            tau(2)=fnr(n+n+1)*(drot(-1,mnm)+drot(1,mnm))
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


      do i=1,4
         do j=1,4
            sp(i,j)=sa(i)*conjg(sa(j))*.5/xv/xv
         enddo
      enddo
      sm(1,1)=sp(1,1)+sp(2,2)+sp(3,3)+sp(4,4)
      sm(1,2)=-sp(1,1)+sp(2,2)-sp(3,3)+sp(4,4)
      sm(2,1)=-sp(1,1)+sp(2,2)+sp(3,3)-sp(4,4)
      sm(2,2)=sp(1,1)+sp(2,2)-sp(3,3)-sp(4,4)
      sm(3,3)=2.*(sp(1,2)+sp(3,4))
      sm(3,4)=-2.*dimag(sp(1,2)+sp(3,4))
      sm(4,3)=2.*dimag(sp(1,2)-sp(3,4))
      sm(4,4)=2.*(sp(1,2)-sp(3,4))
      sm(1,3)=2.*(sp(2,3)+sp(1,4))
      sm(3,1)=2.*(sp(2,4)+sp(1,3))
      sm(1,4)=2.*dimag(sp(2,3)-sp(1,4))
      sm(4,1)=-2.*dimag(sp(2,4)+sp(1,3))
      sm(2,3)=2.*(sp(2,3)-sp(1,4))
      sm(3,2)=2.*(sp(2,4)-sp(1,3))
      sm(2,4)=2.*dimag(sp(2,3)+sp(1,4))
      sm(4,2)=-2.*dimag(sp(2,4)-sp(1,3))

      return
      end
c
c calculation of cluster T matrix via iteration scheme
c
      subroutine amncalc(inew,npart,xp,yp,zp,sni,ski,xi,nodr,
     1            nodrtmax,niter,eps,qeps1,qeps2,meth,
     1            ea,qei,qai,amn0)
      implicit real*8(a-h,o-z)
      include 'scfodim.for'
      parameter(nbd=nod*(nod+2),nbc=4*notd+4,
     1          nbtd=notd*(notd+2),nrd=.5*(npd-1)*(npd-2)+npd-1)
      parameter (nrotd=nod*(2*nod*nod+9*nod+13)/6,
     1           ntrad=nod*(nod*nod+6*nod+5)/6)
      integer nodr(npd),nblk(npd),nodrt(npd),nblkt(npd)
      real*8 xi(*),sni(*),ski(*),rp(npd),qe1(npd),
     1       xp(*),yp(*),zp(*),qei(npd,3),qai(npd,3)
      real*8 ea(2),drott(-nod:nod,0:nbd)
      complex*16 ci,cin,a,an1(2,nod,npd),pfac(npd)
      complex*16 amn(2,nbd,npd,2),amn0(2,nbtd,2)
      complex*16 pmn(2,nbd,npd),pp(2,nbd,2),amnlt(2,nod,nbd)
      real*8 drot(nrotd,nrd),dbet(-1:1,0:nbd)
      complex*16 ephi,anpt(2,nbtd),amnl(2,ntrad,nrd),
     1           ek(nod,nrd),ealpha(-nod:nod)
      common/consts/bcof(0:nbc,0:nbc),fnr(0:2*nbc)
      data ci/(0.d0,1.d0)/

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
      write(*,'('' single sphere max. order:'',i5)') nodrmax
      if(nodrmax.eq.nod) then
         write(*,'('' Warning: single--sphere error tolerance'',
     1   '' may not be attained'')')
         write(*,'('' decrease qeps1 and/or increase nod'')')
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
         write(*,'('' Warning: notd dimension may be too small.'',
     1       '' increase to '',i5)') nodrtmax
      endif
      nodrtmax=min(nodrtmax,notd)
      write(*,'('' estimated cluster expansion order:'',i5)') nodrtmax
      nblktmax=nodrtmax*(nodrtmax+2)
c
      do i=1,npart
         write(*,'(''+assembling interaction matrix row:'',i4,$)') i
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
      write(*,*)

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

      do k=1,2
         write(*,'('' solving for incident state '',i1)') k

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
            itermax=max(itermax,iter)
         endif

14       nodrt1=0
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

      do i=1,npart
         do k=1,3
            qai(i,k)=0.
            qei(i,k)=0.
         enddo
         do n=1,nodr(i)
            nn1=n*(n+1)
            do m=-n,n
               mn=nn1+m
               do ip=1,2
                  cn=dble(1.d0/an1(ip,n,i)-1.d0)
                  qai(i,1)=qai(i,1)
     *                  +cn*cdabs(amn(ip,mn,i,1)+amn(ip,mn,i,2))**2.
                  qai(i,2)=qai(i,2)
     *                  +cn*cdabs(amn(ip,mn,i,1)-amn(ip,mn,i,2))**2.
                  qai(i,3)=qai(i,3)
     *                  +cn*cdabs(amn(ip,mn,i,1)-ci*amn(ip,mn,i,2))**2.
                  a=conjg(pfac(i))
                  qei(i,1)=qei(i,1)+a*(amn(ip,mn,i,1)+amn(ip,mn,i,2))
     *                *conjg(pp(ip,mn,1)+pp(ip,mn,2))
                  qei(i,2)=qei(i,2)+a*(amn(ip,mn,i,1)-amn(ip,mn,i,2))
     *                *conjg(pp(ip,mn,1)-pp(ip,mn,2))
                  qei(i,3)=qei(i,3)+a*(amn(ip,mn,i,1)-ci*amn(ip,mn,i,2))
     *                *conjg(pp(ip,mn,1)-ci*pp(ip,mn,2))
               enddo
            enddo
         enddo
         do k=1,3
            qai(i,k)=4.*qai(i,k)/xi(i)/xi(i)
            qei(i,k)=4.*qei(i,k)/xi(i)/xi(i)
         enddo
      enddo

      write(*,'('' cluster expansion order:'',i5)') nodrtmax

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
  30  enddo
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
50    enddo
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
      write(*,'(''+iteration:'',i4,'' error:'',e13.5,$)') iter,err
      if(err.lt. eps) then
         write(*,*)
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
      write(*,*)
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
350   enddo
      err=err/enorm
      iter=iter+1
      write(*,'(''+iteration:'',i4,'' error:'',e13.5,$)') iter,err
      if((err.gt.eps).and.(iter.lt.niter)) goto 310
      write(*,*)
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
5     nlmax=max(nmax,lmax)
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
5     nlmax=max(nmax,lmax)
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
