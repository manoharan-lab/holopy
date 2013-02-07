C       COMPUTATION OF SPECIAL FUNCTIONS
C 
C          Shanjie Zhang and Jianming Jin
C
C       Copyrighted but permission granted to use code in programs. 
C       Buy their book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
C
C
C      Compiled into a single source file and changed REAL To DBLE throughout.
C
C      Changed according to ERRATA also.
C      
C
C       **********************************

        SUBROUTINE CSPHJY(N,Z,NM,CSJ,CDJ,CSY,CDY)
C
C       ==========================================================
C       Purpose: Compute spherical Bessel functions jn(z) & yn(z)
C                and their derivatives for a complex argument
C       Input :  z --- Complex argument
C                n --- Order of jn(z) & yn(z) ( n = 0,1,2,... )
C       Output:  CSJ(n) --- jn(z)
C                CDJ(n) --- jn'(z)
C                CSY(n) --- yn(z)
C                CDY(n) --- yn'(z)
C                NM --- Highest order computed
C       Routines called:
C                MSTA1 and MSTA2 for computing the starting
C                point for backward recurrence
C       ==========================================================
C
        IMPLICIT COMPLEX*16 (C,Z)
        DOUBLE PRECISION A0
        DIMENSION CSJ(0:N),CDJ(0:N),CSY(0:N),CDY(0:N)
        A0=CDABS(Z)
        NM=N
        IF (A0.LT.1.0D-60) THEN
           DO 10 K=0,N
              CSJ(K)=0.0D0
              CDJ(K)=0.0D0
              CSY(K)=-1.0D+300
10            CDY(K)=1.0D+300
           CSJ(0)=(1.0D0,0.0D0)
           IF (N.GT.0) THEN
              CDJ(1)=(.333333333333333D0,0.0D0)
           ENDIF
           RETURN
        ENDIF
        CSJ(0)=CDSIN(Z)/Z
        CDJ(0)=(CDCOS(Z)-CDSIN(Z)/Z)/Z
        CSY(0)=-CDCOS(Z)/Z
        CDY(0)=(CDSIN(Z)+CDCOS(Z)/Z)/Z
        IF (N.LT.1) THEN
           RETURN
        ENDIF
        CSJ(1)=(CSJ(0)-CDCOS(Z))/Z
        IF (N.GE.2) THEN
           CSA=CSJ(0)
           CSB=CSJ(1)
           M=MSTA1(A0,200)
           IF (M.LT.N) THEN
              NM=M
           ELSE
              M=MSTA2(A0,N,15)
           ENDIF
           CF0=0.0D0
           CF1=1.0D0-100
           DO 15 K=M,0,-1
              CF=(2.0D0*K+3.0D0)*CF1/Z-CF0
              IF (K.LE.NM) CSJ(K)=CF
              CF0=CF1
15            CF1=CF
           IF (CDABS(CSA).GT.CDABS(CSB)) CS=CSA/CF1
           IF (CDABS(CSA).LE.CDABS(CSB)) CS=CSB/CF0
           DO 20 K=0,NM
20            CSJ(K)=CS*CSJ(K)
        ENDIF
        DO 25 K=1,NM
25         CDJ(K)=CSJ(K-1)-(K+1.0D0)*CSJ(K)/Z
        CSY(1)=(CSY(0)-CDSIN(Z))/Z
        CDY(1)=(2.0D0*CDY(0)-CDCOS(Z))/Z
        DO 30 K=2,NM
           IF (CDABS(CSJ(K-1)).GT.CDABS(CSJ(K-2))) THEN
              CSY(K)=(CSJ(K)*CSY(K-1)-1.0D0/(Z*Z))/CSJ(K-1)
           ELSE
              CSY(K)=(CSJ(K)*CSY(K-2)-(2.0D0*K-1.0D0)/Z**3)/CSJ(K-2)
           ENDIF
30      CONTINUE
        DO 35 K=2,NM
35         CDY(K)=CSY(K-1)-(K+1.0D0)*CSY(K)/Z
        RETURN
        END


        INTEGER FUNCTION MSTA1(X,MP)
C
C       ===================================================
C       Purpose: Determine the starting point for backward  
C                recurrence such that the magnitude of    
C                Jn(x) at that point is about 10^(-MP)
C       Input :  x     --- Argument of Jn(x)
C                MP    --- Value of magnitude
C       Output:  MSTA1 --- Starting point   
C       ===================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        N0=INT(1.1D0*A0)+1
        F0=ENVJ(N0,A0)-MP
        N1=N0+5
        F1=ENVJ(N1,A0)-MP
        DO 10 IT=1,20             
           NN=N1-(N1-N0)/(1.0D0-F0/F1)                  
           F=ENVJ(NN,A0)-MP
           IF(ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
 10        F1=F
 20     MSTA1=NN
        RETURN
        END


        INTEGER FUNCTION MSTA2(X,N,MP)
C
C       ===================================================
C       Purpose: Determine the starting point for backward
C                recurrence such that all Jn(x) has MP
C                significant digits
C       Input :  x  --- Argument of Jn(x)
C                n  --- Order of Jn(x)
C                MP --- Significant digit
C       Output:  MSTA2 --- Starting point
C       ===================================================
C
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        A0=DABS(X)
        HMP=0.5D0*MP
        EJN=ENVJ(N,A0)
        IF (EJN.LE.HMP) THEN
           OBJ=MP
           N0=INT(1.1*A0)+1
        ELSE
           OBJ=HMP+EJN
           N0=N
        ENDIF
        F0=ENVJ(N0,A0)-OBJ
        N1=N0+5
        F1=ENVJ(N1,A0)-OBJ
        DO 10 IT=1,20
           NN=N1-(N1-N0)/(1.0D0-F0/F1)
           F=ENVJ(NN,A0)-OBJ
           IF (ABS(NN-N1).LT.1) GO TO 20
           N0=N1
           F0=F1
           N1=NN
10         F1=F
20      MSTA2=NN+10
        RETURN
        END

        REAL*8 FUNCTION ENVJ(N,X)
        DOUBLE PRECISION X
        ENVJ=0.5D0*DLOG10(6.28D0*N)-N*DLOG10(1.36D0*X/N)
        RETURN
        END
