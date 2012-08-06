C---------------------------------------------------------------------
        SUBROUTINE SBESJY  (X,LMAX, J,Y,JP,YP, IFAIL )
C---------------------------------------------------------------------
C   REAL SPHERICAL BESSEL FUNCTIONS AND X DERIVATIVES
C            j , y , j', y'                    FROM L=0 TO L=LMAX
C        FOR REAL X > SQRT(ACCUR) (E.G. 1D-7)    AND INTEGER LMAX
C 
C  J (L)  =      j/L/(X) STORES   REGULAR SPHERICAL BESSEL FUNCTION:
C  JP(L)  = D/DX j/L/(X)            j(0) =  SIN(X)/X
C  Y (L)  =      y/L/(X) STORES IRREGULAR SPHERICAL BESSEL FUNCTION:
C  YP(L)  = D/DX y/L/(X)            y(0) = -COS(X)/X
C                                                
C    IFAIL = -1 FOR ARGUMENTS OUT OF RANGE
C          =  0 FOR ALL RESULTS SATISFACTORY
C 
C   USING LENTZ-THOMPSON EVALUATION OF CONTINUED FRACTION CF1,
C   AND TRIGONOMETRIC FORMS FOR L = 0 SOLUTIONS.
C   LMAX IS LARGEST L NEEDED AND MUST BE <= MAXL, THE ARRAY INDEX.
C   MAXL CAN BE DELETED AND ALL THE ARRAYS DIMENSIONED (0:*)
C   SMALL IS MACHINE DEPENDENT, ABOUT SQRT(MINIMUM REAL NUMBER),
C         SO 1D-150 FOR DOUBLE PRECISION ON VAX, PCS ETC.
C   PRECISION:  RESULTS TO WITHIN 2-3 DECIMALS OF "MACHINE ACCURACY"
C   IN OSCILLATING REGION X .GE.  [ SQRT{LMAX*(LMAX+1)} ]
C   I.E. THE TARGET ACCURACY ACCUR SHOULD BE 100 * ACC8 WHERE ACC8
C   IS THE SMALLEST NUMBER WITH 1+ACC8.NE.1 FOR OUR WORKING PRECISION
C   THIS RANGES BETWEEN 4E-15 AND 2D-17 ON CRAY, VAX, SUN, PC FORTRANS
C   SO CHOOSE A SENSIBLE  ACCUR = 1.0D-14
C   IF X IS SMALLER THAN [ ] ABOVE THE ACCURACY BECOMES STEADILY WORSE:
C   THE VARIABLE ERR IN COMMON /STEED/ HAS AN ESTIMATE OF ACCURACY.
C
C   NOTE: FOR X=1 AND L=100  J = 7.4 E-190     Y = -6.7+E186    1.4.94
C---------------------------------------------------------------------
C   AUTHOR :   A.R.BARNETT       MANCHESTER    12 MARCH 1990/95
C                                AUCKLAND      12 MARCH 1991
C---------------------------------------------------------------------  
        IMPLICIT    NONE
        INTEGER     LIMIT,         MAXL,       LMAX, IFAIL, NFP, L
        PARAMETER ( LIMIT = 20000, MAXL = 1001 )
        DOUBLE PRECISION  J(0:LMAX), Y(0:LMAX), JP(0:LMAX), YP(0:LMAX)
        DOUBLE PRECISION  ZERO,ONE,TWO,THREE,SMALL, ACCUR, TK,SL, ERR
        DOUBLE PRECISION  X,XINV, CF1,DCF1, DEN, C,D, OMEGA, TWOXI
        PARAMETER ( ZERO  = 0.0D0  , ONE   = 1.0D0 , TWO = 2.0D0 )
        PARAMETER ( SMALL = 1.D-150, THREE = 3.0D0 )
C        COMMON /STEDE/    ERR,NFP       ! not required in code
Cf2py   intent(in) X
Cf2py   intent(in) LMAX
Cf2py   intent(out) J
Cf2py   intent(out) Y
Cf2py   intent(out) JP
Cf2py   intent(out) YP
Cf2py   intent(out) IFAIL
        
C-------
        ACCUR = 1.D-14                  ! suitable for double precision
        IFAIL = -1                      ! user to check on exit
        IF (X .LT. DSQRT(ACCUR) )       GOTO 50
C-------TAKES CARE OF NEGATIVE X ... USE REFLECTION FORMULA
C-------BEGIN CALCULATION OF CF1 UNLESS LMAX = 0, WHEN SOLUTIONS BELOW
            XINV  = ONE / X
        IF (LMAX .GT. 0) THEN
            TWOXI =     XINV + XINV
            SL  =  REAL(LMAX)* XINV     ! used also in do loop 3
            TK  =  TWO * SL  + XINV * THREE     
            CF1 =  SL                   ! initial value of CF1
            DEN =  ONE                  ! unnormalised j(Lmax,x)
            IF ( ABS(CF1) .LT. SMALL ) CF1 = SMALL
            C   = CF1                   ! inverse ratio of A convergents
            D   = ZERO                  ! direct  ratio of B convergents   
            DO 10 L = 1,LIMIT
                C   = TK - ONE / C
                D   = TK - D
                        IF ( ABS(C) .LT. SMALL ) C = SMALL
                        IF ( ABS(D) .LT. SMALL ) D = SMALL
                D   = ONE / D
                DCF1= D   * C
                CF1 = CF1 * DCF1
                        IF ( D .LT. ZERO ) DEN = - DEN
                        IF ( ABS(DCF1 - ONE) .LE. ACCUR ) GOTO 20
               TK   = TK + TWOXI
               NFP  = L                 ! ie number in loop
   10       CONTINUE
                    GOTO 50             ! error exit, no convergence
   20       CONTINUE
            ERR = ACCUR * DSQRT(DBLE(NFP))    ! error estimate
                    J (LMAX) = DEN      ! lower-case j's  really
                    JP(LMAX) = CF1 * DEN                         
C------ DOWNWARD RECURSION TO L=0  AS SPHERICAL BESSEL FUNCTIONS
            DO 30  L =  LMAX , 1, -1
                   J (L-1)  = (SL + XINV) * J(L)   + JP(L)
                        SL  =  SL - XINV
                   JP(L-1)  =  SL * J(L-1)          - J(L)
   30       CONTINUE
            DEN = J(0)
        ENDIF                           ! end loop for Lmax GT 0
C------ CALCULATE THE L=0 SPHERICAL BESSEL FUNCTIONS DIRECTLY
        J (0)   =  XINV * DSIN(X)
        Y (0)   = -XINV * DCOS(X)
        JP(0)   = -Y(0) - XINV * J(0)
        YP(0)   =  J(0) - XINV * Y(0)
        IF (LMAX .GT. 0) THEN
                   OMEGA  =  J(0) / DEN
                      SL  = ZERO
            DO 40 L = 1 , LMAX
                    J (L) = OMEGA * J (L)
                    JP(L) = OMEGA * JP(L)
                    Y (L) = SL * Y(L-1)   -   YP(L-1)
                      SL  = SL + XINV
                    YP(L) = Y(L-1)  -  (SL + XINV) * Y(L)
   40       CONTINUE
        ENDIF
        IFAIL = 0                       ! calculations successful
        RETURN
C---------------------------------------------------------------------
C       ERROR TRAPS
C---------------------------------------------------------------------
   50   IF (X .LT. ZERO) THEN
                WRITE(6,1000) X
          ELSEIF (X .EQ. ZERO) THEN
                       IFAIL = 0
                        J(0) = ONE
                DO 60 L = 1, LMAX
                        J(L) = ZERO     ! remaining arrays untouched
   60           CONTINUE            
          ELSE                          ! x .le. sqrt(accur), e.g. 1D-7
                WRITE(6,1001) X
          ENDIF
 1000   FORMAT(' X NEGATIVE !',1PE15.5,'    USE REFLECTION FORMULA'/)
 1001   FORMAT(' WITH X = ',1PE15.5,'    TRY SMALL-X SOLUTIONS',/,
     X  '    j/L/(X)  ->   X**L / (2L+1)!!          AND',/,
     X  '    y/L/(X)  ->  -(2L-1)!! / X**(L+1)'/)
        RETURN
        END
C---------------------------------------------------------------------
C       END OF SUBROUTINE SBESJY 
C---------------------------------------------------------------------
