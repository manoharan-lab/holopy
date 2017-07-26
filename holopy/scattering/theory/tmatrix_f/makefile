#makefile to make S.exe, which is called by HoloPy to run Tmatrix scattering calculations

ifdef COMSPEC
	#Windows
	RM=del S.lp.o lpq.o
else
	#Not Windows
	RM = rm S.lp.o lpq.o
endif

#make files
S.exe : S.lp.o lpq.o
	gfortran -o S.exe S.lp.o lpq.o
	#clean up object files
	$(RM)

S.lp.o : S.lp.f
	gfortran -c S.lp.f

lpq.o : lpq.f
	gfortran -c lpq.f

#end of makefile
