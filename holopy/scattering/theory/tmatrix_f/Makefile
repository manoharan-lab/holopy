#makefile to make S, which is called by HoloPy to run Tmatrix scattering calculations
ifeq ($(OS),Windows_NT)
	TARGET = S.exe
else
	TARGET = S
endif

$(TARGET): S.lp.f lpd.f
	gfortran -o $(TARGET) S.lp.f lpd.f

# requires msys2 on Windows
clean:
	$(RM) $(TARGET) *.o
