# Copyright 2011, Vinothan N. Manoharan, Thomas G. Dimiduk, Rebecca W. Perry,
# Jerome Fung, and Ryan McGorty
#
# This file is part of Holopy.
#
# Holopy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Holopy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Holopy.  If not, see <http://www.gnu.org/licenses/>.
################################################
#
#
# Cython code for MieFieldExtension.
# 
#
#
################################################
import numpy as np
cimport numpy as np

#np.import_array() #One tutorial told me to use this. But it fails to load when I do.

import scipy as sc

#cimport python_unicode #for reading strings

ctypedef np.float64_t dtype_t #Used in a tutorial, but I don't seem to need it. Still here irregardless.


#The following are the functions imported from MieFieldExtension.c and used in this module.
cdef extern from "MieFieldExtension.h":

    #determines number of terms to include in Mie calculations:
    int nstop(float)

    #calculates hologram and stores it to text file:
    void outputhol (double, double, double, double, double, double, double, int, int, double, float, char*)
    
    float hologramFitError (double, double, double, double,
        double, double, double, int, int, int, double, float, char*)

    #calculates hologram and returns it as a numpy array:   
    void outputhol_toarray (double, double, double, double,
        double, double, double, int, int, int, double, double*)

    #calculates hologram:
    void miehol(double*, double, double, double, int, int, double, float, double,
            double*, double*, double, double)
    
    #calculates hologram from two beads:
    void miehol_dimer(double*, double, double, float,
                  float, int, int, double,
                  float, double, 
                  double*, double*,
                  double*, double*,
                  double*, double*,
                  double*, double*)
    
    #calculates fields  
    void flds (double*, double*, double*,
           double*, double*, double*, 
           int, float, int, float,
           float, int, float,
           float, float,
           float*, float*, 
           float*, float*)

    #calculates scattering matrix (on a regularly spaced grid)
    void *scattering_matrix (double*, double*,
           double*, double*, 
           int, float, int, float,
           float, int, float,
           float, float,
           float*, float*, 
           float*, float*)

    #calculating intermediate Mie variables:
    void Dn2 (int, float, float, float, float*, float*)
    void AS(int, float, float*, float*)
    void a_and_b(float, float, float, int, float*, float*, 
        float*, float*, float*, float*, float*, float*)

    #reads in .bin file of DATA
    void *read_hol_bin(double*, char*, int, int)

#memory accouting:
cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t, size)
    size_t strlen(char *s)
    char *strcpy(char *dest, char *src)

#Just a test function really. No need to ever use this.
def nstop_f (float x):
    return nstop(x)


def output_hologram_to_file (double xstart, double ystart, double zparam, double re_np,
        double re_nm, double radius, double alpha, int xdim, int ydim, double mpp,
        float wavelength, filename):
    '''
    This function will output a calculated hologram to a text file.
    To read in that data just go: data_being_read_in = scipy.fromfile('filename',sep='\\n').reshape(X, Y)

    Input Parameters:
        x -- in pixel this will range from 0 to x-dimension
        y -- see above
        z -- z-distance from focal plane. In MICRONS.
        np -- refractive index of particle
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        alpha -- alpha value (scales hologram)
        xdim -- x-dimension (in pixels)
        ydim -- y-dimension (in pixels)
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150
        wavelength
        filename -- where the hologram will be stored

    Example of use:
        Let's say you want to calculate a 512x512 hologram with a 1 micron diameter particle
        in the middle that is 20 microns from the focal plane. Then this is what you might
        want to type:
        %> MFE.output_hologram_to_file(256,256,20,1.58,1.33,0.5,0.8,512,512,0.0681818,658e-9,'output.txt')

    '''
    filename_byte = filename.encode('UTF-8')    
    cdef char* fn = filename_byte
    outputhol(xstart, ystart, zparam, re_np, re_nm, radius, alpha, xdim, ydim, mpp, wavelength, fn)

def outputhol_tonumpy (double xstart, double ystart, double zparam, double re_np, double re_nm, 
            double radius, double alpha, int x_dim, int y_dim, float wavelength, double mpp):
    '''
    This function will output a calculated hologram and return it as a 1D numpy array.
    
    Input Parameters:
        x -- in pixel this will range from 0 to x-dimension
        y -- see above
        z -- z-distance from focal plane. In MICRONS.
        np -- refractive index of particle
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        alpha -- alpha value (scales hologram)
        x_dim -- x-dimension (in pixels)
        y_dim -- y-dimension (in pixels)
        wavelength
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150

    Example of use:
        Let's say you want to calculate a 512x512 hologram with a 1 micron diameter particle
        in the middle that is 20 microns from the focal plane. Then this is what you might
        want to type:
        %> hologram = MFE.outputhol_tonumpy(256,256,20,1.58,1.33,0.5,0.8,512,512,658e-9,0.0681818)

        Then, you could view it by:

        %> pylab.imshow(hologram.reshape(512,512))


    '''
    cdef int nc
    cdef float *ASpsi
    cdef float *ASeta
    cdef float *re_dns
    cdef float *im_dns
    cdef float *re_a
    cdef float *im_a
    cdef float *re_b
    cdef float *im_b
    cdef double z = zparam*1e-6
    cdef double re_m
    cdef double im_m = 0
    cdef double sizeparam
    cdef double *re_Ec1, *re_Ec2, *re_Ec3
    cdef double *im_Ec1, *im_Ec2, *im_Ec3
    cdef unsigned int i=0
    cdef double z_in_pixels, k_in_pixels
    cdef int n = x_dim*y_dim
    cdef np.ndarray holodata = np.zeros((n))
    cdef double *holodata_c
    cdef float xs_sa = 0
    cdef float ys_sa = 0

    sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength)
    re_m = re_np / re_nm

    nterms = nstop(sizeparam)
    if nterms%2 == 0:
        nterms = nterms+1

    re_Ec1 = <double*>malloc(n*sizeof(double))
    re_Ec2 = <double*>malloc(n*sizeof(double))
    re_Ec3 = <double*>malloc(n*sizeof(double))
    im_Ec1 = <double*>malloc(n*sizeof(double))
    im_Ec2 = <double*>malloc(n*sizeof(double))
    im_Ec3 = <double*>malloc(n*sizeof(double))
    ASpsi = <float*>malloc(nterms*sizeof(float))
    ASeta = <float*>malloc(nterms*sizeof(float))
    re_dns = <float*>malloc(nterms*sizeof(float))
    im_dns = <float*>malloc(nterms*sizeof(float))
    re_a = <float*>malloc(nterms*sizeof(float))
    im_a = <float*>malloc(nterms*sizeof(float))
    re_b = <float*>malloc(nterms*sizeof(float))
    im_b = <float*>malloc(nterms*sizeof(float))
    holodata_c = <double*>malloc(n*sizeof(double))

    for i from 0 <= i < nterms:
        ASpsi[i]=0
        ASeta[i]=0
        re_a[i]=0
        im_a[i]=0
        re_b[i]=0
        im_b[i]=0


    nc = nterms

    Dn2( nterms, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0])
    AS(nterms,sizeparam,&ASpsi[0],&ASeta[0])

    for i from 0 <=i < nc:
        im_dns[i] *= -1

    z_in_pixels = z*1e6/mpp
    k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6))
    a_and_b(sizeparam,re_m,im_m,nterms,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0])

    free(ASpsi)
    free(ASeta)
    free(re_dns)
    free(im_dns)

    flds(&re_Ec1[0], &re_Ec2[0], &re_Ec3[0], &im_Ec1[0], &im_Ec2[0], &im_Ec3[0],
nterms,-1*xstart,x_dim,1,-1*ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0])

    miehol(holodata_c, z, 1e-6, re_nm, x_dim, y_dim, mpp, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xs_sa, xs_sa)

    for i from 0<=i<n:
        holodata[i] = holodata_c[i]

    free(re_Ec1)
    free(im_Ec1)
    free(re_Ec2)
    free(im_Ec2)
    free(re_Ec3)
    free(im_Ec3)
    free(re_a)
    free(im_a)
    free(re_b)
    free(im_b)
    free(holodata_c)
    
    return holodata

def fields_tonumpy (double xstart, double ystart, double zparam, double re_np,
            double im_np, double re_nm, double radius, int x_dim, int y_dim,
            float wavelength, double mpp):
    '''
    This function will output a calculated hologram and return it as a 1D numpy array.
    
    Input Parameters:
        x -- in pixel this will range from 0 to x-dimension
        y -- see above
        z -- z-distance from focal plane. In MICRONS.
        np -- refractive index of particle
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        x_dim -- x-dimension (in pixels)
        y_dim -- y-dimension (in pixels)
        wavelength
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150

    Example of use:
        Let's say you want to calculate a 512x512 hologram with a 1 micron diameter particle
        in the middle that is 20 microns from the focal plane. Then this is what you might
        want to type:
        %> hologram = MFE.outputhol_tonumpy(256,256,20,1.58,1.33,0.5,0.8,512,512,658e-9,0.0681818)

        Then, you could view it by:

        %> pylab.imshow(hologram.reshape(512,512))


    '''
    cdef int nc
    cdef float *ASpsi
    cdef float *ASeta
    cdef float *re_dns
    cdef float *im_dns
    cdef float *re_a
    cdef float *im_a
    cdef float *re_b
    cdef float *im_b
    cdef double z = zparam*1e-6
    cdef double re_m
    cdef double im_m 
    cdef double sizeparam
    cdef double *re_Ec1, *re_Ec2, *re_Ec3
    cdef double *im_Ec1, *im_Ec2, *im_Ec3
    cdef unsigned int i=0
    cdef double z_in_pixels, k_in_pixels
    cdef int n = x_dim*y_dim
    cdef np.ndarray eflds = np.zeros((n*6))
    cdef float xs_sa = 0
    cdef float ys_sa = 0

    sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength)
    re_m = re_np / re_nm
    im_m = im_np / re_nm

    nterms = nstop(sizeparam)
    if nterms%2 == 0:
        nterms = nterms+1

    re_Ec1 = <double*>malloc(n*sizeof(double))
    re_Ec2 = <double*>malloc(n*sizeof(double))
    re_Ec3 = <double*>malloc(n*sizeof(double))
    im_Ec1 = <double*>malloc(n*sizeof(double))
    im_Ec2 = <double*>malloc(n*sizeof(double))
    im_Ec3 = <double*>malloc(n*sizeof(double))
    ASpsi = <float*>malloc(nterms*sizeof(float))
    ASeta = <float*>malloc(nterms*sizeof(float))
    re_dns = <float*>malloc(nterms*sizeof(float))
    im_dns = <float*>malloc(nterms*sizeof(float))
    re_a = <float*>malloc(nterms*sizeof(float))
    im_a = <float*>malloc(nterms*sizeof(float))
    re_b = <float*>malloc(nterms*sizeof(float))
    im_b = <float*>malloc(nterms*sizeof(float))

    for i from 0 <= i < nterms:
        ASpsi[i]=0
        ASeta[i]=0
        re_a[i]=0
        im_a[i]=0
        re_b[i]=0
        im_b[i]=0

    nc = nterms

    Dn2( nterms, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0])
    AS(nterms,sizeparam,&ASpsi[0],&ASeta[0])

    for i from 0 <=i < nc:
        im_dns[i] *= -1

    z_in_pixels = z*1e6/mpp
    k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6))
    a_and_b(sizeparam,re_m,im_m,nterms,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0])

    free(ASpsi)
    free(ASeta)
    free(re_dns)
    free(im_dns)

    flds(re_Ec1, re_Ec2, re_Ec3, im_Ec1, im_Ec2, im_Ec3,
nterms,-1*xstart,x_dim,1,-1*ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0])

    for i from 0<=i<n:
        eflds[i] = re_Ec1[i]
    for i from n<=i<2*n:
        eflds[i] = re_Ec2[i-n]
    for i from 2*n<=i<3*n:
        eflds[i] = re_Ec3[i-(2*n)]
    for i from 3*n<=i<4*n:
        eflds[i] = im_Ec1[i-(3*n)]
    for i from 4*n<=i<5*n:
        eflds[i] = im_Ec2[i-(4*n)]
    for i from 5*n<=i<6*n:
        eflds[i] = im_Ec3[i-(5*n)]
    

    free(re_Ec1)
    free(im_Ec1)
    free(re_Ec2)
    free(im_Ec2)
    free(re_Ec3)
    free(im_Ec3)
    free(re_a)
    free(im_a)
    free(re_b)
    free(im_b)
    
    return eflds
    
def outputhol_dimer_tonumpy (double xstart, double ystart, double zparam, 
            double xstart2, double ystart2, double zparam2,
            double re_np, double re_np2, double re_nm, 
            double radius, double radius2,
            double alpha, int x_dim, int y_dim, float wavelength, double mpp):
    '''
    This function will output a calculated hologram and return it as a 1D numpy array.
    
    Input Parameters:
        x -- in pixel this will range from 0 to x-dimension
        y -- see above
        z -- z-distance from focal plane. In MICRONS.
        x2, y2, z3,
        np -- refractive index of particle
        np2
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        radius2
        alpha -- alpha value (scales hologram)
        x_dim -- x-dimension (in pixels)
        y_dim -- y-dimension (in pixels)
        wavelength
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150

    Example of use:
        Let's say you want to calculate a 512x512 hologram with a 1 micron diameter particle
        in the middle that is 20 microns from the focal plane. Then this is what you might
        want to type:
        %> hologram = MFE.outputhol_tonumpy(256,256,20,1.58,1.33,0.5,0.8,512,512,658e-9,0.0681818)

        Then, you could view it by:

        %> pylab.imshow(hologram.reshape(512,512))


    '''
    cdef int nc
    cdef float *ASpsi
    cdef float *ASeta
    cdef float *re_dns
    cdef float *im_dns
    cdef float *re_a
    cdef float *im_a
    cdef float *re_b
    cdef float *im_b
    cdef double z = zparam*1e-6
    cdef double z2 = zparam2*1e-6
    cdef double re_m, re_m2
    cdef double im_m = 0
    cdef double sizeparam, sizeparam2
    cdef double *re_Ec1, *re_Ec2, *re_Ec3
    cdef double *im_Ec1, *im_Ec2, *im_Ec3
    cdef double *re_Ec1_2, *re_Ec2_2, *re_Ec3_2
    cdef double *im_Ec1_2, *im_Ec2_2, *im_Ec3_2
    cdef unsigned int i=0
    cdef double z_in_pixels, z2_in_pixels, k_in_pixels
    cdef int n = x_dim*y_dim
    cdef np.ndarray holodata = np.zeros((n))
    cdef double *holodata_c
    cdef float xs_sa = 0
    cdef float ys_sa = 0

    sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength)
    sizeparam2 = (2*3.14159265*re_nm*radius2)/(1e6 * wavelength)
    re_m = re_np / re_nm
    re_m2 = re_np2 / re_nm

    nterms = nstop(sizeparam)
    if nterms%2 == 0:
        nterms = nterms+1

    re_Ec1 = <double*>malloc(n*sizeof(double))
    re_Ec2 = <double*>malloc(n*sizeof(double))
    re_Ec3 = <double*>malloc(n*sizeof(double))
    im_Ec1 = <double*>malloc(n*sizeof(double))
    im_Ec2 = <double*>malloc(n*sizeof(double))
    im_Ec3 = <double*>malloc(n*sizeof(double))
    ASpsi = <float*>malloc(nterms*sizeof(float))
    ASeta = <float*>malloc(nterms*sizeof(float))
    re_dns = <float*>malloc(nterms*sizeof(float))
    im_dns = <float*>malloc(nterms*sizeof(float))
    re_a = <float*>malloc(nterms*sizeof(float))
    im_a = <float*>malloc(nterms*sizeof(float))
    re_b = <float*>malloc(nterms*sizeof(float))
    im_b = <float*>malloc(nterms*sizeof(float))
    holodata_c = <double*>malloc(n*sizeof(double))
    
    re_Ec1_2 = <double*>malloc(n*sizeof(double))
    re_Ec2_2 = <double*>malloc(n*sizeof(double))
    re_Ec3_2 = <double*>malloc(n*sizeof(double))
    im_Ec1_2 = <double*>malloc(n*sizeof(double))
    im_Ec2_2 = <double*>malloc(n*sizeof(double))
    im_Ec3_2 = <double*>malloc(n*sizeof(double))

    for i from 0 <= i < nterms:
        ASpsi[i]=0
        ASeta[i]=0
        re_a[i]=0
        im_a[i]=0
        re_b[i]=0
        im_b[i]=0

    nc = nterms

    Dn2( nterms, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0])
    AS(nterms,sizeparam,&ASpsi[0],&ASeta[0])

    for i from 0 <=i < nc:
        im_dns[i] *= -1

    z_in_pixels = z*1e6/mpp
    k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6))
    a_and_b(sizeparam,re_m,im_m,nterms,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0])

    free(ASpsi)
    free(ASeta)
    free(re_dns)
    free(im_dns)

    flds(&re_Ec1[0], &re_Ec2[0], &re_Ec3[0], &im_Ec1[0], &im_Ec2[0], &im_Ec3[0],
nterms,-1*xstart,x_dim,1,-1*ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0])


    nterms = nstop(sizeparam2)
    if nterms%2 == 0:
        nterms = nterms+1
        
    free(im_a)
    free(re_a)
    free(im_b)
    free(re_b)
    
    ASpsi = <float*>malloc(nterms*sizeof(float))
    ASeta = <float*>malloc(nterms*sizeof(float))
    re_dns = <float*>malloc(nterms*sizeof(float))
    im_dns = <float*>malloc(nterms*sizeof(float))
    re_a = <float*>malloc(nterms*sizeof(float))
    im_a = <float*>malloc(nterms*sizeof(float))
    re_b = <float*>malloc(nterms*sizeof(float))
    im_b = <float*>malloc(nterms*sizeof(float))
        
    for i from 0 <= i < nterms:
        ASpsi[i]=0
        ASeta[i]=0
        re_a[i]=0
        im_a[i]=0
        re_b[i]=0
        im_b[i]=0

    nc = nterms

    Dn2( nterms, sizeparam2, re_m2, im_m, &re_dns[0], &im_dns[0])
    AS(nterms,sizeparam2,&ASpsi[0],&ASeta[0])

    for i from 0 <=i < nc:
        im_dns[i] *= -1

    z2_in_pixels = z2*1e6/mpp
    k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6))
    a_and_b(sizeparam2,re_m2,im_m,nterms,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0])

    free(ASpsi)
    free(ASeta)
    free(re_dns)
    free(im_dns)

    flds(&re_Ec1_2[0], &re_Ec2_2[0], &re_Ec3_2[0], &im_Ec1_2[0], &im_Ec2_2[0], &im_Ec3_2[0],
nterms,-1*xstart2,x_dim,1,-1*ystart2,y_dim,1,k_in_pixels,z2_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0])

    miehol_dimer(holodata_c, z,z2, 1e-6, re_nm, x_dim, y_dim, mpp, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], &re_Ec1_2[0], &im_Ec1_2[0],&re_Ec2[0], &im_Ec2[0],&re_Ec2_2[0], &im_Ec2_2[0])

    for i from 0<=i<n:
        holodata[i] = holodata_c[i]

    free(re_Ec1)
    free(im_Ec1)
    free(re_Ec2)
    free(im_Ec2)
    free(re_Ec3)
    free(im_Ec3)
    free(re_Ec1_2)
    free(im_Ec1_2)
    free(re_Ec2_2)
    free(im_Ec2_2)
    free(re_Ec3_2)
    free(im_Ec3_2)
    free(re_a)
    free(im_a)
    free(re_b)
    free(im_b)
    free(holodata_c)
    
    return holodata

def hologram_fit_error (double xstart, double ystart, double zparam, double re_np, double re_nm, double radius, double alpha, int n, int x_dim, int y_dim, double mpp, filename):
    '''
    Function used to fit a hologram (saved as a binary data file).

    INPUTS:
        x -- in pixel this will range from 0 to NEGATIVE x-dimension
        y -- see above
        z -- z-distance from focal plane. In MICRONS.
        np -- refractive index of particle
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        alpha -- alpha value (scales hologram)
        n -- number of pixels (x-dimension times y-dimension)
        xdim -- x-dimension (in pixels)
        ydim -- y-dimension (in pixels)
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150
        file name of data -- like 'data_01_1.bin' for instance

    This will return a length n, 1D numpy array [n is total number of pixels]. This is just the difference between
    the data and the calculation at each pixel.  You can then use a minimization function to minimize the sum of the
    squares.

    '''
    cdef int nc
    cdef float *ASpsi
    cdef float *ASeta
    cdef float *re_dns
    cdef float *im_dns
    cdef float *re_a
    cdef float *im_a
    cdef float *re_b
    cdef float *im_b
    cdef double z = zparam*1e-6
    cdef double re_m
    cdef double im_m = 0
    cdef double sizeparam
    cdef double *re_Ec1, *re_Ec2, *re_Ec3
    cdef double *im_Ec1, *im_Ec2, *im_Ec3
    cdef unsigned int i=0
    cdef double z_in_pixels, k_in_pixels
    cdef np.ndarray error = np.zeros((n))
    cdef double *holodata_c
    cdef double *input_holo
    cdef float wavelength = 658e-9
    cdef float xs_sa = 0
    cdef float ys_sa = 0

    filename_byte = filename.encode('UTF-8')    
    cdef char* fn = filename_byte

    input_holo = <double*>malloc(n*sizeof(double))

    read_hol_bin(input_holo, fn, x_dim, y_dim);

    nterms = 50

    sizeparam = (2*3.14159265*re_nm*radius)/(1e6 * wavelength)
    re_m = re_np / re_nm

    re_Ec1 = <double*>malloc(n*sizeof(double))
    re_Ec2 = <double*>malloc(n*sizeof(double))
    re_Ec3 = <double*>malloc(n*sizeof(double))
    im_Ec1 = <double*>malloc(n*sizeof(double))
    im_Ec2 = <double*>malloc(n*sizeof(double))
    im_Ec3 = <double*>malloc(n*sizeof(double))
    ASpsi = <float*>malloc(nterms*sizeof(float))
    ASeta = <float*>malloc(nterms*sizeof(float))
    re_dns = <float*>malloc(nterms*sizeof(float))
    im_dns = <float*>malloc(nterms*sizeof(float))
    re_a = <float*>malloc(nterms*sizeof(float))
    im_a = <float*>malloc(nterms*sizeof(float))
    re_b = <float*>malloc(nterms*sizeof(float))
    im_b = <float*>malloc(nterms*sizeof(float))
    holodata_c = <double*>malloc(n*sizeof(double))

    for i from 0 <= i < nterms:
        ASpsi[i]=0
        ASeta[i]=0
        re_a[i]=0
        im_a[i]=0
        re_b[i]=0
        im_b[i]=0


    nc = nstop(sizeparam)

    Dn2( nc, sizeparam, re_m, im_m, &re_dns[0], &im_dns[0])
    AS(nc,sizeparam,&ASpsi[0],&ASeta[0])

    for i from 0 <=i < nterms:
        im_dns[i] *= -1

    z_in_pixels = z*1e6/mpp
    k_in_pixels = (2 * 3.14159265) / (wavelength / re_nm / (mpp*1e-6))
    a_and_b(sizeparam,re_m,im_m,nc,&re_a[0],&im_a[0],&re_b[0],&im_b[0],&ASpsi[0],&ASeta[0],&re_dns[0],&im_dns[0])

    free(ASpsi)
    free(ASeta)
    free(re_dns)
    free(im_dns)

    flds(&re_Ec1[0], &re_Ec2[0], &re_Ec3[0], &im_Ec1[0], &im_Ec2[0], &im_Ec3[0],
nc,xstart,x_dim,1,ystart,y_dim,1,k_in_pixels,z_in_pixels,&re_a[0],&im_a[0],&re_b[0],&im_b[0])

    miehol(holodata_c, z, 1e-6, re_nm, x_dim, y_dim, mpp, wavelength, alpha, &re_Ec1[0], &im_Ec1[0], xs_sa, xs_sa)

    for i from 0<=i<n:
        error[i] = holodata_c[i] - input_holo[i]

    free(re_Ec1)
    free(im_Ec1)
    free(re_Ec2)
    free(im_Ec2)
    free(re_Ec3)
    free(im_Ec3)
    free(re_a)
    free(im_a)
    free(re_b)
    free(im_b)
    free(holodata_c)
    free(input_holo)    

    return error


def fit_xyz (params, double np, double nm, double rad, double alp, int n, int xd, int yd, double mpp, filename):
    '''
    Function used to fit a hologram (saved as a binary data file).

    INPUTS:
        1D array containing x,y,z in pixels,pixels,microns 
        np -- refractive index of particle
        nm -- refractive index of media
        radius -- radius of particle in MICRONS
        alpha -- alpha value (scales hologram)
        n -- number of pixels (x-dimension times y-dimension)
        xdim -- x-dimension (in pixels)
        ydim -- y-dimension (in pixels)
        microns per pixel -- so, for 150x and 10.6 micron pixels, it'd be 10.6e-6/150
        file name of data -- like 'data_01_1.bin' for instance

    This will return a length n, 1D numpy array [n is total number of pixels]. This is just the difference between
    the data and the calculation at each pixel.  You can then use a minimization function to minimize the sum of the
    squares.

    This just uses hologram_fit_error but puts x,y,z in an array so that scipy.optimize.leastsq can be used.

    Example:
        %> MFE.fit_xyz([-256, -256, 20], 1.5, 1.3, 1., 0.7, 512*512, 512, 512, 0.0681818, 'data_04_10.bin')
    '''
    xs = params[0]
    ys = params[1]
    z = params[2]
    return hologram_fit_error(xs, ys, z, np, nm, rad, alp, n, xd, yd, mpp, filename)
