'''
Calculates holograms of spheres using CUDA. This GPU implementation
capitalizes on the pixel-wise parallelism of hologram calculations.
The use case is currently limited to groups of spheres that all
have the same radius and index of refraction. The c code in the
kernel is a translation of the Fortran code: mieangfuncs.mie_fields().

.. moduleauthor:: Ray Griner <rg1658@yahoo.com>
.. moduleauthor:: Rebecca W. Perry <perry.becca@gmail.com>
'''

from __future__ import division
from time import time
from numpy import uint32, float64, array, zeros
from scipy import sin, cos
from holopy.scattering.scatterer import Sphere, Scatterers
from holopy.scattering.theory.mie_f import miescatlib

# Import the PyCUDA modules
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpu
import pycuda.driver as cu

# Initialize the CUDA device
import pycuda.autoinit

#########################################################
# kernel_holo calculates the scattering matrix and also
#  incorporates code to calculate the Bessel function
#  and pisandtaus. The kernel also calculates all three
#  components of the scattered electric field.
######################################################### 
kernel_holo_source = \
"""
#include <pycuda-complex.hpp>
#define ABS(a)  (((a) < 0) ? -(a) : (a))
typedef pycuda::complex<double> dcmplx;
__global__ void kernel_holo(const unsigned int N, const dcmplx* asbs,
  const unsigned int nstop, const unsigned int colDim, const double pixelsize, 
  const double* x_arr, const double* y_arr, const double* z_arr, 
  const double* r_arr, const dcmplx* index_arr, const double wavevec, 
  const double med_wavelen, const unsigned int numspheres,
  const double* einc, double* holo)
{
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int row = tid/colDim;
  int col = tid%colDim;
  
  dcmplx einc_cx = dcmplx(einc[0],0);
  dcmplx einc_cy = dcmplx(einc[1],0);
  dcmplx inv_pref, hl, dhl;
  dcmplx ci = dcmplx(0,1);
  dcmplx asm0;
  dcmplx asm1;
  dcmplx esx = dcmplx(0,0);
  dcmplx esy = dcmplx(0,0);
  dcmplx esz = dcmplx(0,0);
  dcmplx holoComplex = dcmplx(0,0);
  double mu, thispi, prevpi, prevpi2, thistau;
  double prefactor, kr, theta;
  dcmplx prefactor2;
  int n;
  int limit, l, done;
  double thisj, prevj;
  double thisy, prevy;
  double thisjp, prevjp;
  double thisyp, prevyp;
  double small, tk, sl;
  double xinv, cf1, dcf1, den, c, d, twoxi;
  double cnt;
  int lmax;
  double phi, einc_sph0, einc_sph1;
  double costh, sinth, cosph, sinph;
  dcmplx asph0, asph1;
  double row_coord;
  double col_coord;
  double x;
  double y;
  double z;
  dcmplx ascatm1 = dcmplx(0,0);
  dcmplx ascatm2 = dcmplx(0,0);
  dcmplx exponent;
  dcmplx phasefactor;
  int sphereIncrementer;

  small = .000000000000001;
  limit = 20000;
  lmax = nstop+1;
  
  
  if (tid<N) {
    for (sphereIncrementer = 0; sphereIncrementer<numspheres; sphereIncrementer++) {
      asm0 = dcmplx(0,0);
      asm1 = dcmplx(0,0);
    
      // grab the parameters relevant to this sphere
      x = double(x_arr[sphereIncrementer]);
      y = double(y_arr[sphereIncrementer]);
      z = double(z_arr[sphereIncrementer]);

      // Calculate the spherical coordinates of each pixel-- using
      // the sphere position as the origin.
      row_coord = row*pixelsize-x;
      col_coord = col*pixelsize-y;
      kr = pow(row_coord*row_coord + col_coord*col_coord + z*z, 0.5)*wavevec;
      theta = atan2(pow(row_coord*row_coord + col_coord*col_coord,0.5), z);
      phi = atan2(col_coord, row_coord);
      //convert so that the range of phi is 0 to 2pi
      if (phi < 0){
        phi += 2*3.14159265;
      }
    
      inv_pref = exp(-1.*ci*kr)*kr;
      mu = cos(theta);
      prevpi = 0; //pi from B&H p. 94
      thispi = 1;
      thistau = mu;

      xinv = 1. / kr;
      if (lmax > 0) {
        twoxi = xinv+xinv;
        sl = lmax*xinv;
        tk = 2. * sl + xinv * 3.;
        cf1 = sl;
        den = 1;
        if (ABS(cf1)<small) cf1 = small;
        c = cf1;
        d = 0;
        done = 0;
        for (l=0; l<limit && !done; l++) {
          c = tk-1./c;
          d = tk-d;
          if (ABS(c) < small) c = small;
          if (ABS(d) < small) d = small;
          d = 1./d;
          dcf1 = d*c;
          cf1 = cf1*dcf1;
          if (d<0) den = -den;
          if (ABS(dcf1-1.)<small) done = 1;
          tk = tk + twoxi;
        }
      }
  
      // j(0): spherical bessel function of the first kind
      // y(0): spherical bessel function of the second kind
      // jp(0): derivative of j(0), eqtn. 10.51.2, p. 265 Nist Handbook of Math Func.
      // yp(0): derivative of y(0), eqtn. 10.51.2, p. 265 Nist Handbook of Math Func.
    
      prevj =  xinv * sin(kr); //equivalent to ASpsi
      prevy = -xinv * cos(kr); //equivalent to ASeta
      prevjp = -prevy - xinv*prevj;
      prevyp = prevj - xinv*prevy;
      thisj = prevj; thisjp=prevjp;
      thisy = prevy; thisyp=prevyp;
      den = thisj;

      // Pi and Tau are for calculation of the amplitude scattering matrix
      // see B&H 4.74
      sl = 0;
      for (n=1; n<nstop+1; n++) {
        // First, update the Bessel function values
        thisj = prevj*sl-prevjp;
        thisjp = prevj-(sl+xinv+xinv)*thisj;
        prevj = thisj; prevjp=thisjp;
        thisy = sl*prevy-prevyp;
        sl = sl+xinv;
        thisyp = prevy-(sl+xinv)*thisy;
        prevy = thisy; prevyp = thisyp;
      
        // Done calculating Bessel functions. Now use it
        prefactor = (2.*n+1.)/(n*(n+1.));
        hl = thisj + ci*thisy; //spherical hankel function of the first kind
        dhl = hl/kr + thisjp + ci*thisyp;
        //B&H p. 112, eqtn. 4.74
        asm0 += (prefactor * pow(ci,n) *(asbs[n-1]*thispi*dhl+
                ci*asbs[nstop+n-1]*thistau*hl)); //S1
        asm1 += (prefactor * pow(ci,n) *(asbs[n-1]*thistau*dhl+
                ci*asbs[nstop+n-1]*thispi*hl));  //S2

        // Calculate the pi's and tau's for the next round
        prevpi2 = prevpi;
        prevpi = thispi;
        cnt = n+1;
        thispi = ((2*cnt-1)/(cnt-1))*mu*prevpi-(cnt/(cnt-1))*prevpi2;
        thistau = cnt*mu*thispi-(cnt+1)*prevpi;
      }

      // Apply inverse prefactor so B&H far field formalism can be used
      ascatm1 = asm1*inv_pref;
      ascatm2 = asm0*inv_pref;

      einc_sph0 = einc[0]*cos(phi) + einc[1]*sin(phi);
      einc_sph1 = einc[0]*sin(phi) - einc[1]*cos(phi);
      prefactor2 = ci / kr * exp(ci * kr) ;
      asph0 = prefactor2*ascatm1*einc_sph0;
      asph1 = -prefactor2*ascatm2*einc_sph1;

      // From here to end is the old fieldstocart fn, finalize_fields fn, and fields_to_holo.
      costh = cos(theta);
      sinth = sin(theta);
      cosph = cos(phi);
      sinph = sin(phi);

	  // This accounts for the phase difference of the incident light as it 
	  // hits each sphere.
	  exponent = dcmplx(0, -3.14159265*2*z / med_wavelen);
      phasefactor = pow(2.718281828, exponent);
	
      esx += (costh*cosph*asph0 - sinph*asph1)*phasefactor;
      esy += (costh*sinph*asph0 + cosph*asph1)*phasefactor;
      esz += (-sinth*asph0)*phasefactor;
    }

	// After all the scattered fields have been added together, interfere with
	// the reference wave to generate the final hologram.
	holoComplex = pow(abs(esx + einc_cx), 2.0) + pow(abs(esy + einc_cy), 2.0) + pow(abs(esz), 2.0);
    holo[tid] = holoComplex.real();
  }
}
"""

def cuda_compile(source_string, function_name):
  # Compile the CUDA Kernel at runtime
  source_module = nvcc.SourceModule(source_string, cache_dir='.', keep=True)
  # Return a handle to the compiled CUDA kernel
  return source_module.get_function(function_name)

def scat_coeffs(s, optics):
  x_arr = array([optics.wavevec * s.r])
  m_arr = array([s.n/optics.index])
  # Check that the scatterer is in a range we can compute for
  if x_arr.max() > 1e3:
    raise UnrealizableScatterer(self, s, "radius too large, field "+
                                      "calculation would take forever")
  lmax = miescatlib.nstop(x_arr[0])
  return miescatlib.scatcoeffs(x_arr[0], m_arr[0], lmax)

def compile():
   # Public fn for Compiling the kernel
   cuda_compile(kernel_holo_source,"kernel_holo")
   returndef compile():
   # Public fn for Compiling the kernel
   cuda_compile(kernel_holo_source,"kernel_holo")
   return

###############################################
# scatterer: position, size, and index of refraction of spheres
# schema: describes the detector (e.g. camera)
###############################################
def calc_holo(scatterer, schema):

  # Extract the necessary components from schema
  imsize = uint32(schema.shape[0])
  pxsize = float64(schema.spacing[0])
  wavevec = float64(schema.optics.wavevec)
  med_wavelen = float64(schema.optics.med_wavelen)
  einc = schema.optics.polarization

  start = time()
  # Extract the necessary components from scatterer
  # and calculate scattering coefficients
  # Single sphere case
  if isinstance(scatterer, Sphere):
    asbs = scat_coeffs(scatterer, schema.optics)
    num = uint32(1)
    sphLocation = scatterer.center
    x = float64(array([sphLocation[0]]))
    y = float64(array([sphLocation[1]]))
    z = float64(array([sphLocation[2]]))
    r = float64(array(scatterer.r))
    n = array(scatterer.n, dtype=complex)
  # Multi-sphere case
  else:
    asbs = scat_coeffs(scatterer.scatterers[0], schema.optics)
    num = uint32(len(scatterer.get_component_list()))
    sphLocation = scatterer.centers
    x = float64(array(sphLocation[:,0]))
    y = float64(array(sphLocation[:,1]))
    z = float64(array(sphLocation[:,2]))
    r = float64(array(scatterer.r))
    n = array(scatterer.n, dtype=complex)
  
  npoints = uint32(imsize**2)
  _, nstop = uint32(asbs.shape)

  # 2D array for storing final hologram
  holo = float64(zeros([imsize,imsize]))

  ##################################################### 
  #  Inputs to the kernel call:  
  #  Python var      Size        Py type    GPU type
  #  npoints           1         np.uint32  unsigned int
  #  asbs          2 x nstop     complex    dcmplx
  #  nstop             1         np.uint32  unsigned int
  #  imsize            1         np.uint32  unsigned int
  #  pxsize            1         float64    double
  #  x,y,z,r,n      1 x num      float64    double
  #  wavevec           1         float64    double
  #  med_wavelen       1         float64    double
  #  num               1         np.uint32  unsigned int
  #  einc              2         float64    double
  #  holo       imsize x imsize  float64    double
  #####################################################

  ############################################################
  ## kernel_holo - Replaces asm_fullradial, calc_scat_field, 
  ## and fieldstocart
  ############################################################

  #uncomment to compile and cache the module
  #kernel_holo = cuda_compile(kernel_holo_source,"kernel_holo")

  #uncomment to load from cached file "precompiled.cubin"
  source_module = cu.module_from_file("precompiled.cubin")
  kernel_holo = source_module.get_function("kernel_holo")

  asbs_d = gpu.to_gpu(asbs.copy())
  einc_d = gpu.to_gpu(einc)
  holo_d = gpu.to_gpu(holo)

  x_d = gpu.to_gpu(x)
  y_d = gpu.to_gpu(y)
  z_d = gpu.to_gpu(z)
  r_d = gpu.to_gpu(r)
  n_d = gpu.to_gpu(n.copy())
  
  nblocks = 2**8
  blocksize = (nblocks,1,1)     
  gridsize  = (int((npoints/nblocks)+(npoints%nblocks)),1)

  start_gpu_time = cu.Event()
  end_gpu_time = cu.Event()
  
  start_gpu_time.record()
  start_test = time()
  
  # assume the spheres are the same size, just pass asbs once
  kernel_holo(npoints, asbs_d, nstop, imsize, pxsize,
    x_d, y_d, z_d, r_d, n_d, wavevec, med_wavelen, num,
    einc_d, holo_d, block=blocksize, grid=gridsize)

  end_gpu_time.record()
  end_gpu_time.synchronize()
  gpu_time = start_gpu_time.time_till(end_gpu_time) * 1e-3 
  holo = holo_d.get()

  stop = time()
  print "holo computation took %f sec. of which %f sec. were on the GPU" % (stop-start, gpu_time)

  return holo
