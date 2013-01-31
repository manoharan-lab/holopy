.. _credits:

**********************
References and credits
**********************
Please see the following references:

.. [Fung2012] Fung, J., Perry, R. W., Dimiduk, T. G., & Manoharan, V. N. (2012). Imaging multiple colloidal particles by fitting electromagnetic scattering solutions to digital holograms. Journal of Quantitative Spectroscopy and Radiative Transfer, (0). doi:10.1016/j.jqsrt.2012.06.007

.. [Perry2012] Perry, R. W., Meng, G., Dimiduk, T. G., Fung, J., & Manoharan, V. N. (2012). Real-space studies of the structure and dynamics of self-assembled colloidal clusters. Faraday Discussions. doi:10.1039/c2fd20061a

.. [Fung2011] J\. Fung *et al.*, "Measuring translational, rotational, and vibrational dynamics in colloids with digital holographic microscopy," *Optics Express* **19**, 8051-8065, (2011).

.. [Lee2007] S\. H\. Lee *et al.*, "Characterizing and tracking single colloidal particles with video holographic microscopy," *Optics Express* **15**, 18275-18282, (2007).

.. [Mackowski1996] D\. W\. Mackowski and M\. I\. Mishchenko, "Calculation of the T matrix and the scattering matrix for ensembles of spheres," *J. Opt. Soc. Am. A.* **13**, 2266-2278, (1996).

.. [Yang2003] W\. Yang, "Improved recursive algorithm for light scattering by a multilayered sphere," *Applied Optics* **42**, 1710-1720, (2003).

.. [Yurkin2011] M\. A\. Yurkin and A\. G\. Hoekstra, "The discrete-dipole-approximation code ADDA: Capabilities and known limitations," *J. Quant. Spectrosc. Radiat. Transfer* **112**, 2234-2247 (2011).

If you use HoloPy, we ask that you cite the articles above that are
relevant to your application.

For scattering calculations and formalism, we draw heavily on
the treatise of Bohren & Huffman.  We generally follow their conventions
except where noted.

.. [Bohren1983] C\. F\. Bohren and D\. R\. Huffman, *Absorption and Scattering of Light by Small Particles*, Wiley (1983).

The package includes code from several sources.  We thank Daniel
Mackowski for allowing us to include his T-Matrix code, which computes
scattering from clusters of spheres:  SCSMFO1B_.

.. _SCSMFO1B: ftp://ftp.eng.auburn.edu/pub/dmckwski/scatcodes/index.html

We also make use of a modified version of the Python version of
mpfit_, originally developed by Craig Markwardt. The modified version
we use is drawn from the stsci_python_ package.

.. _mpfit: http://www.physics.wisc.edu/~craigm/idl/fitting.html
.. _stsci_python: http://www.stsci.edu/resources/software_hardware/pyraf/stsci_python

We thank A. Ross Barnett for permitting us to use his routine
SBESJY.FOR_, which computes spherical Bessel functions.

.. _SBESJY.FOR: http://www.fresco.org.uk/programs/barnett/index.htm


We include a copy Michele Simionato's decorator.py_ (v 3.3.3) which
simplifies writing correct function decorators.  If you have that
module installed it will be used preferentially.

  decorator.py is Copyright (c) 2005-2012, Michele Simionato
  All rights reserved.

.. _decorator.py: http://pypi.python.org/pypi/decorator/3.3.3

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   Redistributions of source code must retain the above copyright 
   notice, this list of conditions and the following disclaimer.
   Redistributions in bytecode form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution. 
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
   OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
   TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
   DAMAGE.
