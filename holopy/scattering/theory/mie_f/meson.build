'''
Build instructions for T-Matrix scattering extensions (in fortran77
and fortran90)

We previously used numpy.distutils (now deprecated), which used f2py to
generate the C bindings, automatically detect the fortran compiler, and
compile the extensions. Meson does not integrate with f2py yet, so the C
bindings need to be generated using a "custom_target" directive. This
directive runs (for example, for uts_scmfo:

python -m numpy.f2py uts_scsmfo.for ../../third_party/SBSEJY.F -m uts_scsmfo --lower --build-dir [build-dir] 

Note that f2py by default compiles with optimization
flags.  Not sure about meson.

Ignore compiler warnings of unused variables, unused dummy
arguments, and variables being used uninitialized from compiling
scsmfo_min. The former is relics of how scsmfo was written which I
am not touching. The latter is likely due to some GOTO statements that
could cause a variable to be referenced before it's initialized. Under
normal usage I wouldn't worry about it.
'''

# automatically generate the C wrappers from f2py
# (see https://numpy.org/doc/stable/f2py/buildtools/meson.html)
# Important to specify "--lower" here; otherwise SBESJY routine will not be
# found. Also found I needed to put '--build-dir', '@OUTDIR@'; otherwise the
# generated wrapper files get put in the build directory and can't be found by
# the compiler.
uts_scsmfo_source = custom_target('uts_scsmfomodule.c',
  input: ['uts_scsmfo.for', '../../third_party/SBESJY.F',],
  output: ['uts_scsmfomodule.c',],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'uts_scsmfo', '--lower',
            '--build-dir', '@OUTDIR@'],
)

py.extension_module('uts_scsmfo',
  ['uts_scsmfo.for',
   '../../third_party/SBESJY.F',
   uts_scsmfo_source],
   incdir_f2py / 'fortranobject.c',
   c_args: numpy_nodepr_api,
   include_directories: inc_np,
   install: true,
   dependencies: py_dep,
   subdir: 'holopy/scattering/theory/mie_f'
)

mieangfuncs_source = custom_target(
  'mieangfuncsmodule.c',
  input : ['mieangfuncs.f90',
           'uts_scsmfo.for',
           '../../third_party/SBESJY.F',
           '../../third_party/csphjy.for'],
  output: ['mieangfuncsmodule.c', 'mieangfuncs-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'mieangfuncs',
            '--lower', '--build-dir', '@OUTDIR@'],
)

py.extension_module('mieangfuncs',
  ['mieangfuncs.f90',
   'uts_scsmfo.for',
   '../../third_party/SBESJY.F',
   '../../third_party/csphjy.for',],
   mieangfuncs_source,
   incdir_f2py / 'fortranobject.c',
   c_args: numpy_nodepr_api,
   include_directories: inc_np,
   install: true,
   dependencies: py_dep,
   subdir: 'holopy/scattering/theory/mie_f'
)

scsmfo_source = custom_target(
  'scsmfo_minmodule.c',
  input : ['scsmfo_min.for'],
  output: ['scsmfo_minmodule.c', 'scsmfo_min-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'scsmfo_min',
            '--lower', '--build-dir', '@OUTDIR@'],
)

py.extension_module('scsmfo_min',
  ['scsmfo_min.for'],
  scsmfo_source,
  incdir_f2py / 'fortranobject.c',
  c_args: numpy_nodepr_api,
  include_directories: inc_np,
  install: true,
  dependencies: py_dep,
  subdir: 'holopy/scattering/theory/mie_f'
)

scattering_theory_mie_f_sources = [
 'mie_specfuncs.py',
 'miescatlib.py',
 '__init__.py',
 'multilayer_sphere_lib.py'
]

py.install_sources(scattering_theory_mie_f_sources, subdir: 'holopy/scattering/theory/mie_f')
