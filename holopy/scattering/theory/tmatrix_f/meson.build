S_source = custom_target(
  'Smodule.c',
  input : ['S.f', 'ampld.lp.f', 'lpd.f',],
  output: ['Smodule.c', 'S-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'S',
            '--lower', '--build-dir', '@OUTDIR@'],
)

py.extension_module('S',
  ['S.f',
   'ampld.lp.f',
   'lpd.f',],
  S_source,
  incdir_f2py / 'fortranobject.c',
  c_args: numpy_nodepr_api,
  include_directories: inc_np,
  install: true,
  dependencies: [py_dep, quadmath_dep],
  subdir: 'holopy/scattering/theory/tmatrix_f'
)

scattering_theory_tmatrix_f_sources = [
  '__init__.py'
]

py.install_sources(scattering_theory_tmatrix_f_sources, subdir: 'holopy/scattering/theory/tmatrix_f')

