import sys

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.command import build_ext
    config = Configuration('tmatrix_f', parent_package, top_path)
    if not hasattr(sys, 'real_prefix'):
        #we are not in a virtual_env. 
        #going to compile fortran code
        config.add_extension('S',
                         ['S.f',
                          'ampld.lp.f',
                          'lpd.f']
                         )
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(configuration=configuration)
