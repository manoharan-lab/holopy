import core
from .core import load, save
import scattering
from propagation import propagate
from vis import show

__version__ = 'unknown'
try:
    from _version import __version__
except ImportError:
    # version doesn't exist, or got deleted in bzr
    pass
