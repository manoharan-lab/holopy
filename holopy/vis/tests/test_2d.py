import tempfile
import holopy as hp
from holopy.core.tests.common import get_example_data
from nose.plugins.skip import SkipTest
try:
    import matplotlib.pyplot as plt
    plt.ioff()
except ImportError:
    raise SkipTest()

def test_show():
    d = get_example_data('image0001.npy')
    hp.show(d)
    plt.savefig(tempfile.TemporaryFile(suffix='.pdf'))
