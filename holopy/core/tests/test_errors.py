import unittest

from nose.plugins.attrib import attr

from holopy.core.errors import *

class TestErrors(unittest.TestCase):
    @attr("fast")
    def test_LoadError(self):
        self.assertRaises(LoadError, _raise, LoadError('', ''))

    @attr("fast")
    def test_BadImage(self):
        self.assertRaises(BadImage, _raise, BadImage())

    @attr("fast")
    def test_NoMetadata(self):
        self.assertRaises(NoMetadata, _raise, NoMetadata())

    @attr("fast")
    def test_CoordSysError(self):
        self.assertRaises(CoordSysError, _raise, CoordSysError())

    @attr("fast")
    def test_DependencyMissing(self):
        self.assertRaises(DependencyMissing, _raise, DependencyMissing('', ''))

def _raise(error):
    raise error

if __name__ == '__main__':
    unittest.main()
