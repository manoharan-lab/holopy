import unittest
from holopy.core.errors import *

class TestErrors(unittest.TestCase):
    def test_LoadError(self):
        self.assertRaises(LoadError, _raise, LoadError('', ''))

    def test_BadImage(self):
        self.assertRaises(BadImage, _raise, BadImage())

    def test_NoMetadata(self):
        self.assertRaises(NoMetadata, _raise, NoMetadata())

    def test_CoordSysError(self):
        self.assertRaises(CoordSysError, _raise, CoordSysError())

    def test_DependencyMissing(self):
        self.assertRaises(DependencyMissing, _raise, DependencyMissing('', ''))

def _raise(error):
    raise error

if __name__ == '__main__':
    unittest.main()
