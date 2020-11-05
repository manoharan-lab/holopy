import unittest
import warnings

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

    @attr('fast')
    def test_PerformanceWarning(self):
        self.assertWarns(PerformanceWarning, _warn, PerformanceWarning)


def _raise(error):
    raise error


def _warn(warning_class):
    warnings.warn('', warning_class)

if __name__ == '__main__':
    unittest.main()
