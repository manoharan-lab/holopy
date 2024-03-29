import unittest
import warnings

import pytest

from holopy.core.errors import (
    LoadError, BadImage, NoMetadata, CoordSysError,
    DependencyMissing, PerformanceWarning
)

class TestErrors(unittest.TestCase):
    @pytest.mark.fast
    def test_LoadError(self):
        self.assertRaises(LoadError, _raise, LoadError('', ''))

    @pytest.mark.fast
    def test_BadImage(self):
        self.assertRaises(BadImage, _raise, BadImage())

    @pytest.mark.fast
    def test_NoMetadata(self):
        self.assertRaises(NoMetadata, _raise, NoMetadata())

    @pytest.mark.fast
    def test_CoordSysError(self):
        self.assertRaises(CoordSysError, _raise, CoordSysError())

    @pytest.mark.fast
    def test_DependencyMissing(self):
        self.assertRaises(DependencyMissing, _raise, DependencyMissing('', ''))

    @pytest.mark.fast
    def test_PerformanceWarning(self):
        self.assertWarns(PerformanceWarning, _warn, PerformanceWarning)


def _raise(error):
    raise error


def _warn(warning_class):
    warnings.warn('', warning_class)

if __name__ == '__main__':
    unittest.main()
