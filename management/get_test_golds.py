#!/usr/bin/env python
# Copyright 2011-2013, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, and Ryan McGorty, Anna Wang
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Retrieve full arrays for test comparisons. HoloPy ships with reduced
gold files that only test a few parameters (min, max, std) of arrays. The full
data is helpful for catching some kinds of bugs, but would needlessly bloat
downloadds in general.

.. moduleauthor:: Tom Dimiduk <tdimiduk@physics.havard.edu>
"""

###############################################################################
# This dowload code is adapted from a stackoverflow post:
# http://stackoverflow.com/a/22776/250992
import urllib2

url = "http://manoharan.seas.harvard.edu/holopy/files/holopy-2.0.0_test_gold_data.zip"

file_name = url.split('/')[-1]
u = urllib2.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.getheaders("Content-Length")[0])
print "Downloading: %s Bytes: %s" % (file_name, file_size)

file_size_dl = 0
block_sz = 8192
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
    status = status + chr(8)*(len(status)+1)
    print status,

f.close()
###############################################################################

import zipfile
import sys
import os

zf = zipfile.ZipFile(file_name)
if zf.testzip() is not None:
    print("Test data corrupted, try running this script again to redownload. "
          "If the problem persists, file a bug")
    sys.exit(1)

zf.extractall('..')
print("Full gold data successfully downloaded and extracted")
os.remove(file_name)
