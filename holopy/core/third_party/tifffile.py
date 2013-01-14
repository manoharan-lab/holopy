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

# Copyright (c) 2008-2010, The Regents of the University of California
# Produced by the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read TIFF, STK, LSM and FluoView files and access image data as numpy array.

Only a subset of the TIFF specification is supported, mainly uncompressed and
losslessly compressed 1-32 bit integer as well as 32 and 64-bit float
vimages, which are commonly used in scientific imaging.

TIFF, the Tagged Image File Format, is under the control of Adobe Systems.
STK and LSM are TIFF with custom extensions used by MetaMorph respectively
Carl Zeiss MicroImaging. Currently only primary info records are read
for STK, FluoView, and NIH image formats.

For command line usage run ``python tifffile.py --help``

:Authors:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`__,
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 20091216

Requirements
------------

* `Python 2.6 <http://www.python.org>`__
* `Numpy 1.4 <http://numpy.scipy.org>`__
* `Matplotlib 0.99 <http://matplotlib.sourceforge.net>`__
  (optional for plotting)
* `tifffile.c 20091215 <http://www.lfd.uci.edu/~gohlke/>`__
  (optional for faster decoding of PackBits and LZW encoded strings)

Acknowledgements
----------------
*  Egor Zindy, University of Manchester, for cz_lsm_scan_info specifics.

References
----------

(1) TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    http://partners.adobe.com/public/developer/tiff/
(2) TIFF File Format FAQ. http://www.awaresystems.be/imaging/tiff/faq.html
(3) MetaMorph Stack (STK) Image File Format.
    http://support.meta.moleculardevices.com/docs/t10243.pdf
(4) File Format Description - LSM 5xx Release 2.0.
    http://ibb.gsf.de/homepage/karsten.rodenacker/IDL/Lsmfile.doc
(5) BioFormats. http://www.loci.wisc.edu/ome/formats.html

Examples
--------

>>> tif = TIFFfile('test.tif')
>>> images = tif.asarray()
>>> image0 = tif[0].asarray()
>>> for page in tif:
...     for tag in page.tags.values():
...         t = tag.name, tag.value
...     image = page.asarray()
...     if page.is_rgb: pass
...     if page.is_reduced: pass
...     if page.is_palette:
...         t = page.color_map
...     if page.is_stk:
...         t = page.mm_uic_tags.number_planes
...     if page.is_lsm:
...         t = page.cz_lsm_info
>>> tif.close()

"""

from __future__ import division

import sys
import os
import math
import zlib
import time
import types
import struct
import warnings
from contextlib import contextmanager

import numpy


class TIFFfile(object):
    """Read TIFF, STK, and LSM files. Return image data as NumPy array.

    Attributes
    ----------

    pages : tuple of TIFFpages.

    Examples
    --------

    >>> tif = TIFFfile('test.tif')
    ... try:
    ...     images = tif.asarray()
    ... except Exception, e:
    ...     print e
    ... finally:
    ...     tif.close()

    """

    def __init__(self, filename):
        """Initialize object from file."""
        self._fd = open(filename, 'rb')
        self.fname = filename
        self.fstat = os.fstat(self._fd.fileno())
        try:
            self._fromfile()
        except Exception:
            self._fd.close()
            raise

    def close(self):
        """Close the file object."""
        self._fd.close()
        self._fd = None

    def _fromdata(self, data):
        """Create TIFF header, pages, and tags from numpy array."""
        raise NotImplementedError()

    def _fromfile(self):
        """Read TIFF header and all page records from file."""
        try:
            self.byte_order = TIFF_BYTE_ORDERS[self._fd.read(2)]
        except KeyError:
            raise ValueError("not a valid TIFF file")

        if struct.unpack(self.byte_order+'H', self._fd.read(2))[0] != 42:
            raise ValueError("not a TIFF file")

        self.pages = []
        while 1:
            try:
                self.pages.append(TIFFpage(self))
            except StopIteration:
                break

    def asarray(self, key=None, skipreduced=True, squeeze=True,
                colormapped=True, rgbonly=True):
        """Return image data of multiple TIFF pages as numpy array.

        Raises ValueError if not all pages are of same shape in all but
        first dimension.

        Arguments
        ---------

        key : int or slice
            Defines which pages to return as array.

        skipreduced : bool
            If True any reduced images are skipped.

        squeeze : bool
            If True all length-1 dimensions are squeezed out from result.

        colormapped : bool
            If True color mapping is applied for palette-indexed images.

        rgbonly : bool
            If True return RGB(A) images without extra samples.

        """
        pages = self.pages

        if skipreduced:
            pages = [p for p in pages if p.is_reduced]
        if key:
            pages = pages[key]

        try:
            pages[0]
        except TypeError:
            result = pages.asarray(False, colormapped, rgbonly)
        else:
            if colormapped and self.is_nih:
                result = numpy.vstack(p.asarray(False, False) for p in pages)
                if pages[0].is_palette:
                    result = pages[0].color_map[:, result]
                    result = numpy.swapaxes(result, 0, 1)
            else:
                try:
                    result = numpy.vstack(p.asarray(False, colormapped,
                                                    rgbonly) for p in pages)
                except ValueError:
                    # dimensions of pages don't agree
                    result = pages[0].asarray(False, colormapped, rgbonly)
                p = self.pages[0]
                if p.is_lsm:
                    # adjust LSM data shape
                    lsmi = p.cz_lsm_info
                    order = CZ_SCAN_TYPES[lsmi.scan_type]
                    if p.is_rgb:
                        order = order.replace('C', '').replace('XY', 'XYC')
                    shape = []
                    for i in reversed(order):
                        shape.append(getattr(lsmi, CZ_DIMENSIONS[i]))
                    result.shape = shape

        return result.squeeze() if squeeze else result

    def __len__(self):
        """Return number of image pages in file."""
        return len(self.pages)

    def __getitem__(self, key):
        """Return specified page."""
        return self.pages[key]

    def __iter__(self):
        """Return iterator over pages."""
        return iter(self.pages)

    def __str__(self):
        """Return string containing information about file."""
        fname = os.path.split(self.fname)[-1].capitalize()
        return "%s, %.2f MB, %s, %i pages" % (fname, self.fstat[6]/1048576,
            {'<': 'little endian', '>': 'big endian'}[self.byte_order],
            len(self.pages), )

    def __getattr__(self, name):
        """Return special property."""
        if name in ('is_rgb', 'is_palette', 'is_stk'):
            return all(getattr(p, name) for p in self.pages)
        if name in ('is_lsm', 'is_nih'):
            return getattr(self.pages[0], name)
        if name == 'is_fluoview':
            return 'mm_header' in self.pages[0].tags
        raise AttributeError(name)


@contextmanager
def tifffile(filename):
    """Support for 'with' statement.

    >>> with tifffile('test.tif') as tif:
    ...    image = tif.asarray()

    """
    f = TIFFfile(filename)
    try:
        yield f
    finally:
        f.close()


class TIFFpage(object):
    """A TIFF image file directory (IDF).

    Attributes
    ----------

    shape : tuple of int
        Dimensions of the image array in file.

    dtype : str
        Data type. One of TIFF_SAMPLE_DTYPES.

    tags : TiffTags(Record((dict))
        Tag values are also directly accessible as attributes.

    color_map : numpy array
        Color look up table, palette, if existing.

    mm_uic_tags: Record(dict)
        Consolidated MetaMorph mm_uic# tags, if exists.

    cz_lsm_scan_info: Record(dict)
        LSM scan info attributes, if exists.

    is_rgb : bool
        True if page contains a RGB image.

    is_reduced : bool
        True if page is a reduced image of another image.

    is_palette : bool
        True if page contains a palette-colored image.

    is_stk : bool
        True if page contains MM_UIC2 tag.

    is_lsm : bool
        True if page contains CZ_LSM_INFO tag.

    is_fluoview : bool
        True if page contains MM_STAMP tag.

    is_nih : bool
        True if page contains NIH image header.

    """

    def __init__(self, parent):
        """Initialize object from file."""
        self._parent = parent
        self.shape = ()
        self.tags = TiffTags()
        self._fromfile()
        self._process_tags()

    def _fromfile(self):
        """Read TIFF IDF structure and its tags from file.

        File cursor must be at storage position of IDF offset and is left at
        offset to next IDF.

        Raises StopIteration if offset (first two bytes read) are 0.

        """
        fd = self._parent._fd
        byte_order = self._parent.byte_order
        offset = struct.unpack(byte_order+'I', fd.read(4))[0]
        if not offset:
            raise StopIteration()

        # read standard tags
        tags = self.tags
        fd.seek(offset, 0)
        numtags = struct.unpack(byte_order+'H', fd.read(2))[0]
        for i in xrange(numtags):
            tag = TIFFtag(fd, byte_order=byte_order)
            tags[tag.name] = tag

        # read custom tags
        pos = fd.tell()
        for name, readtag in CUSTOM_TAGS.values():
            if name in tags and readtag:
                value = readtag(fd, byte_order, tags[name])
                if isinstance(value, dict): # numpy.core.records.record
                    value = Record(value)
                tags[name].value = value
        fd.seek(pos)

        # read LSM scan info
        if self.is_lsm:
            pos = fd.tell()
            fd.seek(self.cz_lsm_info['offset_scan_information'])
            try:
                self.cz_lsm_scan_info = read_cz_lsm_scan_info(fd, byte_order)
            except ValueError:
                pass
            fd.seek(pos)

    def _process_tags(self):
        """Validate standard tags and initialize attributes.

        Raise ValueError if tag values not supported.

        """
        tags = self.tags

        for code, (name, default, dtype, count, validate) in TIFF_TAGS.items():
            if not (name in tags or default is None):
                tags[name] = TIFFtag(code, dtype=dtype, count=count,
                                     value=default, name=name)
            if name in tags and validate:
                try:
                    if tags[name].count == 1:
                        setattr(self, name, validate[tags[name].value])
                    else:
                        setattr(self, name, tuple(validate[value]
                                            for value in tags[name].value))
                except KeyError:
                    raise ValueError("%s.value (%s) not supported" %
                                     (name, tags[name].value))

        tag = tags['bits_per_sample']
        if tag.count != 1:
            bps = tag.value[0]
            if all((i-bps for i in tag.value)):
                raise ValueError(
                    "samples must be of same type %s" % str(tag))
            self.bits_per_sample = bps

        tag = tags['sample_format']
        if tag.count != 1:
            fmt = tag.value[0]
            if all((i-fmt for i in tag.value)):
                raise ValueError(
                    "samples must be of same format %s" % str(tag))
            self.sample_format = TIFF_SAMPLE_FORMATS[fmt]

        self.strips_per_image = int(math.floor((self.image_length +
                            self.rows_per_strip - 1) / self.rows_per_strip))

        key = (self.sample_format, self.bits_per_sample)
        try:
            self.dtype = TIFF_SAMPLE_DTYPES[key]
        except KeyError:
            raise ValueError("unsupported sample dtype %s" % str(key))

        if self.is_palette:
            dtype = self.tags['color_map'].dtype[1]
            self.color_map = numpy.array(self.color_map,
                                         dtype).reshape((3, -1))

        planes = 0
        if self.is_stk:
            planes = tags['mm_uic2'].count
            # consolidate mm_uci tags
            self.mm_uic_tags = Record(tags['mm_uic2'].value)
            for t in ('mm_uic3', 'mm_uic4', 'mm_uic1'):
                if t in tags:
                    self.mm_uic_tags.update(tags[t].value)

        if planes:
            if self.planar_configuration == 'contig':
                self.shape = (planes, self.image_length,
                              self.image_width, self.samples_per_pixel)
            else:
                self.shape = (planes, self.samples_per_pixel,
                              self.image_length, self.image_width, 1)
        else:
            if self.planar_configuration == 'contig':
                self.shape = (1, self.image_length, self.image_width,
                              self.samples_per_pixel)
            else:
                self.shape = (self.samples_per_pixel, self.image_length,
                              self.image_width, 1)

        if not self.compression and not 'strip_byte_counts' in tags:
            self.strip_byte_counts = numpy.product(self.shape) * (
                self.bits_per_sample // 8)

    def asarray(self, squeeze=True, colormapped=True, rgbonly=True):
        """Read image data and return as numpy array in native byte order.

        Raise ValueError if format is unsupported.

        Arguments
        ---------

        squeeze : bool
            If True all length-1 dimensions are squeezed out from result.

        colormapped : bool
            If True color mapping is applied for palette-indexed images.

        rgbonly : bool
            If True return RGB(A) image without extra samples.

        """
        fd = self._parent._fd
        if not fd:
            raise IOError("TIFF file is not open")

        if self.compression not in TIFF_DECOMPESSORS:
            raise ValueError("Can't decompress %s" % self.compression)

        strip_offsets = self.strip_offsets
        strip_byte_counts = self.strip_byte_counts
        try:
            strip_offsets[0]
        except TypeError:
            strip_offsets = (self.strip_offsets, )
            strip_byte_counts = (self.strip_byte_counts, )

        byte_order = self._parent.byte_order
        bytes_per_sample = self.bits_per_sample // 8
        typecode = byte_order + self.dtype

        if self.is_stk:
            fd.seek(strip_offsets[0], 0)
            result = numpy.fromfile(fd, typecode, numpy.product(self.shape))
        else:
            # try speed up reading contiguous data by merging all strips
            if not self.compression \
               and self.bits_per_sample in (8, 16, 32, 64) \
               and all(strip_offsets[i] == \
                       strip_offsets[i+1]-strip_byte_counts[i]
                       for i in xrange(len(strip_offsets)-1)):
                strip_byte_counts = (strip_offsets[-1] - strip_offsets[0] +
                                     strip_byte_counts[-1], )
                strip_offsets = (strip_offsets[0], )

            result = numpy.empty(self.shape, self.dtype).reshape(-1)

            runlen = self.image_width
            if self.planar_configuration == 'contig':
                runlen *= self.samples_per_pixel

            if self.bits_per_sample in (8, 16, 32, 64):
                if self.bits_per_sample*runlen % 8:
                    raise ValueError("data and sample size mismatch")
                unpack = lambda data: numpy.fromstring(data, typecode)
            else:
                unpack = lambda data: unpackints(data, typecode,
                                                 self.bits_per_sample, runlen)

            decompress = TIFF_DECOMPESSORS[self.compression]

            index = 0
            for offset, bytecount in zip(strip_offsets, strip_byte_counts):
                fd.seek(offset, 0)
                data = unpack(decompress(fd.read(bytecount)))
                result[index:index+data.size] = data
                index += data.size

        result.shape = self.shape[:]

        if self.predictor == 'horizontal':
            numpy.cumsum(result, axis=2, dtype=self.dtype, out=result)

        if colormapped and self.photometric == 'palette':
            if self.color_map.shape[1] >= 2**self.bits_per_sample:
                result = self.color_map[:, result]
                result = numpy.swapaxes(result, 0, 1)

        if rgbonly and 'extra_samples' in self.tags:
            # return only RGB and first unassociated alpha channel if exists
            extra_samples = self.tags['extra_samples'].value
            if self.tags['extra_samples'].count == 1:
                extra_samples = (extra_samples, )
                for i, es in enumerate(extra_samples):
                    if es == 2: # unassociated alpha channel
                        if self.planar_configuration == 'contig':
                            result = result[..., [0, 1, 2, 3+i]]
                        else:
                            result = result[[0, 1, 2, 3+i]]
                        break
            else:
                if self.planar_configuration == 'contig':
                    result = result[..., :3]
                else:
                    result = result[:3]

        if result.shape[0] != 1:
            result.shape = (1, ) + result.shape

        return result.squeeze() if squeeze else result

    def __getattr__(self, name):
        """Return tag value or special property."""
        tags = self.tags
        if name in tags:
            return tags[name].value
        if name == 'is_rgb':
            return tags['photometric'].value == 2
        if name == 'is_reduced':
            return not tags['new_subfile_type'].value & 1
        if name == 'is_palette':
            return 'color_map' in tags
        if name == 'is_stk':
            return 'mm_uic2' in tags
        if name == 'is_lsm':
            return 'cz_lsm_info' in tags
        if name == 'is_fluoview':
            return 'mm_stamp' in tags
        if name == 'is_nih':
            return 'nih_image_header' in tags
        raise AttributeError(name)

    def __str__(self):
        """Return string containing information about page."""
        t = ','.join(t[3:] for t in (
            'is_stk', 'is_lsm', 'is_nih', 'is_fluoview') if getattr(self, t))
        s = ', '.join(str(i) for i in (
            (' x '.join(str(i) for i in self.shape if i > 1),
            numpy.dtype(self.dtype),
            "%i bit" % self.bits_per_sample,
            self.photometric,
            self.compression if self.compression else 'raw')))
        if t:
            s = ', '.join((s, t))
        return s


class TIFFtag(object):
    """A TIFF tag structure.

    Attributes
    ----------

    name : string
        Attribute name of tag.

    code : int
        Decimal code of tag.

    dtype : str
        Datatype of tag data. One of TIFF_DATA_TYPES.

    count : int
        Number of values.

    value : various types
        Tag data. For codes in CUSTOM_TAGS the 4 bytes file content.

    """
    __slots__ = ('code', 'name', 'count', 'dtype', 'value')

    def __init__(self, arg, **kwargs):
        """Initialize tag from file or arguments."""
        if isinstance(arg, file):
            self._fromfile(arg, **kwargs)
        else:
            self._fromdata(arg, **kwargs)

    def _fromdata(self, code, dtype, count, value, name=None):
        """Initialize tag from arguments."""
        self.code = int(code)
        self.name = name if name else str(code)
        self.dtype = TIFF_DATA_TYPES[dtype]
        self.count = int(count)
        self.value = value

    def _fromfile(self, fd, byte_order):
        """Read tag structure from open file. Advances file cursor 12 bytes."""
        code, dtype, count, value = struct.unpack(byte_order+'HHI4s',
                                                  fd.read(12))

        if code in TIFF_TAGS:
            name = TIFF_TAGS[code][0]
        elif code in CUSTOM_TAGS:
            name = CUSTOM_TAGS[code][0]
        else:
            name = str(code)

        try:
            dtype = TIFF_DATA_TYPES[dtype]
        except KeyError:
            raise ValueError("unknown TIFF tag data type %i" % dtype)

        if not code in CUSTOM_TAGS:
            format = '%s%i%s' % (byte_order, count*int(dtype[0]), dtype[1])
            size = struct.calcsize(format)
            if size <= 4:
                value = struct.unpack(format, value[:size])
            else:
                pos = fd.tell()
                fd.seek(struct.unpack(byte_order+'I', value)[0])
                value = struct.unpack(format, fd.read(size))
                fd.seek(pos)
            if len(value) == 1:
                value = value[0]
            if dtype == '1s':
                value = stripnull(value)

        self.code = code
        self.name = name
        self.dtype = dtype
        self.count = count
        self.value = value

    def __str__(self):
        """Return string containing information about tag."""
        return ' '.join(str(getattr(self, s)) for s in self.__slots__)


class Record(dict):
    """Dictionary with attribute access.

    Can also be initialized with numpy.core.records.record.

    """
    __slots__ = ()

    def __init__(self, arg={}):
        try:
            dict.__init__(self, arg)
        except Exception:
            for i, name in enumerate(arg.dtype.names):
                v = arg[i]
                self[name] = v if v.dtype.char != 'S' else stripnull(v)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self.__setitem__(name, value)

    def __str__(self):
        """Return string with information about all tags."""
        s = []
        lists = []
        list_type = types.ListType
        for k in sorted(self):
            if k.startswith('_'):
                continue
            v = self[k]
            if type(v) == list_type and len(v) and isinstance(v[0], Record):
                lists.append((k, v))
            else:
                s.append(("* %s: %s" % (k, str(v)))[:PRINT_LINE_LEN])
        for k, v in lists:
            l = []
            for i, w in enumerate(v):
                l.append("* %s[%i]\n  %s" % (k, i,
                                             str(w).replace("\n", "\n  ")))
            s.append('\n'.join(l))

        return '\n'.join(s)


class TiffTags(Record):
    """Dictionary of TIFFtags with attribute access."""

    def __str__(self):
        """Return string with information about all tags."""
        s = []
        sortbycode = lambda a, b: cmp(a.code, b.code)
        for tag in sorted(self.itervalues(), sortbycode):
            typecode = "%i%s" % (tag.count * int(tag.dtype[0]), tag.dtype[1])
            line = "* %i %s (%s) %s" % (tag.code, tag.name, typecode,
                                        str(tag.value).split('\n', 1)[0])
            s.append(line[:PRINT_LINE_LEN])
        return '\n'.join(s)


def read_nih_image_header(fd, byte_order, tag):
    """Read NIH_IMAGE_HEADER tag from file and return as dictionary."""
    fd.seek(12 + struct.unpack(byte_order+'I', tag.value)[0])
    return {'version': struct.unpack(byte_order+'H', fd.read(2))[0]}


def read_mm_header(fd, byte_order, tag):
    """Read MM_HEADER tag from file and return as numpy.rec.array."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    return numpy.rec.fromfile(fd, MM_HEADER, 1, byteorder=byte_order)[0]


def read_mm_stamp(fd, byte_order, tag):
    """Read MM_STAMP tag from file and return as numpy.array."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    return numpy.fromfile(fd, byte_order+'8f8', 1)[0]


def read_mm_uic1(fd, byte_order, tag):
    """Read MM_UIC1 tag from file and return as dictionary."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    t = fd.read(8*tag.count)
    t = struct.unpack('%s%iI' % (byte_order, 2*tag.count), t)
    return dict((MM_TAG_IDS[k], v) for k, v in zip(t[::2], t[1::2])
                if k in MM_TAG_IDS)


def read_mm_uic2(fd, byte_order, tag):
    """Read MM_UIC2 tag from file and return as dictionary."""
    result = {'number_planes': tag.count}
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    values = numpy.fromfile(fd, byte_order+'I', 6*tag.count)
    result['z_distance'] = values[0::6] / values[1::6]
    #result['date_created'] = tuple(values[2::6])
    #result['time_created'] = tuple(values[3::6])
    #result['date_modified'] = tuple(values[4::6])
    #result['time_modified'] = tuple(values[5::6])
    return result


def read_mm_uic3(fd, byte_order, tag):
    """Read MM_UIC3 tag from file and return as dictionary."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    t = numpy.fromfile(fd, '%sI' % byte_order, 2*tag.count)
    return {'wavelengths': t[0::2] / t[1::2]}


def read_mm_uic4(fd, byte_order, tag):
    """Read MM_UIC4 tag from file and return as dictionary."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    t = struct.unpack(byte_order + 'hI'*tag.count, fd.read(6*tag.count))
    return dict((MM_TAG_IDS[k], v) for k, v in zip(t[::2], t[1::2])
                if k in MM_TAG_IDS)


def read_cz_lsm_info(fd, byte_order, tag):
    """Read CS_LSM_INFO tag from file and return as numpy.rec.array."""
    fd.seek(struct.unpack(byte_order+'I', tag.value)[0])
    result = numpy.rec.fromfile(fd, CZ_LSM_INFO, 1, byteorder=byte_order)[0]
    {50350412: '1.3', 67127628: '2.0'}[result.magic_number]
    return result


def read_cz_lsm_scan_info(fd, byte_order):
    """Read LSM scan information from file and return as Record."""
    block = Record()
    blocks = [block]
    unpack = struct.unpack

    if 0x10000000 != struct.unpack(byte_order+"I", fd.read(4))[0]:
        raise ValueError("not a lsm_scan_info structure")
    fd.read(8)

    while 1:
        entry, dtype, size = unpack(byte_order+"III", fd.read(12))
        if dtype == 2:
            value = fd.read(size)[:-2]
        elif dtype == 4:
            value = unpack(byte_order+"i", fd.read(4))[0]
        elif dtype == 5:
            value = unpack(byte_order+"d", fd.read(8))[0]
        else:
            value = 0

        if entry in CZ_LSM_SCAN_INFO_ARRAYS:
            blocks.append(block)
            name = CZ_LSM_SCAN_INFO_ARRAYS[entry]
            newobj = []
            setattr(block, name, newobj)
            block = newobj
        elif entry in CZ_LSM_SCAN_INFO_STRUCTS:
            blocks.append(block)
            newobj = Record()
            block.append(newobj)
            block = newobj
        elif entry in CZ_LSM_SCAN_INFO_ATTRIBUTES:
            name = CZ_LSM_SCAN_INFO_ATTRIBUTES[entry]
            setattr(block, name, value)
        elif entry == 0xffffffff:
            block = blocks.pop()
        else:
            setattr(block, "unknown_%x" % entry, value)

        if not blocks:
            break

    return block

def _replace_by(module_function, warn=False):
    """Try replace decorated function by module.function."""

    def decorate(func, module_function=module_function, warn=warn):
        try:
            module, function = module_function.split('.')
            func, oldfunc = getattr(__import__(module), function), func
            globals()['__old_' + func.__name__] = oldfunc
        except Exception, e:
            if warn:
                warnings.warn("Failed to import %s" % module_function)
        return func

    return decorate


@_replace_by('_tifffile.decodepackbits')
def decodepackbits(encoded):
    """Decompress PackBits encoded byte string.

    PackBits is a simple byte-oriented run-length compression scheme.

    """
    result = []
    i = 0
    try:
        while 1:
            n = ord(encoded[i]) + 1
            i += 1
            if n < 129:
                result.extend(encoded[i:i+n])
                i += n
            elif n > 129:
                result.extend(encoded[i:i+1] * (258-n))
                i += 1
    except IndexError:
        return ''.join(result)


@_replace_by('_tifffile.decodelzw')
def decodelzw(encoded):
    """Decompress LZW (Lempel-Ziv-Welch) encoded TIFF strip (byte string).

    The strip must begin with a CLEAR code and end with an EOI code.

    This is an implementation of the LZW decoding algorithm described in (1).
    It is not compatible with old style LZW compressed files like quad-lzw.tif.

    """
    unpack = struct.unpack

    def next_code():
        """Return integer of `bitw` bits at `bitcount` position in encoded."""
        start = bitcount // 8
        s = encoded[start:start+4]
        try:
            code = unpack('>I', s)[0]
        except Exception:
            code = unpack('>I', s + '\x00'*(4-len(s)))[0]
        code = code << (bitcount % 8)
        code = code & mask
        return code >> shr

    switchbitch = { # code: bit-width, shr-bits, bit-mask
        255: (9, 23, int(9*'1'+'0'*23, 2)),
        511: (10, 22, int(10*'1'+'0'*22, 2)),
        1023: (11, 21, int(11*'1'+'0'*21, 2)),
        2047: (12, 20, int(12*'1'+'0'*20, 2)), }
    bitw, shr, mask = switchbitch[255]
    bitcount = 0

    if len(encoded) < 4:
        raise ValueError("strip must be at least 4 characters long")

    if next_code() != 256:
        raise ValueError("strip must begin with CLEAR code")

    code = 0
    result = []
    while 1:
        code = next_code() # ~5% faster when inlining this function
        bitcount += bitw
        if code == 257: # EOI
            break
        if code == 256: # CLEAR
            table = [chr(i) for i in xrange(256)]
            table.extend((0, 0))
            lentable = 258
            bitw, shr, mask = switchbitch[255]
            code = next_code()
            bitcount += bitw
            if code == 257: # EOI
                break
            result.append(table[code])
        else:
            if code < lentable:
                decoded = table[code]
                newcode = table[oldcode] + decoded[0]
            else:
                newcode = table[oldcode]
                newcode += newcode[0]
                decoded = newcode
            result.append(decoded)
            table.append(newcode)
            lentable += 1
        oldcode = code
        if lentable in switchbitch:
            bitw, shr, mask = switchbitch[lentable]

    if code != 257:
        raise ValueError("unexpected end of stream")

    return ''.join(result)


#@_replace_by('_tifffile.unpackints')
def unpackints(data, dtype, intsize, runlen=0):
    """Decompress byte string to array of integers of any bit size <= 32.

    data : str

    dtype : numpy.dtype or str
        A numpy boolean or integer type.

    intsize : int
        Number of bits per integer.

    runlen : int
        Number of consecutive integers, after which to start at next byte

    """
    # bitarray
    if intsize == 1:
        data = numpy.fromstring(data, '|B')
        data = numpy.unpackbits(data)
        if runlen % 8 != 0:
            data = data.reshape(-1, runlen+(8-runlen%8))
            data = data[:, :runlen].reshape(-1)
        return data.astype(dtype)

    dtype = numpy.dtype(dtype)

    if 32 < intsize < 1:
        raise ValueError("intsize out of range")

    if dtype.kind not in "biu":
        raise ValueError("invalid dtype")

    if intsize > dtype.itemsize * 8:
        raise ValueError("dtype.itemsize too small")

    for i in (8, 16, 32):
        if intsize <= i:
            itembytes = i // 8
            break

    if runlen == 0:
        runlen = len(data) // itembytes
    skipbits = runlen*intsize % 8
    if skipbits:
        skipbits = 8 - skipbits
    shrbits = itembytes*8 - intsize
    bitmask = int(intsize*'1'+'0'*shrbits, 2)
    if dtype.byteorder == '|':
        dtypestr = '=' + dtype.char
    else:
        dtypestr = dtype.byteorder + dtype.char
    unpack = struct.unpack

    l = runlen * (len(data)*8 // (runlen*intsize + skipbits))
    result = numpy.empty((l,), dtype)

    bitcount = 0
    for i in xrange(len(result)):
        start = bitcount // 8
        s = data[start:start+itembytes]
        try:
            code = unpack(dtypestr, s)[0]
        except Exception:
            code = unpack(dtypestr, s + '\x00'*(itembytes-len(s)))[0]
        code = code << (bitcount % 8)
        code = code & bitmask
        result[i] = code >> shrbits
        bitcount += intsize
        if (i+1) % runlen == 0:
            bitcount += skipbits

    return result


def stripnull(string):
    """Return string truncated at first null character."""
    i = string.find('\x00')
    return string if (i < 0) else string[:i]


def test_tifffile(directory='testimages', verbose=True):
    """Read all images in directory. Print error message on failure.

    >>> test_tifffile(verbose=False)

    """
    import glob

    successful = 0
    failed = 0
    start = time.time()
    for f in glob.glob(os.path.join(directory, '*.*')):
        if verbose:
            print "\n%s>" % f.lower(),
        t0 = time.time()
        try:
            tif = TIFFfile(f)
        except Exception, e:
            if not verbose:
                print f,
            print "ERROR:", e
            failed += 1
            continue
        try:
            img = tif.asarray()
        except ValueError:
            try:
                img = tif[0].asarray()
            except Exception, e:
                if not verbose:
                    print f,
                print "ERROR:", e
        finally:
            tif.close()
        successful += 1
        if verbose:
            print "%s, %s %s, %s, %.0f ms" % (str(tif), str(img.shape),
                img.dtype, tif[0].compression, (time.time()-t0) * 1e3)

    if verbose:
        print "\nSuccessfully read %i of %i files in %.3f s\n" % (
            successful, successful+failed, time.time()-start)


# TIFF tag structures. Cases that are irrelevant or not implemented are
# commented out.

class TIFF_SUBFILE_TYPES(object):

    def __getitem__(self, key):
        result = []
        if key & 1:
            result.append('reduced_image')
        if key & 2:
            result.append('page')
        if key & 4:
            result.append('mask')
        return tuple(result)

TIFF_OSUBFILE_TYPES = {
    0: 'undefined',
    1: 'image', # full-resolution image data
    2: 'reduced_image', # reduced-resolution image data
    3: 'page', # a single page of a multi-page image
}

TIFF_PHOTOMETRICS = {
    0: 'miniswhite',
    1: 'minisblack',
    2: 'rgb',
    3: 'palette',
    4: 'mask',
    5: 'separated',
    6: 'cielab',
    7: 'icclab',
    8: 'itulab',
    32844: 'logl',
    32845: 'logluv',
}

TIFF_COMPESSIONS = {
    1: None,
    2: 'ccittrle',
    3: 'ccittfax3',
    4: 'cittfax4',
    5: 'lzw',
    6: 'ojpeg',
    7: 'jpeg',
    8: 'adobe_deflate',
    9: 't85',
    10: 't43',
    32766: 'next',
    32771: 'ccittrlew',
    32773: 'packbits',
    32809: 'thunderscan',
    32895: 'it8ctpad',
    32896: 'it8lw',
    32897: 'it8mp',
    32898: 'it8bl',
    32908: 'pixarfilm',
    32909: 'pixarlog',
    32946: 'deflate',
    32947: 'dcs',
    34661: 'jbig',
    34676: 'sgilog',
    34677: 'sgilog24',
    34712: 'jp2000',
}

TIFF_DECOMPESSORS = {
    None: lambda x: x,
    'adobe_deflate': zlib.decompress,
    'deflate': zlib.decompress,
    'packbits': decodepackbits,
    'lzw': decodelzw,
}

TIFF_DATA_TYPES = {
    1: '1B',  # BYTE 8-bit unsigned integer.
    2: '1s',  # ASCII 8-bit byte that contains a 7-bit ASCII code;
              #   the last byte must be NUL (binary zero).
    3: '1H',  # SHORT 16-bit (2-byte) unsigned integer
    4: '1I',  # LONG 32-bit (4-byte) unsigned integer.
    5: '2I',  # RATIONAL Two LONGs: the first represents the numerator of
              #   a fraction; the second, the denominator.
    6: '1b',  # SBYTE An 8-bit signed (twos-complement) integer.
    7: '1B',  # UNDEFINED An 8-bit byte that may contain anything,
              #   depending on the definition of the field.
    8: '1h',  # SSHORT A 16-bit (2-byte) signed (twos-complement) integer.
    9: '1i',  # SLONG A 32-bit (4-byte) signed (twos-complement) integer.
    10: '2i', # SRATIONAL Two SLONGs: the first represents the numerator
              #   of a fraction, the second the denominator.
    11: '1f', # FLOAT Single precision (4-byte) IEEE format.
    12: '1d', # DOUBLE Double precision (8-byte) IEEE format.
}

TIFF_BYTE_ORDERS = {
    'II': '<', # little endian
    'MM': '>', # big endian
}

TIFF_SAMPLE_FORMATS = {
    1: 'uint',
    2: 'int',
    3: 'float',
    #4: 'void',
    #5: 'complex_int',
    #6: 'complex',
}

TIFF_SAMPLE_DTYPES = {
    ('uint', 1): '?', # bitmap
    ('uint', 2): 'B',
    ('uint', 4): 'B',
    ('uint', 6): 'B',
    ('uint', 8): 'B',
    ('uint', 10): 'H',
    ('uint', 12): 'H',
    ('uint', 14): 'H',
    ('uint', 16): 'H',
    ('uint', 24): 'I',
    ('uint', 32): 'I',
    ('int', 8): 'b',
    ('int', 16): 'h',
    ('int', 32): 'i',
    ('float', 32): 'f',
    ('float', 64): 'd',
}

TIFF_PREDICTORS = {
    1: None,
    2: 'horizontal',
    #3: 'floatingpoint',
}

TIFF_ORIENTATIONS = {
    1: 'top_left',
    2: 'top_right',
    3: 'bottom_right',
    4: 'bottom_left',
    5: 'left_top',
    6: 'right_top',
    7: 'right_bottom',
    8: 'left_bottom',
}

TIFF_FILLORDERS = {
    1: 'msb2lsb',
    2: 'lsb2msb',
}

TIFF_RESUNITS = {
    1: 'none',
    2: 'inch',
    3: 'centimeter',
}

TIFF_PLANARCONFIGS = {
    1: 'contig',
    2: 'separate',
}

TIFF_EXTRA_SAMPLES = {
    0: 'unspecified',
    1: 'assocalpha',
    2: 'unassalpha',
}

# MetaMorph STK tags
MM_TAG_IDS = {
    0: 'auto_scale',
    1: 'min_scale',
    2: 'max_scale',
    3: 'spatial_calibration',
    #4: 'x_calibration',
    #5: 'y_calibration',
    #6: 'calibration_units',
    #7: 'name',
    8: 'thresh_state',
    9: 'thresh_state_red',
    11: 'thresh_state_green',
    12: 'thresh_state_blue',
    13: 'thresh_state_lo',
    14: 'thresh_state_hi',
    15: 'zoom',
    #16: 'create_time',
    #17: 'last_saved_time',
    18: 'current_buffer',
    19: 'gray_fit',
    20: 'gray_point_count',
    #21: 'gray_x',
    #22: 'gray_y',
    #23: 'gray_min',
    #24: 'gray_max',
    #25: 'gray_unit_name',
    26: 'standard_lut',
    27: 'wavelength',
    #28: 'stage_position',
    #29: 'camera_chip_offset',
    #30: 'overlay_mask',
    #31: 'overlay_compress',
    #32: 'overlay',
    #33: 'special_overlay_mask',
    #34: 'special_overlay_compress',
    #35: 'special_overlay',
    36: 'image_property',
    #37: 'stage_label',
    #38: 'autoscale_lo_info',
    #39: 'autoscale_hi_info',
    #40: 'absolute_z',
    #41: 'absolute_z_valid',
    #42: 'gamma',
    #43: 'gamma_red',
    #44: 'gamma_green',
    #45: 'gamma_blue',
    #46: 'camera_bin',
    47: 'new_lut',
    #48: 'image_property_ex',
    49: 'plane_property',
    #50: 'user_lut_table',
    51: 'red_autoscale_info',
    #52: 'red_autoscale_lo_info',
    #53: 'red_autoscale_hi_info',
    54: 'red_minscale_info',
    55: 'red_maxscale_info',
    56: 'green_autoscale_info',
    #57: 'green_autoscale_lo_info',
    #58: 'green_autoscale_hi_info',
    59: 'green_minscale_info',
    60: 'green_maxscale_info',
    61: 'blue_autoscale_info',
    #62: 'blue_autoscale_lo_info',
    #63: 'blue_autoscale_hi_info',
    64: 'blue_min_scale_info',
    65: 'blue_max_scale_info',
    #66: 'overlay_plane_color',
}

# Olymus Fluoview
MM_DIMENSION = [
    ('name', 'a16'),
    ('size', 'i4'),
    ('origin', 'f8'),
    ('resolution', 'f8'),
    ('unit', 'a64'),
]

MM_HEADER = [
    ('header_flag', 'i2'),
    ('image_type', 'u1'),
    ('image_name', 'a257'),
    ('offset_data', 'u4'),
    ('palette_size', 'i4'),
    ('offset_palette0', 'u4'),
    ('offset_palette1', 'u4'),
    ('comment_size', 'i4'),
    ('offset_comment', 'u4'),
    ('dimensions', MM_DIMENSION, 10),
    ('offset_position', 'u4'),
    ('map_type', 'i2'),
    ('map_min', 'f8'),
    ('map_max', 'f8'),
    ('min_value', 'f8'),
    ('max_value', 'f8'),
    ('offset_map', 'u4'),
    ('gamma', 'f8'),
    ('offset', 'f8'),
    ('gray_channel', MM_DIMENSION),
    ('offset_thumbnail', 'u4'),
    ('voice_field', 'i4'),
    ('offset_voice_field', 'u4'),
]

# Carl Zeiss LSM record
CZ_LSM_INFO = [
    ('magic_number', 'i4'),
    ('structure_size', 'i4'),
    ('dimension_x', 'i4'),
    ('dimension_y', 'i4'),
    ('dimension_z', 'i4'),
    ('dimension_channels', 'i4'),
    ('dimension_time', 'i4'),
    ('dimension_data_type', 'i4'),
    ('thumbnail_x', 'i4'),
    ('thumbnail_y', 'i4'),
    ('voxel_size_x', 'f8'),
    ('voxel_size_y', 'f8'),
    ('voxel_size_z', 'f8'),
    ('origin_x', 'f8'),
    ('origin_y', 'f8'),
    ('origin_z', 'f8'),
    ('scan_type', 'u2'),
    ('spectral_scan', 'u2'),
    ('data_type', 'u4'),
    ('offset_vector_overlay', 'u4'),
    ('offset_input_lut', 'u4'),
    ('offset_output_lut', 'u4'),
    ('offset_channel_colors', 'u4'),
    ('time_interval', 'f8'),
    ('offset_channel_data_types', 'u4'),
    ('offset_scan_information', 'u4'),
    ('offset_ks_data', 'u4'),
    ('offset_time_stamps', 'u4'),
    ('offset_event_list', 'u4'),
    ('offset_roi', 'u4'),
    ('offset_bleach_roi', 'u4'),
    ('offset_next_recording', 'u4'),
    ('display_aspect_x', 'f8'),
    ('display_aspect_y', 'f8'),
    ('display_aspect_z', 'f8'),
    ('display_aspect_time', 'f8'),
    ('offset_mean_of_roi_overlay', 'u4'),
    ('offset_topo_isoline_overlay', 'u4'),
    ('offset_topo_profile_overlay', 'u4'),
    ('offset_linescan_overlay', 'u4'),
    ('offset_toolbar_flags', 'u4'),
]

# Map cz_lsm_info.scan_type to dimension order
CZ_SCAN_TYPES = {
    0: 'XYZCT', # x-y-z scan
    1: 'XYZCT', # z scan (x-z plane)
    2: 'XYZCT', # line scan
    3: 'XYTCZ', # time series x-y
    4: 'XYZTC', # time series x-z
    5: 'XYTCZ', # time series 'Mean of ROIs'
    6: 'XYZTC', # time series x-y-z
    7: 'XYCTZ', # spline scan
    8: 'XYCZT', # spline scan x-z
    9: 'XYTCZ', # time series spline plane x-z
    10: 'XYZCT', # point mode
}

# Map dimension codes to cz_lsm_info attribute
CZ_DIMENSIONS = {
    'X': 'dimension_x',
    'Y': 'dimension_y',
    'Z': 'dimension_z',
    'C': 'dimension_channels',
    'T': 'dimension_time',
}

# Descriptions of cz_lsm_info.data_type
CZ_DATA_TYPES = {
    0: 'varying data types',
    2: '12 bit unsigned integer',
    5: '32 bit float',
    #default: '8 bit unsigned integer',
}

CZ_LSM_SCAN_INFO_ARRAYS = {
    0x20000000: "tracks",
    0x30000000: "lasers",
    0x60000000: "detectionchannels",
    0x80000000: "illuminationchannels",
    0xa0000000: "beamsplitters",
    0xc0000000: "datachannels",
    0x13000000: "markers",
    0x11000000: "timers",
}

CZ_LSM_SCAN_INFO_STRUCTS = {
    0x40000000: "tracks",
    0x50000000: "lasers",
    0x70000000: "detectionchannels",
    0x90000000: "illuminationchannels",
    0xb0000000: "beamsplitters",
    0xd0000000: "datachannels",
    0x14000000: "markers",
    0x12000000: "timers",
}

CZ_LSM_SCAN_INFO_ATTRIBUTES = {
    0x10000001: "name",
    0x10000002: "description",
    0x10000003: "notes",
    0x10000004: "objective",
    0x10000005: "processing_summary",
    0x10000006: "special_scan_mode",
    0x10000007: "oledb_recording_scan_type",
    0x10000008: "oledb_recording_scan_mode",
    0x10000009: "number_of_stacks",
    0x1000000a: "lines_per_plane",
    0x1000000b: "samples_per_line",
    0x1000000c: "planes_per_volume",
    0x1000000d: "images_width",
    0x1000000e: "images_height",
    0x1000000f: "images_number_planes",
    0x10000010: "images_number_stacks",
    0x10000011: "images_number_channels",
    0x10000012: "linscan_xy_size",
    0x10000013: "scan_direction",
    0x10000014: "time_series",
    0x10000015: "original_scan_data",
    0x10000016: "zoom_x",
    0x10000017: "zoom_y",
    0x10000018: "zoom_z",
    0x10000019: "sample_0x",
    0x1000001a: "sample_0y",
    0x1000001b: "sample_0z",
    0x1000001c: "sample_spacing",
    0x1000001d: "line_spacing",
    0x1000001e: "plane_spacing",
    0x1000001f: "plane_width",
    0x10000020: "plane_height",
    0x10000021: "volume_depth",
    0x10000023: "nutation",
    0x10000034: "rotation",
    0x10000035: "precession",
    0x10000036: "sample_0time",
    0x10000037: "start_scan_trigger_in",
    0x10000038: "start_scan_trigger_out",
    0x10000039: "start_scan_event",
    0x10000040: "start_scan_time",
    0x10000041: "stop_scan_trigger_in",
    0x10000042: "stop_scan_trigger_out",
    0x10000043: "stop_scan_event",
    0x10000044: "stop_scan_time",
    0x10000045: "use_rois",
    0x10000046: "use_reduced_memory_rois",
    0x10000047: "user",
    0x10000048: "use_bccorrection",
    0x10000049: "position_bccorrection1",
    0x10000050: "position_bccorrection2",
    0x10000051: "interpolation_y",
    0x10000052: "camera_binning",
    0x10000053: "camera_supersampling",
    0x10000054: "camera_frame_width",
    0x10000055: "camera_frame_height",
    0x10000056: "camera_offset_x",
    0x10000057: "camera_offset_y",
    # lasers
    0x50000001: "name",
    0x50000002: "acquire",
    0x50000003: "power",
    # tracks
    0x40000001: "multiplex_type",
    0x40000002: "multiplex_order",
    0x40000003: "sampling_mode",
    0x40000004: "sampling_method",
    0x40000005: "sampling_number",
    0x40000006: "acquire",
    0x40000007: "sample_observation_time",
    0x4000000b: "time_between_stacks",
    0x4000000c: "name",
    0x4000000d: "collimator1_name",
    0x4000000e: "collimator1_position",
    0x4000000f: "collimator2_name",
    0x40000010: "collimator2_position",
    0x40000011: "is_bleach_track",
    0x40000012: "is_bleach_after_scan_number",
    0x40000013: "bleach_scan_number",
    0x40000014: "trigger_in",
    0x40000015: "trigger_out",
    0x40000016: "is_ratio_track",
    0x40000017: "bleach_count",
    0x40000018: "spi_center_wavelength",
    0x40000019: "pixel_time",
    0x40000021: "condensor_frontlens",
    0x40000023: "field_stop_value",
    0x40000024: "id_condensor_aperture",
    0x40000025: "condensor_aperture",
    0x40000026: "id_condensor_revolver",
    0x40000027: "condensor_filter",
    0x40000028: "id_transmission_filter1",
    0x40000029: "id_transmission1",
    0x40000030: "id_transmission_filter2",
    0x40000031: "id_transmission2",
    0x40000032: "repeat_bleach",
    0x40000033: "enable_spot_bleach_pos",
    0x40000034: "spot_bleach_posx",
    0x40000035: "spot_bleach_posy",
    0x40000036: "spot_bleach_posz",
    0x40000037: "id_tubelens",
    0x40000038: "id_tubelens_position",
    0x40000039: "transmitted_light",
    0x4000003a: "reflected_light",
    0x4000003b: "simultan_grab_and_bleach",
    0x4000003c: "bleach_pixel_time",
    # detection_channels
    0x70000001: "integration_mode",
    0x70000002: "special_mode",
    0x70000003: "detector_gain_first",
    0x70000004: "detector_gain_last",
    0x70000005: "amplifier_gain_first",
    0x70000006: "amplifier_gain_last",
    0x70000007: "amplifier_offs_first",
    0x70000008: "amplifier_offs_last",
    0x70000009: "pinhole_diameter",
    0x7000000a: "counting_trigger",
    0x7000000b: "acquire",
    0x7000000c: "point_detector_name",
    0x7000000d: "amplifier_name",
    0x7000000e: "pinhole_name",
    0x7000000f: "filter_set_name",
    0x70000010: "filter_name",
    0x70000013: "integrator_name",
    0x70000014: "detection_channel_name",
    0x70000015: "detection_detector_gain_bc1",
    0x70000016: "detection_detector_gain_bc2",
    0x70000017: "detection_amplifier_gain_bc1",
    0x70000018: "detection_amplifier_gain_bc2",
    0x70000019: "detection_amplifier_offset_bc1",
    0x70000020: "detection_amplifier_offset_bc2",
    0x70000021: "detection_spectral_scan_channels",
    0x70000022: "detection_spi_wavelength_start",
    0x70000023: "detection_spi_wavelength_stop",
    0x70000026: "detection_dye_name",
    0x70000027: "detection_dye_folder",
    # illumination_channels
    0x90000001: "name",
    0x90000002: "power",
    0x90000003: "wavelength",
    0x90000004: "aquire",
    0x90000005: "detchannel_name",
    0x90000006: "power_bc1",
    0x90000007: "power_bc2",
    # beam_splitters
    0xb0000001: "filter_set",
    0xb0000002: "filter",
    0xb0000003: "name",
    # data_channels
    0xd0000001: "name",
    0xd0000003: "acquire",
    0xd0000004: "color",
    0xd0000005: "sample_type",
    0xd0000006: "bits_per_sample",
    0xd0000007: "ratio_type",
    0xd0000008: "ratio_track1",
    0xd0000009: "ratio_track2",
    0xd000000a: "ratio_channel1",
    0xd000000b: "ratio_channel2",
    0xd000000c: "ratio_const1",
    0xd000000d: "ratio_const2",
    0xd000000e: "ratio_const3",
    0xd000000f: "ratio_const4",
    0xd0000010: "ratio_const5",
    0xd0000011: "ratio_const6",
    0xd0000012: "ratio_first_images1",
    0xd0000013: "ratio_first_images2",
    0xd0000014: "dye_name",
    0xd0000015: "dye_folder",
    0xd0000016: "spectrum",
    0xd0000017: "acquire",
    # markers
    0x14000001: "name",
    0x14000002: "description",
    0x14000003: "trigger_in",
    0x14000004: "trigger_out",
    # timers
    0x12000001: "name",
    0x12000002: "description",
    0x12000003: "interval",
    0x12000004: "trigger_in",
    0x12000005: "trigger_out",
    0x12000006: "activation_time",
    0x12000007: "activation_number",
}

# Map TIFF tag codes to attribute names, default value, type, count, validator
TIFF_TAGS = {
    254: ('new_subfile_type', 0, 4, 1, TIFF_SUBFILE_TYPES()),
    255: ('subfile_type', None, 3, 1, TIFF_OSUBFILE_TYPES),
    256: ('image_width', None, 4, 1, None),
    257: ('image_length', None, 4, 1, None),
    258: ('bits_per_sample', 1, 3, 1, None),
    259: ('compression', 1, 3, 1, TIFF_COMPESSIONS),
    262: ('photometric', None, 3, 1, TIFF_PHOTOMETRICS),
    266: ('fill_order', 1, 3, 1, TIFF_FILLORDERS),
    269: ('document_name', None, 2, None, None),
    270: ('image_description', None, 2, None, None),
    271: ('make', None, 2, None, None),
    272: ('model', None, 2, None, None),
    273: ('strip_offsets', None, 4, None, None),
    274: ('orientation', 1, 3, 1, TIFF_ORIENTATIONS),
    277: ('samples_per_pixel', 1, 3, 1, None),
    278: ('rows_per_strip', 2**32-1, 4, 1, None),
    279: ('strip_byte_counts', None, 4, None, None), # required
    #280: ('min_sample_value', 0, 3, None, None),
    #281: ('max_sample_value', None, 3, None, None), # 2**bits_per_sample
    282: ('x_resolution', None, 5, 1, None),
    283: ('y_resolution', None, 5, 1, None),
    284: ('planar_configuration', 1, 3, 1, TIFF_PLANARCONFIGS),
    285: ('page_name', None, 2, None, None),
    296: ('resolution_unit', 2, 4, 1, TIFF_RESUNITS),
    305: ('software', None, 2, None, None),
    306: ('datetime', None, 2, None, None),
    315: ('artist', None, 2, None, None),
    316: ('host_computer', None, 2, None, None),
    317: ('predictor', 1, 3, 1, TIFF_PREDICTORS),
    320: ('color_map', None, 3, None, None),
    338: ('extra_samples', None, 3, None, TIFF_EXTRA_SAMPLES),
    339: ('sample_format', 1, 3, 1, TIFF_SAMPLE_FORMATS),
    33432: ('copyright', None, 2, None, None),
    32997: ('image_depth', None, 4, 1, None),
    32998: ('tile_depth', None, 4, 1, None),
}

# Map custom TIFF tag codes to attribute names and import functions
CUSTOM_TAGS = {
    33628: ('mm_uic1', read_mm_uic1),
    33629: ('mm_uic2', read_mm_uic2),
    33630: ('mm_uic3', read_mm_uic3),
    33631: ('mm_uic4', read_mm_uic4),
    34361: ('mm_header', read_mm_header),
    34362: ('mm_stamp', read_mm_stamp),
    34386: ('mm_user_block', None),
    34412: ('cz_lsm_info', read_cz_lsm_info),
    43314: ('nih_image_header', read_nih_image_header),
}

# Max line length of printed output
PRINT_LINE_LEN = 79

def imshow(data, title=None, isrgb=True, vmin=0, vmax=None,
           cmap=None, photometric='rgb', interpolation='bilinear',
           dpi=96, figure=None, subplot=111, maxdim=4096, **kwargs):
    """Plot n-dimensional images using matplotlib.pyplot.

    Return figure, subplot and plot axis.
    Requires pyplot already imported ``from matplotlib import pyplot``.

    Arguments
    ---------

    isrgb : bool
        If True, data will be displayed as RGB(A) images if possible.

    photometric : str
        'miniswhite', 'minisblack', 'rgb', or 'palette'

    title : str
        Window and subplot title.

    figure : a matplotlib.figure.Figure instance (optional).

    subplot : int
        A matplotlib.pyplot.subplot axis.

    maxdim: int
        maximum image size in any dimension.

    Other arguments are same as for matplotlib.pyplot.imshow.

    """

    if photometric not in ('miniswhite', 'minisblack', 'rgb', 'palette'):
        raise ValueError("Can't handle %s photometrics" % self.photometric)

    data = data.squeeze()
    data = data[(slice(0, maxdim), ) * len(data.shape)]

    dims = len(data.shape)
    if dims < 2:
        raise ValueError("not an image")
    if dims == 2:
        dims = 0
        isrgb = False
    else:
        if (isrgb and data.shape[-3] in (3, 4)):
            data = numpy.swapaxes(data, -3, -2)
            data = numpy.swapaxes(data, -2, -1)
        elif (not isrgb and data.shape[-1] in (3, 4)):
            data = numpy.swapaxes(data, -3, -1)
            data = numpy.swapaxes(data, -2, -1)
        isrgb = isrgb and data.shape[-1] in (3, 4)
        dims -= 3 if isrgb else 2

    datamax = data.max()
    if data.dtype in (numpy.int8, numpy.int16, numpy.int32,
                      numpy.uint8, numpy.uint16, numpy.uint32):
        for bits in (1, 2, 4, 8, 10, 12, 14, 16, 24, 32):
            if datamax <= 2**bits:
                datamax = 2**bits
                break
        if isrgb:
            data *= (255.0 / datamax) # better use digitize()
            data = data.astype('B')
    elif isrgb:
        data /= datamax

    if not isrgb and vmax is None:
        vmax = datamax

    pyplot = sys.modules['matplotlib.pyplot']

    if figure is None:
        pyplot.rc('font', family='sans-serif', weight='normal', size=8)
        figure = pyplot.figure(dpi=dpi, figsize=(10.3, 6.3), frameon=True,
                               facecolor='1.0', edgecolor='w')
        try:
            figure.canvas.manager.window.title(title)
        except Exception:
            pass
        pyplot.subplots_adjust(bottom=0.03*(dims+2), top=0.925,
                               left=0.1, right=0.95, hspace=0.05, wspace=0.0)
    subplot = pyplot.subplot(subplot)

    if title:
        pyplot.title(title, size=11)

    if cmap is None:
        if photometric=='miniswhite':
            cmap = pyplot.cm.binary
        else:
            cmap = pyplot.cm.gray

    image = pyplot.imshow(data[(0, ) * dims].squeeze(), vmin=vmin, vmax=vmax,
                          cmap=cmap, interpolation=interpolation, **kwargs)

    if not isrgb:
        pyplot.colorbar()

    def format_coord(x, y):
        """Callback to format coordinate display in toolbar."""
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            if dims:
                return "%s @ %s [%4i, %4i]" % (cur_ax_dat[1][y, x],
                                               current, x, y)
            else:
                return "%s @ [%4i, %4i]" % (data[y, x], x, y)
        except IndexError:
            return ""

    pyplot.gca().format_coord = format_coord

    if dims:
        current = list((0, ) * dims)
        cur_ax_dat = [0, data[tuple(current)].squeeze()]
        sliders = [pyplot.Slider(
            pyplot.axes([0.125, 0.03*(axis+1), 0.725, 0.025]),
            'Dimension %i' % axis, 0, data.shape[axis]-1, 0, facecolor='0.5',
            valfmt='%%.0f of %i' % data.shape[axis]) for axis in range(dims)]
        for slider in sliders:
            slider.drawon = False

        def set_image(current, sliders=sliders, data=data):
            """Change image and redraw canvas."""
            cur_ax_dat[1] = data[tuple(current)].squeeze()
            image.set_data(cur_ax_dat[1])
            for ctrl, index in zip(sliders, current):
                ctrl.eventson = False
                ctrl.set_val(index)
                ctrl.eventson = True
            figure.canvas.draw()

        def on_changed(index, axis, data=data, image=image, figure=figure,
                       current=current):
            """Callback for slider change event."""
            index = int(round(index))
            cur_ax_dat[0] = axis
            if index == current[axis]:
                return
            if index >= data.shape[axis]:
                index = 0
            elif index < 0:
                index = data.shape[axis] - 1
            current[axis] = index
            set_image(current)

        def on_keypressed(event, data=data, current=current):
            """Callback for key press event."""
            key = event.key
            axis = cur_ax_dat[0]
            if str(key) in '0123456789':
                on_changed(key, axis)
            elif key == 'right':
                on_changed(current[axis] + 1, axis)
            elif key == 'left':
                on_changed(current[axis] - 1, axis)
            elif key == 'up':
                cur_ax_dat[0] = 0 if axis == len(data.shape)-1 else axis + 1
            elif key == 'down':
                cur_ax_dat[0] = len(data.shape)-1 if axis == 0 else axis - 1
            elif key == 'end':
                on_changed(data.shape[axis] - 1, axis)
            elif key == 'home':
                on_changed(0, axis)

        figure.canvas.mpl_connect('key_press_event', on_keypressed)
        for axis, ctrl in enumerate(sliders):
            ctrl.on_changed(lambda k, a=axis: on_changed(k, a))

    return figure, subplot, image


def main(argv=None):
    """Command line usage main function."""
    if float(sys.version[0:3]) < 2.5:
        print "This script requires Python version 2.5 or better."
        print "This is Python version %s" % sys.version
        return 0
    if argv is None:
        argv = sys.argv

    import re
    import optparse
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot

    search_doc = lambda r, d: re.search(r, __doc__).group(1) if __doc__ else d
    parser = optparse.OptionParser(
        usage="usage: %prog [options] path",
        description=search_doc("\n\n([^|]*?)\n\n", ''),
        version="%%prog %s" % search_doc(":Version: (.*)", "Unknown"))
    opt = parser.add_option
    opt('-p', '--page', dest='page', type='int', default=-1,
        help="display single page")
    opt('--noplot', dest='noplot', action='store_true', default=False,
        help="don't display images")
    opt('--norgb', dest='norgb', action='store_true', default=False,
        help="don't try display as RGB(A) color images")
    opt('--nocolmap', dest='nocolmap', action='store_true', default=False,
        help="don't apply color mapping to paletted images")
    opt('--interpol', dest='interpol', metavar='INTERPOL', default='bilinear',
        help="image interpolation method")
    opt('--dpi', dest='dpi', type='int', default=96,
        help="set plot resolution")
    opt('--test', dest='test', action='store_true', default=False,
        help="try read all images in path")
    opt('--doctest', dest='doctest', action='store_true', default=False,
        help="runs the internal tests")
    opt('-v', '--verbose', dest='verbose', action='store_true', default=True)
    opt('-q', '--quiet', dest='verbose', action='store_false')

    settings, path = parser.parse_args()
    path = ' '.join(path)

    if settings.doctest:
        import doctest
        doctest.testmod()
        return 0

    if not path:
        parser.error("No file specified")

    if settings.test:
        test_tifffile(path, settings.verbose)
        return 0

    print "Reading file structure...",
    start = time.time()
    tif = TIFFfile(path)
    print "%.3f ms" % ((time.time()-start) * 1e3)

    img = None
    if not settings.noplot:
        print "Reading image data... ",
        start = time.time()
        try:
            if settings.page < 0:
                img = tif.asarray(colormapped=not settings.nocolmap,
                                  rgbonly=not settings.norgb)
            else:
                img = tif[settings.page].asarray(rgbonly=not settings.norgb,
                                            colormapped=not settings.nocolmap)
            print "%.3f ms" % ((time.time()-start) * 1e3)
        except ValueError, e:
            print e #; raise
    tif.close()

    print "\nTIFF file:", tif
    page = 0 if settings.page < 0 else settings.page
    print "\nPAGE %i:" % page, tif[page]
    page = tif[page]
    print page.tags
    if page.is_palette:
        print "\nColor Map:", page.color_map.shape, page.color_map.dtype

    for attr in ('cz_lsm_info', 'cz_lsm_scan_info', 'mm_uic_tags',
                 'mm_header', 'nih_image_header'):
        if hasattr(page, attr):
            print "\n", attr.upper(), "\n", Record(getattr(page, attr))

    if img is not None and not settings.noplot:
        imshow(img, title=', '.join((str(tif), str(tif[0]))),
               photometric=page.photometric,
               interpolation=settings.interpol,
               dpi=settings.dpi, isrgb=not settings.norgb)
        pyplot.show()


# Documentation in HTML format can be generated with Epydoc
__docformat__ = "restructuredtext en"

if __name__ == "__main__":
    sys.exit(main())
