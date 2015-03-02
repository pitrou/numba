from __future__ import print_function, division, absolute_import

import array
import sys

import numpy as np

from numba import unittest_support as unittest
from numba import jit
from .support import TestCase


@jit(nopython=True)
def len_usecase(buf):
    return len(buf)


@jit(nopython=True)
def getitem_usecase(buf, i):
    return buf[i]


@jit(nopython=True)
def getslice_usecase(buf, i, j):
    s = buf[i:j]
    return s[0] + 2 * s[-1]


@jit(nopython=True)
def setitem_usecase(buf, i, v):
    buf[i] = v


@jit(nopython=True)
def iter_usecase(buf):
    res = 0.0
    for i, x in enumerate(buf):
        res += x
        res *= i + 1
    return res


# On Python 2, array.array doesn't support the PEP 3118 buffer API
array_supported = sys.version_info >= (3,)
# On Python 2, bytes is really the str object
bytes_supported = sys.version_info >= (3,)
# On Python 2, indexing a memoryview returns bytes
memoryview_structured_indexing = sys.version_info >= (3,)


@unittest.skipIf(sys.version_info < (2, 7),
                 "buffer protocol not supported on Python 2.6")
class TestBufferProtocol(TestCase):

    def _arrays(self):
        n = 10
        for letter, offset in [
            ('b', -3),
            ('B', 0),
            ('h', -5000),
            ('H', 40000),
            ('i', -100000),
            ('I', 1000000),
            ('l', -100000),
            ('L', 1000000),
            ('q', -2**60),
            ('Q', 2**63 + 1),
            ('f', 1.5),
            ('d', -1.5),
            ]:
            yield array.array(letter, [i + offset for i in range(n)])

    def _memoryviews(self):
        n = 10
        yield memoryview(bytearray(b"abcdefghi"))
        yield memoryview(b"abcdefghi")
        for dtype, start, stop in [
            ('int8', -10, 10),
            ('uint8', 0, 10),
            ('int16', -5000, 1000),
            ('uint16', 40000, 50000),
            ('int32', -100000, 100000),
            ('uint32', 0, 1000000),
            ('int64', -2**60, 10),
            ('uint64', 0, 2**64 - 10),
            ('float32', 1.5, 3.5),
            ('float64', 1.5, 3.5),
            ('complex64', -8j, 12 + 5j),
            ('complex128', -8j, 12 + 5j),
            ]:
            yield memoryview(np.linspace(start, stop, n, dtype=dtype))

    def _check_unary(self, jitfunc, *args):
        pyfunc = jitfunc.py_func
        self.assertPreciseEqual(jitfunc(*args), pyfunc(*args))

    def check_len(self, obj):
        self._check_unary(len_usecase, obj)

    def check_iter(self, obj):
        self._check_unary(iter_usecase, obj)

    def check_getitem(self, obj):
        for i in range(len(obj)):
            try:
                expected = obj[i]
            except NotImplementedError:
                if isinstance(obj, memoryview):
                    # The memoryview object doesn't support all codes yet,
                    # fall back on the underlying object.
                    expected = obj.obj[i]
            self.assertPreciseEqual(getitem_usecase(obj, i), expected)

    def check_setitem(self, obj):
        for i in range(len(obj)):
            orig = list(obj)
            val = obj[i] // 2 + 1
            setitem_usecase(obj, i, val)
            self.assertEqual(obj[i], val)
            for j, val in enumerate(orig):
                if j != i:
                    self.assertEqual(obj[j], val)

    def check_getslice(self, obj):
        self._check_unary(getslice_usecase, obj, 1, len(obj) - 1)

    def test_len(self):
        self.check_len(bytearray(5))
        if bytes_supported:
            self.check_len(b"xyz")
        for mem in self._memoryviews():
            self.check_len(mem)
        if array_supported:
            for arr in self._arrays():
                self.check_len(arr)

    def test_getitem(self):
        self.check_getitem(bytearray(b"abc"))
        if bytes_supported:
            self.check_getitem(b"xyz")
        if memoryview_structured_indexing:
            for mem in self._memoryviews():
                self.check_getitem(mem)
        if array_supported:
            for arr in self._arrays():
                self.check_getitem(arr)

    def test_getslice(self):
        with self.assertTypingError():
            self.check_getslice(bytearray(b"abcde"))
        if bytes_supported:
            self.check_getslice(b"xyzuvw")
        if memoryview_structured_indexing:
            self.check_getslice(memoryview(b"xyzuvw"))
        if array_supported:
            with self.assertTypingError():
                self.check_getslice(array.array('i', range(10)))

    def test_setitem(self):
        self.check_setitem(bytearray(b"abcdefghi"))
        if memoryview_structured_indexing:
            self.check_setitem(memoryview(b"abcdefghi"))
        if array_supported:
            for arr in self._arrays():
                self.check_setitem(arr)

    def test_iter(self):
        self.check_iter(bytearray(b"abc"))
        if bytes_supported:
            self.check_iter(b"xyz")
        if memoryview_structured_indexing:
            self.check_iter(memoryview(b"xyz"))
        if array_supported:
            for arr in self._arrays():
                self.check_iter(arr)


if __name__ == '__main__':
    unittest.main()