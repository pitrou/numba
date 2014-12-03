"""
Tests for the _dispatcher.HashWrapper object.
"""

from __future__ import print_function, absolute_import

import gc
import weakref

import numpy as np

from numba import unittest_support as unittest
from numba import _dispatcher


wrap = _dispatcher.make_wrapper


class HashWrapper(unittest.TestCase):

    def test_basics(self):
        v = wrap(None)
        self.assertIsInstance(v, _dispatcher.HashWrapper)
        w = wrap(v)
        self.assertIs(v, w)
        del v, w

    def test_eq(self):
        # Equality between two wrappers reflect equality between
        # the respective wrapped objects.
        def eq(a, b):
            self.assertTrue(a == b, (a, b))
            self.assertFalse(a != b, (a, b))
            self.assertTrue(b == a, (a, b))
            self.assertFalse(b != a, (a, b))
        def ne(a, b):
            self.assertTrue(a != b, (a, b))
            self.assertFalse(a == b, (a, b))
            self.assertTrue(b != a, (a, b))
            self.assertFalse(b == a, (a, b))
        u = wrap(1)
        v = wrap(2)
        w = wrap(1)
        x = wrap(1.0)
        ne(u, v)
        eq(u, w)
        eq(u, x)
        # Comparison with unwrapped objects succeeds
        eq(u, 1)
        eq(u, 1.0)
        ne(u, 2)
        ne(u, None)

    def test_hash(self):
        u = wrap(1)
        for i in range(5):
            self.assertEqual(hash(u), hash(1))

    def test_hash_caching(self):
        called = []
        class Dummy(object):
            def __hash__(self):
                called.append(None)
                return 42

        a = wrap(Dummy())
        b = wrap(Dummy())
        for i in range(5):
            self.assertEqual(hash(a), 42)
        self.assertEqual(len(called), 1)
        for i in range(5):
            self.assertEqual(hash(b), 42)
        self.assertEqual(len(called), 2)

    def test_hash_error(self):
        v = wrap([])
        with self.assertRaises(TypeError):
            hash(v)
        with self.assertRaises(TypeError):
            hash(v)

        class Dummy(object):
            def __hash__(self):
                1/0

        v = wrap(Dummy())
        with self.assertRaises(ZeroDivisionError):
            hash(v)
        with self.assertRaises(ZeroDivisionError):
            hash(v)

    def test_dict(self):
        d = {}
        u = wrap(1)
        v = wrap(2)
        w = wrap(1)
        x = wrap(1.0)
        y = wrap(3)
        d[u] = 'u'
        d[v] = 'v'
        self.assertEqual(d[u], 'u')
        self.assertEqual(d[v], 'v')
        self.assertEqual(d[w], 'u')
        self.assertEqual(d[x], 'u')
        self.assertEqual(d[1], 'u')
        self.assertEqual(d[1.0], 'u')
        self.assertNotIn(y, d)

        # Force collisions
        class Dummy(object):
            def __hash__(self):
                return 42
            def __eq__(self, other):
                return self is other

        a = wrap(Dummy())
        b = wrap(Dummy())
        d[a] = 'a'
        self.assertEqual(d[a], 'a')
        self.assertNotIn(b, d)
        d[b] = 'b'
        self.assertEqual(d[a], 'a')
        self.assertEqual(d[b], 'b')

    def test_dict_numpy_dtypes(self):
        # This checks the concrete use case for HashWrapper objects.
        #
        d = {}
        a = np.dtype('int32')
        b = np.dtype('int32')
        c = np.dtype('float64')
        u = wrap(a)
        v = wrap(b)
        w = wrap(c)
        d[u] = 'a'
        d[w] = 'c'
        self.assertEqual(d[u], 'a')
        self.assertEqual(d[v], 'a')
        self.assertEqual(d[w], 'c')
        self.assertEqual(d[a], 'a')
        self.assertEqual(d[b], 'a')
        self.assertEqual(d[c], 'c')
        # The below fails because of https://github.com/numpy/numpy/issues/5345
        #d = {}
        #d[a] = 'a'
        #d[c] = 'c'
        #self.assertEqual(d[u], 'a')
        #self.assertEqual(d[v], 'a')
        #self.assertEqual(d[w], 'c')

    def check_refcount(self, wrapfunc):
        class Dummy(object):
            pass

        d = Dummy()
        v = wrapfunc(d)
        wr = weakref.ref(d)
        del d
        gc.collect()
        self.assertIsNot(wr(), None)
        del v
        gc.collect()
        self.assertIs(wr(), None)

    def test_refcount(self):
        self.check_refcount(wrap)

    def test_refcount_double_wrap(self):
        def double_wrap(v):
            return wrap(wrap(v))
        self.check_refcount(double_wrap)

    def test_cyclic_gc(self):
        class Dummy(object):
            pass
        d = Dummy()
        d.d = d
        v = wrap(d)
        wr = weakref.ref(d)
        del d, v
        gc.collect()
        self.assertIs(wr(), None)


if __name__ == '__main__':
    unittest.main()
