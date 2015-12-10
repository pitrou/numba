from __future__ import print_function, division, absolute_import

from collections import namedtuple
import math

import numpy as np

from numba import unittest_support as unittest
from numba import jit, types, errors, typeof
from numba.compiler import compile_isolated
from .support import TestCase

from numba.extending import (typeof_impl, type_callable,
                             builtin, builtin_cast,
                             implement, overlay,
                             models, register_model)


# Define a function's typing and implementation using the classical
# two-step API

def func1(x=None):
    raise NotImplementedError

@type_callable(func1)
def type_func1(context):
    def typer(x=None):
        if x in (None, types.none):
            # 0-arg or 1-arg with None
            return types.int32
        elif isinstance(x, types.Float):
            # 1-arg with float
            return x

    return typer

@builtin
@implement(func1)
@implement(func1, types.none)
def func1_nullary(context, builder, sig, args):
    return context.get_constant(sig.return_type, 42)

@builtin
@implement(func1, types.Float)
def func1_unary(context, builder, sig, args):
    def func1_impl(x):
        return math.sqrt(2 * x)
    return context.compile_internal(builder, func1_impl, sig, args)

def call_func1_nullary():
    return func1()

def call_func1_unary(x):
    return func1(x)


# Define a custom type and an implicit cast on it

class MyDummy(object):
    pass

class MyDummyType(types.Opaque):

    def can_convert_to(self, context, toty):
        if isinstance(toty, types.Number):
            from numba.typeconv import Conversion
            return Conversion.safe

mydummy_type = MyDummyType('mydummy')
mydummy = MyDummy()

@typeof_impl.register(MyDummy)
def typeof_mydummy(val, c):
    return mydummy_type

@builtin_cast(MyDummyType, types.Number)
def mydummy_to_number(context, builder, fromty, toty, val):
    return context.get_constant(toty, 42)

def get_dummy():
    return mydummy

register_model(MyDummyType)(models.OpaqueModel)


# Define an overlaid function (combined API)

def where(cond, x, y):
    raise NotImplementedError

def np_where(cond, x, y):
    """
    Wrap np.where() to allow for keyword arguments
    """
    return np.where(cond, x, y)

def call_where(cond, x, y):
    return where(cond, x, y)

@overlay(where)
def overlay_where(cond, x, y):
    """
    Implement where().
    """

    # Choose implementation based on argument types.
    if isinstance(cond, types.Array):
        if x.dtype != y.dtype:
            raise errors.TypingError("x and y should have the same dtype")

        # Array where() => return an array of the same shape
        if all(ty.layout == 'C' for ty in (cond, x, y)):
            def where_impl(cond, x, y):
                """
                Fast implementation for C-contiguous arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                cf = cond.flat
                xf = x.flat
                yf = y.flat
                rf = res.flat
                for i in range(cond.size):
                    rf[i] = xf[i] if cf[i] else yf[i]
                return res
        else:
            def where_impl(cond, x, y):
                """
                Generic implementation for other arrays
                """
                shape = cond.shape
                if x.shape != shape or y.shape != shape:
                    raise ValueError("all inputs should have the same shape")
                res = np.empty_like(x)
                for idx, c in np.ndenumerate(cond):
                    res[idx] = x[idx] if c else y[idx]
                return res

    else:
        def where_impl(cond, x, y):
            """
            Scalar where() => return a 0-dim array
            """
            scal = x if cond else y
            return np.full_like(scal, scal)

    return where_impl


class TestLowLevelExtending(TestCase):

    # We check with both @jit and compile_isolated(), to exercise the
    # registration logic.

    def test_func1(self):
        pyfunc = call_func1_nullary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(), 42)
        pyfunc = call_func1_unary
        cfunc = jit(nopython=True)(pyfunc)
        self.assertPreciseEqual(cfunc(None), 42)
        self.assertPreciseEqual(cfunc(18.0), 6.0)

    def test_func1_isolated(self):
        pyfunc = call_func1_nullary
        cr = compile_isolated(pyfunc, ())
        self.assertPreciseEqual(cr.entry_point(), 42)
        pyfunc = call_func1_unary
        cr = compile_isolated(pyfunc, (types.float64,))
        self.assertPreciseEqual(cr.entry_point(18.0), 6.0)

    def test_cast_mydummy(self):
        pyfunc = get_dummy
        cr = compile_isolated(pyfunc, (), types.float64)
        self.assertPreciseEqual(cr.entry_point(), 42.0)


class TestHighLevelExtending(TestCase):

    def test_where(self):
        pyfunc = call_where
        cfunc = jit(nopython=True)(pyfunc)

        def check(*args, **kwargs):
            expected = np_where(*args, **kwargs)
            got = cfunc(*args, **kwargs)
            self.assertPreciseEqual

        check(True, 3, 8)
        check(np.bool_([True, False, True]), np.int32([1, 2, 3]),
              np.int32([4, 5, 5]))
        check(x=3, cond=True, y=8)

        # The typing error is propagated
        with self.assertRaises(errors.TypingError) as raises:
            cfunc(np.bool_([]), np.int32([]), np.int64([]))
        self.assertIn("x and y should have the same dtype",
                      str(raises.exception))


if __name__ == '__main__':
    unittest.main()
