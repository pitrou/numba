from __future__ import print_function, division, absolute_import

import math

from numba import unittest_support as unittest
from numba import jit, types
from numba.compiler import compile_isolated
from .support import TestCase

from numba.extending import (typeof_impl, type_callable,
                             builtin, implement, overlay)


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


def where(cond, x=None, y=None):
    raise NotImplementedError

@overlay(where)
def where(cond, x, y):
    """
    Implement where().
    """
    # Choose implementation based on argument types.
    if isinstance(cond, types.BaseTuple):
        n = len(cond)
        if n != len(x) or n != len(y):
            raise TypingError("where() arguments must be the same length")
        #if n == 0:
            #def where_impl(cond, x, y):
                #return ()
        #else:
            #def where_impl(
        #def where_impl(cond, x, y):

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


if __name__ == '__main__':
    unittest.main()
