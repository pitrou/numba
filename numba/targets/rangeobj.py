"""
Implementation of the range object for fixed-size integers.
"""

import llvm.core as lc

from numba import errcode
from numba import types, typing, cgutils
from numba.targets.imputils import builtin, implement, iterator_impl


def make_range(range_state_type, range_iter_type, int_type):

    class RangeState(cgutils.Structure):
        _fields = [('start', int_type),
                   ('stop', int_type),
                   ('step', int_type)]

    @builtin
    @implement(types.range_type, int_type)
    def range1_impl(context, builder, sig, args):
        """
        range(stop: int) -> range object
        """
        [stop] = args
        state = RangeState(context, builder)
        state.start = context.get_constant(int_type, 0)
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return state._getvalue()

    @builtin
    @implement(types.range_type, int_type, int_type)
    def range2_impl(context, builder, sig, args):
        """
        range(start: int, stop: int) -> range object
        """
        start, stop = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = context.get_constant(int_type, 1)
        return state._getvalue()

    @builtin
    @implement(types.range_type, int_type, int_type, int_type)
    def range3_impl(context, builder, sig, args):
        """
        range(start: int, stop: int, step: int) -> range object
        """
        [start, stop, step] = args
        state = RangeState(context, builder)
        state.start = start
        state.stop = stop
        state.step = step
        return state._getvalue()

    @builtin
    @implement('getiter', range_state_type)
    def getiter_range32_impl(context, builder, sig, args):
        """
        range.__iter__
        """
        (value,) = args
        state = RangeState(context, builder, value)
        return RangeIter.from_range_state(context, builder, state)._getvalue()

    @iterator_impl(range_state_type, range_iter_type)
    class RangeIter(cgutils.Structure):

        _fields = [('iter', types.CPointer(int_type)),
                   ('stop', int_type),
                   ('step', int_type),
                   ('count', types.CPointer(int_type))]

        @classmethod
        def from_range_state(cls, context, builder, state):
            """
            Create a RangeIter initialized from the given RangeState *state*.
            """
            self = cls(context, builder)
            start = state.start
            stop = state.stop
            step = state.step

            startptr = cgutils.alloca_once(builder, start.type)
            builder.store(start, startptr)

            countptr = cgutils.alloca_once(builder, start.type)

            self.iter = startptr
            self.stop = stop
            self.step = step
            self.count = countptr

            diff = builder.sub(stop, start)
            zero = context.get_constant(int_type, 0)
            one = context.get_constant(int_type, 1)
            pos_diff = builder.icmp(lc.ICMP_SGT, diff, zero)
            pos_step = builder.icmp(lc.ICMP_SGT, step, zero)
            sign_differs = builder.xor(pos_diff, pos_step)
            zero_step = builder.icmp(lc.ICMP_EQ, step, zero)

            with cgutils.if_unlikely(builder, zero_step):
                # step shouldn't be zero
                context.return_errcode(builder, errcode.ASSERTION_ERROR)

            with cgutils.ifelse(builder, sign_differs) as (then, orelse):
                with then:
                    builder.store(zero, self.count)

                with orelse:
                    rem = builder.srem(diff, step)
                    uneven = builder.icmp(lc.ICMP_SGT, rem, zero)
                    newcount = builder.add(builder.sdiv(diff, step),
                                           builder.select(uneven, one, zero))
                    builder.store(newcount, self.count)

            return self

        def iternext(self, context, builder):
            res = builder.load(self.iter)
            one = context.get_constant(int_type, 1)

            countptr = self.count
            builder.store(builder.sub(builder.load(countptr), one), countptr)

            builder.store(builder.add(res, self.step), self.iter)

            return res

        def itervalid(self, context, builder):
            zero = context.get_constant(int_type, 0)
            gt = builder.icmp(lc.ICMP_SGE, builder.load(self.count), zero)
            return gt

    return RangeState, RangeIter


RangeState32, RangeIter32 = make_range(types.range_state32_type,
                                       types.range_iter32_type, types.int32)
RangeState64, RangeIter64 = make_range(types.range_state64_type,
                                       types.range_iter64_type, types.int64)
