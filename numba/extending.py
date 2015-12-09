
from numba import types

from .typing.typeof import typeof_impl

from .targets.imputils import (builtin, builtin_attr, implement,
                               impl_attribute, impl_attribute_generic)


def type_callable(func):
    from numba.typing.templates import CallableTemplate, builtin, builtin_global
    try:
        func_name = func.__name__
    except AttributeError:
        func_name = str(func)

    def decorate(typing_func):
        def generic(self):
            return typing_func(self.context)

        name = "%s_CallableTemplate" % (func_name,)
        bases = (CallableTemplate,)
        class_dict = dict(key=func, generic=generic)
        template = type(name, bases, class_dict)
        builtin(template)
        builtin_global(func, types.Function(template))

    return decorate
