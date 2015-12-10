
from numba import types, typing

# Re-exported symbols
from .typing.typeof import typeof_impl
from .targets.imputils import builtin, implement


def type_callable(func):
    """
    Decorate a function as implementing typing for the callable *func*.
    """
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


def overlay(func):
    from numba.typing.templates import builtin_global

    def decorate(overlay_func):
        ty = types.OverlayFunction(func, overlay_func)
        builtin_global(func, ty)
        return overlay_func

    return decorate
