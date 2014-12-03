/*
 * A simple object wrapping another one and caching its hash value,
 * for performance.
 */

#include "_pymodule.h"
#include "_hashwrap.h"


typedef struct {
    PyObject_HEAD
    PyObject *wrapped;
    Py_hash_t hash;
} WrapperObject;

static PyTypeObject WrapperType;

#define WRAPPER_CHECK(v) (Py_TYPE(v) == &WrapperType)


static PyObject *
wrapper_richcompare(PyObject *v, PyObject *w, int op)
{
    WrapperObject *obj = (WrapperObject *) v;
    if (WRAPPER_CHECK(w))
        w = ((WrapperObject *) w)->wrapped;
    return PyObject_RichCompare(obj->wrapped, w, op);
}

static Py_hash_t
wrapper_hash(WrapperObject *obj)
{
    if (obj->hash == -1)
        obj->hash = PyObject_Hash(obj->wrapped);
    return obj->hash;
}

static int
wrapper_traverse(WrapperObject *obj, visitproc visit, void *arg)
{
    Py_VISIT(obj->wrapped);
    return 0;
}

static void
wrapper_dealloc(WrapperObject *obj)
{
    Py_DECREF(obj->wrapped);
    _PyObject_GC_UNTRACK((PyObject *) obj);
    Py_TYPE(obj)->tp_free((PyObject *) obj);
}


static PyTypeObject WrapperType = {
#if (PY_MAJOR_VERSION < 3)
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#else
    PyVarObject_HEAD_INIT(NULL, 0)
#endif
    "_dispatcher.HashWrapper", /*tp_name*/
    sizeof(WrapperObject),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor) wrapper_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    (hashfunc) wrapper_hash,   /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, /*tp_flags*/
    0,                         /* tp_doc */
    (traverseproc) wrapper_traverse, /* tp_traverse */
    0,                         /* tp_clear */
    wrapper_richcompare,       /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};

PyTypeObject *hashwrap_type = &WrapperType;


/* Create a new wrapper object */
PyObject *
hashwrap_make_wrapper(PyObject *self, PyObject *arg)
{
    WrapperObject *obj;

    if (Py_TYPE(arg) == &WrapperType) {
        Py_INCREF(arg);
        return arg;
    }
    obj = (WrapperObject *) PyType_GenericNew(&WrapperType, NULL, NULL);
    if (obj == NULL)
        return NULL;
    Py_INCREF(arg);
    obj->wrapped = arg;
    obj->hash = -1;
    return (PyObject *) obj;
}
