#ifndef NUMBA_HASHWRAP_H_
#define NUMBA_HASHWRAP_H_

#include "_pymodule.h"

#ifdef __cplusplus
extern "C" {
#endif

extern PyTypeObject *hashwrap_type;

extern PyObject *
hashwrap_make_wrapper(PyObject *self, PyObject *arg);

#ifdef __cplusplus
}
#endif

#endif  /* NUMBA_HASHWRAP_H_ */
