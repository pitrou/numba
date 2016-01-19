/*
Implement parallel vectorize workqueue.

This keeps a set of worker threads running all the time.
They wait and spin on a task queue for jobs.

**WARNING**
This module is not thread-safe.  Adding task to queue is not protected from
race condition.
*/

#ifdef _MSC_VER
    /* Windows */
    #include <windows.h>
    #include <process.h>
    #define NUMBA_WINTHREAD
#else
    /* PThread */
    #include <pthread.h>
    #include <unistd.h>
    #include <semaphore.h>
    #include <fcntl.h>
    #define NUMBA_PTHREAD
#endif

#include <string.h>
#include <stdio.h>
#include "workqueue.h"
#include "../_pymodule.h"

static cas_function_t *cas = NULL;

static void
cas_wait(volatile int *ptr, const int old, const int repl) {
    int out = repl;
    int timeout = 1;   /* starting from 1us nap */
    static const int MAX_WAIT_TIME = 20 * 1000; /* max wait is 20ms */

    while (1) {
        if (cas) { /* protect against CAS function being released by LLVM during
                      interpreter teardown. */
            out = cas(ptr, old, repl);
            if (out == old) return;
        }

        take_a_nap(timeout);

        /* Exponentially increase the wait time until the max has reached*/
        timeout <<= 1;
        if (timeout >= MAX_WAIT_TIME) {
            timeout = MAX_WAIT_TIME;
        }
    }
}

/* As the thread-pool isn't inherited by children,
   free the task-queue, too. */
static void reset_after_fork(void);

/* PThread */
#ifdef NUMBA_PTHREAD

static thread_pointer
numba_new_thread(void *worker, void *arg)
{
    int status;
    pthread_attr_t attr;
    pthread_t th;

    pthread_atfork(0, 0, reset_after_fork);

    /* Create detached threads */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    status = pthread_create(&th, &attr, worker, arg);

    if (status != 0){
        return NULL;
    }

    pthread_attr_destroy(&attr);
    return (thread_pointer)th;
}

static void
take_a_nap(int usec) {
    usleep(usec);
}


#endif

/* Win Thread */
#ifdef NUMBA_WINTHREAD

/* Adapted from Python/thread_nt.h */
typedef struct {
    void (*func)(void*);
    void *arg;
} callobj;

static unsigned __stdcall
bootstrap(void *call)
{
    callobj *obj = (callobj*)call;
    void (*func)(void*) = obj->func;
    void *arg = obj->arg;
    HeapFree(GetProcessHeap(), 0, obj);
    func(arg);
    _endthreadex(0);
    return 0;
}

static thread_pointer
numba_new_thread(void *worker, void *arg)
{
    uintptr_t handle;
    unsigned threadID;
    callobj *obj;

    if (sizeof(handle) > sizeof(void*))
        return 0;

    obj = (callobj*)HeapAlloc(GetProcessHeap(), 0, sizeof(*obj));
    if (!obj)
        return NULL;

    obj->func = worker;
    obj->arg = arg;

    handle = _beginthreadex(NULL, 0, bootstrap, obj, 0, &threadID);
    if (handle == -1)
        return 0;
    return (thread_pointer)handle;
}

static void
take_a_nap(int usec) {
    /* Note that Sleep(0) will relinquish the current time slice, allowing
       other threads to run. */
    Sleep(usec / 1000);
}


#endif

typedef struct Task{
    void (*func)(void *args, void *dims, void *steps, void *data);
    void *args, *dims, *steps, *data;
} Task;

typedef struct {
    volatile int lock;
    Task task;
} Queue;


static Queue *queues = NULL;
static int queue_count;
static int queue_pivot = 0;

static void
set_cas(void *ptr) {
    cas = ptr;
}

static void
add_task(void *fn, void *args, void *dims, void *steps, void *data) {
    void (*func)(void *args, void *dims, void *steps, void *data) = fn;

    Queue *queue = &queues[queue_pivot];

    Task *task = &queue->task;
    task->func = func;
    task->args = args;
    task->dims = dims;
    task->steps = steps;
    task->data = data;

    /* Move pivot */
    if ( ++queue_pivot == queue_count ) {
        queue_pivot = 0;
    }
}

static
void thread_worker(void *arg) {
    Queue *queue = (Queue*)arg;
    Task *task;

    while (1) {
        cas_wait(&queue->lock, READY, RUNNING);

        task = &queue->task;
        task->func(task->args, task->dims, task->steps, task->data);

        cas_wait(&queue->lock, RUNNING, DONE);
    }
}

static void launch_threads(int count) {
    if (!queues) {
        /* If queues are not yet allocated,
           create them, one for each thread. */
       int i;
       size_t sz = sizeof(Queue) * count;

       queues = malloc(sz);     /* this memory will leak */
       memset(queues, 0, sz);
       queue_count = count;

       for (i = 0; i < count; ++i) {
            numba_new_thread(thread_worker, &queues[i]);
       }
    }
}

static void synchronize(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, DONE, IDLE);
    }
}

static void ready(void) {
    int i;
    for (i = 0; i < queue_count; ++i) {
        cas_wait(&queues[i].lock, IDLE, READY);
    }
}

static void reset_after_fork(void)
{
    free(queues);
    queues = NULL;
}

/*
 * Benchmark functions for synchronization primitives
 */

static volatile int dummy = 0;

static PyObject *
bench_cas_wait(PyObject *self, PyObject *args)
{
    int i, n;

    if (!PyArg_ParseTuple(args, "i:bench_cas_wait", &n))
        return NULL;

    for (i = 0; i < n; i++) {
        cas_wait(&dummy, dummy, dummy ^ 1);
    }
    Py_RETURN_NONE;
}

#ifdef NUMBA_PTHREAD

static sem_t dummy_sem_val;
static sem_t *dummy_sem;
static pthread_mutex_t dummy_mutex;

static PyObject *
bench_posix_semaphore(PyObject *self, PyObject *args)
{
    int i, n;
    int err = 0;

    if (!PyArg_ParseTuple(args, "i:bench_semaphore", &n))
        return NULL;

#ifdef __APPLE__
    dummy_sem = sem_open("/numba_bench_sem", O_CREAT, 0600, 10);
    err = (dummy_sem == SEM_FAILED);
#else
    dummy_sem = &dummy_sem_val;
    err = sem_init(dummy_sem, 0, 10);
#endif

    if (!err) {
        for (i = 0; i < n; i++) {
            if ((err |= sem_wait(dummy_sem)))
                break;
            if ((err |= sem_post(dummy_sem)))
                break;
        }
#ifdef __APPLE__
        err |= sem_close(dummy_sem);
        err |= sem_unlink("/numba_bench_sem");
#else
        err |= sem_destroy(dummy_sem);
#endif
    }

    if (err) {
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    else {
        Py_RETURN_NONE;
    }
}

static PyObject *
bench_posix_mutex(PyObject *self, PyObject *args)
{
    int i, n;
    int err = 0;

    if (!PyArg_ParseTuple(args, "i:bench_mutex", &n))
        return NULL;

    err = pthread_mutex_init(&dummy_mutex, NULL);
    if (!err) {
        for (i = 0; i < n; i++) {
            if ((err |= pthread_mutex_lock(&dummy_mutex)))
                break;
            if ((err |= pthread_mutex_unlock(&dummy_mutex)))
                break;
        }
        err |= pthread_mutex_destroy(&dummy_mutex);
    }

    if (err) {
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    else {
        Py_RETURN_NONE;
    }
}

#ifdef __linux__
static pthread_spinlock_t dummy_spinlock;

static PyObject *
bench_posix_spinlock(PyObject *self, PyObject *args)
{
    int i, n;
    int err = 0;

    if (!PyArg_ParseTuple(args, "i:bench_spinlock", &n))
        return NULL;

    err = pthread_spin_init(&dummy_spinlock, PTHREAD_PROCESS_PRIVATE);
    if (!err) {
        for (i = 0; i < n; i++) {
            if ((err |= pthread_spin_lock(&dummy_spinlock)))
                break;
            if ((err |= pthread_spin_unlock(&dummy_spinlock)))
                break;
        }
        err |= pthread_spin_destroy(&dummy_spinlock);
    }

    if (err) {
        return PyErr_SetFromErrno(PyExc_OSError);
    }
    else {
        Py_RETURN_NONE;
    }
}
#endif

#endif

#ifdef NUMBA_WINTHREAD

static HANDLE dummy_sem;
static CRITICAL_SECTION dummy_cs;
static SRWLOCK dummy_srw_lock;

static PyObject *
bench_win_semaphore(PyObject *self, PyObject *args)
{
    int i, n;
    int success = 0;

    if (!PyArg_ParseTuple(args, "i:bench_semaphore", &n))
        return NULL;

    dummy_sem = CreateSemaphore(NULL, 10, 20, NULL);
    if (dummy_sem) {
        for (i = 0; i < n; i++) {
            if (WaitForSingleObject(dummy_sem, INFINITE) == WAIT_FAILED)
                break;
            if (!ReleaseSemaphore(dummy_sem, 1, NULL))
                break;
        }
        success = (i == n);
        success &= CloseHandle(dummy_sem);
    }

    if (!success) {
        return PyErr_SetFromWindowsErr(GetLastError());
    }
    else {
        Py_RETURN_NONE;
    }
}

static PyObject *
bench_win_critical_section(PyObject *self, PyObject *args)
{
    int i, n;

    if (!PyArg_ParseTuple(args, "i:bench_critical_section", &n))
        return NULL;

    /* NOTE: a critical section can be used in tandem with a Windows
       condition variable, for e.g. queue management. */
    InitializeCriticalSection(&dummy_cs);
    for (i = 0; i < n; i++) {
        EnterCriticalSection(&dummy_cs);
        dummy = i;
        LeaveCriticalSection(&dummy_cs);
    }
    DeleteCriticalSection(&dummy_cs);

    Py_RETURN_NONE;
}

static PyObject *
bench_win_srw_lock(PyObject *self, PyObject *args)
{
    int i, n;

    if (!PyArg_ParseTuple(args, "i:bench_srw_lock", &n))
        return NULL;

    InitializeSRWLock(&dummy_srw_lock);
    for (i = 0; i < n; i++) {
        AcquireSRWLockExclusive(&dummy_srw_lock);
        dummy = i;
        ReleaseSRWLockExclusive(&dummy_srw_lock);
    }

    Py_RETURN_NONE;
}

#endif


static PyMethodDef methods[] = {
    { "bench_cas_wait", (PyCFunction) bench_cas_wait, METH_VARARGS, NULL },
#ifdef NUMBA_PTHREAD
    { "bench_mutex", (PyCFunction) bench_posix_mutex, METH_VARARGS, NULL },
    { "bench_semaphore", (PyCFunction) bench_posix_semaphore, METH_VARARGS, NULL },
#ifdef __linux__
    { "bench_spinlock", (PyCFunction) bench_posix_spinlock, METH_VARARGS, NULL },
#endif
#endif
#ifdef NUMBA_WINTHREAD
    { "bench_critical_section", (PyCFunction) bench_win_critical_section, METH_VARARGS, NULL },
    { "bench_semaphore", (PyCFunction) bench_win_semaphore, METH_VARARGS, NULL },
    { "bench_srw_lock", (PyCFunction) bench_win_srw_lock, METH_VARARGS, NULL },
#endif
    { NULL },
};


MOD_INIT(workqueue) {
    PyObject *m;
    MOD_DEF(m, "workqueue", "No docs", methods)
    if (m == NULL)
        return MOD_ERROR_VAL;

    PyObject_SetAttrString(m, "set_cas",
                           PyLong_FromVoidPtr(&set_cas));
    PyObject_SetAttrString(m, "launch_threads",
                           PyLong_FromVoidPtr(&launch_threads));
    PyObject_SetAttrString(m, "synchronize",
                           PyLong_FromVoidPtr(&synchronize));
    PyObject_SetAttrString(m, "ready",
                           PyLong_FromVoidPtr(&ready));
    PyObject_SetAttrString(m, "add_task",
                           PyLong_FromVoidPtr(&add_task));

    return MOD_SUCCESS_VAL(m);
}
