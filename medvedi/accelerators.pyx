# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib $ORIGIN

import cython

from cpython cimport PyObject
from cython.operator cimport dereference as deref
from libc.stddef cimport wchar_t
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy
from numpy cimport (
    NPY_ARRAY_C_CONTIGUOUS,
    NPY_OBJECT,
    PyArray_DATA,
    PyArray_DESCR,
    PyArray_DescrFromType,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    PyArray_STRIDE,
    dtype as np_dtype,
    import_array,
    ndarray,
    npy_bool,
    npy_intp,
)

from medvedi.native.cpython cimport (
    Py_INCREF,
    Py_None,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
)
from medvedi.native.mi_heap_destroy_stl_allocator cimport (
    mi_heap_destroy_stl_allocator,
    mi_unordered_map,
    mi_unordered_set,
    pair,
)
from medvedi.native.numpy cimport (
    PyArray_Descr,
    PyArray_DescrNew,
    PyArray_NewFromDescr,
    PyArray_Type,
)
from medvedi.native.optional cimport optional
from medvedi.native.string_view cimport string_view

import numpy as np

import_array()


cdef extern from "memnrchr.h" nogil:
    char *memnrchr(const char *, char, size_t)
    wchar_t *wmemnrchr(const wchar_t *, wchar_t, size_t)


def unordered_unique(ndarray arr not None) -> np.ndarray:
    cdef:
        np_dtype dtype = <np_dtype>PyArray_DESCR(arr)
    assert PyArray_NDIM(arr) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr)
    if dtype.kind == b"S" or dtype.kind == b"U":
        return _unordered_unique_str(arr, dtype)
    elif dtype.kind == b"i" or dtype.kind == b"u":
        if dtype.itemsize == 8:
            return _unordered_unique_int[int64_t](arr, dtype, 0)
        elif dtype.itemsize == 4:
            return _unordered_unique_int[int64_t](arr, dtype, 4)
        else:
            raise AssertionError(f"dtype {dtype} is not supported")
    elif dtype.kind == b"O":
        return _unordered_unique_pystr(arr)
    else:
        raise AssertionError(f"dtype {dtype} is not supported")


@cython.cdivision(True)
cdef ndarray _unordered_unique_pystr(ndarray arr):
    cdef:
        PyObject **data_in = <PyObject **>PyArray_DATA(arr)
        PyObject **data_out
        PyObject *str_obj
        char *str_data
        unsigned int str_kind
        Py_ssize_t str_len
        int64_t i, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0) >> 3
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_map[string_view, int64_t]] hashtable
        pair[string_view, int64_t] it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            str_obj = data_in[i * stride]
            if str_obj == Py_None:
                continue
            str_data = <char *> PyUnicode_DATA(str_obj)
            str_len = PyUnicode_GET_LENGTH(str_obj)
            str_kind = PyUnicode_KIND(str_obj)
            deref(hashtable).try_emplace(string_view(str_data, str_len * str_kind), i)

    result = np.empty(deref(hashtable).size(), dtype=object)
    data_out = <PyObject **>PyArray_DATA(result)
    i = 0
    for it in deref(hashtable):
        str_obj = data_in[it.second]
        data_out[i] = str_obj
        Py_INCREF(str_obj)
        i += 1
    return result


@cython.cdivision(True)
cdef ndarray _unordered_unique_str(ndarray arr, np_dtype dtype):
    cdef:
        char *data = <char *>PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        optional[mi_heap_destroy_stl_allocator[string_view]] alloc
        optional[mi_unordered_set[string_view]] hashtable
        string_view it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            deref(hashtable).emplace(data + i * stride, itemsize)

    result = np.empty(deref(hashtable).size(), dtype=dtype)

    with nogil:
        data = <char *>PyArray_DATA(result)
        i = 0
        for it in deref(hashtable):
            memcpy(data + i * itemsize, it.data(), itemsize)
            i += 1
    return result


ctypedef fused varint:
    int64_t
    int32_t


@cython.cdivision(True)
cdef ndarray _unordered_unique_int(ndarray arr, np_dtype dtype, varint _):
    cdef:
        char *data = <char *> PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        optional[mi_heap_destroy_stl_allocator[varint]] alloc
        optional[mi_unordered_set[varint]] hashtable
        varint it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            deref(hashtable).emplace((<varint *>(data + i * stride))[0])

    result = np.empty(deref(hashtable).size(), dtype=dtype)

    with nogil:
        data = <char *> PyArray_DATA(result)
        i = 0
        for it in deref(hashtable):
            (<varint *>(data + i * itemsize))[0] = it
            i += 1
    return result


def in1d_str(
    ndarray trial not None,
    ndarray dictionary not None,
    bint verbatim = False,
    bint invert = False,
) -> np.ndarray:
    cdef:
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
        char kind = dtype_trial.kind
    assert PyArray_NDIM(trial) == 1
    assert PyArray_NDIM(dictionary) == 1
    assert kind == b"S" or kind == b"U"
    assert kind == dtype_dict.kind
    return _in1d_str(trial, dictionary, kind == b"S", verbatim, invert)


cdef ndarray _in1d_str(
    ndarray trial,
    ndarray dictionary,
    bint is_char,
    bint verbatim,
    bint invert,
):
    cdef:
        char *data_trial = <char *>PyArray_DATA(trial)
        char *data_dictionary = <char *> PyArray_DATA(dictionary)
        char *output
        char *s
        char *tail
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
        int64_t i, size, \
            itemsize = dtype_dict.itemsize, \
            length = PyArray_DIM(dictionary, 0), \
            stride = PyArray_STRIDE(dictionary, 0)
        optional[mi_heap_destroy_stl_allocator[string_view]] alloc
        optional[mi_unordered_set[string_view]] hashtable
        mi_unordered_set[string_view].iterator end
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length * 4)
        for i in range(length):
            s = data_dictionary + i * stride
            if verbatim:
                size = itemsize
            else:
                if is_char:
                    tail = <char *> memnrchr(s, 0, itemsize)
                else:
                    tail = <char *> wmemnrchr(<wchar_t *> s, 0, itemsize >> 2)
                if tail == NULL:
                    tail = s
                size = tail - s
            deref(hashtable).emplace(s, size)
        itemsize = dtype_trial.itemsize
        length = PyArray_DIM(trial, 0)
        stride = PyArray_STRIDE(trial, 0)

    result = np.empty(length, dtype=bool)

    with nogil:
        output = <char *>PyArray_DATA(result)
        end = deref(hashtable).end()
        for i in range(length):
            s = data_trial + i * stride
            if verbatim:
                size = itemsize
            else:
                if is_char:
                    tail = <char *> memnrchr(s, 0, itemsize)
                else:
                    tail = <char *> wmemnrchr(<wchar_t *> s, 0, itemsize >> 2)
                if tail == NULL:
                    tail = s
                size = tail - s
            if invert:
                output[i] = deref(hashtable).find(string_view(s, size)) == end
            else:
                output[i] = deref(hashtable).find(string_view(s, size)) != end
    return result


def is_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.zeros(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_null_vec(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_null_vec(
    const char *obj_arr,
    const long stride,
    const long size,
    npy_bool *out_arr,
) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None == (<const PyObject **> (obj_arr + i * stride))[0]


def is_not_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.ones(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_not_null(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_not_null(
    const char *obj_arr,
    const long stride,
    const long size,
    npy_bool *out_arr,
) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None != (<const PyObject **> (obj_arr + i * stride))[0]


def array_of_objects(int length, fill_value) -> ndarray:
    cdef:
        ndarray arr
        np_dtype objdtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_OBJECT))
        npy_intp nplength = length, i
        PyObject **data
        PyObject *obj = <PyObject *> fill_value

    arr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &nplength,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(<PyObject *> objdtype)
    data = <PyObject **> PyArray_DATA(arr)
    for i in range(nplength):
        data[i] = obj
    obj.ob_refcnt += nplength
    return arr
