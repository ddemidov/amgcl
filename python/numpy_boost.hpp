/*
Copyright (c) 2012, Michael Droettboom
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * The names of its contributors may not be used to endorse or
      promote products derived from this software without specific
      prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __NUMPY_BOOST_HPP__
#define __NUMPY_BOOST_HPP__

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <boost/multi_array.hpp>
#include <boost/cstdint.hpp>
#include <complex>
#include <algorithm>

/* numpy_type_map<T>

   Provides a mapping from C++ datatypes to Numpy type
   numbers. */
namespace detail {
  template<class T>
  class numpy_type_map {
  public:
    static const int typenum;
  };

  template<>
  const int numpy_type_map<float>::typenum = NPY_FLOAT;

  template<>
  const int numpy_type_map<std::complex<float> >::typenum = NPY_CFLOAT;

  template<>
  const int numpy_type_map<double>::typenum = NPY_DOUBLE;

  template<>
  const int numpy_type_map<std::complex<double> >::typenum = NPY_CDOUBLE;

  template<>
  const int numpy_type_map<long double>::typenum = NPY_LONGDOUBLE;

  template<>
  const int numpy_type_map<std::complex<long double> >::typenum = NPY_CLONGDOUBLE;

  template<>
  const int numpy_type_map<boost::int8_t>::typenum = NPY_INT8;

  template<>
  const int numpy_type_map<boost::uint8_t>::typenum = NPY_UINT8;

  template<>
  const int numpy_type_map<boost::int16_t>::typenum = NPY_INT16;

  template<>
  const int numpy_type_map<boost::uint16_t>::typenum = NPY_UINT16;

  template<>
  const int numpy_type_map<boost::int32_t>::typenum = NPY_INT32;

  template<>
  const int numpy_type_map<boost::uint32_t>::typenum = NPY_UINT32;

  template<>
  const int numpy_type_map<boost::int64_t>::typenum = NPY_INT64;

  template<>
  const int numpy_type_map<boost::uint64_t>::typenum = NPY_UINT64;
}

class python_exception : public std::exception {

};

/* An array that acts like a boost::multi_array, but is backed by the
   memory of a Numpy array.  Provides nice C++ interface to a Numpy
   array without any copying of the data.

   It may be constructed one of two ways:

     1) With an existing Numpy array.  The boost::multi_array will
        then have the data, dimensions and strides of the Numpy array.

     2) With a list of dimensions, in which case a new contiguous
        Numpy array will be created and the new boost::array will
        point to it.

 */
template<class T, int NDims>
class numpy_boost : public boost::multi_array_ref<T, NDims>
{
public:
  typedef numpy_boost<T, NDims>            self_type;
  typedef boost::multi_array_ref<T, NDims> super;
  typedef typename super::size_type        size_type;
  typedef T*                               TPtr;

private:
  PyArrayObject* array;

  void init_from_array(PyArrayObject* a) throw() {
    /* Upon calling init_from_array, a should already have been
       incref'd for ownership by this object. */

    /* Store a reference to the Numpy array so we can DECREF it in the
       destructor. */
    array = a;

    /* Point the boost::array at the Numpy array data.

       We don't need to worry about free'ing this pointer, because it
       will always point to memory allocated as part of the data of a
       Numpy array.  That memory is managed by Python reference
       counting. */
    super::base_ = (TPtr)PyArray_DATA(a);

    /* Set the storage order.

       It would seem like we would want to choose C or Fortran
       ordering here based on the flags in the Numpy array.  However,
       those flags are purely informational, the actually information
       about storage order is recorded in the strides. */
    super::storage_ = boost::c_storage_order();

    /* Copy the dimensions from the Numpy array to the boost::array. */
    boost::detail::multi_array::copy_n(PyArray_DIMS(a), NDims, super::extent_list_.begin());

    /* Copy the strides from the Numpy array to the boost::array.

       Numpy strides are in bytes.  boost::array strides are in
       elements, so we need to divide. */
    for (size_t i = 0; i < NDims; ++i) {
      super::stride_list_[i] = PyArray_STRIDE(a, i) / sizeof(T);
    }

    /* index_base_list_ stores the bases of the indices in each
       dimension.  Since we want C-style and Numpy-style zero-based
       indexing, just fill it with zeros. */
    std::fill_n(super::index_base_list_.begin(), NDims, 0);

    /* We don't want any additional offsets.  If they exist, Numpy has
       already handled that for us when calculating the data pointer
       and strides. */
    super::origin_offset_ = 0;
    super::directional_offset_ = 0;

    /* Calculate the number of elements.  This has nothing to do with
       memory layout. */
    super::num_elements_ = std::accumulate(super::extent_list_.begin(),
                                           super::extent_list_.end(),
                                           size_type(1),
                                           std::multiplies<size_type>());
  }

public:
  /* Construct from an existing Numpy array */
  numpy_boost(PyObject* obj) :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {
    PyArrayObject* a;

    a = (PyArrayObject*)PyArray_FromObject(
        obj, ::detail::numpy_type_map<T>::typenum, NDims, NDims);
    if (a == NULL) {
      throw boost::python::error_already_set();
    }

    init_from_array(a);
  }

  /* Copy constructor */
  numpy_boost(const self_type &other) throw() :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {
    Py_INCREF(other.array);
    init_from_array(other.array);
  }

  /* Construct a new array based on the given dimensions */
  template<class ExtentsList>
  explicit numpy_boost(const ExtentsList& extents) :
    super(NULL, std::vector<typename super::index>(NDims, 0)),
    array(NULL)
  {
    npy_intp shape[NDims];
    PyArrayObject* a;

    boost::detail::multi_array::copy_n(extents, NDims, shape);

    a = (PyArrayObject*)PyArray_SimpleNew(
        NDims, shape, ::detail::numpy_type_map<T>::typenum);
    if (a == NULL) {
      throw boost::python::error_already_set();
    }

    init_from_array(a);
  }

  /* Destructor */
  ~numpy_boost() {
    /* Dereference the numpy array. */
    Py_XDECREF(array);
  }

  /* Assignment operator */
  void operator=(const self_type &other) throw() {
    Py_INCREF(other.array);
    Py_DECREF(array);
    init_from_array(other.array);
  }

  /* Return the underlying Numpy array object.  [Borrowed
     reference] */
  PyObject*
  py_ptr() const throw() {
    return (PyObject*)array;
  }
};

#endif
