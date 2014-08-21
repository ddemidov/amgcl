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

/*
This header defines type conversions for use with Boost.Python.  If
not using numpy_boost with Boost.Python, do not include this header.

Each numpy_boost type that you wish to pass or return from a function
must be declared in the init function by calling
numpy_boost_python_register_type.

For example, if your module returns numpy_boost<int, 2> types, the
module's init function must call:

    numpy_boost_python_register_type<int, 2>;
*/

#ifndef __NUMPY_BOOST_PYTHON_HPP__
#define __NUMPY_BOOST_PYTHON_HPP__

#include "numpy_boost.hpp"
#include <boost/python.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

template<class T, int NDims>
struct numpy_boost_to_python
{
  static PyObject*
  convert(const numpy_boost<T, NDims>& o)
  {
    return boost::python::incref(o.py_ptr());
  }
};

template<class T, int NDims>
struct numpy_boost_from_python
{
  static void*
  convertible(PyObject* obj_ptr)
  {
    PyArrayObject* a;

    a = (PyArrayObject*)PyArray_FromObject(
        obj_ptr, ::detail::numpy_type_map<T>::typenum, NDims, NDims);
    if (a == NULL) {
      return 0;
    }
    Py_DECREF(a);
    return obj_ptr;
  }

  static void
  construct(
    PyObject* obj_ptr,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
      // Grab pointer to memory into which to construct the new numpy_boost object
      void* storage = (
        (boost::python::converter::rvalue_from_python_storage<numpy_boost<T, NDims> >*)
        data)->storage.bytes;

      // in-place construct the new numpy_boost object using the character data
      // extraced from the python object
      new (storage) numpy_boost<T, NDims>(obj_ptr);

      // Stash the memory chunk pointer for later use by boost.python
      data->convertible = storage;
   }

   numpy_boost_from_python()
   {
     boost::python::converter::registry::push_back(
       &convertible,
       &construct,
         boost::python::type_id<numpy_boost<T, NDims> >());
   }
};


template<class T, int NDims>
void
numpy_boost_python_register_type()
{
  boost::python::to_python_converter<
    numpy_boost<T, NDims>,
    numpy_boost_to_python<T, NDims> >();

  numpy_boost_from_python<T, NDims> converter;
}

#endif /* __NUMPY_BOOST_PYTHON_HPP__ */
