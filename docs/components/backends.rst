Backends
========

A backend in AMGCL is a class that binds datatypes like matrix and vector with
parallel primitives like matrix-vector product, linear combination of vectors,
or inner product computation. The backend system is implemented using free
functions and partial template specializations, which allows to decouple the
implementation of common parallel primitives from the specific datatypes used
in the supported backends. This makes it possible to adopt third-party or
user-defined datatypes for use within AMGCL without any modification of the
core library code. 

Algorithm setup in AMGCL is performed using internal data structures. As soon
as the setup is completed, the necessary objects (mostly matrices and vectors)
are transferred to the backend datatypes. Solution phase of the algorithms is
expressed in terms of the predefined parallel primitives which makes it
possible to switch parallelization technology (such as OpenMP_, CUDA_, or
OpenCL_) simply by changing the backend template parameter of the algorithm.
For example, the norm of the residual :math:`\epsilon = ||f - Ax||` in AMGCL is
computed with ``amgcl::backend::residual()`` and
``amgcl::backend::inner_product()`` primitives:

.. code-block:: cpp

    backend::residual(f, A, x, r);
    auto e = sqrt(backend::inner_product(r, r));

.. _OpenMP: https://www.openmp.org/
.. _OpenCL: https://www.khronos.org/opencl/
.. _CUDA: https://developer.nvidia.com/cuda-toolkit

The backends currenly supported by AMGCL are listed below.

OpenMP (builtin) backend
------------------------


.. cpp:class:: template <class ValueType> \
                amgcl::backend::builtin

    Include ``<amgcl/backend/builtin.hpp>``.

    This is the bultin backend that does not have any external dependencies and
    uses OpenMP_-parallelization for its primitives. As with any backend in
    AMGCL, it is defined with a :doc:`value type <valuetypes>` template
    parameter, which specifies the type of the system matrix elements. The
    backend is also used internally by AMGCL during construction phase, and so
    moving the constructed datatypes (matrices and vectors) to the backend has
    no overhead.  The backend has no parameters (the ``params`` subtype is an
    empty struct).

    .. cpp:class:: params

NVIDIA CUDA backend
-------------------

.. cpp:class:: template <class ValueType> \
                amgcl::backend::cuda

   Include ``<amgcl/backend/cuda.hpp>``.

   The backend uses the NVIDIA CUDA_ technology for the parallelization of its
   primitives. It depends on the Thrust_ and cuSPARSE_ libraries. The code
   using the backend has to be compiled with NVIDIA's nvcc_ compiler. The user
   needs to initialize the cuSPARSE_ library with a call to the
   `cusparseCreate()` function and pass the returned handle to AMGCL in the
   backend parameters.

   .. cpp:class:: params

      .. cpp:member:: cusparseHandle_t cusparse_handle         

         cuSPARSE_ handle created with the `cusparseCreate()`_ function.

.. _Thrust: https://docs.nvidia.com/cuda/thrust/index.html
.. _cuSPARSE: https://docs.nvidia.com/cuda/cusparse/index.html
.. _nvcc: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
.. _cusparseCreate(): https://docs.nvidia.com/cuda/cusparse/index.html#cusparseCreate

VexCL backend
-------------

.. cpp:class:: template <class ValueType> \
                amgcl::backend::vexcl

   Include ``<amgcl/backend/vexcl.hpp>``.

   The backend uses the VexCL_ library for the implementation of its
   primitives. VexCL_ provides OpenMP, OpenCL, or CUDA parallelization,
   selected at compile time with a preprocessor definition. The user has to
   initialize the VexCL_ context and pass it to AMGCL via the backend
   parameter.

   .. cpp:class:: params

      .. cpp:member:: std::vector<vex::backend::command_queue> q

         VexCL command queues identifying the compute devices in the compute
         context.

      .. cpp:member:: bool fast_matrix_setup = true

         Transform the CSR matrices into the internal VexCL format on the
         GPU. This is faster, but temporarily requires more memory on the GPU.

.. _VexCL: https://github.com/ddemidov/vexcl

ViennaCL backend
----------------

.. cpp:class:: template <class Matrix> \
                amgcl::backend::viennacl

   Include ``<amgcl/backend/viennacl.hpp>``.

   The backend uses the ViennaCL_ library for the implementation of its
   primitives. ViennaCL_ is a free open-source linear algebra library for
   computations on many-core architectures (GPUs, MIC) and multi-core CPUs. The
   library is written in C++ and supports CUDA, OpenCL, and OpenMP (including
   switches at runtime). The template parameter for the backend specifies
   ViennaCL_ matrix class to use.  Possible choices are
   ``viannacl::compressed_matrix<T>``, ``viennacl::ell_matrix<T>``, and
   ``viennacl::hyb_matrix<T>``. The backend has no runtime parameters.

   .. cpp:class:: params

.. _ViennaCL: http://viennacl.sourceforge.net/

Eigen backend
-------------

.. cpp:class:: template <class ValueType> \
                amgcl::backend::eigen

   Include ``<amgcl/backend/eigen.hpp>``.

   The backend uses Eigen_ library datatypes for implementation of its
   primitives. It could be useful in case the user already works with the
   Eigen_ library, for example, to assemble the linear system to be solved with
   AMGCL. AMGCL also provides an Eigen :doc:`matrix adapter <adapters>`, so
   that Eigen matrices may be transparently used with AMGCL solvers.

   .. cpp:class:: params

.. _Eigen: http://eigen.tuxfamily.org

Blaze backend
-------------

.. cpp:class:: template <class ValueType> \
                amgcl::backend::blaze

   Include ``<amgcl/backend/blaze.hpp>``.

   The backend uses Blaze_ library datatypes for implementation of its
   primitives. It could be useful in case the user already works with the
   Blaze_ library, for example, to assemble the linear system to be solved with
   AMGCL.

   .. cpp:class:: params

.. _Blaze: https://bitbucket.org/blaze-lib/blaze
