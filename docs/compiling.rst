Compilation issues
==================

AMGCL is a header-only library, so one does not need to compile it in order to
use the library. However, there are some dependencies coming with the library:

1. The runtime interface of AMGCL depends on the header-only
   `Boost.property_tree`_ library that allows the solvers and preconditioners
   to accept dynamically formed parameters. When the runtime interface is not
   used, it is possible to get rid of the `Boost.property_tree`_ dependency by
   defining the preprocessor macro ``AMGCL_NO_BOOST``.
2. AMGCL uses OpenMP_ during the setup of the provided solvers and
   preconditioners, and also for the :cpp:class:`amgcl::backend::builtin`
   backend. OpenMP is supported by most, if not all, of the relatively modern
   C++ compilers, so that should not be a problem. One just has to remember to
   enable the OpenMP support during the compilation of the project that uses
   AMGCL.

   With OpenMP, the reductions in the :cpp:class:`amgcl::backend::builtin`
   backend (``inner_product`` and the sparse matrix product used to build the
   coarse levels) may give results that differ in the last bits for different
   numbers of threads. Define the preprocessor macro
   ``AMGCL_DETERMINISTIC_REDUCTIONS`` to make them independent of the thread
   count: ``inner_product`` then sums in fixed-size blocks combined in order,
   and the matrix product always uses ``spgemm_saad`` instead of switching to
   ``spgemm_rmerge`` above 16 threads, which may be slower on machines with
   many cores. The macro is not defined by default.
3. Each of the AMGCL backends brings its own set of dependencies. For example,
   the :cpp:class:`amgcl::backend::vexcl` backend depends on the header-only
   VexCL_ library, which in turn depends on some Boost libraries and either on
   CUDA or OpenCL support. The :cpp:class:`amgcl::backend::cuda` backend
   depends on the CUDA support and the CUSPARSE_ and Thrust_ libraries.

.. _Boost.property_tree: https://www.boost.org/doc/libs/release/libs/property_tree
.. _OpenMP:  https://www.openmp.org/
.. _CUSPARSE: https://docs.nvidia.com/cuda/cusparse/index.html
.. _Thrust: https://docs.nvidia.com/cuda/thrust/index.html
.. _VexCL: https://github.com/ddemidov/vexcl

If your project already uses CMake as the build system, then using AMGCL
should be easy. Here is a concise example that shows how to compile a project
using AMGCL with the builtin backend:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.1)
    project(example)

    find_package(amgcl)

    add_executable(example example.cpp)
    target_link_libraries(example amgcl::amgcl)

And here is an example of adding the support for the VexCL_ backend:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.1)
    project(example)

    find_package(amgcl)
    find_package(VexCL)

    add_executable(example example.cpp)
    target_link_libraries(example amgcl::amgcl VexCL::OpenCL)

``find_package(amgcl)`` may be used when the cmake support for AMGCL was
installed either system-wide, or in the current user home directory. If that is
not the case, one can simply copy the amgcl folder into a subdirectory of the
main project and replace the ``find_package(amgcl)`` line with
``add_subdirectory(amgcl)``.

Finally, in order to compile the AMGCL tests and examples, the following script
may be used:

.. code-block:: shell

    git clone https://github.com/ddemidov/amgcl
    cd ./amgcl
    cmake -Bbuild -DAMGCL_BUILD_TESTS=ON -DAMGCL_BUILD_EXAMPLES=ON .
    cmake --build build

After this, the compiled tests and examples may be found in the ``build`` folder.
