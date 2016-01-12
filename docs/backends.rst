Backends
========

A backend in AMGCL is a class that defines matrix and vector types together
with several operations on them, such as creation, matrix-vector products,
elementwise vector operations, inner products etc. The
`\<amgcl/backend/interface.hpp>`_ file defines an interface that each backend
should extend.  The AMG hierarchy is moved to the specified backend upon
construction. The solution phase then uses types and operations defined in the
backend. This enables transparent acceleration of the solution phase with
OpenMP, OpenCL, CUDA, or any other technologies.

In order to use a backend, user must include its definition from the
corresponding file inside `amgcl/backend`_ folder. On the user side of things,
only the types of the right-hand side and the solution vectors should be
affected by the choice of AMGCL backend. Here is an example of using the
:cpp:class:`builtin <amgcl::backend::builtin>` backend. First, we need to
include the appropriate header:

.. code-block:: cpp

    #include <amgcl/backend/builtin.hpp>

Then, we need to construct the solver and apply it to the vector types
supported by the backend:

.. code-block:: cpp

    typedef amgcl::backend::builtin<double> Backend;

    typedef amgcl::make_solver<
        amgcl::amg<Backend, amgcl::coarsening::aggregation, amgcl::relaxation::spai0>,
        amgcl::solver::gmres<Backend>
        > Solver;

    Solver solve(A);

    std::vector<double> rhs, x; // Initialized elsewhere

    solve(rhs, x);

Now, if we want to switch to a different backend, for example, in order to
accelerate the solution phase with a powerful GPU, we just need to include
another backend header, and change the definitions of ``Backend``, ``rhs``,
and ``x``. Here is an example of what needs to be done to use the
:cpp:class:`VexCL <amgcl::backend::vexcl>` backend.

Include the correct header:

.. code-block:: cpp

    #include <amgcl/backend/builtin.hpp>

Change the definition of ``Backend``:

.. code-block:: cpp

    typedef amgcl::backend::vexcl<double> Backend;

Change the definition of the vectors:

.. code-block:: cpp

    vex::vector<double> rhs, x;

That's it! Well, almost. In case the backend requires some parameters, we also
need to provide those. In particular, the VexCL backend should know what
VexCL context to use:

.. code-block:: cpp

    // Initialize VexCL context on a single GPU:
    vex::Context ctx(vex::Filter::GPU && vex::Filter::Count(1));

    // Create backend parameters:
    Backend::params backend_prm;
    backend_prm.q = ctx;

    // Pass the parameters to the solver constructor:
    Solver solve(A, Solver::params(), backend_prm);

.. _`amgcl/backend`: https://github.com/ddemidov/amgcl/blob/master/amgcl/backend/
.. _`\<amgcl/backend/interface.hpp>`: https://github.com/ddemidov/amgcl/blob/master/amgcl/backend/interface.hpp

Builtin
-------

``#include`` `\<amgcl/backend/builtin.hpp>`_

.. doxygenstruct:: amgcl::backend::builtin
    :members:

VexCL
-----

``#include`` `\<amgcl/backend/vexcl.hpp>`_

.. doxygenstruct:: amgcl::backend::vexcl
    :members:

.. _\<amgcl/backend/builtin.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/backend/builtin.hpp
.. _\<amgcl/backend/vexcl.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/backend/vexcl.hpp
