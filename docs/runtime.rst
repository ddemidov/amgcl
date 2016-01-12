Runtime interface
=================

The compile-time configuration of the solvers used in AMGCL is not always
convenient, especially if the solvers are used inside a software package or
another library. That is why AMGCL provides runtime interface, which allows to
postpone the configuration until, well, runtime. The classes inside
:cpp:any:`amgcl::runtime` namespace correspond to their compile-time
alternatives, but the only template parameter they have is the backend to use.

Since there is no way of knowing the parameter structure at compile time, the
runtime classes accept parameters only in form of
``boost::property_tree::ptree``. The actual components of the method are set
through the parameter tree as well. The runtime interface provides some
enumerations for this purpose. For example, to select smoothed aggregation for
coarsening, we could do this:

.. code-block:: cpp

    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.type", amgcl::runtime::coarsening::smoothed_aggregation);

The enumerations provide functions for converting to/from strings, so the
following would work as well:

.. code-block:: cpp

    prm.put("precond.coarsening.type", "smoothed_aggregation");

Here is an example of using a runtime-configurable solver:

.. code-block:: cpp

    #include <amgcl/backend/builtin.hpp>
    #include <amgcl/runtime.hpp>

    ...

    boost::property_tree::ptree prm;
    prm.put("precond.coarsening.type", amgcl::runtime::coarsening::smoothed_aggregation);
    prm.put("precond.relaxation.type", amgcl::runtime::relaxation::spai0);
    prm.put("solver.type",             amgcl::runtime::solver::gmres);

    amgcl::make_solver<
        amgcl::runtime::amg<Backend>,
        amgcl::runtime::iterative_solver<Backend>
        > solve(A, prm);


Classes
-------

AMG preconditioner
##################

.. doxygenclass:: amgcl::runtime::amg
    :members:

.. doxygenenum:: amgcl::runtime::coarsening::type

.. doxygenenum:: amgcl::runtime::relaxation::type

Iterative solver
################

.. doxygenclass:: amgcl::runtime::iterative_solver
    :members:

.. doxygenenum:: amgcl::runtime::solver::type
