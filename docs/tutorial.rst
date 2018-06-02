Getting started
===============

The easiest way to solve a problem with AMGCL is to use the
:cpp:class:`amgcl::make_solver` class. It has two
template parameters: the first one specifies a :doc:`preconditioner
<preconditioners>` to use, and the second chooses an :doc:`iterative solver
<solvers>`. The class constructor takes the system matrix in one of supported
:doc:`formats <adapters>` and parameters for the chosen algorithms and for the
:doc:`backend <backends>`.

Solving Poisson's equation
--------------------------

Let us consider a simple example of `Poisson's equation`_ in a unit square.
Here is how the problem may be solved with AMGCL. We will use BiCGStab solver
preconditioned with smoothed aggregation multigrid with SPAI(0) for relaxation
(smoothing). First, we include the necessary headers. Each of those brings in
the corresponding component of the method:

.. _Poisson's equation: https://en.wikipedia.org/wiki/Poisson%27s_equation

.. code-block:: cpp

    #include <amgcl/make_solver.hpp>
    #include <amgcl/solver/bicgstab.hpp>
    #include <amgcl/amg.hpp>
    #include <amgcl/coarsening/smoothed_aggregation.hpp>
    #include <amgcl/relaxation/spai0.hpp>
    #include <amgcl/adapter/crs_tuple.hpp>

Next, we assemble sparse matrix for the Poisson's equation on a uniform
1000x1000 grid. See :doc:`poisson` for the source code of the
:cpp:func:`poisson` function:

.. code-block:: cpp

    std::vector<int>    ptr, col;
    std::vector<double> val, rhs;
    int n = poisson(1000, ptr, col, val, rhs);

For this example, we select the :cpp:class:`builtin <amgcl::backend::builtin>`
backend with double precision numbers as value type:

.. code-block:: cpp

    typedef amgcl::backend::builtin<double> Backend;

Now we can construct the solver for our system matrix. We use the convenient
adapter for boost tuples here and just tie together the matrix size and its CRS
components:

.. code-block:: cpp

    typedef amgcl::make_solver<
        // Use AMG as preconditioner:
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        // And BiCGStab as iterative solver:
        amgcl::solver::bicgstab<Backend>
        > Solver;

    Solver solve( std::tie(n, ptr, col, val) );

Once the solver is constructed, we can apply it to the right-hand side to
obtain the solution. This may be repeated multiple times for different
right-hand sides. Here we start with a zero initial approximation. The solver
returns a boost tuple with number of iterations and norm of the achieved
residual:

.. code-block:: cpp

    std::vector<double> x(n, 0.0);
    int    iters;
    double error;
    std::tie(iters, error) = solve(rhs, x);

That's it! Vector ``x`` contains the solution of our problem now.

Input formats
-------------

We used STL vectors to store the matrix components in the above axample. This
may seem too restrictive if you want to use AMGCL with your own types.  But the
`crs_tuple` adapter will take anything that the Boost.Range_ library recognizes
as a random access range. For example, you can wrap raw pointers to your data
into a `boost::iterator_range`_:

.. _Boost.Range: http://www.boost.org/doc/libs/release/libs/range/
.. _`boost::iterator_range`: http://www.boost.org/doc/libs/release/libs/range/doc/html/range/reference/utilities/iterator_range.html

.. code-block:: cpp

    Solver solve( std::make_tuple(
        n,
        boost::make_iterator_range(ptr.data(), ptr.data() + ptr.size()),
        boost::make_iterator_range(col.data(), col.data() + col.size()),
        boost::make_iterator_range(val.data(), val.data() + val.size())
        ) );

Same applies to the right-hand side and the solution vectors. And if that is
still not general enough, you can provide your own adapter for your matrix
type. See :doc:`adapters` for further information on this.

Setting parameters
------------------

Any component in AMGCL defines its own parameters by declaring a ``param``
subtype. When a class wraps several subclasses, it includes parameters of its
children into its own ``param``. For example, parameters for the
:cpp:class:`amgcl::make_solver\<Precond, Solver>` are declared as

.. code-block:: cpp

    struct params {
        typename Precond::params precond;
        typename Solver::params solver;
    };

Knowing that, we can easily set the parameters for individual components. For
example, we can set the desired tolerance for the iterative solver in the above
example like this:

.. code-block:: cpp

    Solver::params prm;
    prm.solver.tol = 1e-3;
    Solver solve( std::tie(n, ptr, col, val), prm );

Parameters may also be initialized with a `boost::property_tree::ptree`_. This
is especially convenient when :doc:`runtime` is used, and the exact structure
of the parameters is not known at compile time:

.. code-block:: cpp

    boost::property_tree::ptree prm;
    prm.put("solver.tol", 1e-3);
    Solver solve( std::tie(n, ptr, col, val), prm );

.. _`boost::property_tree::ptree`: http://www.boost.org/doc/libs/release/doc/html/property_tree.html


The ``make_solver`` class
-------------------------

.. doxygenclass:: amgcl::make_solver
    :members:
