# AMGCL

[<img src="https://travis-ci.org/ddemidov/amgcl.svg?branch=master" alt="Build Status" />](https://travis-ci.org/ddemidov/amgcl)
[<img src="https://coveralls.io/repos/ddemidov/amgcl/badge.png?branch=master" alt="Coverage Status" />](https://coveralls.io/r/ddemidov/amgcl)


AMGCL is a C++ header only library for constructing an algebraic [multigrid][]
(AMG) hierarchy.  AMG is one the most effective methods for solution of large
sparse unstructured systems of equations, arising, for example, from
discretization of PDEs on unstructured grids [5,6]. The method can be used as a
black-box solver for various computational problems, since it does not require
any information about the underlying geometry. AMG is often used not as a
standalone solver but as a preconditioner within an iterative solver (e.g.
Conjugate Gradients,  BiCGStab, or GMRES).

AMGCL builds the AMG hierarchy on a CPU and then transfers it to one of the
provided backends.  This allows for transparent acceleration of the solution
phase with help of OpenCL, CUDA, or OpenMP technologies. Users may provide
their own backends which enables tight integration between AMGCL and the user
code.

The library source code is available under MIT license at
https://github.com/ddemidov/amgcl.  Doxygen-generated documentation is located
at http://ddemidov.github.io/amgcl.

### Table of contents

* [Getting started](#getting-started)
    * [Backends](#backends)
    * [Matrix adapters](#matrix-adapters)
    * [Coarsening strategies](#coarsening)
    * [Relaxation schemes](#relaxation)
    * [Solvers](#solvers)
* [Extending AMGCL](#extending)
    * [Adding backends](#adding-backends)
    * [Adding coarseners](#adding-coarseners)
    * [Adding smoothers](#adding-smoothers)
* [References](#references)
* [Projects using AMGCL](#projects)

## <a name="getting-started"></a>Getting started

The main class of the library is `amgcl::amg<Backend, Coarsening, Relaxation>`
which is defined in [amgcl/amgcl.hpp][].  It has three template parameters that
allow the user to select the exact components of the method:

1. **Backend** to transfer the constructed hierarchy to,
2. **Coarsening** strategy for hierarchy construction, and
3. **Relaxation** scheme (smoother to use during the solution phase).

See below for the list of available choices for each of the template
parameters. Instance of the class builds the AMG hierarchy for the given system
matrix and is intended to be used as a preconditioner. Here is the complete
example that solves a linear system of equations with BiCGstab method on a
multicore CPU, uses classic Ruge-Stuben algorithm for coarsening and damped
Jacobi smoother for relaxation:

~~~{.cpp}
#include <iostream>

// Definition of the main class
#include <amgcl/amgcl.hpp>

// Builtin backend: works on CPU, uses OpenMP for parallelization
#include <amgcl/backend/builtin.hpp>

// Allows to specify system matrix as a tuple of sizes and ranges (as in Boost.Range).
#include <amgcl/adapter/crs_tuple.hpp>

// Classic Ruge-Stuben coarsening algorithm
#include <amgcl/coarsening/ruge_stuben.hpp>

// Damped Jacobi relaxation
#include <amgcl/relaxation/damped_jacobi.hpp>

// BiCGStab iterative solver
#include <amgcl/solver/bicgstab.hpp>

int main() {
    // Sparse matrix in CRS format (the assembling is omitted for clarity):
    int n;                   // Matrix size
    std::vector<double> val; // Values of nonzero entries.
    std::vector<int>    col; // Column numbers of nonzero entries.
    std::vector<int>    ptr; // Points to the start of each row in the above arrays.
    std::vector<double> rhs; // Right-hand side of the system of equations.

    // Define the AMG type:
    typedef amgcl::amg<
        amgcl::backend::builtin<double>,
        amgcl::coarsening::ruge_stuben,
        amgcl::relaxation::damped_jacobi
        > AMG;

    // Construct the AMG hierarchy.
    // Note that this step only depends on the matrix. Hence, the constructed
    // instance may be reused for several right-hand sides.
    // The matrix is specified as a tuple of sizes and ranges.
    AMG amg( boost::tie(n, ptr, col, val) );

    // Output some information about the constructed hierarchy:
    std::cout << amg << std::endl;

    // Use BiCGStab as an iterative solver:
    typedef amgcl::solver::bicgstab<
        amgcl::backend::builtin<double>
        > Solver;

    // Construct the iterative solver. It needs size of the system to
    // preallocate the required temporary structures:
    Solver solve(n);

    // The solution vector. Use zero as initial approximation.
    std::vector<double> x(n, 0);

    // Solve the system. Returns number of iterations made and the achieved residual.
    int    iters;
    double resid;
    boost::tie(iters, resid) = solve(amg, rhs, x);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;
}
~~~

There is a convenience class
`amgcl::make_solver<Backend, Coarsening, Relaxation, Solver>` which wraps both
an AMG preconditioner and an iterative solver. By using the class the above
example could be made a bit shorter:

~~~{.cpp}
// Construct the AMG hierarchy and create the iterative solver.
amgcl::make_solver<
    amgcl::backend::builtin<double>,
    amgcl::coarsening::ruge_stuben,
    amgcl::relaxation::damped_jacobi,
    amgcl::solver::bicgstab
    > solve( boost::tie(n, ptr, col, val) );

// ...

// Solve the linear system.
boost::tie(iters, resid) = solve(rhs, x);
~~~

### <a name="backends"></a>Backends

A backend in AMGCL is a class that defines matrix and vector types together
with several operations on them, such as creation, matrix-vector products,
vector sums, inner products etc. See [Adding backends](#adding-backends) for
more detailed description of a backend implementation. The AMG hierarchy is
moved to the specified backend upon construction. The solution phase then uses
types and operations defined in the backend. This enables transparent
acceleration of the solution phase with OpenMP, OpenCL, CUDA, or any other
technologies.

Here is the list of backends currently implemented in the library:

- `amgcl::backend::builtin<value_type>` (defined in
  [amgcl/backend/builtin.hpp][]).  The `builtin` backend does not have any
  external dependencies (except for the [Boost][] libraries), and uses OpenMP
  for parallelization. It uses CRS format for storing matrices. Vectors are
  instances of `std::vector<value_type>`.  There is no usual overhead for
  moving the constructed hierarchy to the builtin backend, since the backend
  is always used internally during construction.
- `amgcl::backend::block_crs<value_type>` ([amgcl/backend/block_crs.hpp][]).
  The `block_crs` backend is similar to the `builtin` backend. The only
  difference is that it uses Block CRS format for storing matrices. The format
  is well suited for matrices that have block structure. This is usualy the
  case when a system of coupled PDEs is discretized.
- `amgcl::backend::eigen<value_type>` ([amgcl/backend/eigen.hpp][]).
  The `eigen` backend uses types and operations from [Eigen][] library. Eigen
  is a C++ template library for linear algebra. It works on a CPU and as of
  this writing is single threaded.
- `amgcl::backend::blaze<value_type>` ([amgcl/backend/blaze.hpp][]).
  The `blaze` backend uses types and operations from [Blaze][] library. Blaze
  is an open-source, high-performance C++ math library for dense and sparse
  arithmetic.
- `amgcl::backend::vexcl<value_type>` ([amgcl/backend/vexcl.hpp][]).
  The `vexcl` backend uses [VexCL][] library for accelerating the solution
  phase. VexCL is a C++ vector expression template library for OpenCL/CUDA.
  VexCL is able to utilize several compute devices at once.
- `amgcl::backend::viennacl<matrix_type>` ([amgcl/backend/viennacl.hpp][]).
  The `viennacl` backend is built on top of [ViennaCL][] library which is a
  free open-source linear algebra library for computations on many-core
  architectures (GPUs, MIC) and multi-core CPUs.
- `amgcl::backend::cuda<value_type>` ([amgcl/backend/cuda.hpp][]).
  Uses CUDA libraries CUSPARSE and Thrust for matrix and vector operations.

### <a name="matrix-adapters"></a>Matrix adapters

AMGCL provides several adapters for common sparse matrix formats. An adapter
allows to construct an AMG hierarchy directly from the matrix format it adapts.

- `amgcl::adapter::crs_tuple<value_type>` ([amgcl/adapter/crs_tuple.hpp][]).
  The `crs_tuple` adapter facilitates construction of `amgcl::amg<>` instances from user matrices
  strored in CRS format. A `boost::tuple` of matrix size and ranges of
  nonzero values, columns, and row pointers may be used. The example below
  constructs an AMG preconditioner from a matrix stored in raw pointers:
~~~{.cpp}
int n;       // Matrix size.
double *val; // Values.
int    *col; // Column numbers.
int    *ptr  // Row pointers into the above arrays.

AMG amg( boost::make_tuple(
            n,
            boost::make_iterator_range(ptr, ptr + n + 1),
            boost::make_iterator_range(col, col + ptr[n]),
            boost::make_iterator_range(val, val + ptr[n])
            )
        );
~~~
- `amgcl::adapter::crs_builder<RowBuilder>`
  ([amgcl/adapter/crs_builder.hpp][]). The adapter
    backend does not need fully constructed matrix in CRS format (which would
    be copied into AMG anyway), but builds matrix rows as needed.
    This results in reduced memory requirements. There is a convenience
    function `make_matrix(const RowBuilder&)` that returns a
    `crs_builder<RowBuilder>` instance. Here is an example of 2D poisson
    problem construction:
~~~{.cpp}
struct poisson_2d {
    typedef double val_type;
    typedef long   col_type;

    poisson_2d(size_t n) : n(n), h2i((n - 1) * (n - 1)) {}

    // Number of rows in the constructed matrix:
    size_t rows() const { return n * n; }

    // Estimated number of nonzeros in the problem:
    size_t nonzeros() const { return 5 * rows(); }

    // Fills column numbers and values of nonzero elements in the given matrix row.
    void operator()(size_t row,
            std::vector<col_type> &col,
            std::vector<val_type> &val
            ) const
    {
        size_t i = row % n;
        size_t j = row / n;

        if (j > 0) {
            col.push_back(row - n);
            val.push_back(-h2i);
        }

        if (i > 0) {
            col.push_back(row - 1);
            val.push_back(-h2i);
        }

        col.push_back(row);
        val.push_back(4 * h2i);

        if (i + 1 < n) {
            col.push_back(row + 1);
            val.push_back(-h2i);
        }

        if (j + 1 < n) {
            col.push_back(row + n);
            val.push_back(-h2i);
        }
    }

    private:
        size_t n;
        double h2i;
};

amgcl::adapter::make_solver<
    Backend, Coarsening, Relaxation, IterativeSolver
    > solve( amgcl::backend::make_matrix( poisson_2d(m) ) );
~~~

### <a name="coarsening"></a>Coarsening strategies

A coarsener in AMGCL is a class that takes a system matrix and returns three
operators:

1. Restriction operator `R` that downsamples the residual error to a coarser
   level in AMG hierarchy,
2. Prolongation operator `P` that interpolates a correction computed on a
   coarser grid into a finer grid,
3. System matrix `A'` at a coarser level that is usually computed as a Galerkin
   operator `A' = R A P`.

The AMG hierarchy is constructed by recursive invocation of the selected
coarsener. Below is the list of coarsening strategies implemented in the
library.

- Classic Ruge-Stuben coarsening implemented as
  `amgcl::coarsening::ruge_stuben` class (defined in
  [amgcl/coarsening/ruge_stuben.hpp]). Ruge-Stuben coarsening usually results
  in a more efficient multigrid cycles at the price of increased construction
  time and higher memory requirements.
- Aggregation based coarsening strategies. Aggregation based coarseners are
  implemented as class templates with a single template parameter. The
  parameter controls how fine-level variables are subdivided into aggregates.
  Possible choices are `amgcl::coarsening::plain_aggregates`
  ([amgcl/coarsening/plain_aggregates.hpp][]) and
  `amgcl::coarsening::pointwise_aggregates`
  ([amgcl/coarsening/pointwise_aggregates.hpp][]). The latter may be used when
  a system of coupled PDEs is solved. In this case the aggregation
  acts on grid points instead of individual variables.
  - Non-smoothed aggregation: `amgcl::coarsening::aggregation<Aggregates>`
    ([amgcl/coarsening/aggregation.hpp][]).
  - Smoothed aggregation:
    `amgcl::coarsening::smoothed_aggregation<Aggregates>`
    ([amgcl/coarsening/smoothed_aggregation.hpp][]).
  - Smoothed aggregation with energy minimization (see [6]):
    `amgcl::coarsening::smoothed_aggr_emin<Aggregates>`
    ([amgcl/coarsening/smoothed_aggr_emin.hpp][]).

In many cases the best choice is the smoothed aggregation coarsening. It
results in quick construction with low memory consumption and is well suited
for backends with GPGPU acceleration.

### <a name="relaxation"></a>Relaxation schemes

- Gauss-Seidel relaxation: `amgcl::relaxation::gauss_seidel`
  ([amgcl/relaxation/gauss_seidel.hpp][]).
- ILU0 smoother: `amgcl::relaxation::ilu0` ([amgcl/relaxation/ilu0.hpp][]).
- Damped Jacobi relaxation: `amgcl::relaxation::damped_jacobi`
  ([amgcl/relaxation/damped_jacobi.hpp][]).
- Sparse approximate inverse smoother: `amgcl::relaxation::spai0`
  ([amgcl/relaxation/spai0.hpp][]).
- Chebyshev polynomial smoother: `amgcl::relaxation::chebyshev`
  ([amgcl/relaxation/chebyshev.hpp][]).

_Note that Gauss-Seidel and ILU0 smoothers are serial in nature and thus are
only implemented for CPU-based backends that offer iteration over matrix rows
(currently `builtin` and `eigen` backends)._

### <a name="solvers"></a>Solvers

AMGCL provides several iterative solvers, but it should be easy to use it as a
preconditioner with a user-provided solver.

- Conjugate Gradients solver: `amgcl::solver::cg<Backend>`
  ([amgcl/solver/cg.hpp][]).
- BiCGStab: `amgcl::solver::bicgstab<Backend>`
  ([amgcl/solver/bicgstab.hpp][]).
- BiCGStab(L): `amgcl::solver::bicgstabl<Backend>`
  ([amgcl/solver/bicgstabl.hpp][]).
- GMRES: `amgcl::solver::gmres<Backend>` ([amgcl/solver/gmres.hpp][]).

Each solver in AMGCL is a class template. Its single template parameter
specifies the backend to use. This allows to preallocate necessary resources at
class construction. Obviously, the solver backend has to coincide with the AMG
backend.

Solvers provide two versions of function call operator. The simpler version
takes a constructed AMG instance, a right-hand side vector, and a solution
vector. This version solves the same system of equations that was used for the
construction of AMG hierarchy.

The other version also takes a system matrix as first parameter. This version
may be used for the solution of non-stationary problems with slowly changing
coefficients. There is a strong chance that AMG built for one time step will
act as a reasonably good preconditioner for several subsequent time steps [3].

Both versions return a tuple of number of iterations made and a residual error
achieved.

## <a name="extending"></a>Extending AMGCL

### <a name="adding-backends"></a>Adding backends

A backend is a class that defines a matrix and a vector types, and provides
static member functions for cloning and creating matrices and vectors. Users
also have to define several operations acting on backend matrices and vectors:

- Getting matrix dimensions and number of nonzero values,
- Implementation of matrix vector product,
- Implementation of residual error computation,
- Clear vector elments, copy vector elements from another vector,
- Implementation of linear combinations for vectors.

See [amgcl/backend/vexcl.hpp][] for a complete example of backend
implementation.

### <a name="adding-coarseners"></a>Adding coarseners

A coarsener in AMGCL should have the following interface:

~~~{.cpp}
class my_coarsener {
public:
    struct params {
        // Coarsener parameters, if any.
    };

    // Constructs transfer operators.
    // Takes system matrix in builtin format and parameters.
    // Operators are returned in a tuple of shared pointers.
    template <typename Val, typename Col, typename Ptr>
    static boost::tuple<
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >,
        boost::shared_ptr< backend::crs<Val, Col, Ptr> >
        >
    transfer_operators(
        const amgcl::backend::crs<Val, Col, Ptr> &A,
        const params &prm
        );

    // Returns system matrix for the coarser level.
    // Takes system matrix and transfer operators for the current level.
    template <typename Val, typename Col, typename Ptr>
    static boost::shared_ptr< backend::crs<Val, Col, Ptr> >
    coarse_operator(
            const backend::crs<Val, Col, Ptr> &A,
            const backend::crs<Val, Col, Ptr> &P,
            const backend::crs<Val, Col, Ptr> &R,
            const params &prm
            );
};
~~~

Have a look at [amgcl/coarsening/aggregation.hpp][] for an example.

### <a name="adding-smoothers"></a>Adding smoothers

Here is the interface of an AMGCL smoother:

~~~{.cpp}
template <class Backend>
class my_smoother {
public:
    struct params {
        // Smoother parameters, if any.
    };

    // Constructor.
    // Takes system matrix in builtin format, smoother parameters, and backend
    // parameters.
    template <typename Val, typename Col, typename Ptr>
    my_smoother(
        const backend::crs<Val, Col, Ptr> &A,
        const params &prm,
        const typename Backend::params &backend_prm
        );

    // Pre-relaxation.
    // Takes system matrix in backend format, right-hand side and solution
    // vectors, and a vector for temporary storage.
    void apply_pre(
            typename Backend::matrix const &A,
            typename Backend::vector const &rhs,
            typename Backend::vector const &x,
            typename Backend::vector       &tmp,
            const params &prm
            ) const;

    // Post-relaxation.
    // In case the smoother is symmetric, may just call apply_pre.
    void apply_post(
            typename Backend::matrix const &A,
            typename Backend::vector const &rhs,
            typename Backend::vector const &x,
            typename Backend::vector       &tmp,
            const params &prm
            ) const;
};
~~~

Have a look at [amgcl/relaxation/damped_jacobi.hpp][] for an example.

## <a name="references"></a>References

1. R. Barrett, M. Berry, T. F. Chan, J. Demmel, J. Donato, J. Dongarra, V.
   Eijkhout, R. Pozo, C. Romine, and H. Van der Vorst. Templates for the
   Solution of Linear Systems: Building Blocks for Iterative Methods, 2nd
   Edition. SIAM, Philadelphia, PA, 1994.
2. O. Bröker and M. J. Grote. Sparse approximate inverse smoothers for
   geometric and algebraic multigrid. Applied numerical mathematics,
   41(1):61–80, 2002.
3. D. E. Demidov and D. V. Shevchenko. Modification of algebraic multigrid for
   effective gpgpu-based solution of nonstationary hydrodynamics problems.
   Journal of Computational Science, 3(6):460–462, 2012.
4. J. Frank and C. Vuik. On the construction of deflation-based
   preconditioners. SIAM Journal on Scientific Computing, 23(2):442–462, 2001.
5. Pascal Hénon, Pierre Ramet, and Jean Roman. Pastix: a high-performance
   parallel direct solver for sparse symmetric positive definite systems.
   Parallel Computing, 28(2):301–321, 2002.
6. M. Sala and R. S. Tuminaro. A new petrov-galerkin smoothed aggregation
   preconditioner for nonsymmetric linear systems. SIAM Journal on Scientific
   Computing, 31(1):143–166, 2008.
7. G. L. G. Sleijpen and D. R. Fokkema. Bicgstab (l) for linear equations
   involving unsymmetric matrices with complex spectrum. Electronic
   Transactions on Numerical Analysis, 1(11):2000, 1993.
8. K. Stuben. Algebraic multigrid (AMG): an introduction with applications. GMD
   Report 70, GMD, Sankt Augustin, Germany, 1999.
9. U. Trottenberg, C. Oosterlee, and A. Schüller. Multigrid. Academic Press,
   London, 2001. 631 p.
10. P. Vanek, J. Mandel, and M. Brezina. Algebraic multigrid by smoothed
   aggregation for second and fourth order elliptic problems. Computing,
   56(3):179–196, 1996.
11. P. Vanek, M. Brezina, J. Mandel, and others. Convergence of algebraic
    multigrid based on smoothed aggregation. Numerische Mathematik,
    88(3):559–579, 2001.

## <a name="projects"></a>Projects using AMGCL

- [Kratos Multi-Physics][Kratos] (an open source framework for the
  implementation of numerical methods for the solution of engineering
  problems) is using AMGCL for solution of discretized PDEs.
- [PARALUTION][] (a library allowing to employ various sparse iterative
  solvers and preconditioners on multi/many-core CPU and GPU devices) uses
  adopted AMGCL code to build an algebraic multigrid hierarchy.

----------------------------
_This work is a joint effort of [Supercomputer Center of Russian Academy of
Sciences][JSCC] (Kazan branch) and [Kazan Federal University][KPFU]. It is
partially supported by RFBR grants No 12-07-0007 and 12-01-00033._

[multigrid]:  http://en.wikipedia.org/wiki/Multigrid_method
[Boost]:      http://www.boost.org
[Eigen]:      http://eigen.tuxfamily.org
[Blaze]:      https://code.google.com/p/blaze-lib
[VexCL]:      http://github.com/ddemidov/vexcl
[ViennaCL]:   http://viennacl.sourceforge.net
[Kratos]:     http://www.cimne.com/kratos
[PARALUTION]: http://www.paralution.com

[amgcl/amgcl.hpp]: amgcl/amgcl.hpp

[amgcl/backend/builtin.hpp]:     amgcl/backend/builtin.hpp
[amgcl/backend/block_crs.hpp]:   amgcl/backend/block_crs.hpp
[amgcl/backend/eigen.hpp]:       amgcl/backend/eigen.hpp
[amgcl/backend/blaze.hpp]:       amgcl/backend/blaze.hpp
[amgcl/backend/vexcl.hpp]:       amgcl/backend/vexcl.hpp
[amgcl/backend/viennacl.hpp]:    amgcl/backend/viennacl.hpp
[amgcl/backend/cuda.hpp]:        amgcl/backend/cuda.hpp

[amgcl/coarsening/ruge_stuben.hpp]:          amgcl/coarsening/ruge_stuben.hpp
[amgcl/coarsening/aggregation.hpp]:          amgcl/coarsening/aggregation.hpp
[amgcl/coarsening/smoothed_aggregation.hpp]: amgcl/coarsening/smoothed_aggregation.hpp
[amgcl/coarsening/smoothed_aggr_emin.hpp]:   amgcl/coarsening/smoothed_aggr_emin.hpp
[amgcl/coarsening/plain_aggregates.hpp]:     amgcl/coarsening/plain_aggregates.hpp
[amgcl/coarsening/pointwise_aggregates.hpp]: amgcl/coarsening/pointwise_aggregates.hpp

[amgcl/relaxation/damped_jacobi.hpp]: amgcl/relaxation/damped_jacobi.hpp
[amgcl/relaxation/spai0.hpp]:         amgcl/relaxation/spai0.hpp
[amgcl/relaxation/chebyshev.hpp]:     amgcl/relaxation/chebyshev.hpp
[amgcl/relaxation/gauss_seidel.hpp]:  amgcl/relaxation/gauss_seidel.hpp
[amgcl/relaxation/ilu0.hpp]:          amgcl/relaxation/ilu0.hpp

[amgcl/solver/cg.hpp]:        amgcl/solver/cg.hpp
[amgcl/solver/bicgstab.hpp]:  amgcl/solver/bicgstab.hpp
[amgcl/solver/bicgstabl.hpp]: amgcl/solver/bicgstabl.hpp
[amgcl/solver/gmres.hpp]:     amgcl/solver/gmres.hpp

[amgcl/adapter/crs_tuple.hpp]:   amgcl/adapter/crs_tuple.hpp
[amgcl/adapter/crs_builder.hpp]: amgcl/adapter/crs_builder.hpp

[JSCC]:       http://www.jscc.ru/eng/index.shtml
[KPFU]:       http://www.kpfu.ru
