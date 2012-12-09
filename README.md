amgcl
=====

This is a simple and generic AMG hierarchy builder (and a work in progress).
May be used as a standalone solver or as a preconditioner. CG and BiCGStab
iterative solvers are provided. Solvers from [ViennaCL][ViennaCL] are
supported as well.

[VexCL][VexCL], [ViennaCL][ViennaCL], or [Eigen][Eigen] matrix/vector
containers may be used with built-in and ViennaCL's solvers. See
[examples/vexcl.cpp][ex1], [examples/viennacl.cpp][ex2] and
[examples/eigen.cpp][ex3] for respective examples.

Doxygen-generated documention is available at http://ddemidov.github.com/amgcl.

[VexCL]:    https://github.com/ddemidov/vexcl
[ViennaCL]: http://viennacl.sourceforge.net
[Eigen]:    http://eigen.tuxfamily.org

[ex1]: https://github.com/ddemidov/amgcl/blob/master/examples/vexcl.cpp
[ex2]: https://github.com/ddemidov/amgcl/blob/master/examples/viennacl.cpp
[ex3]: https://github.com/ddemidov/amgcl/blob/master/examples/eigen.cpp

AMG hierarchy building
----------------------

Constructor of `amgcl::solver<>` object builds the multigrid hierarchy based on
algebraic information contained in the system matrix:

```C++
// amgcl::sparse::matrix<double, int> A;
// or
// amgcl::sparse::matrix_map<double, int> A;
amgcl::solver<
    double,                 // Scalar type
    int,                    // Index type of the matrix
    amgcl::interp::classic, // Interpolation kind
    amgcl::level::cpu       // Where to store the hierarchy
> amg(A);
```

Currently supported interpolation schemes are `amgcl::interp::classic` and
`amgcl::interp::aggregation<amgcl::aggr::plain>`. The aggregation scheme uses
less memory and is set up faster than classic interpolation, but its
convergence rate is slower. It is well suited for VexCL or ViennaCL containers,
where solution phase is accelerated by the OpenCL technology and, therefore,
the cost of the setup phase is much more important.

```C++
amgcl::solver<
    double, int,
    amgcl::interp::aggregation<amgcl::aggr::plain>,
    amgcl::level::vexcl
> amg(A);
```

Solution
--------

Once the hierarchy is constructed, it may be repeatedly used to solve the
linear system for different right-hand sides:

```C++
// std::vector<double> rhs, x;

auto conv = amg.solve(rhs, x);

std::cout << "Iterations: " << std::get<0>(conv) << std::endl
          << "Error:      " << std::get<1>(conv) << std::endl;
```

Using the AMG as a preconditioner with a Krylov subspace method like conjugate
gradients works even better:
```C++
// Eigen::VectorXd rhs, x;

auto conv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
```

Types of right-hand side and solution vectors should be compatible with the
level type used for construction of the AMG hierarchy. For example,
if `amgcl::level::vexcl` is used as a storage backend, then `vex::SpMat` and
`vex::vector` types have to be used when solving:

```C++
// vex::SpMat<double,int> Agpu;
// vex::vector<double> rhs, x;

auto conv = amgcl::solve(Agpu, rhs, amg, x, amgcl::cg_tag());
```

Installation
------------

The library is header-only, so there is nothing to compile or link to. You just
need to copy amgcl folder somewhere and tell your compiler to scan it for
include files.

