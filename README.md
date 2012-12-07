amgcl
=====

Simple AMG hierarchy builder. May be used as a standalone solver or as a
preconditioner. CG and BiCGStab iterative solvers are provided. Solvers from
from ViennaCL (http://viennacl.sourceforge.net) are supported as well.

Eigen (http://eigen.tuxfamily.org) or VexCL (https://github.com/ddemidov/vexcl)
matrix/vector containers may be used. See examples/eigen.cpp and
examples/vexcl.cpp for respective examples.

AMG hierarchy building
----------------------

Constructor of `amg::solver<>` object builds the multigrid hierarchy based on
algebraic information contained in the system matrix:

```C++
// amg::sparse::matrix<double, int> A;
// or
// amg::sparse::matrix_map<double, int> A;
amg::solver<
    double,                 // Scalar type
    int,                    // Index type of the matrix
    amg::interp::classic,   // Interpolation kind
    amg::level::cpu         // Where to store the hierarchy
> amg(A);
```

Currently supported interpolation schemes are `amg::interp::classic` and
`amg::interp::aggregation<amg::aggr::plain>`. The aggregation scheme uses less
memory and is set up faster than classic interpolation, but its convergence
rate is slower. It is well suited for VexCL containers, where solution phase is
accelerated by the OpenCL technology and, therefore, the cost of the setup
phase is much more important.

```C++
amg::solver<
    double, int,
    amg::interp::aggregation<amg::aggr::plain>,
    amg::level::vexcl
> amg(A);
```

Solution
--------

After the hierarchy is constructed, it may be repeatedly used to solve the
linear system for different right-hand sides:

```C++
// std::vector<double> rhs, x;

auto conv = amg.solve(rhs, x);

std::cout << "Iterations: " << std::get<0>(conv) << std::endl
          << "Error:      " << std::get<1>(conv) << std::endl;
```

Using the amg as a preconditioner with a Krylov subspace mathod like conjugate
gradients works even better:
```C++
// Eigen::VectorXd rhs, x;

auto conv = amg::solve(A, rhs, x, cg_tag());
```

If `amg::level::vexcl` is used as a storage backend, then `vex::SpMat` and
`vex::vector` types have to be used when solving:

```C++
// vex::SpMat<double,int> Agpu;
// vex::vector<double> rhs, x;

auto conv = amg::solve(Agpu, rhs, x, cg_tag());
```
