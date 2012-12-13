amgcl
=====

This is a simple and generic algebraic multigrid (AMG) hierarchy builder (and a
work in progress).  May be used as a standalone solver or as a preconditioner.
CG and BiCGStab iterative solvers are provided. Solvers from
[ViennaCL][ViennaCL] are supported as well.

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

See documentation for [Interpolation][interp] module to see the list of
supported interpolation schemes. The aggregation schemes use less memory and
are set up faster than classic interpolation, but their convergence rate is
slower. They are well suited for VexCL or ViennaCL containers, where solution
phase is accelerated by the OpenCL technology and, therefore, the cost of the
setup phase is much more important.

[interp]: http://ddemidov.github.com/amgcl/group__interpolation.html

```C++
amgcl::solver<
    double, int,
    amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
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
Performance
-----------

Here is output of `utest` program (see examples folder), solving 2D 2048x2048
Poisson problem generated with `./genproblem 2048`.

The first run is CPU-only (`--level=2`, see `./utest -help` for the options
list). The CPU is Intel Core i7 920:
```
$ ./utest --level 2
Reading "problem.dat"...
Done

Number of levels:    6
Operator complexity: 1.34
Grid complexity:     1.19

level     unknowns       nonzeros
---------------------------------
    0      4194304       20938768 (74.75%)
    1       698198        6278320 (22.41%)
    2        77749         701425 ( 2.50%)
    3         8814          82110 ( 0.29%)
    4          988           9362 ( 0.03%)
    5          115           1149 ( 0.00%)

Iterations: 25
Error:      6.679105e-09

[Profile:            7.562 sec.] (100.00%)
[ self:              0.011 sec.] (  0.15%)
[  Read problem:     0.130 sec.] (  1.72%)
[  setup:            1.020 sec.] ( 13.49%)
[  solve:            6.401 sec.] ( 84.65%)
```

The second run is VexCL-based, the GPU is NVIDIA Tesla C2075:
```
$ ./utest --level=1
Reading "problem.dat"...
Done

1. Tesla C2075

Number of levels:    6
Operator complexity: 1.34
Grid complexity:     1.19

level     unknowns       nonzeros
---------------------------------
    0      4194304       20938768 (74.75%)
    1       698198        6278320 (22.41%)
    2        77749         701425 ( 2.50%)
    3         8814          82110 ( 0.29%)
    4          988           9362 ( 0.03%)
    5          115           1149 ( 0.00%)

Iterations: 25
Error:      6.679105e-09

[Profile:                     3.592 sec.] (100.00%)
[ self:                       0.437 sec.] ( 12.16%)
[  OpenCL initialization:     0.051 sec.] (  1.41%)
[  Read problem:              0.130 sec.] (  3.61%)
[  setup:                     2.179 sec.] ( 60.65%)
[  solve:                     0.796 sec.] ( 22.16%)
```

Setup time has increased, because data structures have to be transfered to GPU
memory. But due to the accelerated solution the total time is reduced. Further
time savings may be expected if the preconditioner is reused for solution with
different right-hand sides.

Installation
------------

The library is header-only, so there is nothing to compile or link to. You just
need to copy amgcl folder somewhere and tell your compiler to scan it for
include files.

