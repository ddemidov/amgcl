amgcl
=====

amgcl is a simple and generic algebraic [multigrid][amg] (AMG) hierarchy builder
(and a work in progress).  The constructed hierarchy may be used as a
standalone solver or as a preconditioner with some iterative solver.  Several
[iterative solvers][solvers] are provided, and it is also possible to use
generic solvers from other libraries, e.g. [ViennaCL][ViennaCL].

The setup phase is completely CPU-based. The constructed levels of AMG
hierarchy may be stored and used through several [backends][levels]. This
allows for transparent acceleration of the solution phase with help of OpenCL,
CUDA, or OpenMP technologies.  See [examples/vexcl.cpp][ex1],
[examples/viennacl.cpp][ex2] and [examples/eigen.cpp][ex3] for examples of
using amgcl with [VexCL][VexCL], [ViennaCL][ViennaCL], and CPU
backends.

Doxygen-generated documentation is available at http://ddemidov.github.com/amgcl.

[amg]:      http://en.wikipedia.org/wiki/Multigrid_method
[solvers]:  http://ddemidov.github.com/amgcl/group__iterative.html
[levels]:   http://ddemidov.github.com/amgcl/group__levels.html
[interp]:   http://ddemidov.github.com/amgcl/group__interpolation.html
[ex1]:      https://github.com/ddemidov/amgcl/blob/master/examples/vexcl.cpp
[ex2]:      https://github.com/ddemidov/amgcl/blob/master/examples/viennacl.cpp
[ex3]:      https://github.com/ddemidov/amgcl/blob/master/examples/eigen.cpp
[VexCL]:    https://github.com/ddemidov/vexcl
[ViennaCL]: http://viennacl.sourceforge.net
[Eigen]:    http://eigen.tuxfamily.org

Overview
--------

You can use amgcl to solve large sparse system of linear equations in three
simple steps: first, you have to select method components (this is a compile
time decision); second, the AMG hierarchy has to be constructed from a system
matrix; and third, the hierarchy is used to solve the equation system for a
given right-hand side.

The list of interpolation schemes and available backends may be found in
[Interpolation][interp] and [Level Storage Backends][levels] documentation
modules.  The aggregation and smoothed-aggregation interpolation schemes use
less memory and are set up faster than classic interpolation, but their
convergence rate is slower. They are well suited for GPU-accelerated backends,
where the cost of the setup phase is much more important.

Here is the complete code example showing each step in action:
```C++
// First, we need to include relevant headers. Each header basically
// corresponds to an AMG component. Let's say we want to use conjugate gradient
// method preconditioned with smoothed aggregation AMG with VexCL backend:

// This is generic hierarchy builder.
#include <amgcl/amgcl.hpp>
// It will use the following components:

// Interpolation scheme based on smoothed aggregation.
#include <amgcl/interp_smoothed_aggr.hpp>
// Aggregates will be constructed with plain aggregation:
#include <amgcl/aggr_plain.hpp>
// VexCL will be used as a backend:
#include <amgcl/level_vexcl.hpp>
// The definition of conjugate gradient method:
#include <amgcl/cg.hpp>

int main() {
    // VexCL context initialization (let's use all GPUs that support double precision):
    vex::Context ctx( vex::Filter::Type(CL_DEVICE_TYPE_GPU) && vex::Filter::DoublePrecision );

    // Here, the system matrix and right-hand side are somehow constructed. The
    // system matrix data is stored in compressed row storage format in vectors
    // row, col, and val.
    int size;
    std::vector<int>    row, col;
    std::vector<double> val, rhs;

    // We wrap the matrix data into amgcl-compatible type.
    // No data is copied here:
    auto A = amgcl::sparse::map(size, size, row.data(), col.data(), val.data());

    // The AMG builder type. Note the use of damped Jacobi relaxation (smoothing) on each level.
    typedef amgcl::solver<
        double /* matrix value type */, int /* matrix index type */,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::vexcl<amgcl::relax::damped_jacobi>
    > AMG;

    // The parameters. Most of the parameters have some reasonable defaults.
    // VexCL backend needs to know what context to use:
    AMG::params prm;
    prm.level.ctx = &ctx;

    // Here we construct the hierarchy:
    AMG amg(A, prm);

    // Now let's solve the system of equations. We need to transfer matrix,
    // right-hand side, and initial approximation to GPUs. The matrix part may
    // be omitted though, since AMG already has it as part of the hierarchy:
    std::vector<double> x(size, 0.0);

    vex::vector<double> f(ctx.queue(), rhs);
    vex::vector<double> u(ctx.queue(), x);

    // Call AMG-preconditioned CG method:
    auto cnv = amgcl::solve(amg.top_matrix(), f, amg, u, amgcl::cg_tag());

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl;

    // Copy the solution back to host:
    vex::copy(u, x);
}
```
The following command line would compile the example:
```
g++ -o example -std=c++0x -O3 -fopenmp example.cpp -I<path/to/vexcl> -I<path/to/amgcl> -lOpenCL
```
The C++11 support is enabled here (by `-std=c++0x` flag) because it is required
by VexCL library. amgcl relies on Boost instead. Also note the use of
`-fopenmp` switch. It enables an OpenMP-based parallelization of the setup
stage.


Performance
-----------

Here is output of `utest` benchmark (see examples folder), solving 2D 2048x2048
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

