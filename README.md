# amgcl

[![Build Status](https://travis-ci.org/ddemidov/amgcl.png?branch=master)](https://travis-ci.org/ddemidov/amgcl)

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

Doxygen-generated documentation is available at http://ddemidov.github.io/amgcl.

## Overview

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
~~~{.cpp}
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
~~~
The following command line would compile the example:
~~~
g++ -o example -std=c++0x -O3 -fopenmp example.cpp -I<path/to/vexcl> -I<path/to/amgcl> -lOpenCL
~~~
The C++11 support is enabled here (by `-std=c++0x` flag) because it is required
by VexCL library. amgcl relies on Boost instead. Also note the use of
`-fopenmp` switch. It enables an OpenMP-based parallelization of the setup
stage.


## Performance

Here is output of `utest` benchmark (see examples folder), solving 2D 2048x2048
Poisson problem generated with `./genproblem2d 2048`.

The first run is CPU-only (`--level=2`, see `./utest --help` for the options
list). The CPU is Intel Core i7 920:
~~~
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

Iterations: 23
Error:      4.12031e-09


[utest:              7.159 sec.] (100.00%)
[ self:              0.010 sec.] (  0.14%)
[  Read problem:     0.131 sec.] (  1.82%)
[  setup:            1.104 sec.] ( 15.42%)
[  solve:            5.915 sec.] ( 82.62%)
~~~

The second run is VexCL-based, the GPU is NVIDIA Tesla C2075:
~~~
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

Iterations: 23
Error:      4.12031e-09


[utest:                       3.030 sec.] (100.00%)
[ self:                       0.028 sec.] (  0.93%)
[  OpenCL initialization:     0.062 sec.] (  2.05%)
[  Read problem:              0.133 sec.] (  4.37%)
[  setup:                     2.082 sec.] ( 68.72%)
[  solve:                     0.725 sec.] ( 23.93%)
~~~

Setup time has increased, because data structures have to be transfered to GPU
memory. But due to the accelerated solution the total time is reduced. Further
time savings may be expected if the preconditioner is reused for solution with
different right-hand sides.

## Installation

The library is header-only, so there is nothing to compile or link to. You just
need to copy amgcl folder somewhere and tell your compiler to scan it for
include files.

## Projects using amgcl

1. [Kratos Multi-Physics][kratos] (an open source framework for the
   implementation of numerical methods for the solution of engineering
   problems) is using amgcl for solution of discretized PDEs. 
2. [PARALUTION][] (a library allowing to employ various sparse iterative
   solvers and preconditioners on multi/many-core CPU and GPU devices) uses
   adopted amgcl code to build an algebraic multigrid hierarchy.

## References

1. _U. Trottenberg, C. Oosterlee, A. Shuller,_ Multigrid, Academic Press,
   London, 2001.
2. _K. Stuben,_ Algebraic multigrid (AMG): an introduction with applications,
   Journal of Computational and Applied Mathematics,  2001, Vol. 128, Pp.
   281-309.
3. _P. Vanek, J. Mandel, M. Brezina,_ Algebraic multigrid by smoothed
   aggregation for second and fourth order elliptic problems, Computing 56,
   1996, Pp. 179-196.
4. _Y. Notay, P. Vassilevski,_ Recursive Krylov-based multigrid cycles, Numer.
   Linear Algebra Appl. 2008; 15:473-487.
5. _R. Barrett, M. Berry, T. F. Chan et al._ Templates for the Solution of
   Linear Systems: Building Blocks for Iterative Methods, 2nd Edition, SIAM,
   Philadelphia, PA, 1994.
6. _O. Broeker, M. Grote,_ Sparse approximate inverse smoothers for geometric
   and algebraic multigrid, Applied Numerical Mathematics, Volume 41, Issue 1,
   April 2002, Pages 61–80.
7. _M. Sala, R. Tuminaro,_ A new Petrov-Galerkin smoothed aggregation
   preconditioner for nonsymmetric linear systems.  SIAM J. Sci. Comput. 2008,
   Vol. 31, No.1, pp. 143-166.


[amg]:      http://en.wikipedia.org/wiki/Multigrid_method
[solvers]:  http://ddemidov.github.io/amgcl/group__iterative.html
[levels]:   http://ddemidov.github.io/amgcl/group__levels.html
[interp]:   http://ddemidov.github.io/amgcl/group__interpolation.html
[ex1]:      https://github.com/ddemidov/amgcl/blob/master/examples/vexcl.cpp
[ex2]:      https://github.com/ddemidov/amgcl/blob/master/examples/viennacl.cpp
[ex3]:      https://github.com/ddemidov/amgcl/blob/master/examples/eigen.cpp
[VexCL]:    https://github.com/ddemidov/vexcl
[ViennaCL]: http://viennacl.sourceforge.net
[Eigen]:    http://eigen.tuxfamily.org

[kratos]: http://www.cimne.com/kratos
[PARALUTION]: http://www.paralution.com/

[jscc]: http://www.jscc.ru/eng/index.shtml
[kpfu]: http://www.kpfu.ru

----------------------------
_This work is a joint effort of [Supercomputer Center of Russian Academy of
Sciences][jscc] (Kazan branch) and [Kazan Federal University][kpfu]. It is
partially supported by RFBR grants No 12-07-0007 and 12-01-00033._

