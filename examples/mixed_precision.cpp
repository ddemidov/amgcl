#include <vector>
#include <tuple>

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/profiler.hpp>

#if defined(SOLVER_BACKEND_VEXCL)
#  include <amgcl/backend/vexcl.hpp>
   typedef amgcl::backend::vexcl<float>  fBackend;
   typedef amgcl::backend::vexcl<double> dBackend;
#else
#  ifndef SOLVER_BACKEND_BUILTIN
#    define SOLVER_BACKEND_BUILTIN
#  endif
#  include <amgcl/backend/builtin.hpp>
   typedef amgcl::backend::builtin<float>  fBackend;
   typedef amgcl::backend::builtin<double> dBackend;
#endif

#include "sample_problem.hpp"

namespace amgcl { profiler<> prof; }
using amgcl::prof;

int main() {
    // Combine single-precision preconditioner with a
    // double-precision Krylov solver.
    typedef
        amgcl::amg<
            fBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >
        Precond;

    // Solver is in double precision:
    typedef
        amgcl::solver::cg< dBackend >
        Solver;

    std::vector<ptrdiff_t> ptr, col;
    std::vector<double>    val_d;
    std::vector<double>    rhs;

    fBackend::params bprm_f;
    dBackend::params bprm_d;

#ifdef SOLVER_BACKEND_VEXCL
    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;

    bprm_f.q = ctx;
    bprm_d.q = ctx;
#endif

    prof.tic("assemble");
    int n = sample_problem(128, val_d, col, ptr, rhs);
    prof.toc("assemble");

    std::vector<float>  val_f(val_d.begin(), val_d.end());

    auto A_f = std::tie(n, ptr, col, val_f);

#if defined(SOLVER_BACKEND_VEXCL)
    dBackend::matrix A_d(ctx, n, n, ptr, col, val_d);

    vex::vector<double> f(ctx, rhs);
    vex::vector<double> x(ctx, n);
    x = 0;
#elif defined(SOLVER_BACKEND_BUILTIN)
    auto A_d = std::tie(n, ptr, col, val_d);
    std::vector<double> &f = rhs;
    std::vector<double> x(n, 0.0);
#endif

    prof.tic("setup");
    Solver S(n, Solver::params(), bprm_d);
    Precond P(A_f, Precond::params(), bprm_f);
    prof.toc("setup");

    std::cout << P << std::endl;

    int iters;
    double error;
    prof.tic("solve");
    std::tie(iters, error) = S(A_d, P, f, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << prof << std::endl;
}
