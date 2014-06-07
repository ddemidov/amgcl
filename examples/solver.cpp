#include <iostream>

#include <amgcl/amgcl.hpp>

#include <amgcl/backend/builtin.hpp>

#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>

#include <amgcl/solver/bicgstab.hpp>

#include <amgcl/profiler.hpp>

#include "sample_problem.hpp"

namespace amgcl {
    profiler<> prof("v2");
}

int main() {
    using amgcl::prof;

    typedef amgcl::amg<
        amgcl::backend::builtin<double>,
        amgcl::coarsening::smoothed_aggregation<
            amgcl::coarsening::plain_aggregates
            >,
        amgcl::relaxation::spai0
        > AMG;

    amgcl::backend::crs<double, int> A;
    std::vector<double> rhs;

    prof.tic("assemble");
    int n = A.nrows = A.ncols = sample_problem(128, A.val, A.col, A.ptr, rhs);
    prof.toc("assemble");

    prof.tic("build");
    AMG amg(A);
    prof.toc("build");

    std::cout << amg << std::endl;

    boost::shared_ptr<AMG::vector> f = AMG::backend_type::copy_vector(rhs, amg.prm.backend);
    boost::shared_ptr<AMG::vector> x = AMG::backend_type::create_vector(n, amg.prm.backend);

    amgcl::backend::clear(*x);

    amgcl::solver::bicgstab<AMG::backend_type> solve(n);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(amg, *f, *x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << amgcl::prof << std::endl;
}
