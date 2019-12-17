#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

#include <amgcl/backend/block_crs.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof("v2");
}

int main() {
    using amgcl::prof;

    typedef amgcl::backend::block_crs<double> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::bicgstab<Backend>
        > Solver;

    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    prof.tic("read");
    {
        std::istream_iterator<int>    iend;
        std::istream_iterator<double> dend;

        std::ifstream fptr("rows.txt");
        std::ifstream fcol("cols.txt");
        std::ifstream fval("values.txt");
        std::ifstream frhs("rhs.txt");

        amgcl::precondition(fptr, "rows.txt not found");
        amgcl::precondition(fcol, "cols.txt not found");
        amgcl::precondition(fval, "values.txt not found");
        amgcl::precondition(frhs, "rhs.txt not found");

        std::istream_iterator<int>    iptr(fptr);
        std::istream_iterator<int>    icol(fcol);
        std::istream_iterator<double> ival(fval);
        std::istream_iterator<double> irhs(frhs);

        ptr.assign(iptr, iend);
        col.assign(icol, iend);
        val.assign(ival, dend);
        rhs.assign(irhs, dend);
    }

    int n = ptr.size() - 1;
    prof.toc("read");

    prof.tic("build");
    Solver::params prm;
    prm.precond.coarsening.aggr.eps_strong = 0;
    prm.precond.coarsening.aggr.block_size = 4;
    prm.precond.npre = 2;
    prm.precond.npost = 2;

    Backend::params bprm;
    bprm.block_size = 4;

    Solver solve(std::tie(n, ptr, col, val), prm, bprm);
    prof.toc("build");

    std::cout << solve << std::endl;

    std::vector<double> x(n, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    std::tie(iters, resid) = solve(rhs, x);
    prof.toc("solve");

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << resid << std::endl
              << std::endl;

    std::cout << prof << std::endl;
}
