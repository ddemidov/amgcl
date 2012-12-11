#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define AMGCL_PROFILING
//#define AGGREGATION

#include <amgcl/amgcl.hpp>
#ifdef AGGREGATION
#  include <amgcl/aggr_plain.hpp>
#  include <amgcl/interp_aggr.hpp>
#else
#  include <amgcl/interp_classic.hpp>
#endif
#include <amgcl/operations_eigen.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

#include "read.hpp"

namespace amgcl {
profiler<> prof;
}
using amgcl::prof;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    // Read matrix and rhs from a binary file.
    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;
    Eigen::VectorXd     rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Wrap the matrix into Eigen Map.
    Eigen::MappedSparseMatrix<double, Eigen::RowMajor, int> A(
            n, n, row.back(), row.data(), col.data(), val.data()
            );

    // Build the preconditioner:
    typedef amgcl::solver<
        double, int,
#ifdef AGGREGATION
        amgcl::interp::aggregation<amgcl::aggr::plain>,
#else
        amgcl::interp::classic,
#endif
        amgcl::level::cpu
        > AMG;

    typename AMG::params prm;
#ifdef AGGREGATION
    prm.level.kcycle = 1;
#endif

    prof.tic("setup");
    AMG amg(amgcl::sparse::map(A), prm);
    prof.toc("setup");


    // Solve the problem with CG method. Use AMG as a preconditioner:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    prof.tic("solve (cg)");
    auto cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    // Solve the problem with BiCGStab method. Use AMG as a preconditioner:
    x.setZero();
    prof.tic("solve (bicg)");
    cnv = amgcl::solve(A, rhs, amg, x, amgcl::bicg_tag());
    prof.toc("solve (bicg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
