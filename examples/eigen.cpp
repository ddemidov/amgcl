#include <iostream>
#include <cstdlib>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_sa_emin.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;
typedef Eigen::Matrix<real, Eigen::Dynamic, 1> EigenVector;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }
    amgcl::profiler<> prof(argv[0]);

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;
    EigenVector       rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Wrap the matrix into Eigen Map.
    Eigen::MappedSparseMatrix<real, Eigen::RowMajor, int> A(
            n, n, row.back(), row.data(), col.data(), val.data()
            );

    // Build the preconditioner:
    typedef amgcl::solver<
        real, int,
        amgcl::interp::sa_emin<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::damped_jacobi>
        > AMG;

    // Use K-Cycle on each level to improve convergence:
    AMG::params prm;
    prm.level.kcycle = 1;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    EigenVector x = EigenVector::Zero(n);
    prof.tic("solve (cg)");
    std::pair<int,real> cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
