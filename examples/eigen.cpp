#include <iostream>
#include <fstream>
#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define AMGCL_PROFILING
//#define AGGREGATION

#include <amgcl/amgcl.hpp>
#ifdef AGGREGATION
#  include <amgcl/aggregation.hpp>
#else
#  include <amgcl/interp_classic.hpp>
#endif
#include <amgcl/operations_eigen.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

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
    std::ifstream pfile(argv[1], std::ios::binary);
    int n;
    pfile.read((char*)&n, sizeof(int));

    std::vector<int> row(n + 1);
    pfile.read((char*)row.data(), row.size() * sizeof(int));

    std::vector<int>    col(row.back());
    std::vector<double> val(row.back());
    Eigen::VectorXd     rhs(n);

    pfile.read((char*)col.data(), col.size() * sizeof(int));
    pfile.read((char*)val.data(), val.size() * sizeof(double));
    pfile.read((char*)rhs.data(), rhs.size() * sizeof(double));

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
