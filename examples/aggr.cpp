#include <iostream>
#include <fstream>
#include <cstdlib>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>

namespace amg {
amg::profiler<> prof;
}
using amg::prof;

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

    // Wrap the matrix into amg::sparse::map and build the preconditioner:
    prof.tic("setup");
    amg::params prm;
    prm.npre = 2;
    prm.npost = 2;
    amg::solver<
        double, int,
        amg::interp::aggr_plain, amg::level::cpu
        > amg(
            amg::sparse::map(n, n, row.data(), col.data(), val.data()),
            prm
            );
    prof.toc("setup");

    // Wrap the matrix into Eigen Map.
    Eigen::MappedSparseMatrix<double, Eigen::RowMajor, int> A(
            n, n, row.back(), row.data(), col.data(), val.data()
            );


    // Solve the problem with CG method. Use AMG as a preconditioner:
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    prof.tic("solve (cg)");
    auto cnv = amg::solve(A, rhs, amg, x, amg::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    // Solve the problem with BiCGStab method. Use AMG as a preconditioner:
    x.setZero();
    prof.tic("solve (bicg)");
    cnv = amg::solve(A, rhs, amg, x, amg::bicg_tag());
    prof.toc("solve (bicg)");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;

    std::cout << prof;
}
