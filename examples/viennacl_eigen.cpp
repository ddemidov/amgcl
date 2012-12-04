#include <iostream>
#include <fstream>
#include <cstdlib>
#include <amgcl/amgcl.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define VIENNACL_HAVE_EIGEN
#include <viennacl/linalg/cg.hpp>

namespace viennacl { namespace traits {

// This is needed for ViennaCL to recognize MappedSparseMatrix as Eigen type.
template <class T>
struct tag_of<T,
    typename std::enable_if<
            std::is_base_of<Eigen::EigenBase<T>, T>::value
        >::type
    >
{
  typedef viennacl::tag_eigen  type;
};

} }

namespace amg {
amg::profiler<> prof;
}
using amg::prof;

// Simple wrapper around amg::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const amg::params &prm = amg::params())
        : amg(A, prm), r(amg::sparse::matrix_rows(A))
    { }

    // Use one V-cycle with zero initial approximation as a preconditioning step.
    template <class vector>
    void apply(vector &x) const {
        std::fill(r.begin(), r.end(), static_cast<double>(0));
        amg.cycle(x, r);
        std::copy(r.begin(), r.end(), &x[0]);
    }

    mutable amg::solver<double, int> amg;
    mutable std::vector<double> r;
};

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

    // Wrap the matrix into amg::sparse::map and build the preconditioner:
    prof.tic("setup");
    amg_precond amg(amg::sparse::map(n, n, row.data(), col.data(), val.data()));
    prof.toc("setup");

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    Eigen::VectorXd x = viennacl::linalg::solve(A, rhs, tag, amg);
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl
              << "Real error: " << (rhs - A * x).norm() / rhs.norm() << std::endl;

    std::cout << prof;
}
