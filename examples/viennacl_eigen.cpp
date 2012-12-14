#include <iostream>
#include <cstdlib>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define VIENNACL_HAVE_EIGEN
#include <viennacl/linalg/cg.hpp>

#include "read.hpp"

// This is needed for ViennaCL to recognize MappedSparseMatrix as Eigen type.
namespace viennacl { namespace traits {

template <class T>
struct tag_of<T,
    typename std::enable_if< std::is_base_of<Eigen::EigenBase<T>, T>::value >::type
    >
{
  typedef viennacl::tag_eigen  type;
};

} }

// Simple wrapper around amgcl::solver that provides ViennaCL's preconditioner
// interface.
struct amg_precond {
    typedef amgcl::solver<
        double, int,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu
        > AMG;
    typedef typename AMG::params params;

    // Build AMG hierarchy.
    template <class matrix>
    amg_precond(const matrix &A, const params &prm = params())
        : amg(A, prm), r(amgcl::sparse::matrix_rows(A))
    {
        std::cout << amg << std::endl;
    }


    // Use one V-cycle with zero initial approximation as a preconditioning step.
    template <class vector>
    void apply(vector &x) const {
        std::fill(r.begin(), r.end(), static_cast<double>(0));
        amg.apply(x, r);
        std::copy(r.begin(), r.end(), &x[0]);
    }

    mutable AMG amg;
    mutable std::vector<double> r;
};

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    amgcl::profiler<> prof;

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

    // Use K-Cycle on each level to improve convergence:
    typename amg_precond::AMG::params prm;
    prm.level.kcycle = 1;

    // Build the preconditioner:
    prof.tic("setup");
    amg_precond amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    Eigen::VectorXd x = viennacl::linalg::solve(A, rhs, tag, amg);
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;
}
