#include <iostream>
#include <cstdlib>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define VIENNACL_HAVE_EIGEN

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/operations_viennacl.hpp>
#include <amgcl/profiler.hpp>

#include <viennacl/linalg/cg.hpp>

#include "read.hpp"

typedef double real;
typedef Eigen::Matrix<real, Eigen::Dynamic, 1> EigenVector;

// This is needed for ViennaCL to recognize MappedSparseMatrix as Eigen type.
namespace viennacl { namespace traits {

template <class T>
struct tag_of<T,
    typename boost::enable_if< boost::is_base_of<Eigen::EigenBase<T>, T> >::type
    >
{
  typedef viennacl::tag_eigen  type;
};

} }

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
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu
        > AMG;

    // Use K-Cycle on each level to improve convergence:
    typename AMG::params prm;
    prm.level.kcycle = 1;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), prm );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method from ViennaCL. Use AMG as a
    // preconditioner:
    prof.tic("solve");
    viennacl::linalg::cg_tag tag(1e-8, n);
    EigenVector x = viennacl::linalg::solve(A, rhs, tag,
            amgcl::make_viennacl_precond<EigenVector>(amg));
    prof.toc("solve");

    std::cout << "Iterations: " << tag.iters() << std::endl
              << "Error:      " << tag.error() << std::endl;

    std::cout << prof;
}
