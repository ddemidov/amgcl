#include <iostream>
#include <cstdlib>
#include <utility>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/operations_ublas.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;
typedef boost::numeric::ublas::compressed_matrix<real, boost::numeric::ublas::row_major> ublas_matrix;
typedef boost::numeric::ublas::vector<real> ublas_vector;

namespace amgcl {
    profiler<> prof("ublas");
}
using amgcl::prof;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <problem.dat>" << std::endl;
        return 1;
    }

    // Read matrix and rhs from a binary file.
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;
    ublas_vector rhs;
    int n = read_problem(argv[1], row, col, val, rhs);

    // Create ublas matrix with the data.
    ublas_matrix A(n, n);
    A.reserve(row[n]);

    for(int i = 0; i < n; ++i)
        for(int j = row[i], e = row[i+1]; j < e; ++j)
            A.push_back(i, col[j], val[j]);

    // Build the preconditioner:
    typedef amgcl::solver<
        real, ptrdiff_t,
        amgcl::interp::smoothed_aggregation<amgcl::aggr::plain>,
        amgcl::level::cpu<amgcl::relax::spai0>
        > AMG;

    prof.tic("setup");
    AMG amg( amgcl::sparse::map(A), AMG::params() );
    prof.toc("setup");

    std::cout << amg << std::endl;

    // Solve the problem with CG method. Use AMG as a preconditioner:
    ublas_vector x(n, 0);
    prof.tic("solve (cg)");
    std::pair<int,real> cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
    prof.toc("solve (cg)");

    std::cout << "Iterations: " << cnv.first  << std::endl
              << "Error:      " << cnv.second << std::endl
              << std::endl;

    std::cout << prof;
}
