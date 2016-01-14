#include <iostream>
#include <vector>
#include <map>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

class sparse_matrix {
    public:
        typedef std::map<int, double> sparse_row;

        sparse_matrix(int n, int m) : _n(n), _m(m), _rows(n) { }

        int nrows() const { return _n; }
        int ncols() const { return _m; }

        // Get a value at row i and column j
        double operator()(int i, int j) const {
            sparse_row::const_iterator elem = _rows[i].find(j);
            return elem == _rows[i].end() ? 0.0 : elem->second;
        }

        // Get reference to a value at row i and column j
        double& operator()(int i, int j) { return _rows[i][j]; }

        // Access the whole row
        const sparse_row& operator[](int i) const { return _rows[i]; }
    private:
        int _n, _m;
        std::vector<sparse_row> _rows;
};

namespace amgcl {
namespace backend {

// Let AMGCL know the value type of our matrix:
template <> struct value_type<sparse_matrix> {
    typedef double type;
};

// Let AMGCL know the size of our matrix:
template<> struct rows_impl<sparse_matrix> {
    static int get(const sparse_matrix &A) { return A.nrows(); }
};

template<> struct cols_impl<sparse_matrix> {
    static int get(const sparse_matrix &A) { return A.ncols(); }
};

template<> struct nonzeros_impl<sparse_matrix> {
    static int get(const sparse_matrix &A) {
        int n = A.nrows(), nnz = 0;
        for(int i = 0; i < n; ++i)
            nnz += A[i].size();
        return nnz;
    }
};

// Allow AMGCL to iterate over the rows of our matrix:
template<> struct row_iterator<sparse_matrix> {
    struct iterator {
        sparse_matrix::sparse_row::const_iterator _it, _end;

        iterator(const sparse_matrix &A, int row)
            : _it(A[row].begin()), _end(A[row].end()) { }

        // Check if we are at the end of the row.
        operator bool() const {
            return _it != _end;
        }

        // Advance to the next nonzero element.
        iterator& operator++() {
            ++_it;
            return *this;
        }

        // Column number of the current nonzero element.
        int col() const { return _it->first; }

        // Value of the current nonzero element.
        double value() const { return _it->second; }
    };

    typedef iterator type;
};

template<> struct row_begin_impl<sparse_matrix> {
    typedef typename row_iterator<sparse_matrix>::type iterator;
    static iterator get(const sparse_matrix &A, int row) {
        return iterator(A, row);
    }
};

} // namespace backend
} // namespace amgcl

int main() {
    // Discretize a 1D Poisson problem
    const int n = 10000;

    sparse_matrix A(n, n);
    for(int i = 0; i < n; ++i) {
        if (i == 0 || i == n - 1) {
            // Dirichlet boundary condition
            A(i,i) = 1.0;
        } else {
            // Internal point.
            A(i, i-1) = -1.0;
            A(i, i)   =  2.0;
            A(i, i+1) = -1.0;
        }
    }

    // Create an AMGCL solver for the problem.
    typedef amgcl::backend::builtin<double> Backend;
    amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::aggregation,
            amgcl::relaxation::spai0
            >,
        amgcl::solver::cg<Backend>
        > solve( A );

    std::cout << solve.precond() << std::endl;
}
