Adapting custom matrix class
----------------------------

This example shows how to adapt a custom matrix class for use with AMGCL
solvers. Let's say the user application declares the following class to store
its sparse matrices:

.. code-block:: cpp

    class sparse_matrix {
        public:
            typedef std::map<int, double> sparse_row;

            sparse_matrix(int n, int m) : _n(n), _m(m), _rows(n) { }

            int nrows()    const { return _n; }
            int ncols()    const { return _m; }
            int nonzeros() const {
                int nnz = 0;
                for(auto &row : _rows) nnz += row.size();
                return nns;
            }

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

Using ``std::map`` to store sparse rows is probably not the best idea
performance-wise, but it may be convenient during assembly phase. Also, AMGCL
will copy the matrix to internal structures during construction, so setup
performance should not be affected by the choice of the input matrix type too
much.

In order to make the above class work as input matrix for AMGCL, we have to
specialize a few templates inside ``amgcl::backend`` namespace (the templates
are declared inside `\<amgcl/backend/interface.hpp>`_). First we need to let
AMGCL know the value type of the matrix:

.. code-block:: cpp

    namespace amgcl {
    namespace backend {

    template <> struct value_type<sparse_matrix> {
        typedef double type;
    };

We also need to tell how to get the dimensions of the matrix and the number of
its nonzero elements:

.. code-block:: cpp

    // Number of rows in the matrix
    template<> struct rows_impl<sparse_matrix> {
        static int get(const sparse_matrix &A) { return A.nrows(); }
    };

    // Number of cols in the matrix
    template<> struct cols_impl<sparse_matrix> {
        static int get(const sparse_matrix &A) { return A.ncols(); }
    };

    // Number of nonzeros in the matrix. This may be just a rough estimate.
    template<> struct nonzeros_impl<sparse_matrix> {
        static int get(const sparse_matrix &A) { return A.nonzeros(); }
    };

The last and the most involved part is providing a row iterator for the custom
matrix type. In order to do this we need to define a
``row_iterator<sparse_matrix>::type`` class and specialize
``row_begin_impl<sparse_matrix>`` template that would return the iterator over
the given row of the matrix. Here goes:

.. code-block:: cpp

    // Here we define row_iterator<sparse_matrix>::type
    template<> struct row_iterator<sparse_matrix> {
        struct iterator {
            sparse_matrix::sparse_row::const_iterator _it, _end;

            // Take the matrix and the row number:
            iterator(const sparse_matrix &A, int row)
                : _it(A[row].begin()), _end(A[row].end()) { }

            // Check if the iterator is valid:
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

    // Provide a way to obtain the row iterator for the given matrix row:
    template<> struct row_begin_impl<sparse_matrix> {
        typedef typename row_iterator<sparse_matrix>::type iterator;
        static iterator get(const sparse_matrix &A, int row) {
            return iterator(A, row);
        }
    };

    } // namespace backend
    } // namespace amgcl


After this, we can directly use our matrix type to create an AMGCL solver:

.. code-block:: cpp

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


.. note::

    The complete source code of the example may be found at
    `examples/custom_adapter.cpp`_.

.. _\<amgcl/backend/interface.hpp>: https://github.com/ddemidov/amgcl/blob/master/amgcl/backend/interface.hpp
.. _examples/custom_adapter.cpp: https://github.com/ddemidov/amgcl/blob/master/examples/custom_adapter.cpp
