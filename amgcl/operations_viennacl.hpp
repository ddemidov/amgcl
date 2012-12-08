#ifndef AMGCL_OPERATIONS_VIENNACL_HPP
#define AMGCL_OPERATIONS_VIENNACL_HPP

#include <type_traits>

#include <amgcl/spmat.hpp>
#include <viennacl/compressed_matrix.hpp>

namespace amgcl {

template <class spmat>
void copy(
    const spmat &A,
    viennacl::compressed_matrix< typename sparse::matrix_value<spmat>::type > &B
    )
{
    typedef typename sparse::matrix_value<spmat>::type value_t;

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    auto n   = sparse::matrix_rows(A);
    auto m   = sparse::matrix_cols(A);
    auto nnz = sparse::matrix_nonzeros(A);

    std::vector<unsigned int> row(Arow, Arow + n + 1);
    std::vector<unsigned int> col(Acol, Acol + nnz);

    B.set(row.data(), col.data(), const_cast<value_t*>(Aval), n, m, nnz);
}

template <class spmat>
class viennacl_matrix_adapter {
    public:
        typedef typename sparse::matrix_index<spmat>::type index_type;
        typedef typename sparse::matrix_value<spmat>::type value_type;

        class const_iterator1;

        class const_iterator2 {
            public:
                bool operator!=(const const_iterator2 &it) const {
                    return pos != it.pos;
                }

                const const_iterator2& operator++() {
                    ++pos;
                    return *this;
                }

                index_type index1() const {
                    return row;
                }

                index_type index2() const {
                    return col[pos];
                }

                value_type operator*() const {
                    return val[pos];
                }
            private:
                const_iterator2(index_type row, index_type pos,
                        const index_type *col, const value_type *val)
                    : row(row), pos(pos), col(col), val(val)
                { }

                index_type row;
                index_type pos;
                const index_type *col;
                const value_type *val;

                friend class const_iterator1;
        };

        class const_iterator1 {
            public:
                bool operator!=(const const_iterator1 &it) const {
                    return pos != it.pos;
                }

                const const_iterator1& operator++() {
                    ++pos;
                    return *this;
                }

                const const_iterator2 begin() const {
                    return const_iterator2(pos, row[pos], col, val);
                }

                const const_iterator2 end() const {
                    return const_iterator2(pos, row[pos + 1], col, val);
                }
            private:
                const_iterator1(index_type pos,
                        const index_type *row,
                        const index_type *col,
                        const value_type *val
                        )
                    : pos(pos), row(row), col(col), val(val)
                { }

                index_type pos;
                const index_type *row;
                const index_type *col;
                const value_type *val;

                friend class viennacl_matrix_adapter;
        };

        viennacl_matrix_adapter(const spmat &A)
            : rows(sparse::matrix_rows(A)),
              cols(sparse::matrix_cols(A)),
              row(sparse::matrix_outer_index(A)),
              col(sparse::matrix_inner_index(A)),
              val(sparse::matrix_values(A))
        { }

        const_iterator1 begin1() const {
            return const_iterator1(0, row, col, val);
        }

        const_iterator1 end1() const {
            return const_iterator1(rows, row, col, val);
        }

        index_type size1() const {
            return rows;
        }

        index_type size2() const {
            return cols;
        }
    private:
        index_type rows;
        index_type cols;

        const index_type *row;
        const index_type *col;
        const value_type *val;
};

namespace sparse {

template <class spmat>
viennacl_matrix_adapter<spmat> viennacl_map(const spmat &A) {
    return viennacl_matrix_adapter<spmat>(A);
}

} // namespace sparse
} // namespace amgcl

#endif
