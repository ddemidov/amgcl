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

} // namespace amgcl

#endif
