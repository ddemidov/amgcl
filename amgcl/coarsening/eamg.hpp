#ifndef AMGCL_COARSENING_EAMG_HPP
#define AMGCL_COARSENING_EAMG_HPP

// Testing EAMG coarsining/prolongation approach by Li and Zheng

#include <algorithm>
#include <numeric>

#include <tuple>
#include <memory>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/scaled_galerkin.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace coarsening {

template <class Backend>
struct eamg {
    typedef amgcl::detail::empty_params params;

    eamg(const params& = params()) {}

    // Take matrix A, return prolongation P and restriction R=P^T
    template <class Matrix>
    std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
    transfer_operators(const Matrix &A) const {
        typedef typename backend::value_type<Matrix>::type value_type;

        ptrdiff_t n = backend::rows(A);
        ptrdiff_t N = 0;

        std::vector<ptrdiff_t> Cidx(n, -1);
        std::vector<char>      mark(n, 'U');

        // Split variables into C and M sets.
        for(ptrdiff_t i = 0; i < n; ++i) {
            if (mark[i] != 'U') continue;

            mark[i] = 'C';
            Cidx[i] = N++;

            for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                ptrdiff_t c = A.col[j];
                if (mark[c] == 'U') {
                    mark[c] = 'M';
                }
            }
        }

        auto P = std::make_shared<Matrix>();
        P->set_size(n, N, true);

        // Count non-zeros in the prolongation matrix P.
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            if (mark[i] == 'C') {
                P->ptr[i+1] = 1;
                continue;
            }

            for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                ptrdiff_t c = A.col[j];
                if (mark[c] == 'C') {
                    ++P->ptr[i+1];
                }
            }
        }

        P->set_nonzeros(P->scan_row_sizes());

        // Fill the prolongation operator P.
#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            ptrdiff_t row_head = P->ptr[i];

            if (mark[i] == 'C') {
                P->col[row_head] = Cidx[i];
                P->val[row_head] = 1.0;
                continue;
            }

            value_type denom = math::zero<value_type>();

            for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                ptrdiff_t c = A.col[j];
                ptrdiff_t v = A.val[j];
                if (c == i || mark[c] == 'M') {
                    denom += v;
                }
            }

            denom = math::inverse(denom);

            for(ptrdiff_t j = A.ptr[i], e = A.ptr[i+1]; j < e; ++j) {
                ptrdiff_t c = A.col[j];
                ptrdiff_t v = A.val[j];
                if (mark[c] == 'C') {
                    P->col[row_head] = Cidx[c];
                    P->val[row_head] = -denom * v;
                    ++row_head;
                }
            }
        }

        return std::make_tuple(P, transpose(*P));
    }

    template <class Matrix>
    std::shared_ptr<Matrix>
    coarse_operator(const Matrix &A, const Matrix &P, const Matrix &R) const {
        return detail::galerkin(A, P, R);
    }
};

} // namespace coarsening

namespace backend {

template <class Backend>
struct coarsening_is_supported<
    Backend,
    coarsening::eamg,
    typename std::enable_if<
        !std::is_arithmetic<typename backend::value_type<Backend>::type>::value
        >::type
    > : std::false_type
{};

} // namespace backend
} // namespace amgcl


#endif
