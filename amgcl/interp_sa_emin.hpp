#ifndef AMGCL_INTERP_SA_EMIN_HPP
#define AMGCL_INTERP_SA_EMIN_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   interp_sa_emin.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Interpolation scheme based on smoothed aggregation with energy minimization.
 */

#include <vector>
#include <algorithm>
#include <functional>

#include <boost/typeof/typeof.hpp>

#include <amgcl/spmat.hpp>
#include <amgcl/aggr_connect.hpp>
#include <amgcl/tictoc.hpp>

namespace amgcl {

namespace interp {

/// Interpolation scheme based on smoothed aggregation with energy minimization.
/**
 * See \ref Sala_2008 "Sala (2008)"
 *
 * \param aggr_type \ref aggregation "Aggregation scheme".
 *
 * \ingroup interpolation
 */
template <class aggr_type>
struct sa_emin {

/// Parameters controlling aggregation.
struct params {
    /// Parameter \f$\varepsilon_{str}\f$ defining strong couplings.
    /**
     * Variable \f$i\f$ is defined to be strongly coupled to another variable,
     * \f$j\f$, if \f[|a_{ij}| \geq \varepsilon\sqrt{a_{ii} a_{jj}}\quad
     * \text{with fixed} \quad \varepsilon = \varepsilon_{str} \left(
     * \frac{1}{2} \right)^l,\f]
     * where \f$l\f$ is level number (finest level is 0).
     */
    mutable float eps_strong;

    params() : eps_strong(0.08f) {}
};

/// Constructs coarse level by aggregation.
/**
 * Returns interpolation operator, which is enough to construct system matrix
 * at coarser level.
 *
 * \param A   system matrix.
 * \param prm parameters.
 *
 * \returns interpolation operator.
 */
template < class value_t, class index_t >
static std::pair<
    sparse::matrix<value_t, index_t>,
    sparse::matrix<value_t, index_t>
    >
interp(const sparse::matrix<value_t, index_t> &A, const params &prm) {
    TIC("aggregates");
    BOOST_AUTO(aggr, aggr_type::aggregates(A, aggr::connect(A, prm.eps_strong)));
    prm.eps_strong *= 0.5;
    TOC("aggregates");

    const index_t n = sparse::matrix_rows(A);

    index_t nc = std::max(
            static_cast<index_t>(0),
            *std::max_element(aggr.begin(), aggr.end()) + static_cast<index_t>(1)
            );

    TIC("tentative interpolation");
    sparse::matrix<value_t, index_t> P_tent(n, nc);
    P_tent.col.reserve(n);
    P_tent.val.reserve(n);

    P_tent.row[0] = 0;
    for(index_t i = 0; i < n; ++i) {
        if (aggr[i] >= 0) {
            P_tent.row[i + 1] = P_tent.row[i] + 1;
            P_tent.col.push_back(aggr[i]);
            P_tent.val.push_back(static_cast<value_t>(1));
        } else {
            P_tent.row[i + 1] = P_tent.row[i];
        }
    }
    BOOST_AUTO(R_tent, sparse::transpose(P_tent));
    TOC("tentative interpolation");

    // Compute smoothed nterpolation and restriction operators.
    static std::pair<
        sparse::matrix<value_t, index_t>,
        sparse::matrix<value_t, index_t>
    > PR;

    TIC("smoothed interpolation");
    improve_tentative_interp(A, P_tent, aggr).swap(PR.first);
    TOC("smoothed interpolation");

    TIC("smoothed restriction");
    sparse::transpose(
            improve_tentative_interp(sparse::transpose(A), P_tent, aggr)
            ).swap(PR.second);
    TOC("smoothed restriction");

    return PR;
}

private:

template <class spmat>
static std::vector<typename sparse::matrix_value<spmat>::type>
colwise_inner_prod(const spmat &A, const spmat &B) {
    typedef typename sparse::matrix_value<spmat>::type value_t;
    typedef typename sparse::matrix_index<spmat>::type index_t;

    const index_t n = sparse::matrix_rows(A);
    const index_t m = sparse::matrix_cols(A);

    assert(n == sparse::matrix_rows(B));
    assert(m == sparse::matrix_cols(B));

    BOOST_AUTO(Arow, sparse::matrix_outer_index(A));
    BOOST_AUTO(Acol, sparse::matrix_inner_index(A));
    BOOST_AUTO(Aval, sparse::matrix_values(A));

    BOOST_AUTO(Brow, sparse::matrix_outer_index(B));
    BOOST_AUTO(Bcol, sparse::matrix_inner_index(B));
    BOOST_AUTO(Bval, sparse::matrix_values(B));

    std::vector<value_t> sum(m, static_cast<value_t>(0));

    for(index_t i = 0; i < n; ++i) {
        for(
                index_t ja = Arow[i], ea = Arow[i + 1],
                        jb = Brow[i], eb = Brow[i + 1];
                ja < ea && jb < eb;
           )
        {
            index_t ca = Acol[ja];
            index_t cb = Bcol[jb];

            if (ca < cb)
                ++ja;
            else if (cb < ca)
                ++jb;
            else /*ca == cb*/ {
                sum[ca] += Aval[ja] * Bval[jb];
                ++ja;
                ++jb;
            }
        }
    }

    return sum;
}

template <class spmat>
static std::vector<typename sparse::matrix_value<spmat>::type>
colwise_norm(const spmat &A) {
    typedef typename sparse::matrix_value<spmat>::type value_t;
    typedef typename sparse::matrix_index<spmat>::type index_t;

    const index_t n = sparse::matrix_rows(A);
    const index_t m = sparse::matrix_cols(A);

    BOOST_AUTO(Arow, sparse::matrix_outer_index(A));
    BOOST_AUTO(Acol, sparse::matrix_inner_index(A));
    BOOST_AUTO(Aval, sparse::matrix_values(A));

    std::vector<value_t> sum(m, static_cast<value_t>(0));

    for(index_t i = 0; i < n; ++i)
        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j)
            sum[Acol[j]] += Aval[j] * Aval[j];

    return sum;
}

template <class spmat>
static spmat improve_tentative_interp(const spmat &A, const spmat &P_tent,
        const std::vector<typename sparse::matrix_index<spmat>::type> &aggr)
{
    typedef typename sparse::matrix_value<spmat>::type value_t;
    typedef typename sparse::matrix_index<spmat>::type index_t;

    const index_t n = sparse::matrix_rows(A);
    const index_t m = sparse::matrix_cols(P_tent);

    BOOST_AUTO(D, sparse::diagonal(A));
    BOOST_AUTO(AP, sparse::prod(A, P_tent));
    BOOST_AUTO(DAP, AP);
    for(index_t i = 0; i < n; ++i) {
        value_t dinv = 1 / D[i];
        D[i] = dinv;
        for(index_t j = DAP.row[i], e = DAP.row[i + 1]; j < e; ++j)
            DAP.val[j] *= dinv;
    }
    BOOST_AUTO(ADAP, sparse::prod(A, DAP));

    sparse::sort_rows(AP);
    sparse::sort_rows(ADAP);

    BOOST_AUTO(num, colwise_inner_prod(AP, ADAP));
    BOOST_AUTO(den, colwise_norm(ADAP));

    std::vector<value_t> omega(m);
    std::transform(num.begin(), num.end(), den.begin(), omega.begin(),
            std::divides<value_t>());

    // Update DAP to obtain P.
    for(index_t i = 0; i < n; ++i) {
        for(index_t j = DAP.row[i], e = DAP.row[i + 1]; j < e; ++j) {
            index_t c = DAP.col[j];
            DAP.val[j] *= -omega[c];
            if (c == aggr[i]) DAP.val[j] += 1;
        }
    }

    return DAP;
}

};

} // namespace interp
} // namespace amgcl



#endif
