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

    /// Controls whether system matrix should be filtered.
    /**
     * If set, interpolation and resriction operators will be based on the
     * filtered system matrix given by
     * \f[
     * a_{ij}^F = \begin{cases} a_{ij}, \quad \text{if} j \in
     * N_i^l(\varepsilon_{str}) \\ 0, \quad \text{otherwise}  \end{cases},
     * \quad \text{if} i\neq j, \quad a_{ii}^F = a_ii - \sum\limits_{j\neq i}
     * \left( a_{ij} - a_{ij}^F \right).
     * \f]
     */
    bool filter_matrix;

    /// Controls whether restriction should use its own damping parameters.
    /**
     * If not set, restriction operator reuses damping parameters computed for
     * prolongation operator, thus reducing the setup cost.
     */
    bool separate_damping;

    params() : eps_strong(0.08f), filter_matrix(false), separate_damping(false) {}
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
    const index_t n = sparse::matrix_rows(A);

    BOOST_AUTO(S, aggr::connect(A, prm.eps_strong));
    prm.eps_strong *= 0.5;

    TIC("aggregates");
    BOOST_AUTO(aggr, aggr_type::aggregates(A, S));
    TOC("aggregates");

    index_t nc = std::max(
            static_cast<index_t>(0),
            *std::max_element(aggr.begin(), aggr.end()) + static_cast<index_t>(1)
            );

    // Filtered matrix and its diagonal.
    sparse::matrix<value_t, index_t> Af;
    std::vector<value_t> Dinv;

    if (prm.filter_matrix) {
        Af.resize(n, n);
        Dinv.resize(n);

        Af.col.reserve(sparse::matrix_nonzeros(A));
        Af.val.reserve(sparse::matrix_nonzeros(A));

        Af.row[0] = 0;
        for(index_t i = 0; i < n; ++i) {
            value_t dia = 0;
            for(index_t j = A.row[i], e = A.row[i + 1]; j < e; ++j) {
                index_t c = A.col[j];
                value_t v = A.val[j];

                if (c == i)
                    dia += v;
                else if (!S[j])
                    dia -= v;
                else {
                    Af.col.push_back(c);
                    Af.val.push_back(v);
                }
            }

            Dinv[i] = 1 / dia;

            Af.col.push_back(i);
            Af.val.push_back(dia);

            Af.row[i + 1] = Af.col.size();
        }
    } else {
        sparse::diagonal(A).swap(Dinv);
        for(BOOST_AUTO(d, Dinv.begin()); d != Dinv.end(); ++d) *d = 1 / *d;
    }

    // Compute smoothed interpolation and restriction operators.
    static std::pair<
        sparse::matrix<value_t, index_t>,
        sparse::matrix<value_t, index_t>
    > PR;

    std::vector<value_t> omega(n);

    TIC("smoothed interpolation");
    smoothed_interpolation(prm.filter_matrix ? Af : A,
            Dinv, aggr, nc, omega).swap(PR.first);
    TOC("smoothed interpolation");

    TIC("smoothed restriction");
    if (prm.separate_damping)
        sparse::transpose(
                smoothed_interpolation(sparse::transpose(prm.filter_matrix ? Af : A),
                    Dinv, aggr, nc, omega)
                ).swap(PR.second);
    else
        smoothed_restriction(prm.filter_matrix ? Af : A,
                Dinv, aggr, nc, omega).swap(PR.second);
    TOC("smoothed restriction");

    return PR;
}

private:

template <typename value_t, typename index_t>
static sparse::matrix<value_t, index_t>
smoothed_interpolation(const sparse::matrix<value_t, index_t> &A,
        const std::vector<value_t> &Dinv, const std::vector<index_t> &aggr,
        index_t nc, std::vector<value_t> &omega)
{
    const index_t n = sparse::matrix_rows(A);

    sparse::matrix<value_t, index_t> AP(n, nc);
    std::fill(AP.row.begin(), AP.row.end(), static_cast<index_t>(0));


    std::vector<value_t> omega_p(nc, static_cast<value_t>(0));
    std::vector<value_t> denum(nc, static_cast<value_t>(0));


#pragma omp parallel
    {
#ifdef _OPENMP
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();

        index_t chunk_size  = (n + nt - 1) / nt;
        index_t chunk_start = tid * chunk_size;
        index_t chunk_end   = std::min(n, chunk_start + chunk_size);
#else
        index_t chunk_start = 0;
        index_t chunk_end   = n;
#endif

        std::vector<index_t> marker(nc, static_cast<index_t>(-1));

        // Compute A * P_tent product. P_tent is stored implicitly in aggr.
        // 1. Structure of the product result:
        for(index_t i = chunk_start; i < chunk_end; ++i) {
            for(index_t j = A.row[i], e = A.row[i + 1]; j < e; ++j) {
                index_t g = aggr[A.col[j]]; if (g < 0) continue;

                if (marker[g] != i) {
                    marker[g] = i;
                    ++AP.row[i + 1];
                }
            }
        }

        std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));

#pragma omp barrier
#pragma omp single
        {
            std::partial_sum(AP.row.begin(), AP.row.end(), AP.row.begin());
            AP.reserve(AP.row.back());
        }

        // 2. Compute the product result.
        for(index_t i = chunk_start; i < chunk_end; ++i) {
            index_t row_beg = AP.row[i];
            index_t row_end = row_beg;
            for(index_t j = A.row[i], e = A.row[i + 1]; j < e; ++j) {
                index_t g = aggr[A.col[j]]; if (g < 0) continue;

                if (marker[g] < row_beg) {
                    marker[g] = row_end;
                    AP.col[row_end] = g;
                    AP.val[row_end] = A.val[j];
                    ++row_end;
                } else {
                    AP.val[marker[g]] += A.val[j];
                }
            }

            sparse::insertion_sort(&AP.col[row_beg], &AP.val[row_beg], row_end - row_beg);
        }

        std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));
        std::vector< std::pair<index_t, value_t> > adap(128);

#pragma omp barrier

        // Compute A * Dinv * AP row by row and compute columnwise scalar products
        // necessary for computation of omega_p. The actual results of
        // matrix-matrix product are not stored.
        for(index_t ia = chunk_start; ia < chunk_end; ++ia) {
            adap.clear();

            // Form current row of ADAP matrix.
            for(index_t ja = A.row[ia], ea = A.row[ia + 1]; ja < ea; ++ja) {
                index_t ca = A.col[ja];
                value_t va = A.val[ja];
                value_t di = Dinv[ca];

                for(index_t jb = AP.row[ca], eb = AP.row[ca + 1]; jb < eb; ++jb) {
                    index_t cb = AP.col[jb];
                    value_t vb = AP.val[jb] * di;

                    if (marker[cb] < 0) {
                        marker[cb] = adap.size();
                        adap.push_back(std::make_pair(cb, va * vb));
                    } else {
                        adap[marker[cb]].second += va * vb;
                    }
                }
            }

            std::sort(adap.begin(), adap.end());

            // Update columnwise scalar products (AP,ADAP) and (ADAP,ADAP).
            // 1. (AP, ADAP)
            for(
                    index_t ja = AP.row[ia], ea = AP.row[ia + 1],
                    jb = 0, eb = adap.size();
                    ja < ea && jb < eb;
               )
            {
                index_t ca = AP.col[ja];
                index_t cb = adap[jb].first;

                if (ca < cb)
                    ++ja;
                else if (cb < ca)
                    ++jb;
                else /*ca == cb*/ {
#pragma omp atomic
                    omega_p[ca] += AP.val[ja] * adap[jb].second;
                    ++ja;
                    ++jb;
                }
            }

            // 2. (ADAP, ADAP) (and clear marker)
            for(index_t j = 0, e = adap.size(); j < e; ++j) {
                index_t c = adap[j].first;
                value_t v = adap[j].second;
#pragma omp atomic
                denum[c] += v * v;
                marker[c] = -1;
            }
        }
    }

    std::transform(omega_p.begin(), omega_p.end(), denum.begin(), omega_p.begin(),
            std::divides<value_t>());

    // Convert omega from (4.13) to (4.14) (Sala, Tuminaro, 2008):
#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        value_t w = -1;
        for(index_t j = A.row[i], e = A.row[i + 1]; j < e; ++j) {
            index_t g = aggr[A.col[j]]; if (g < 0) continue;
            if (omega_p[g] < w || w < 0) w = omega_p[g];
        }
        omega[i] = std::max(w, static_cast<value_t>(0));
    }

    // Update AP to obtain P.
#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        value_t wd = omega[i] * Dinv[i];
        for(index_t j = AP.row[i], e = AP.row[i + 1]; j < e; ++j)
            AP.val[j] = (AP.col[j] == aggr[i] ? 1 : 0) - wd * AP.val[j];
    }

    return AP;
}

template <typename value_t, typename index_t>
static sparse::matrix<value_t, index_t>
smoothed_restriction(const sparse::matrix<value_t, index_t> &A,
        const std::vector<value_t> &Dinv, const std::vector<index_t> &aggr,
        index_t nc, const std::vector<value_t> &omega)
{
    const index_t n = sparse::matrix_rows(A);

    // Get structure of R_tent from aggr
    std::vector<index_t> R_tent_row(nc + 1, static_cast<index_t>(0));
    for(index_t i = 0; i < n; ++i) {
        index_t g = aggr[i]; if (g < 0) continue;
        ++R_tent_row[g + 1];
    }

    std::partial_sum(R_tent_row.begin(), R_tent_row.end(), R_tent_row.begin());
    std::vector<index_t> R_tent_col(R_tent_row.back());

    for(index_t i = 0; i < n; ++i) {
        index_t g = aggr[i]; if (g < 0) continue;
        R_tent_col[R_tent_row[g]++] = i;
    }

    std::rotate(R_tent_row.begin(), R_tent_row.end() - 1, R_tent_row.end());
    R_tent_row[0] = 0;

    sparse::matrix<value_t, index_t> R(nc, n);
    std::fill(R.row.begin(), R.row.end(), static_cast<index_t>(0));

    // Compute R_tent * A * Dinv.
#pragma omp parallel
    {
#ifdef _OPENMP
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();

        index_t chunk_size  = (nc + nt - 1) / nt;
        index_t chunk_start = tid * chunk_size;
        index_t chunk_end   = std::min(nc, chunk_start + chunk_size);
#else
        index_t chunk_start = 0;
        index_t chunk_end   = nc;
#endif

        std::vector<index_t> marker(n, static_cast<index_t>(-1));
        for(index_t ir = chunk_start; ir < chunk_end; ++ir) {
            for(index_t jr = R_tent_row[ir], er = R_tent_row[ir + 1]; jr < er; ++jr) {
                index_t cr = R_tent_col[jr];
                for(index_t ja = A.row[cr], ea = A.row[cr + 1]; ja < ea; ++ja) {
                    index_t ca = A.col[ja];

                    if (marker[ca] != ir) {
                        marker[ca] = ir;
                        ++R.row[ir + 1];
                    }
                }
            }
        }

        std::fill(marker.begin(), marker.end(), static_cast<index_t>(-1));

#pragma omp barrier
#pragma omp single
        {
            std::partial_sum(R.row.begin(), R.row.end(), R.row.begin());
            R.reserve(R.row.back());
        }

        for(index_t ir = chunk_start; ir < chunk_end; ++ir) {
            index_t row_beg = R.row[ir];
            index_t row_end = row_beg;

            for(index_t jr = R_tent_row[ir], er = R_tent_row[ir + 1]; jr < er; ++jr) {
                index_t cr = R_tent_col[jr];

                for(index_t ja = A.row[cr], ea = A.row[cr + 1]; ja < ea; ++ja) {
                    index_t ca = A.col[ja];
                    value_t va = A.val[ja] * Dinv[ca];

                    if (marker[ca] < row_beg) {
                        marker[ca] = row_end;
                        R.col[row_end] = ca;
                        R.val[row_end] = va;
                        ++row_end;
                    } else {
                        R.val[marker[ca]] += va;
                    }
                }
            }
        }
    }

    // Update R.
#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < nc; ++i) {
        for(index_t j = R.row[i], e = R.row[i + 1]; j < e; ++j) {
            index_t c = R.col[j];
            R.val[j] = (aggr[c] == i ? 1 : 0) - omega[c] * R.val[j];
        }
    }

    return R;
}

};

} // namespace interp
} // namespace amgcl



#endif
