#ifndef AMGCL_INTERP_HPP
#define AMGCL_INTERP_HPP

#include <vector>
#include <tuple>
#include <algorithm>
#include <amgcl/spmat.hpp>
#include <amgcl/params.hpp>
#include <amgcl/profiler.hpp>

namespace amg {

#ifdef AMGCL_PROFILING
extern amg::profiler<> prof;
#  define TIC(what) prof.tic(what);
#  define TOC(what) prof.tic(what);
#else
#  define TIC(what)
#  define TOC(what)
#endif

// Extract strong connections from a system matrix.
template < class spmat >
std::tuple<
    std::vector<typename sparse::matrix_index<spmat>::type>,
    std::vector<typename sparse::matrix_index<spmat>::type>,
    std::vector<char>
    >
connect(const spmat &A, const params &prm, std::vector<char> &cf) {
    typedef typename sparse::matrix_index<spmat>::type index_t;
    typedef typename sparse::matrix_value<spmat>::type value_t;

    const index_t n = sparse::matrix_rows(A);

    std::tuple<
        std::vector<index_t>,
        std::vector<index_t>,
        std::vector<char>
        > S;

    auto &Srow = std::get<0>(S);
    auto &Scol = std::get<1>(S);
    auto &Sval = std::get<2>(S);

    Srow.resize(n + 1, 0);
    Sval.resize(sparse::matrix_nonzeros(A), false);

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        value_t a_min = 0;

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j)
            if (Acol[j] != i && Aval[j] < a_min) a_min = Aval[j];

        if (fabs(a_min) < 1e-32) {
            cf[i] = 'F';
            continue;
        }

        a_min *= prm.eps_strong;

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j)
            if (Acol[j] != i && Aval[j] < a_min) Sval[j] = true;
    }

    for(index_t i = 0, nnz = Arow[n]; i < nnz; ++i)
        if (Sval[i]) Srow[Acol[i] + 1]++;

    std::partial_sum(Srow.begin(), Srow.end(), Srow.begin());

    Scol.resize(Srow.back());

    for(index_t i = 0; i < n; ++i)
        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j)
            if (Sval[j]) Scol[ Srow[ Acol[j] ]++ ] = i;

    for(index_t i = n; i > 0; --i) Srow[i] = Srow[i-1];

    return S;
}

// Split variables into C(oarse) and F(ine) sets.
template < class spmat >
void cfsplit(
        const spmat &A,
        const std::tuple<
                    std::vector<typename sparse::matrix_index<spmat>::type>,
                    std::vector<typename sparse::matrix_index<spmat>::type>,
                    std::vector<char>
                    > &S,
        std::vector<char> &cf
        )
{
    typedef typename sparse::matrix_index<spmat>::type index_t;

    const index_t n = sparse::matrix_rows(A);

    auto &Srow = std::get<0>(S);
    auto &Scol = std::get<1>(S);
    auto &Sval = std::get<2>(S);

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);

    std::vector<index_t> lambda(n);

    // Initialize lambdas:
    for(index_t i = 0; i < n; ++i) {
        index_t temp = 0;
        for(index_t j = Srow[i], e = Srow[i + 1]; j < e; ++j)
            temp += (cf[Scol[j]] == 'U' ? 1 : 2);
        lambda[i] = temp;
    }

    // Keep track of variable groups of equal lambda values.
    // ptr - start of a group;
    // cnt - size of a group;
    // i2n - variable number;
    // n2i - vaiable position in a group.
    std::vector<index_t> ptr(n+1, static_cast<index_t>(0));
    std::vector<index_t> cnt(n, static_cast<index_t>(0));
    std::vector<index_t> i2n(n);
    std::vector<index_t> n2i(n);

    for(index_t i = 0; i < n; ++i) ptr[lambda[i] + 1]++;

    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());

    for(index_t i = 0; i < n; ++i) {
        index_t lam = lambda[i];
        index_t idx = ptr[lam] + cnt[lam]++;
        i2n[idx] = i;
        n2i[i] = idx;
    }

    static_assert(std::is_signed<index_t>::value, "Matrix index type should be signed");

    // Process variables by decreasing lambda value.
    // 1. The vaiable with maximum value of lambda becomes next C-variable.
    // 2. Its neighbours from S' become F-variables.
    // 3. Keep lambda values in sync.
    for(index_t top = n - 1; top >= 0; --top) {
        index_t i = i2n[top];
        index_t lam = lambda[i];

        if (lam == 0) {
            std::replace(cf.begin(), cf.end(), 'U', 'C');
            break;
        }

        // Remove tne variable from its group.
        cnt[lam]--;

        if (cf[i] == 'F') continue;
        assert(cf[i] == 'U');

        // Mark the variable as 'C'.
        cf[i] = 'C';

        // Its neighbours from S' become F-variables.
        for(index_t j = Srow[i], e = Srow[i + 1]; j < e; ++j) {
            index_t c = Scol[j];

            if (cf[c] != 'U') continue;

            cf[c] = 'F';

            // Increase lambdas of the newly created F's neighbours.
            for(index_t jj = Arow[c], ee = Arow[c + 1]; jj < ee; ++jj) {
                if (!Sval[jj]) continue;

                index_t cc = Acol[jj];
                index_t lam_cc = lambda[cc];

                if (cf[cc] != 'U' || lam_cc >= n - 1) continue;

                index_t old_pos = n2i[cc];
                index_t new_pos = ptr[lam_cc] + cnt[lam_cc] - 1;

                n2i[i2n[old_pos]] = new_pos;
                n2i[i2n[new_pos]] = old_pos;

                std::swap(i2n[old_pos], i2n[new_pos]);

                --cnt[lam_cc];
                ++cnt[lam_cc + 1];
                ptr[lam_cc + 1] = ptr[lam_cc] + cnt[lam_cc];

                ++lambda[cc];
            }
        }

        // Decrease lambdas of the newly create C's neighbours.
        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; j++) {
            if (!Sval[j]) continue;

            index_t c = Acol[j];
            index_t lam = lambda[c];

            if (cf[c] != 'U' || lam == 0) continue;

            index_t old_pos = n2i[c];
            index_t new_pos = ptr[lam];

            n2i[i2n[old_pos]] = new_pos;
            n2i[i2n[new_pos]] = old_pos;

            std::swap(i2n[old_pos], i2n[new_pos]);

            --cnt[lam];
            ++cnt[lam - 1];
            ++ptr[lam];
            --lambda[c];

            assert(ptr[lam - 1] == ptr[lam] - cnt[lam - 1]);
        }
    }
}

// Compute prolongation operator from a system matrix.
template < class value_t, class index_t >
sparse::matrix<value_t, index_t> interp(
        const sparse::matrix<value_t, index_t> &A, const params &prm
        )
{
    const index_t n = sparse::matrix_rows(A);

    std::vector<char> cf(n, 'U');

    TIC("conn");
    auto S = connect(A, prm, cf);
    TOC("conn");

    TIC("split");
    cfsplit(A, S, cf);
    TOC("split");

    TIC("interpolation");
    index_t nc = 0;
    std::vector<index_t> cidx(n);

    for(index_t i = 0; i < n; i++)
        if (cf[i] == 'C') cidx[i] = nc++;

    auto Arow = sparse::matrix_outer_index(A);
    auto Acol = sparse::matrix_inner_index(A);
    auto Aval = sparse::matrix_values(A);

    auto &Sval = std::get<2>(S);

    sparse::matrix<value_t, index_t> P(n, nc);
    std::fill(P.row.begin(), P.row.end(), static_cast<index_t>(0));

    std::vector<value_t> Amin, Amax;

    if (prm.trunc_int) {
        Amin.resize(n);
        Amax.resize(n);
    }

#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        if (cf[i] == 'C') {
            ++P.row[i + 1];
            continue;
        }

        if (prm.trunc_int) {
            value_t amin = 0, amax = 0;

            for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
                if (!Sval[j] || cf[Acol[j]] != 'C') continue;

                amin = std::min(amin, Aval[j]);
                amax = std::max(amax, Aval[j]);
            }

            Amin[i] = amin = amin * prm.eps_tr;
            Amax[i] = amax = amax * prm.eps_tr;

            for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
                if (!Sval[j] || cf[Acol[j]] != 'C') continue;

                if (Aval[j] <= amin || Aval[j] >= amax)
                    ++P.row[i + 1];
            }
        } else {
            for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j)
                if (Sval[j] && cf[Acol[j]] == 'C')
                    ++P.row[i + 1];
        }
    }

    std::partial_sum(P.row.begin(), P.row.end(), P.row.begin());

    P.reserve(P.row.back());

#pragma omp parallel for schedule(dynamic, 1024)
    for(index_t i = 0; i < n; ++i) {
        index_t row_head = P.row[i];

        if (cf[i] == 'C') {
            P.col[row_head] = cidx[i];
            P.val[row_head] = 1;
            continue;
        }

        value_t diag  = 0;
        value_t a_num = 0, a_den = 0;
        value_t b_num = 0, b_den = 0;
        value_t d_neg = 0, d_pos = 0;

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            index_t c = Acol[j];
            value_t v = Aval[j];

            if (c == i) {
                diag = v;
                continue;
            }

            if (v < 0) {
                a_num += v;
                if (Sval[j] && cf[c] == 'C') {
                    a_den += v;
                    if (prm.trunc_int && v > Amin[i]) d_neg += v;
                }
            } else {
                b_num += v;
                if (Sval[j] && cf[c] == 'C') {
                    b_den += v;
                    if (prm.trunc_int && v < Amax[i]) d_pos += v;
                }
            }
        }

        value_t cf_neg = 1;
        value_t cf_pos = 1;

        if (prm.trunc_int) {
            if (fabs(a_den - d_neg) > 1e-32) cf_neg = a_den / (a_den - d_neg);
            if (fabs(b_den - d_pos) > 1e-32) cf_pos = b_den / (b_den - d_pos);
        }

        if (b_num > 0 && fabs(b_den) < 1e-32) diag += b_num;

        value_t alpha = fabs(a_den) > 1e-32 ? -cf_neg * a_num / (diag * a_den) : 0;
        value_t beta  = fabs(b_den) > 1e-32 ? -cf_pos * b_num / (diag * b_den) : 0;

        for(index_t j = Arow[i], e = Arow[i + 1]; j < e; ++j) {
            index_t c = Acol[j];
            value_t v = Aval[j];

            if (!Sval[j] || cf[c] != 'C') continue;
            if (prm.trunc_int && v > Amin[i] && v < Amax[i]) continue;

            P.col[row_head] = cidx[c];
            P.val[row_head] = (v < 0 ? alpha : beta) * v;
            ++row_head;
        }
    }
    TOC("interpolation");

    return P;
}

} // namespace amg

#endif
