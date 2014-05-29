#ifndef AMGCL_LEVEL_BLAZE_HPP
#define AMGCL_LEVEL_BLAZE_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   level_blaze.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Level of an AMG hierarchy for use with Blaze library.
 *
 * See https://code.google.com/p/blaze-lib.
 */

#include <array>
#include <memory>

#include <amgcl/level_params.hpp>
#include <amgcl/spmat.hpp>
#include <amgcl/spai.hpp>
#include <amgcl/chebyshev.hpp>
#include <amgcl/operations_blaze.hpp>
#include <amgcl/gmres.hpp>

#include <blaze/Math.h>

namespace amgcl {
namespace level {

struct blaze_damped_jacobi {
    struct params {
        float damping;
        params(float w = 0.72) : damping(w) {}
    };

    template <typename value_t, typename index_t>
    struct instance {
        instance() {}

        template <class spmat>
        instance(const spmat &A, const params&)
            : dia(sparse::matrix_rows(A), sparse::diagonal(A).data())
        {}

        template <class spmat, class vector>
        void apply(const spmat &A, const vector &rhs, vector &x, vector &tmp, const params &prm) const {
            tmp = rhs - A * x;
            x += prm.damping * tmp / dia;
        }

        blaze::DynamicVector<value_t> dia;
    };
};

struct blaze_spai0 {
    struct params { };

    template <typename value_t, typename index_t>
    struct instance {
        instance() {}

        template <class spmat>
        instance(const spmat &A, const params&)
            : M(sparse::matrix_rows(A), spai::level0(A).data())
        { }

        template <class spmat, class vector>
        void apply(const spmat &A, const vector &rhs, vector &x, vector &tmp, const params&) const {
            tmp = rhs - A * x;
            x += M * tmp;
        }

        blaze::DynamicVector<value_t> M;
    };
};

struct blaze_chebyshev {
    struct params {
        unsigned degree;
        float    lower;

        params(unsigned degree = 5, float lower = 1.0f / 30.0f)
            : degree(degree), lower(lower)
        {}
    };

    template <typename value_t, typename index_t>
    struct instance {
        instance() {}

        template <class spmat>
        instance(const spmat &A, const params &prm)
            : p(sparse::matrix_rows(A)),
              q(sparse::matrix_rows(A))
        {
            value_t r = spectral_radius(A);
            C = chebyshev_coefficients(prm.degree, r * prm.lower, r);
        }

        template <class spmat, class vector>
        void apply(const spmat &A, const vector &rhs, vector &x, vector &res, const params&) const {
            res = rhs - A * x;

            p = C[0] * res;

            for(auto c = C.begin() + 1; c != C.end(); ++c) {
                q = A * p;
                p = (*c) * res + q;
            }

            x += p;
        }

        std::vector<value_t> C;
        mutable blaze::DynamicVector<value_t> p, q;
    };
};

template <relax::scheme Relaxation>
struct blaze_relax_scheme;

AMGCL_REGISTER_RELAX_SCHEME(blaze, damped_jacobi);
AMGCL_REGISTER_RELAX_SCHEME(blaze, spai0);
AMGCL_REGISTER_RELAX_SCHEME(blaze, chebyshev);

/// Blaze-based AMG hierarchy.
/**
 * Level of an AMG hierarchy for use with Blaze library.
 *
 * \ingroup levels
 *
 * \param Relaxation Relaxation \ref relax::scheme "scheme" (smoother) to use
 *                   inside V-cycles.
 */
template <relax::scheme Relaxation = relax::damped_jacobi>
struct blaze {

/// Parameters for Blaze-based level storage scheme.
struct params : public amgcl::level::params
{
    typename blaze_relax_scheme<Relaxation>::type::params relax;
};

template <typename value_t, typename index_t = long long>
class instance {
    public:
        typedef sparse::matrix<value_t, index_t>   cpu_matrix;
        typedef ::blaze::CompressedMatrix<value_t> matrix;
        typedef ::blaze::DynamicVector<value_t>    vector;

#define COPY_MATRIX(A, B) \
            A.reserve(B.row.back()); \
            for(index_t i = 0; i < B.rows; ++i) { \
                for(index_t j = B.row[i]; j < B.row[i+1]; ++j) { \
                    A.append(i, B.col[j], B.val[j]); \
                } \
                A.finalize(i); \
            }

        // Construct complete multigrid level from system matrix (a),
        // prolongation (p) and restriction (r) operators.
        // The matrices are moved into the local members.
        instance(cpu_matrix &a, cpu_matrix &p, cpu_matrix &r, const params &prm, unsigned nlevel)
            : A(a.rows, a.cols), P(p.rows, p.cols), R(r.rows, r.cols),
              t(a.rows), relax(a, prm.relax)
        {
            COPY_MATRIX(A, a);
            COPY_MATRIX(P, p);
            COPY_MATRIX(R, r);

            if (nlevel) {
                u.resize(a.rows);
                f.resize(a.rows);

                if (prm.kcycle && nlevel % prm.kcycle == 0)
                    gmres.reset(new gmres_data<vector>(prm.kcycle_iterations, a.rows));
            }

            a.clear();
            p.clear();
            r.clear();
        }

        // Construct the coarsest hierarchy level from system matrix (a) and
        // its inverse (ai).
        instance(cpu_matrix &a, cpu_matrix &ai, const params &prm, unsigned /*nlevel*/)
            : A(a.rows, a.cols), Ainv(ai.rows, ai.cols),
              u(a.rows), f(a.rows), t(a.rows)
        {
            COPY_MATRIX(A, a);
            COPY_MATRIX(Ainv, ai);

            a.clear();
            ai.clear();
        }

#undef COPY_MATRIX

        // Returns reference to the system matrix
        const matrix& get_matrix() const {
            return A;
        }

        // Compute residual value.
        value_t resid(const vector &rhs, vector &x) const {
            t = rhs - A * x;

            return sqrt( (t, t) );
        }

        // Perform one V-cycle. Coarser levels are cycled recursively. The
        // coarsest level is solved directly.
        template <class Iterator>
        static void cycle(Iterator plvl, Iterator end, const params &prm,
                const vector &rhs, vector &x)
        {
            Iterator pnxt = plvl; ++pnxt;

            instance *lvl = plvl->get();
            instance *nxt = pnxt->get();

            if (pnxt != end) {
                for(unsigned j = 0; j < prm.ncycle; ++j) {
                    for(unsigned i = 0; i < prm.npre; ++i)
                        lvl->relax.apply(lvl->A, rhs, x, lvl->t, prm.relax);

                    lvl->t = rhs - lvl->A * x;
                    nxt->f = lvl->R * lvl->t;
                    nxt->u = 0;

                    if (nxt->gmres)
                        kcycle(pnxt, end, prm, nxt->f, nxt->u);
                    else
                        cycle(pnxt, end, prm, nxt->f, nxt->u);

                    x += lvl->P * nxt->u;

                    for(unsigned i = 0; i < prm.npost; ++i)
                        lvl->relax.apply(lvl->A, rhs, x, lvl->t, prm.relax);
                }
            } else {
                x = lvl->Ainv * rhs;
            }
        }

        template <class Iterator>
        static void kcycle(Iterator plvl, Iterator end, const params &prm,
                const vector &rhs, vector &x)
        {
            Iterator pnxt = plvl; ++pnxt;

            instance *lvl = plvl->get();

            if (pnxt != end) {
                cycle_precond<Iterator> p(plvl, end, prm);

                lvl->gmres->restart(lvl->A, rhs, p, x);

                for(int i = 0; i < lvl->gmres->M; ++i)
                    lvl->gmres->iteration(lvl->A, p, i);

                lvl->gmres->update(x, lvl->gmres->M - 1);
            } else {
                x = lvl->Ainv * rhs;
            }
        }

        index_t size() const {
            return A.rows();
        }

        index_t nonzeros() const {
            return A.nonZeros();
        }
    private:
        matrix A;
        matrix P;
        matrix R;
        matrix Ainv;

        mutable vector u;
        mutable vector f;
        mutable vector t;

        typename blaze_relax_scheme<Relaxation>::type::template instance<value_t, index_t> relax;

        mutable std::unique_ptr< gmres_data<vector> > gmres;

        template <class Iterator>
        struct cycle_precond {
            cycle_precond(Iterator lvl, Iterator end, const params &prm)
                : lvl(lvl), end(end), prm(prm) {}

            void apply(const vector &r, vector &x) const {
                cycle(lvl, end, prm, r, x);
            }

            Iterator lvl, end;
            const params &prm;
        };
};

};

} // namespace level
} // namespace amgcl

#endif
