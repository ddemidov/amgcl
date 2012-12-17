#ifndef AMGCL_LEVEL_VEXCL_HPP
#define AMGCL_LEVEL_VEXCL_HPP

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
 * \file   level_vexcl.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Level of an AMG hierarchy for use with VexCL vectors.
 */

#include <amgcl/level_params.hpp>
#include <amgcl/spmat.hpp>

#include <vexcl/vexcl.hpp>

namespace amgcl {
namespace level {

/// VexCL-based AMG hierarchy.
/**
 * Level of an AMG hierarchy for use with VexCL vectors.
 * \ingroup levels
 */
struct vexcl {

/// Parameters for VexCL-based level storage scheme.
struct params
    : public amgcl::level::params
{
    vex::Context *ctx;  ///< VexCL Context for VexCL objects creation.

    params() : ctx(0) { }
};

template <typename value_t, typename index_t = long long>
class instance {
    public:
        typedef sparse::matrix<value_t, index_t>      cpu_matrix;
        typedef vex::SpMat<value_t, index_t, index_t> matrix;
        typedef vex::vector<value_t>                  vector;

        // Construct complete multigrid level from system matrix (a),
        // prolongation (p) and restriction (r) operators.
        // The matrices are moved into the local members.
        instance(cpu_matrix &a, cpu_matrix &p, cpu_matrix &r, const params &prm, unsigned nlevel)
            : A(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(),
                    a.rows, a.cols, a.row.data(), a.col.data(), a.val.data()),
              P(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(),
                      p.rows, p.cols, p.row.data(), p.col.data(), p.val.data()),
              R(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(),
                          r.rows, r.cols, r.row.data(), r.col.data(), r.val.data()),
              d(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              t(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              sum(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue())
        {
            vex::copy(diagonal(a), d);

            if (nlevel) {
                u.resize(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows);
                f.resize(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows);

                if (prm.kcycle && nlevel % prm.kcycle == 0)
                    cg.resize(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows);
            }

            a.clear();
            p.clear();
            r.clear();
        }

        // Construct the coarsest hierarchy level from system matrix (a) and
        // its inverse (ai).
        instance(cpu_matrix &a, cpu_matrix &ai, const params &prm, unsigned nlevel)
            : A(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(),
                        a.rows, a.cols, a.row.data(), a.col.data(), a.val.data()),
              Ainv(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(),
                          ai.rows, ai.cols, ai.row.data(), ai.col.data(), ai.val.data()),
              d(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              u(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              f(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              t(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue(), a.rows),
              sum(prm.ctx ? prm.ctx->queue() : vex::StaticContext<>::get().queue())
        {
            vex::copy(diagonal(a), d);

            a.clear();
            ai.clear();
        }

        // Returns reference to the system matrix
        const matrix& get_matrix() const {
            return A;
        }

        // Perform one relaxation (smoothing) step.
        void relax(const vector &rhs, vector &x, const params &prm) const {
            const index_t n = x.size();

            t = rhs - A * x;
            x += prm.relax_factor * t / d;
        }

        // Compute residual value.
        value_t resid(const vector &rhs, vector &x) const {
            t = rhs - A * x;

            return sqrt(sum(t * t));
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
                    for(unsigned i = 0; i < prm.npre; ++i) lvl->relax(rhs, x, prm);

                    lvl->t = rhs - lvl->A * x;
                    nxt->f = lvl->R * lvl->t;
                    nxt->u = 0;

                    if (nxt->cg.size())
                        kcycle(pnxt, end, prm, nxt->f, nxt->u);
                    else
                        cycle(pnxt, end, prm, nxt->f, nxt->u);

                    x += lvl->P * nxt->u;

                    for(unsigned i = 0; i < prm.npost; ++i) lvl->relax(rhs, x, prm);
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
            instance *nxt = pnxt->get();

            if (pnxt != end) {
                vex::vector<value_t> &r = lvl->cg(0);
                vex::vector<value_t> &s = lvl->cg(1);
                vex::vector<value_t> &p = lvl->cg(2);
                vex::vector<value_t> &q = lvl->cg(3);

                r = rhs;

                value_t rho1 = 0, rho2 = 0;

                for(int iter = 0; iter < 2; ++iter) {
                    s = 0;
                    cycle(plvl, end, prm, r, s);

                    rho2 = rho1;
                    rho1 = lvl->sum(r * s);

                    if (iter)
                        p = s + (rho1 / rho2) * p;
                    else
                        p = s;

                    q = lvl->A * p;

                    value_t alpha = rho1 / lvl->sum(q * p);

                    x += alpha * p;
                    r -= alpha * q;
                }
            } else {
                x = lvl->Ainv * rhs;
            }
        }

        index_t size() const {
            return A.rows();
        }

        index_t nonzeros() const {
            return A.nonzeros();
        }
    private:
        matrix A;
        matrix P;
        matrix R;
        matrix Ainv;

        vector d;

        vex::Reductor<value_t, vex::SUM> sum;

        mutable vector u;
        mutable vector f;
        mutable vector t;

        mutable vex::multivector<value_t,4> cg;
};

};

} // namespace level
} // namespace amgcl

#endif
