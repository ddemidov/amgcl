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

#include <amgcl/spmat.hpp>
#include <vexcl/vexcl.hpp>

namespace amg {
namespace level {

// CPU-based AMG hierarchy. vex::StaticContext is used for the VexCL types
// construction.
struct vexcl {

template <typename value_t, typename index_t = long long>
class instance {
    public:
        typedef sparse::matrix<value_t, index_t>      cpu_matrix;
        typedef vex::SpMat<value_t, index_t, index_t> matrix;
        typedef vex::vector<value_t>                  vector;

        // Construct complete multigrid level from system matrix (a),
        // prolongation (p) and restriction (r) operators.
        // The matrices are moved into the local members.
        instance(cpu_matrix &&a, cpu_matrix &&p, cpu_matrix &&r, bool parent = true)
            : A(new matrix(vex::StaticContext<>::get().queue(), a.rows, a.cols, a.row.data(), a.col.data(), a.val.data())),
              P(new matrix(vex::StaticContext<>::get().queue(), p.rows, p.cols, p.row.data(), p.col.data(), p.val.data())),
              R(new matrix(vex::StaticContext<>::get().queue(), r.rows, r.cols, r.row.data(), r.col.data(), r.val.data())),
              t(a.rows), d(a.rows)
        {
            if (parent) {
                u.resize(a.rows);
                f.resize(a.rows);
            }

            extract_diagonal(a);

            a.clear();
            p.clear();
            r.clear();
        }

        // Construct the coarsest hierarchy level from system matrix (a) and
        // its inverse (ai).
        instance(cpu_matrix &&a, cpu_matrix &&ai)
            : A(new matrix(vex::StaticContext<>::get().queue(), a.rows, a.cols, a.row.data(), a.col.data(), a.val.data())),
              Ainv(new matrix(vex::StaticContext<>::get().queue(), ai.rows, ai.cols, ai.row.data(), ai.col.data(), ai.val.data())),
              u(a.rows), f(a.rows), t(a.rows), d(a.rows)
        {
            extract_diagonal(a);

            a.clear();
            ai.clear();
        }

        // Perform one relaxation (smoothing) step.
        void relax(const vector &rhs, vector &x) {
            const index_t n = x.size();

            t = rhs - (*A) * x;
            x += 0.72 * (t / d);
        }

        // Compute residual value.
        value_t resid(const vector &rhs, vector &x) const {
	    static vex::Reductor<value_t, vex::SUM> sum(
		    vex::StaticContext<>::get().queue()
		    );

            t = rhs - (*A) * x;

            return sqrt(sum(t * t));
        }

        // Perform one V-cycle. Coarser levels are cycled recursively. The
        // coarsest level is solved directly.
        template <class Iterator>
        static void cycle(Iterator lvl, Iterator end, const amg::params &prm,
                const vector &rhs, vector &x)
        {
            Iterator nxt = lvl; ++nxt;

            if (nxt != end) {
                for(unsigned j = 0; j < prm.ncycle; ++j) {
                    for(unsigned i = 0; i < prm.npre; ++i) lvl->relax(rhs, x);

                    lvl->t = rhs - (*lvl->A) * x;
                    nxt->f = (*lvl->R) * lvl->t;
                    nxt->u = 0;

                    cycle(nxt, end, prm, nxt->f, nxt->u);

                    x += (*lvl->P) * nxt->u;

                    for(unsigned i = 0; i < prm.npost; ++i) lvl->relax(rhs, x);
                }
            } else {
                x = (*lvl->Ainv) * rhs;
            }
        }
    private:
        std::unique_ptr<matrix> A;
        std::unique_ptr<matrix> P;
        std::unique_ptr<matrix> R;
        std::unique_ptr<matrix> Ainv;

        vector u;
        vector f;
        vector t;

        vector d;

        void extract_diagonal(const cpu_matrix &a) {
            std::vector<value_t> diag(a.rows);

            for(index_t i = 0; i < a.rows; ++i) {
                for(index_t j = a.row[i], e = a.row[i + 1]; j < e; ++j) {
                    if (a.col[j] == i) {
                        diag[i] = a.val[j];
                        break;
                    }
                }
            }

            vex::copy(diag, d);
        }
};

};

} // namespace level
} // namespace amg

#endif
