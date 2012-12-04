#ifndef AMGCL_AMGCL_HPP
#define AMGCL_AMGCL_HPP

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

#include <iostream>
#include <iomanip>
#include <list>
#include <memory>

#include <amgcl/spmat.hpp>
#include <amgcl/params.hpp>
#include <amgcl/interp.hpp>
#include <amgcl/cpu_level.hpp>
#include <amgcl/profiler.hpp>

namespace amg {

#ifdef AMGCL_PROFILING
extern amg::profiler<> prof;
#  define TIC(what) prof.tic(what);
#  define TOC(what) prof.toc(what);
#else
#  define TIC(what)
#  define TOC(what)
#endif

// Algebraic multigrid method. The hierarchy by default is built for a CPU. The
// other possibility is VexCL-based representation
// ( level_t = level::vexcl<value_t, index_t> ).
template <
    typename value_t,
    typename index_t = long long,
    class interp_t = interp::classic,   // Interpolation scheme.
    class level_t = level::cpu          // Where to build the hierarchy
    >
class solver {
    public:
        typedef sparse::matrix<value_t, index_t> matrix;
        typedef typename level_t::template instance<value_t, index_t> level_type;

        // The input matrix is copied here and may be freed afterwards.
        solver(matrix A, const params &prm = params()) : prm(prm)
        {
            build_level(std::move(A), prm);
        }

        // Use the AMG hierarchy as a standalone solver. The vector types should
        // be compatible with level_type.
        // 1. Any type with operator[] should work on a CPU.
        // 2. vex::vector<value_t> should be used with VexCL-based hierarchy.
        template <class vector1, class vector2>
        bool solve(const vector1 &rhs, vector2 &x) {
            for(size_t iter = 0; iter < prm.maxiter; iter++) {
                cycle(rhs, x);

                value_t r = hier.front().resid(rhs, x);

                if (r <= prm.tol) return true;
            }
            return false;
        }

        // Perform 1 V-cycle. May be used as a preconditioning step.
        template <class vector1, class vector2>
        void cycle(const vector1 &rhs, vector2 &x) {
            level_type::cycle(hier.begin(), hier.end(), prm, rhs, x);
        }

        template <class vector1, class vector2>
        void apply(const vector1 &rhs, vector2 &x) {
            cycle(rhs, x);
        }

    private:
        void build_level(matrix &&A, const params &prm, bool parent = false)
        {
            if (A.rows <= prm.coarse_enough) {
                hier.emplace_back(std::move(A), std::move(sparse::inverse(A)));
            } else {
                TIC("interp");
                matrix P = interp_t::interp(A, prm);
                TOC("interp");

                TIC("transp");
                matrix R = sparse::transpose(P);
                TOC("transp");

                TIC("prod");
                matrix a = sparse::prod(sparse::prod(R, A), P);
                TOC("prod");

                hier.emplace_back(std::move(A), std::move(P), std::move(R), parent);

                build_level(std::move(a), prm, true);
            }
        }

        params prm;
        std::list< level_type > hier;
};

} // namespace amg

#endif
