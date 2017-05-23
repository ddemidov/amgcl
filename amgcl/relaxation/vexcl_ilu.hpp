#ifndef AMGCL_RELAXATION_VEXCL_ILU_HPP
#define AMGCL_RELAXATION_VEXCL_ILU_HPP

/*
The MIT License

Copyright (c) 2012-2017 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/relaxation/vexcl_ilu.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  VexCL-spacific implementation of solver for sparse triangular
 *         systems obtained as a result of an incomplete LU factorization.
 */

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/relaxation/detail/ilu_solve.hpp>

namespace amgcl {
namespace relaxation {
namespace detail {

template <class value_type, class DS>
class ilu_solve< backend::vexcl<value_type, DS> > {
    public:
        typedef backend::vexcl<value_type, DS> Backend;
        typedef typename Backend::matrix matrix;
        typedef typename Backend::vector vector;
        typedef typename Backend::matrix_diagonal matrix_diagonal;
        typedef typename backend::builtin<value_type>::matrix build_matrix;
        typedef typename Backend::rhs_type rhs_type;
        typedef typename math::scalar_of<value_type>::type scalar_type;

        template <class Params>
        ilu_solve(
                boost::shared_ptr<build_matrix> L,
                boost::shared_ptr<build_matrix> U,
                boost::shared_ptr<backend::numa_vector<value_type> > D,
                const Params &, const typename Backend::params &bprm
                ) : lower(bprm.q, *L, D->data()), upper(bprm.q, *U, D->data())
        { }

        template <class Vector>
        void solve(Vector &x) {
            lower.solve(x);
            upper.solve(x);
        }

    private:
        template <bool lower>
        struct sptr_solve {
            static const int block_size = 64;

            // reordered matrix
            vex::vector<ptrdiff_t>  I; // original row number
            vex::vector<ptrdiff_t>  C; // col
            vex::vector<value_type> V; // val
            vex::vector<value_type> D; // dia

            // row intervals for each block:
            ptrdiff_t nblocks, pitch, max_width;
            vex::vector<ptrdiff_t> B;
            vex::vector<ptrdiff_t> E;

            // dependants of each block:
            vex::vector<int> deps;
            mutable vex::vector<int> tmp_deps;
            vex::vector<int> dep_ptr;
            vex::vector<int> dep_idx;

            vex::backend::command_queue q;

            template <class Matrix>
            sptr_solve(const std::vector<vex::backend::command_queue> &_q,
                    const Matrix &A, const value_type *_D = 0
                    ) : nblocks(0), q(_q[0])
            {
                precondition(_q.size() == 1, "ILU is only supported for single-device vexcl contexts");

                ptrdiff_t nlev = 0;
                ptrdiff_t n = A.nrows;

                pitch = vex::alignup(n, 16);
                max_width = 0;

                std::vector<ptrdiff_t> level(n, 0);
                std::vector<ptrdiff_t> order(n, 0);


                // 1. split rows into levels.
                AMGCL_TIC("color");
                {
                    ptrdiff_t beg = lower ? 0 : n-1;
                    ptrdiff_t end = lower ? n :  -1;
                    ptrdiff_t inc = lower ? 1 :  -1;

                    for(ptrdiff_t i = beg; i != end; i += inc) {
                        ptrdiff_t l = level[i];

                        for(ptrdiff_t j = A.ptr[i]; j < A.ptr[i+1]; ++j)
                            l = std::max(l, level[A.col[j]]+1);

                        level[i] = l;
                        nlev = std::max(nlev, l+1);
                        max_width = std::max(max_width, A.ptr[i+1] - A.ptr[i]);
                    }
                }
                AMGCL_TOC("color");


                // 2. reorder matrix rows, count number of blocks.
                AMGCL_TIC("sort rows");
                std::vector<ptrdiff_t> start(nlev+1, 0);

                for(ptrdiff_t i = 0; i < n; ++i)
                    ++start[level[i]+1];

                for(ptrdiff_t i = 0; i < nlev; ++i) {
                    nblocks += (start[i+1] + block_size - 1) / block_size;
                    start[i+1] += start[i];
                }

                for(ptrdiff_t i = 0; i < n; ++i)
                    order[start[level[i]]++] = i;

                std::rotate(start.begin(), start.end() - 1, start.end());
                start[0] = 0;
                AMGCL_TOC("sort rows");


                // 4. reorder matrix rows
                AMGCL_TIC("reorder matrix");
                std::vector<ptrdiff_t>  col(pitch * max_width, -1);
                std::vector<value_type> val(pitch * max_width);
                std::vector<value_type> dia; if (!lower) dia.resize(n);

                for(ptrdiff_t i = 0; i < n; ++i) {
                    ptrdiff_t r = order[i];
                    for(ptrdiff_t j = A.ptr[r], c = 0; j < A.ptr[r+1]; ++j, ++c) {
                        col[c * pitch + i] = A.col[j];
                        val[c * pitch + i] = A.val[j];
                    }

                    if (!lower) dia[i] = _D[r];
                }
                AMGCL_TOC("reorder matrix");

                // 4. organize matrix rows into blocks:
                //    each level is split into blocks,
                //    and each block is solved by a single workgroup.
                AMGCL_TIC("create blocks");
                std::vector<ptrdiff_t> block_beg; block_beg.reserve(nblocks+1); block_beg.push_back(0);
                std::vector<ptrdiff_t> block_id(n, -1);

                for(ptrdiff_t i = 0; i < nlev; ++i) {
                    ptrdiff_t lev_beg = start[i];
                    ptrdiff_t lev_end = start[i+1];
                    for(ptrdiff_t j = lev_beg; j < lev_end; j += block_size) {
                        ptrdiff_t id = block_beg.size()-1;
                        ptrdiff_t beg = j;
                        ptrdiff_t end = std::min(j+block_size, lev_end);
                        block_beg.push_back(end);

                        for(ptrdiff_t k = beg; k < end; ++k)
                            block_id[order[k]] = id;
                    }
                }
                AMGCL_TOC("create blocks");

                // 5. build dependency graph between tasks.
                AMGCL_TIC("dependency graph");
                AMGCL_TIC("create");
                std::vector<int> parent_ptr(nblocks + 1, 0);
                std::vector<int> marker(nblocks, -1);

                for(ptrdiff_t b = 0; b < nblocks; ++b) {
                    for(ptrdiff_t i = block_beg[b]; i < block_beg[b+1]; ++i) {
                        for(ptrdiff_t j = 0; j < max_width; ++j) {
                            ptrdiff_t c = col[j * pitch + i]; if (c < 0) continue;
                            ptrdiff_t id = block_id[c];
                            if (marker[id] != b) {
                                marker[id] = b;
                                ++parent_ptr[b+1];
                            }
                        }
                    }
                }

                std::partial_sum(parent_ptr.begin(), parent_ptr.end(), parent_ptr.begin());
                std::vector<int> parent(parent_ptr.back());
                std::fill(marker.begin(), marker.end(), -1);

                std::vector<int> nparents(nblocks);

                for(ptrdiff_t b = 0; b < nblocks; ++b) {
                    ptrdiff_t h = parent_ptr[b];
                    nparents[b] = parent_ptr[b+1] - parent_ptr[b];

                    for(ptrdiff_t i = block_beg[b]; i < block_beg[b+1]; ++i) {
                        for(ptrdiff_t j = 0; j < max_width; ++j) {
                            ptrdiff_t c = col[j * pitch + i]; if (c < 0) continue;
                            ptrdiff_t id = block_id[c];
                            if (marker[id] != b) {
                                marker[id] = b;
                                parent[h++] = id;
                            }
                        }
                    }
                }
                AMGCL_TOC("create");

                AMGCL_TIC("transpose");
                std::vector<int> child_ptr(nblocks+1,0);

                for(ptrdiff_t i = 0; i < nblocks; ++i) {
                    for(ptrdiff_t j = parent_ptr[i]; j < parent_ptr[i+1]; ++j) {
                        ++child_ptr[parent[j]+1];
                    }
                }

                std::partial_sum(child_ptr.begin(), child_ptr.end(), child_ptr.begin());
                std::vector<int> children(child_ptr.back());
                for(ptrdiff_t i = 0; i < nblocks; ++i) {
                    for(ptrdiff_t j = parent_ptr[i]; j < parent_ptr[i+1]; ++j) {
                        children[child_ptr[parent[j]]++]=i;
                    }
                }
                std::rotate(child_ptr.begin(), child_ptr.end()-1, child_ptr.end());
                child_ptr[0] = 0;
                AMGCL_TOC("transpose");
                AMGCL_TOC("dependency graph");

                B.resize(_q, block_beg);
                I.resize(_q, order);
                C.resize(_q, col);
                V.resize(_q, val);

                if (!lower) D.resize(_q, dia);

                deps.resize(_q, nparents);
                tmp_deps.resize(_q, nblocks);

                dep_ptr.resize(_q, child_ptr);
                dep_idx.resize(_q, children);
            }

            template <class Vector>
            void solve(Vector &x) const {
                tmp_deps = deps;

                auto K = solve_kernel(q);

                K.push_arg(nblocks);
                K.push_arg(pitch);
                K.push_arg(max_width);
                K.push_arg(B(0));
                K.push_arg(I(0));
                K.push_arg(C(0));
                K.push_arg(V(0));
                if (!lower) K.push_arg(D(0));
                K.push_arg(tmp_deps(0));
                K.push_arg(dep_ptr(0));
                K.push_arg(dep_idx(0));
                K.push_arg(x(0));

                K.config(nblocks, block_size)(q);
            }

            static vex::backend::kernel& solve_kernel(const vex::backend::command_queue &q) {
                using namespace vex;
                using namespace vex::detail;

                static kernel_cache cache;

                auto kernel = cache.find(q);
                if (kernel == cache.end()) {
                    vex::backend::source_generator src(q);

                    src.begin_kernel("sptr_solve");
                    src.begin_kernel_parameters();
                    src.template parameter<ptrdiff_t>("nblocks");
                    src.template parameter<ptrdiff_t>("pitch");
                    src.template parameter<ptrdiff_t>("width");
                    src.template parameter< global_ptr<ptrdiff_t> >("B");
                    src.template parameter< global_ptr<ptrdiff_t> >("I");
                    src.template parameter< global_ptr<ptrdiff_t> >("C");
                    src.template parameter< global_ptr<value_type> >("V");
                    if (!lower)
                        src.template parameter< global_ptr<value_type> >("D");
                    src.template parameter< global_ptr<int> >("deps");
                    src.template parameter< global_ptr<int> >("dep_ptr");
                    src.template parameter< global_ptr<int> >("dep_idx");
                    src.template parameter< global_ptr<value_type> >("x");
                    src.end_kernel_parameters();

                    src.new_line() << "volatile " << type_name<global_ptr<int>>() << " d = deps;";
                    src.new_line() << "int t_id = " << src.local_id(0) << ";";
                    src.new_line() << "for(int block_id = " << src.group_id(0) << "; block_id < nblocks; block_id += " << src.num_groups(0) << ")";
                    src.open("{");
                    src.new_line() << "while(d[block_id] > 0);";
                    src.new_line() << "int i = B[block_id] + t_id;";
                    src.new_line() << "if (i < B[block_id+1])";
                    src.open("{");
                    src.new_line() << type_name<rhs_type>() << " s = 0;";
                    src.new_line() << "for(" << type_name<ptrdiff_t>() << " j = 0; j < width; ++j)";
                    src.open("{");
                    src.new_line() << type_name<ptrdiff_t>() << " c = C[j*pitch+i]; if (c < 0) continue;";
                    src.new_line() << "s += V[j*pitch+i] * x[c];";
                    src.close("}");
                    if (lower)
                        src.new_line() << "x[I[i]] -= s;";
                    else
                        src.new_line() << "x[I[i]] = D[i] * (x[I[i]] - s);";
                    src.close("}");
                    src.new_line() << "for(int j = dep_ptr[block_id] + t_id; j < dep_ptr[block_id+1]; j += " << src.local_size(0) << ")";
                    src.open("{");
#ifdef VEXCL_BACKEND_CUDA
                    src.new_line() << "atomicSub(deps + dep_idx[j], 1);";
#else
                    src.new_line() << "atomic_dec(deps + dep_idx[j]);";
#endif
                    src.close("}");
                    src.close("}");
                    src.end_kernel();

                    kernel = cache.insert(q, vex::backend::kernel(q, src.str(), "sptr_solve"));
                }

                return kernel->second;
            }
        };

        sptr_solve<true>  lower;
        sptr_solve<false> upper;
};

} // namespace detail
} // namespace relaxation
} // namespace amgcl

#endif
