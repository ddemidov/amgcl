#ifndef AMGCL_BACKEND_VEXCL_HPP
#define AMGCL_BACKEND_VEXCL_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/backend/vexcl.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  VexCL backend.
 */

#include <iostream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <amgcl/solver/skyline_lu.hpp>
#include <vexcl/vexcl.hpp>
#include <vexcl/sparse/matrix.hpp>
#include <vexcl/sparse/distributed.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {

template <class V, class C, class P>
using vex_SpMat = vex::sparse::ell<V, C, P>;

namespace solver {

/** Wrapper around solver::skyline_lu for use with the VexCL backend.
 * Copies the rhs to the host memory, solves the problem using the host CPU,
 * then copies the solution back to the compute device(s).
 */
template <class value_type>
struct vexcl_skyline_lu : solver::skyline_lu<value_type> {
    typedef solver::skyline_lu<value_type> Base;
    typedef typename math::rhs_of<value_type>::type rhs_type;

    mutable std::vector<rhs_type> _rhs, _x;

    template <class Matrix, class Params>
    vexcl_skyline_lu(const Matrix &A, const Params&)
        : Base(*A), _rhs(backend::rows(*A)), _x(backend::rows(*A))
    { }

    template <class Vec1, class Vec2>
    void operator()(const Vec1 &rhs, Vec2 &x) const {
        vex::copy(rhs, _rhs);
        static_cast<const Base*>(this)->operator()(_rhs, _x);
        vex::copy(_x, x);
    }
};

}

namespace backend {

/**
 * The backend uses the <a href="https://github.com/ddemidov/vexcl">VexCL</a>
 * library for accelerating solution on the modern GPUs and multicore
 * processors with the help of OpenCL or CUDA technologies.
 * The VexCL backend stores the system matrix as ``vex_SpMat<real>`` and
 * expects the right hand side and the solution vectors to be instances of the
 * ``vex::vector<real>`` type.
 */
template <typename real, class DirectSolver = solver::vexcl_skyline_lu<real> >
struct vexcl {
    typedef real      value_type;
    typedef ptrdiff_t index_type;

    typedef vex_SpMat<value_type, index_type, index_type> matrix;
    typedef typename math::rhs_of<value_type>::type rhs_type;
    typedef vex::vector<rhs_type>                          vector;
    typedef vex::vector<value_type>                        matrix_diagonal;
    typedef DirectSolver                                   direct_solver;

    struct provides_row_iterator : boost::false_type {};

    /// The VexCL backend parameters.
    struct params {

        std::vector< vex::backend::command_queue > q; ///< Command queues that identify compute devices to use with VexCL.

        /// Do CSR to ELL conversion on the GPU side.
        /** This will result in faster setup, but will require more GPU memory. */
        bool fast_matrix_setup;

        params() : fast_matrix_setup(true) {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_CHILD(p, fast_matrix_setup)
        {
            std::vector<vex::backend::command_queue> *ptr = 0;
            ptr = p.get("q", ptr);
            if (ptr) q = *ptr;
            AMGCL_PARAMS_CHECK(p, (q)(fast_matrix_setup));
        }

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            p.put(path + "q", &q);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, fast_matrix_setup);
        }

        const std::vector<vex::backend::command_queue>& context() const {
            if (q.empty())
                return vex::current_context().queue();
            else
                return q;

        }
    };

    static std::string name() { return "vexcl"; }

    // Copy matrix from builtin backend.
    static boost::shared_ptr<matrix>
    copy_matrix(boost::shared_ptr< typename builtin<real>::matrix > A, const params &prm)
    {
        precondition(!prm.context().empty(), "Empty VexCL context!");
        return boost::make_shared<matrix>(prm.context(), rows(*A), cols(*A), A->ptr, A->col, A->val, prm.fast_matrix_setup);
    }

    // Copy vector from builtin backend.
    template <class T>
    static boost::shared_ptr< vex::vector<T> >
    copy_vector(const std::vector<T> &x, const params &prm)
    {
        precondition(!prm.context().empty(), "Empty VexCL context!");
        return boost::make_shared< vex::vector<T> >(prm.context(), x);
    }

    // Copy vector from builtin backend.
    template <class T>
    static boost::shared_ptr< vex::vector<T> >
    copy_vector(boost::shared_ptr< std::vector<T> > x, const params &prm)
    {
        return copy_vector(*x, prm);
    }

    // Create vector of the specified size.
    static boost::shared_ptr<vector>
    create_vector(size_t size, const params &prm)
    {
        precondition(!prm.context().empty(), "Empty VexCL context!");

        return boost::make_shared<vector>(prm.context(), size);
    }

    struct gather {
        mutable vex::gather<value_type> G;
        mutable std::vector<value_type> tmp;

        gather(size_t src_size, const std::vector<ptrdiff_t> &I, const params &prm)
            : G(prm.context(), src_size, std::vector<size_t>(I.begin(), I.end())) { }

        void operator()(const vector &src, vector &dst) const {
            G(src, tmp);
            vex::copy(tmp, dst);
        }

        void operator()(const vector &vec, std::vector<value_type> &vals) const {
            G(vec, vals);
        }
    };

    struct scatter {
        mutable vex::scatter<value_type> S;
        mutable std::vector<value_type> tmp;

        scatter(size_t size, const std::vector<ptrdiff_t> &I, const params &prm)
            : S(prm.context(), size, std::vector<size_t>(I.begin(), I.end()))
            , tmp(I.size())
        { }

        void operator()(const vector &src, vector &dst) const {
            vex::copy(src, tmp);
            S(tmp, dst);
        }
    };


    // Create direct solver for coarse level
    static boost::shared_ptr<direct_solver>
    create_solver(boost::shared_ptr< typename builtin<real>::matrix > A, const params &prm)
    {
        return boost::make_shared<direct_solver>(A, prm);
    }
};

//---------------------------------------------------------------------------
// Backend interface implementation
//---------------------------------------------------------------------------
template < typename V, typename C, typename P >
struct rows_impl< vex_SpMat<V, C, P> > {
    static size_t get(const vex_SpMat<V, C, P> &A) {
        return A.rows();
    }
};

template < typename V, typename C, typename P >
struct cols_impl< vex_SpMat<V, C, P> > {
    static size_t get(const vex_SpMat<V, C, P> &A) {
        return A.cols();
    }
};

template < typename V, typename C, typename P >
struct nonzeros_impl< vex_SpMat<V, C, P> > {
    static size_t get(const vex_SpMat<V, C, P> &A) {
        return A.nonzeros();
    }
};


template <int B>
vex::backend::kernel& blocked_spmv_kernel2(const vex::backend::command_queue &q) {
    using namespace vex;
    using namespace vex::detail;
    static kernel_cache cache;

    auto K = cache.find(q);
    if (K == cache.end()) {
        vex::backend::source_generator src(q);

         /* The following kernel uses B threads for each B*B block -> more continguous memory reads for A
         * Performances are from a Tesla C2070.
         */
       src.kernel("blocked_spmv2").open("(")
            .template parameter<int>("N")
            .template parameter<int>("ell_width")
            .template parameter<int>("ell_pitch")
            .template parameter< global_ptr<const long> >("ell_col")
            .template parameter< global_ptr<const double> >("ell_val")
            .template parameter< global_ptr<const double> >("x")
            .template parameter< global_ptr<double> >("y")
            .close(")").open("{");

        src.new_line() << " size_t global_id   = " << src.global_id(0) << ";";
        src.new_line() << " size_t global_size = " << src.global_size(0) << ";";
 
        src.new_line() << " #define subwarp_size " << B;
        src.new_line() << " const size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
        src.new_line() << " const size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";

        src.new_line().smem_static_var("double", "row_A[256*subwarp_size]");
#ifdef VEXCL_BACKEND_OPENCL
        src.new_line().smem_static_var("double", "*my_A = row_A + subwarp_gid * subwarp_size * subwarp_size");
#else
        src.new_line() << " double *my_A = row_A + subwarp_gid * subwarp_size * subwarp_size;";
#endif
        src.new_line() << " double my_x, my_y;";

        src.new_line() << " size_t loop_iters = (N-1) / (global_size / subwarp_size) + 1;";

        src.new_line() << " for (size_t iter = 0; iter < loop_iters; ++iter)";
        src.open("{");
        src.new_line() << "   size_t row = (global_id + iter * global_size) / subwarp_size;";
        src.new_line() << "   my_y = 0;";
        src.new_line() << "   size_t offset = min((int)row, (int)N-1);";
        src.new_line() << "   for (size_t i = 0; i < ell_width; ++i, offset += ell_pitch) {";
        src.new_line() << "     int c = ell_col[offset];";

        src.new_line() << "     size_t ell_val_offset = subwarp_size * subwarp_size * offset + subwarp_idx;";
        src.new_line() << "     my_x = (c >= 0) ? x[subwarp_size * c + subwarp_idx] : 0.0;";
        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k) ";
        src.new_line() << "       my_A[k * subwarp_size + subwarp_idx] = (c >= 0) ? ell_val[ell_val_offset + k * subwarp_size] * my_x : 0.0;";
        src.new_line().barrier();

        src.new_line() << "     for (size_t k=0; k<subwarp_size; ++k)";
        src.new_line() << "       my_y += my_A[subwarp_idx * subwarp_size + k];";
        src.new_line() << "   }";

        src.new_line() << "   if (row < N)";
        src.new_line() << "     y[subwarp_size*row+subwarp_idx] = my_y;";
        src.close("}"); // for

        
        src.close("}"); // kernel

        K = cache.insert(q, vex::backend::kernel(q, src.str(), "blocked_spmv2"));
        K->second.config(256, 256);
    }

    return K->second;
}


template < typename Alpha, typename Beta, typename VA, typename C, typename P, typename VX >
struct spmv_impl<
    Alpha, vex_SpMat<VA, C, P>, vex::vector<VX>,
    Beta,  vex::vector<VX>
    >
{
    typedef vex_SpMat<VA, C, P> matrix;
    typedef vex::vector<VX>      vector;

    static void apply(Alpha alpha, const matrix &A, const vector &x,
            Beta beta, vector &y)
    {
        if (!math::is_zero(beta))
            y = alpha * (A * x) + beta * y;
        else if (alpha == 1)
        {
            auto &K = blocked_spmv_kernel2<amgcl::math::static_rows<VA>::value>(x.queue_list()[0]);
            K(x.queue_list()[0], (int)y.size(), (int)A.ell_width, (int)A.ell_pitch, A.ell_col, A.ell_val, x(0), y(0));
        } else {
            apply(1.0, A, x, beta, y);
            y *= alpha;
        }
    }
};

template < typename VA, typename C, typename P, typename VX >
struct residual_impl<
    vex_SpMat<VA, C, P>,
    vex::vector<VX>,
    vex::vector<VX>,
    vex::vector<VX>
    >
{
    typedef vex_SpMat<VA, C, P> matrix;
    typedef vex::vector<VX>      vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x,
            vector &r)
    {
        spmv(1.0, A, x, 0.0, r);
        r = rhs - r;
    }
};

template < typename V >
struct clear_impl< vex::vector<V> >
{
    static void apply(vex::vector<V> &x)
    {
        x = V();
    }
};

template < typename V >
struct copy_impl<
    vex::vector<V>,
    vex::vector<V>
    >
{
    static void apply(const vex::vector<V> &x, vex::vector<V> &y)
    {
        y = x;
    }
};

template < typename V >
struct copy_to_backend_impl<
    vex::vector<V>
    >
{
    static void apply(const std::vector<V> &data, vex::vector<V> &x)
    {
        vex::copy(data, x);
    }
};

template < typename V >
struct inner_product_impl<
    vex::vector<V>,
    vex::vector<V>
    >
{
    typedef typename math::inner_product_impl<V>::return_type return_type;

    static return_type get(const vex::vector<V> &x, const vex::vector<V> &y)
    {
        vex::Reductor<return_type, vex::SUM_Kahan> sum( x.queue_list() );
        return sum(x * y);
    }
};

template < typename A, typename B, typename V >
struct axpby_impl<
    A, vex::vector<V>,
    B, vex::vector<V>
    > {
    static void apply(A a, const vex::vector<V> &x, B b, vex::vector<V> &y)
    {
        if (!math::is_zero(b))
            y = a * x + b * y;
        else
            y = a * x;
    }
};

template < typename A, typename B, typename C, typename V >
struct axpbypcz_impl<
    A, vex::vector<V>,
    B, vex::vector<V>,
    C, vex::vector<V>
    >
{
    static void apply(
            A a, const vex::vector<V> &x,
            B b, const vex::vector<V> &y,
            C c,       vex::vector<V> &z
            )
    {
        if (!math::is_zero(c))
            z = a * x + b * y + c * z;
        else
            z = a * x + b * y;
    }
};

template < typename A, typename B, typename V1, typename V2 >
struct vmul_impl<
    A, vex::vector<V1>, vex::vector<V2>,
    B, vex::vector<V2>
    >
{
    static void apply(A a, const vex::vector<V1> &x, const vex::vector<V2> &y,
            B b, vex::vector<V2> &z)
    {
        if (!math::is_zero(b))
            z = a * x * y + b * z;
        else
            z = a * x * y;
    }
};

} // namespace backend
} // namespace amgcl

#endif
