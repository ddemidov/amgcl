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
#include <vexcl/sparse/csr.hpp>
#include <vexcl/sparse/ell.hpp>

#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>

namespace amgcl {

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
 * The VexCL backend stores the system matrix as ``vex::SpMat<real>`` and
 * expects the right hand side and the solution vectors to be instances of the
 * ``vex::vector<real>`` type.
 */
template <typename real, class DirectSolver = solver::vexcl_skyline_lu<real> >
struct vexcl {
    typedef real      value_type;
    typedef ptrdiff_t index_type;

    typedef typename math::rhs_of<value_type>::type rhs_type;

    typedef vex::sparse::ell<value_type, index_type, index_type> matrix;
    typedef vex::vector<rhs_type>                          vector;
    typedef vex::vector<value_type>                        matrix_diagonal;
    typedef DirectSolver                                   direct_solver;

    struct provides_row_iterator : boost::false_type {};

    /// The VexCL backend parameters.
    struct params {

        std::vector< vex::backend::command_queue > q; ///< Command queues that identify compute devices to use with VexCL.

        params() {}

        params(const boost::property_tree::ptree &p) {
            std::vector<vex::backend::command_queue> *ptr = 0;
            ptr = p.get("q", ptr);
            if (ptr) q = *ptr;
            AMGCL_PARAMS_CHECK(p, (q));
        }

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            p.put(path + "q", &q);
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

        const typename builtin<real>::matrix &a = *A;

        BOOST_AUTO(Aptr, a.ptr_data());
        BOOST_AUTO(Acol, a.col_data());
        BOOST_AUTO(Aval, a.val_data());

        const size_t n   = backend::rows(a);
        const size_t nnz = backend::nonzeros(a);

        return boost::make_shared<matrix>(prm.context(),
                boost::make_iterator_range(Aptr, Aptr + n+1),
                boost::make_iterator_range(Acol, Acol + nnz),
                boost::make_iterator_range(Aval, Aval + nnz)
                );
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
struct rows_impl< vex::sparse::ell<V, C, P> > {
    static size_t get(const vex::sparse::ell<V, C, P> &A) {
        return A.rows();
    }
};

template < typename V, typename C, typename P >
struct cols_impl< vex::sparse::ell<V, C, P> > {
    static size_t get(const vex::sparse::ell<V, C, P> &A) {
        return A.cols();
    }
};

template < typename V, typename C, typename P >
struct nonzeros_impl< vex::sparse::ell<V, C, P> > {
    static size_t get(const vex::sparse::ell<V, C, P> &A) {
        return A.nonzeros();
    }
};

template < typename Alpha, typename Beta, typename VA, typename C, typename P, typename VX >
struct spmv_impl<
    Alpha, vex::sparse::ell<VA, C, P>, vex::vector<VX>,
    Beta,  vex::vector<VX>
    >
{
    typedef vex::sparse::ell<VA, C, P> matrix;
    typedef vex::vector<VX>      vector;

    static void apply(Alpha alpha, const matrix &A, const vector &x,
            Beta beta, vector &y)
    {
        if (!math::is_zero(beta))
            y = alpha * (A * x) + beta * y;
        else
            y = alpha * (A * x);
    }
};

template < typename VA, typename C, typename P, typename VX >
struct residual_impl<
    vex::sparse::ell<VA, C, P>,
    vex::vector<VX>,
    vex::vector<VX>,
    vex::vector<VX>
    >
{
    typedef vex::sparse::ell<VA, C, P> matrix;
    typedef vex::vector<VX>      vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x,
            vector &r)
    {
        r = rhs - A * x;
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
