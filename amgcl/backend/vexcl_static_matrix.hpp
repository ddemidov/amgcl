#ifndef AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP
#define AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP

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
 * \file   amgcl/backend/vexcl_static_matrix.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Static matrix support for the VexCL backend.
 */

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/value_type/static_matrix.hpp>

namespace vex {

template <typename T, int N, int M>
struct is_cl_native< amgcl::static_matrix<T, N, M> > : std::true_type {};

template <typename T, int N, int M>
struct type_name_impl< amgcl::static_matrix<T, N, M> >
{
    static std::string get() {
        std::ostringstream s;
        s << "amgcl_matrix_" << type_name<T>() << "_" << N << "x" << M;
        return s.str();
    }
};

template <typename T, int N, int M>
struct cl_scalar_of< amgcl::static_matrix<T, N, M> > {
    typedef T type;
};

namespace sparse {

template <typename T, int N>
struct rhs_of< amgcl::static_matrix<T, N, N> > {
    typedef amgcl::static_matrix<T, N, 1> type;
};

template <typename T, int N>
struct spmv_ops_impl<amgcl::static_matrix<T,N,N>, amgcl::static_matrix<T,N,1>> {
    typedef amgcl::static_matrix<T,N,N> matrix_value;
    typedef amgcl::static_matrix<T,N,1> vector_value;

    static void decl_accum_var(backend::source_generator &src, const std::string &name)
    {
        src.new_line() << type_name<vector_value>() << " " << name << ";";
        for(int i = 0; i < N; ++i) {
            src.new_line() << name << ".data[" << i << "][0] = 0;";
        }
    }

    static void append(backend::source_generator &src,
            const std::string &sum, const std::string &val)
    {
        for(int i = 0; i < N; ++i)
            src.new_line() << sum << ".data[" << i << "][0] += " << val << ".data[" << i << "][0];";
    }

    static void append_product(backend::source_generator &src,
            const std::string &sum, const std::string &mat_val, const std::string &vec_val)
    {
        src.open("{");
        src.new_line() << type_name<vector_value>() << " v = " << vec_val << ";";
        for(int i = 0; i < N; ++i) {
            src.new_line() << sum << ".data[" << i << "][0] += ";
            for(int j = 0; j < N; ++j) {
                if (j) src << " + ";
                src << mat_val << ".data[" << i << "][" << j << "] * v.data[" << j << "][0]";
            }
            src << ";";
        }
        src.close("}");
    }
};

} // namespace sparse
} // namespace vex

namespace amgcl {
namespace backend {

template <typename T, int N>
std::string vexcl_static_matrix_declaration() {
    std::ostringstream s;
    s << "typedef struct { " << vex::type_name<T>() << " data[" << N << "][" << N << "]; } "
         "amgcl_matrix_" << vex::type_name<T>() << "_" << N << "x" << N << ";\n";
    if (N != 1)
    s << "typedef struct { " << vex::type_name<T>() << " data[" << N << "][" << 1 << "]; } "
         "amgcl_matrix_" << vex::type_name<T>() << "_" << N << "x" << 1 << ";\n";
    return s.str();
}

template <typename T, int N>
struct vex_scale {
    typedef static_matrix<T,N,1> vector;

    struct apply_type : vex::UserFunction<apply_type, vector(T, vector)> {
        apply_type() {}

        static std::string name() {
            return "scale_" + vex::type_name<vector>();
        }

        static void define(vex::backend::source_generator &src, const std::string &name = name()) {
            src.begin_function<vector>(name);
            src.begin_function_parameters();
            src.parameter<T>("a");
            src.parameter<vector>("m");
            src.end_function_parameters();
            for(int i = 0; i < N; ++i)
                src.new_line() << "m.data[" << i << "][0] *= a;";
            src.new_line() << "return m;";
            src.end_function();
        }
    } const apply;
};

template <typename T, int N>
struct vex_add {
    typedef static_matrix<T,N,1> vector;

    struct apply_type : vex::UserFunction<apply_type, vector(vector, vector)> {
        apply_type() {}

        static std::string name() {
            return "add_" + vex::type_name<vector>();
        }

        static void define(vex::backend::source_generator &src, const std::string &name = name()) {
            src.begin_function<vector>(name);
            src.begin_function_parameters();
            src.parameter<vector>("a");
            src.parameter<vector>("b");
            src.end_function_parameters();
            for(int i = 0; i < N; ++i)
                src.new_line() << "a.data[" << i << "][0] += "
                               << "b.data[" << i << "][0];";
            src.new_line() << "return a;";
            src.end_function();
        }
    } const apply;
};

template <typename T, int N>
struct vex_sub {
    typedef static_matrix<T,N,1> vector;

    struct apply_type : vex::UserFunction<apply_type, vector(vector, vector)> {
        apply_type() {}

        static std::string name() {
            return "sub_" + vex::type_name<vector>();
        }

        static void define(vex::backend::source_generator &src, const std::string &name = name()) {
            src.begin_function<vector>(name);
            src.begin_function_parameters();
            src.parameter<vector>("a");
            src.parameter<vector>("b");
            src.end_function_parameters();
            for(int i = 0; i < N; ++i)
                src.new_line() << "a.data[" << i << "][0] -= "
                               << "b.data[" << i << "][0];";
            src.new_line() << "return a;";
            src.end_function();
        }
    } const apply;
};

template <typename T, int N>
struct vex_mul {
    typedef static_matrix<T,N,N> matrix;
    typedef static_matrix<T,N,1> vector;

    struct apply_type : vex::UserFunction<apply_type, vector(matrix, vector)> {
        apply_type() {}

        static std::string name() {
            return "mul_" + vex::type_name<matrix>();
        }

        static void define(vex::backend::source_generator &src, const std::string &name = name()) {
            src.begin_function<vector>(name);
            src.begin_function_parameters();
            src.parameter<matrix>("a");
            src.parameter<vector>("b");
            src.end_function_parameters();
            src.new_line() << vex::type_name<vector>() << " c;";
            for(int i = 0; i < N; ++i) {
                src.new_line() << "c.data[" << i << "][0] = ";
                for(int j = 0; j < N; ++j) {
                    if (j) src << " + ";
                    src << "a.data[" << i << "][" << j << "] * b.data[" << j << "][0]";
                }
                src << ";";
            }
            src.new_line() << "return c;";
            src.end_function();
        }
    } const apply;
};

template <typename Alpha, typename Beta, typename T, int B>
struct spmv_impl<Alpha,
    vex::sparse::distributed<vex::sparse::matrix<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>>,
    vex::vector<static_matrix<T,B,1>>, Beta, vex::vector<static_matrix<T,B,1>>>
{
    typedef vex::sparse::distributed<vex::sparse::matrix<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>> matrix;
    typedef vex::vector<static_matrix<T,B,1>> vector;

    static void apply(Alpha alpha, const matrix &A, const vector &x, Beta beta, vector &y)
    {
        if (beta)
            y = vex_add<T,B>().apply(vex_scale<T,B>().apply(alpha, A * x), vex_scale<T,B>().apply(beta, y));
        else
            y = vex_scale<T,B>().apply(alpha, A * x);
    }
};

template <typename T, int B>
struct residual_impl<
    vex::sparse::distributed<vex::sparse::matrix<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>>,
    vex::vector<static_matrix<T,B,1>>,
    vex::vector<static_matrix<T,B,1>>,
    vex::vector<static_matrix<T,B,1>>
    >
{
    typedef vex::sparse::distributed<vex::sparse::matrix<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>> matrix;
    typedef vex::vector<static_matrix<T,B,1>> vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x, vector &r)
    {
        r = vex_sub<T,B>().apply(rhs, A * x);
    }
};

template < typename Alpha, typename Beta, typename T, int B >
struct vmul_impl<
    Alpha, vex::vector< static_matrix<T,B,B> >,
    vex::vector< static_matrix<T,B,1> >,
    Beta, vex::vector< static_matrix<T,B,1> >
    >
{
    typedef vex::vector< static_matrix<T,B,B> > matrix;
    typedef vex::vector< static_matrix<T,B,1> > vector;

    static void apply(Alpha a, const matrix &x, const vector &y, Beta b, vector &z)
    {
        if (b)
            z = vex_add<T,B>().apply(vex_scale<T,B>().apply(a, vex_mul<T,B>().apply(x, y)), vex_scale<T,B>().apply(b, z));
        else
            z = vex_scale<T,B>().apply(a, vex_mul<T,B>().apply(x, y));
    }
};

template < typename T, int B >
struct clear_impl< vex::vector< static_matrix<T,B,1> > >
{
    typedef static_matrix<T,B,1> vector_value;
    typedef vex::vector<vector_value> vector;

    static void apply(vector &x) {
        const vector_value zero = {0};
        x = zero;
    }
};

template < typename T, int B >
struct copy_impl<
    vex::vector< static_matrix<T,B,1> >,
    vex::vector< static_matrix<T,B,1> >
    >
{
    typedef vex::vector< static_matrix<T,B,1> > vector;

    static void apply(const vector &x, vector &y) {
        y = x;
    }
};

template < typename A, typename B, typename T, int N >
struct axpby_impl<
    A, vex::vector< static_matrix<T, N, 1> >,
    B, vex::vector< static_matrix<T, N, 1> >
    >
{
    typedef vex::vector< static_matrix<T,N,1> > vector;

    static void apply(A a, const vector &x, B b, vector &y) {
        if (b)
            y = vex_add<T,N>().apply(vex_scale<T,N>().apply(a, x), vex_scale<T,N>().apply(b, y));
        else
            y = vex_scale<T,N>().apply(a, x);
    }
};

template < typename A, typename B, typename C, typename T, int N >
struct axpbypcz_impl<
    A, vex::vector< static_matrix<T, N, 1> >,
    B, vex::vector< static_matrix<T, N, 1> >,
    C, vex::vector< static_matrix<T, N, 1> >
    >
{
    typedef vex::vector< static_matrix<T,N,1> > vector;

    static void apply(A a, const vector &x, B b, const vector &y, C c, vector &z) {
        if (c)
            z = vex_add<T,N>().apply(vex_add<T,N>().apply(vex_scale<T,N>().apply(a, x), vex_scale<T,N>().apply(b, y)), vex_scale<T,N>().apply(c, z));
        else
            z = vex_add<T,N>().apply(vex_scale<T,N>().apply(a, x), vex_scale<T,N>().apply(b, y));
    }
};

template < typename T, int B >
struct inner_product_impl<
    vex::vector< static_matrix<T,B,1> >,
    vex::vector< static_matrix<T,B,1> >
    >
{
    typedef T return_type;
    typedef static_matrix<T,B,1> vector_value;
    typedef vex::vector<vector_value> vector;

    struct BlockProduct : vex::UserFunction< BlockProduct, double(vector_value, vector_value) > {
        BlockProduct() {}

        static std::string name() {
            return "inner_prod_" + vex::type_name<vector_value>();
        }

        static void define(vex::backend::source_generator &src, const std::string &name = name()) {
            src.begin_function<T>(name);
            src.begin_function_parameters();
            src.parameter<vector_value>("a");
            src.parameter<vector_value>("b");
            src.end_function_parameters();
            src.new_line() << "return ";
            for(int i = 0; i < B; ++i) {
                if (i) src << " + ";
                src.new_line() << "a.data[" << i << "][0] * "
                               << "b.data[" << i << "][0]";
            }
            src << ";";
            src.end_function();
        }
    };

    static return_type get(const vector &x, const vector &y) {
        static const BlockProduct bp;

        vex::Reductor<T, vex::SUM_Kahan> sum( x.queue_list() );
        return sum( bp(x, y) );
    }
};

} // namespace backend
} // namespace amgcl

#endif
