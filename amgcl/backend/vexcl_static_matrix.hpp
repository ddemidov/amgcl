#ifndef AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP
#define AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP

#include <amgcl/value_type/static_matrix.hpp>
#include <vexcl/vexcl.hpp>

namespace amgcl {
namespace backend {

inline void enable_static_matrix_for_vexcl(
        const std::vector<vex::backend::command_queue> &q)
{
#if !defined(VEXCL_BACKEND_CUDA)
    vex::Filter::Platform  AMD("AMD");
    vex::Filter::CLVersion v12(1, 2);
    for(size_t i = 0; i < q.size(); ++i) {
        const vex::backend::device &d = vex::backend::get_device(q[i]);
        precondition(AMD(d) && v12(d),
                "VexCL only supports static_matrix on AMD platforms for OpenCL >= 1.2");
    }
#endif

    vex::push_compile_options(q, "-x clc++");
    vex::push_program_header(q, VEX_STRINGIZE_SOURCE(
template <class T, int N, int M>
struct amgcl_static_matrix {
    T buf[N * M];

    T operator()(int i, int j) const {
        return buf[i * M + j];
    }

    T& operator()(int i, int j) {
        return buf[i * M + j];
    }

    T operator()(int i) const {
        return buf[i];
    }

    T& operator()(int i) {
        return buf[i];
    }

    const amgcl_static_matrix& operator+=(amgcl_static_matrix y) {
        for(int i = 0; i < N * M; ++i)
            buf[i] += y.buf[i];
        return *this;
    }

    const amgcl_static_matrix& operator-=(amgcl_static_matrix y) {
        for(int i = 0; i < N * M; ++i)
            buf[i] -= y.buf[i];
        return *this;
    }

    const amgcl_static_matrix& operator*=(T c) {
        for(int i = 0; i < N * M; ++i)
            buf[i] *= c;
        return *this;
    }

    friend amgcl_static_matrix operator+(amgcl_static_matrix x, amgcl_static_matrix y)
    {
        return x += y;
    }

    friend amgcl_static_matrix operator-(amgcl_static_matrix x, amgcl_static_matrix y)
    {
        return x -= y;
    }

    friend amgcl_static_matrix operator*(T a, amgcl_static_matrix x)
    {
        return x *= a;
    }

    friend amgcl_static_matrix operator-(amgcl_static_matrix x)
    {
        for(int i = 0; i < N * M; ++i)
            x.buf[i] = -x.buf[i];
        return x;
    }
};

template <typename T, int N, int K, int M>
amgcl_static_matrix<T, N, M> operator*(
        amgcl_static_matrix<T, N, K> a,
        amgcl_static_matrix<T, K, M> b
        )
{
    amgcl_static_matrix<T, N, M> c;
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < M; ++j)
            c(i,j) = T();
        for(int k = 0; k < K; ++k) {
            T aik = a(i,k);
            for(int j = 0; j < M; ++j)
                c(i,j) += aik * b(k,j);
        }
    }
    return c;
}

template <typename T, int N>
T operator*(
        amgcl_static_matrix<T, N, 1> a,
        amgcl_static_matrix<T, N, 1> b
        )
{
    T sum = T();
    for(int i = 0; i < N; ++i)
        sum += a(i) * b(i);
    return sum;
}
));

}

} // namespace backend
} // namespace amgcl

namespace vex {

template <typename T, int N, int M>
struct is_cl_native< amgcl::static_matrix<T, N, M> > : std::true_type {};

template <typename T, int N, int M>
struct type_name_impl< amgcl::static_matrix<T, N, M> >
{
    static std::string get() {
        std::ostringstream s;
        s << "amgcl_static_matrix<" << type_name<T>() << "," << N << "," << M << ">";
        return s.str();
    }
};

template <typename T, int N, int M>
struct cl_scalar_of< amgcl::static_matrix<T, N, M> > {
    typedef T type;
};

} // namespace vex

#endif
