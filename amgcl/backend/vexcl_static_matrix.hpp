#ifndef AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP
#define AMGCL_BACKEND_VEXCL_STATIC_MATRIX_HPP

#include <amgcl/value_type/static_matrix.hpp>
#include <vexcl/vexcl.hpp>
#include <vexcl/sparse/ell.hpp>
#include <vexcl/sparse/distributed.hpp>

namespace amgcl {
namespace backend {

inline void enable_static_matrix_for_vexcl(
        const std::vector<vex::backend::command_queue> &q)
{
#if !defined(VEXCL_BACKEND_CUDA)
    vex::Filter::Platform  AMD("AMD");
    vex::Filter::CLVersion v12(1, 2);
    for(size_t i = 0; i < q.size(); ++i) {
        auto d = vex::backend::get_device(q[i]);
        precondition(AMD(d) && v12(d),
                "VexCL only supports static_matrix on AMD platforms for OpenCL >= 1.2");
    }

    vex::push_compile_options(q, "-x clc++");
#endif

    vex::backend::source_generator src(q[0], false);

    src.new_line() << "template <class T, int N, int M>";
    src.new_line() << "struct amgcl_static_matrix ";
    src.open("{");

    src.new_line() << "T buf[N * M];";

    src.function("T", "operator()")
        .open("(")
        .parameter<int>("i")
        .parameter<int>("j")
        .close(") const")
        .open("{")
        .new_line() << "return buf[i * M + j];";
    src.close("}");

    src.function("T&", "operator()")
        .open("(")
        .parameter<int>("i")
        .parameter<int>("j")
        .close(")")
        .open("{")
        .new_line() << "return buf[i * M + j];";
    src.close("}");

    src.function("T", "operator()")
        .open("(")
        .parameter<int>("i")
        .close(") const")
        .open("{")
        .new_line() << "return buf[i];";
    src.close("}");

    src.function("T&", "operator()")
        .open("(")
        .parameter<int>("i")
        .close(")")
        .open("{")
        .new_line() << "return buf[i];";
    src.close("}");

    src.function("const amgcl_static_matrix&", "operator+=")
        .open("(")
        .parameter("amgcl_static_matrix", "y")
        .close(")")
        .open("{");
    src.new_line() << "for(int i = 0; i < N * M; ++i) buf[i] += y.buf[i];";
    src.new_line() << "return *this;";
    src.close("}");

    src.function("const amgcl_static_matrix&", "operator-=")
        .open("(")
        .parameter("amgcl_static_matrix", "y")
        .close(")")
        .open("{");
    src.new_line() << "for(int i = 0; i < N * M; ++i) buf[i] -= y.buf[i];";
    src.new_line() << "return *this;";
    src.close("}");

    src.function("const amgcl_static_matrix&", "operator*=")
        .open("(")
        .parameter("T", "c")
        .close(")")
        .open("{");
    src.new_line() << "for(int i = 0; i < N * M; ++i) buf[i] *= c;";
    src.new_line() << "return *this;";
    src.close("}");

    src.new_line() << "friend";
    src.function("amgcl_static_matrix", "operator+")
        .open("(")
        .parameter("amgcl_static_matrix", "x")
        .parameter("amgcl_static_matrix", "y")
        .close(")")
        .open("{")
        .new_line() << "return x += y;";
    src.close("}");

    src.new_line() << "friend";
    src.function("amgcl_static_matrix", "operator-")
        .open("(")
        .parameter("amgcl_static_matrix", "x")
        .parameter("amgcl_static_matrix", "y")
        .close(")")
        .open("{")
        .new_line() << "return x -= y;";
    src.close("}");

    src.new_line() << "friend";
    src.function("amgcl_static_matrix", "operator*")
        .open("(")
        .parameter("T", "a")
        .parameter("amgcl_static_matrix", "x")
        .close(")")
        .open("{")
        .new_line() << "return x *= a;";
    src.close("}");

    src.new_line() << "friend";
    src.function("amgcl_static_matrix", "operator-")
        .open("(")
        .parameter("amgcl_static_matrix", "x")
        .close(")")
        .open("{");
    src.new_line() << "for(int i = 0; i < N * M; ++i) x.buf[i] = -x.buf[i];";
    src.new_line() << "return x;";
    src.close("}");
    src.close("};");

    src.new_line() << "template <typename T, int N, int K, int M>";
    src.function("amgcl_static_matrix<T, N, M>", "operator*")
        .open("(")
        .parameter("amgcl_static_matrix<T, N, K>", "a")
        .parameter("amgcl_static_matrix<T, K, M>", "b")
        .close(")")
        .open("{");

    src.new_line() << "amgcl_static_matrix<T, N, M> c;";
    src.new_line() << "for(int i = 0; i < N; ++i)";
    src.open("{");
    src.new_line() << "for(int j = 0; j < M; ++j) c(i,j) = T();";
    src.new_line() << "for(int k = 0; k < K; ++k)";
    src.open("{");
    src.new_line() << "T aik = a(i,k);";
    src.new_line() << "for(int j = 0; j < M; ++j) c(i,j) += aik * b(k,j);";
    src.close("}");
    src.close("}");
    src.new_line() << "return c;";
    src.close("}");

    src.new_line() << "template <typename T, int N>";
    src.function("T", "operator*")
        .open("(")
        .parameter("amgcl_static_matrix<T, N, 1>", "a")
        .parameter("amgcl_static_matrix<T, N, 1>", "b")
        .close(")")
        .open("{");
    src.new_line() << "T sum = T();";
    src.new_line() << "for(int i = 0; i < N; ++i) sum += a(i) * b(i);";
    src.new_line() << "return sum;";
    src.close("}");

    vex::push_program_header(q, src.str());
}

} // namespace backend
} // namespace amgcl

namespace vex {
namespace sparse {

template <typename T, int N>
struct rhs_of< amgcl::static_matrix<T, N, N> > {
    typedef amgcl::static_matrix<T, N, 1> type;
};

} // namespace sparse

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
