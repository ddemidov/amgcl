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

template <typename Alpha, typename Beta, typename T, int B>
struct spmv_impl<
    Alpha, vex_SpMat<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>,
    vex::vector<static_matrix<T,B,1>>, Beta, vex::vector<static_matrix<T,B,1>>>
{
    typedef vex_SpMat<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t> matrix;
    typedef vex::vector<static_matrix<T,B,1>>                     vector;

    static void apply(Alpha alpha, const matrix &A, const vector &x, Beta beta, vector &y)
    {
        auto &K = spmv_kernel(x.queue_list()[0]);

        K.push_arg((int)y.size());
        K.push_arg((T)alpha);
        K.push_arg((T)beta);
        K.push_arg((int)A.ell_width);
        K.push_arg((int)A.ell_pitch);
        K.push_arg(A.ell_col);
        K.push_arg(A.ell_val);
        K.push_arg(x(0));
        K.push_arg(y(0));

        K(x.queue_list()[0]);
    }

    static vex::backend::kernel& spmv_kernel(const vex::backend::command_queue &q) {
        using namespace vex;
        using namespace vex::detail;
        static kernel_cache cache;

        auto K = cache.find(q);
        if (K == cache.end()) {
            vex::backend::source_generator src(q);

            /* The following kernel uses B threads for each B*B block ->
             * more continguous memory reads for A.
             */
            src.kernel("blocked_spmv").open("(")
                .template parameter<int>("N")
                .template parameter<T>("alpha")
                .template parameter<T>("beta")
                .template parameter<int>("ell_width")
                .template parameter<int>("ell_pitch")
                .template parameter< global_ptr<const long> >("ell_col")
                .template parameter< global_ptr<const T> >("ell_val")
                .template parameter< global_ptr<const T> >("x")
                .template parameter< global_ptr<T> >("y")
                .close(")").open("{");

            src.new_line() << " size_t global_id   = " << src.global_id(0) << ";";
            src.new_line() << " size_t global_size = " << src.global_size(0) << ";";

            src.new_line() << " #define subwarp_size " << B;
            src.new_line() << " const size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
            src.new_line() << " const size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";

            src.new_line().smem_static_var(type_name<T>(), "row_A[256*subwarp_size]");
            src.new_line() << type_name< shared_ptr<T> >() << " my_A = row_A + subwarp_gid * subwarp_size * subwarp_size;";
            src.new_line() << type_name<T>() << " my_x, my_y;";

            src.new_line() << "int loop_iters = (N-1) / (global_size / subwarp_size) + 1;";

            src.new_line() << "for (size_t iter = 0; iter < loop_iters; ++iter)";
            src.open("{");
            src.new_line() << "int row = (global_id + iter * global_size) / subwarp_size;";
            src.new_line() << "my_y = 0;";
            src.new_line() << "int offset = min(row, N-1);";
            src.new_line() << "for (int i = 0; i < ell_width; ++i, offset += ell_pitch) {";
            src.new_line() << "  int c = ell_col[offset];";

            src.new_line() << "  int ell_val_offset = subwarp_size * subwarp_size * offset + subwarp_idx;";
            src.new_line() << "  my_x = (c >= 0) ? x[subwarp_size * c + subwarp_idx] : 0.0;";
            src.new_line() << "  for (size_t k=0; k<subwarp_size; ++k) ";
            src.new_line() << "    my_A[k * subwarp_size + subwarp_idx] = (c >= 0) ? ell_val[ell_val_offset + k * subwarp_size] * my_x : 0.0;";
            src.new_line().barrier();

            src.new_line() << "  for (size_t k=0; k<subwarp_size; ++k)";
            src.new_line() << "    my_y += my_A[subwarp_idx * subwarp_size + k];";
            src.new_line() << "}";

            src.new_line() << "if (row < N)";
            src.new_line() << "  y[subwarp_size*row+subwarp_idx] = beta * y[subwarp_size*row+subwarp_idx] + alpha * my_y;";
            src.close("}");
            src.close("}");

            K = cache.insert(q, vex::backend::kernel(q, src.str(), "blocked_spmv"));
            K->second.config(256, 256);
        }

        return K->second;
    }

};

template <typename T, int B>
struct residual_impl<
    vex_SpMat<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t>,
    vex::vector<static_matrix<T,B,1>>,
    vex::vector<static_matrix<T,B,1>>,
    vex::vector<static_matrix<T,B,1>>
    >
{
    typedef vex_SpMat<static_matrix<T,B,B>, ptrdiff_t, ptrdiff_t> matrix;
    typedef vex::vector<static_matrix<T,B,1>>                     vector;

    static void apply(const vector &rhs, const matrix &A, const vector &x, vector &r)
    {
        spmv(1, A, x, 0, r);
        r = rhs - r;
    }
};

template < typename Alpha, typename Beta, typename T, int B >
struct vmul_impl<
    Alpha, vex::vector< static_matrix<T,B,B> >,
    vex::vector< static_matrix<T,B,1> >,
    Beta, vex::vector< static_matrix<T,B,1> >
    >
{
    typedef vex::vector< static_matrix<T,B,B> > vector1;
    typedef vex::vector< static_matrix<T,B,1> > vector2;

    static void apply(Alpha a, const vector1 &x, const vector2 &y, Beta b, vector2 &z)
    {
        auto &K = vmul_kernel(x.queue_list()[0]);

        K.push_arg((int)y.size());
        K.push_arg(static_cast<T>(a));
        K.push_arg(static_cast<T>(b));
        K.push_arg(x(0));
        K.push_arg(y(0));
        K.push_arg(z(0));

        K(x.queue_list()[0]);
    }

    static vex::backend::kernel& vmul_kernel(const vex::backend::command_queue &q) {
        using namespace vex;
        using namespace vex::detail;
        static kernel_cache cache;

        auto K = cache.find(q);
        if (K == cache.end()) {
            vex::backend::source_generator src(q);

            src.kernel("vmul").open("(")
                .template parameter<int>("N")
                .template parameter<T>("alpha")
                .template parameter<T>("beta")
                .template parameter< global_ptr<const T> >("x")
                .template parameter< global_ptr<const T> >("y")
                .template parameter< global_ptr<T> >("z")
                .close(")").open("{");

            src.new_line() << "size_t global_id   = " << src.global_id(0) << ";";
            src.new_line() << "size_t global_size = " << src.global_size(0) << ";";

            src.new_line() << "#define subwarp_size " << B;
            src.new_line() << "const size_t subwarp_gid = " << src.local_id(0) << " / subwarp_size;";
            src.new_line() << "const size_t subwarp_idx = " << src.local_id(0) << " % subwarp_size;";

            src.new_line().smem_static_var(type_name<T>(), "row_A[256*subwarp_size]");
            src.new_line() << type_name<shared_ptr<T>>() << " my_A = row_A + subwarp_gid * subwarp_size * subwarp_size;";
            src.new_line() << type_name<T>() << " my_y, my_z;";

            src.new_line() << "size_t loop_iters = (N-1) / (global_size / subwarp_size) + 1;";

            src.new_line() << "for (size_t iter = 0; iter < loop_iters; ++iter)";
            src.open("{");
            src.new_line() << "size_t row = (global_id + iter * global_size) / subwarp_size;";
            src.new_line() << "my_z = 0;";

            src.new_line() << "size_t row2 = min((int)row, (int)N-1);";
            src.new_line() << "size_t x_offset = subwarp_size * subwarp_size * row2 + subwarp_idx;";
            src.new_line() << "my_y = y[subwarp_size * row2 + subwarp_idx];";
            src.new_line() << "for (size_t k=0; k<subwarp_size; ++k) ";
            src.new_line() << "  my_A[k * subwarp_size + subwarp_idx] = x[x_offset + k * subwarp_size] * my_y;";
            src.new_line().barrier();

            src.new_line() << "for (size_t k=0; k<subwarp_size; ++k)";
            src.new_line() << "  my_z += my_A[subwarp_idx * subwarp_size + k];";

            src.new_line() << "if (row < N)";
            src.new_line() << "  z[subwarp_size*row+subwarp_idx] = beta * z[subwarp_size*row+subwarp_idx] + alpha * my_z;";
            src.close("}");
            src.close("}");

            K = cache.insert(q, vex::backend::kernel(q, src.str(), "vmul"));
            K->second.config(256, 256);
        }

        return K->second;
    }
};
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
