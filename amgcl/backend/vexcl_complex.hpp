#ifndef AMGCL_BACKEND_VEXCL_COMPLEX_HPP
#define AMGCL_BACKEND_VEXCL_COMPLEX_HPP

#include <complex>
#include <vexcl/vexcl.hpp>

namespace amgcl {
namespace backend {

inline void enable_complex_for_vexcl(
        const std::vector<vex::backend::command_queue> &q)
{
#if !defined(VEXCL_BACKEND_CUDA)
    vex::Filter::Platform  AMD("AMD");
    vex::Filter::CLVersion v12(1, 2);
    for(size_t i = 0; i < q.size(); ++i) {
        auto d = vex::backend::get_device(q[i]);
        precondition(AMD(d) && v12(d),
                "VexCL only supports std::complex on AMD platforms for OpenCL >= 1.2");
    }

    vex::push_compile_options(q, "-x clc++");
#endif

    vex::backend::source_generator src(q[0], false);

    src.new_line() << "template <class T>";
    src.new_line() << "struct std_complex ";
    src.open("{");

    src.new_line() << "T real, imag;";

    src.new_line() << "std_complex(T real = 0, T imag = 0) : real(real), imag(imag) {}";

    src.function("const std_complex&", "operator+=")
        .open("(")
        .parameter("std_complex", "y")
        .close(")")
        .open("{");
    src.new_line() << "real += y.real;";
    src.new_line() << "imag += y.imag;";
    src.new_line() << "return *this;";
    src.close("}");

    src.function("const std_complex&", "operator-=")
        .open("(")
        .parameter("std_complex", "y")
        .close(")")
        .open("{");
    src.new_line() << "real -= y.real;";
    src.new_line() << "imag -= y.imag;";
    src.new_line() << "return *this;";
    src.close("}");

    src.function("const std_complex&", "operator*=")
        .open("(")
        .parameter("T", "c")
        .close(")")
        .open("{");
    src.new_line() << "real *= c;";
    src.new_line() << "imag *= c;";
    src.new_line() << "return *this;";
    src.close("}");

    src.function("const std_complex&", "operator*=")
        .open("(")
        .parameter("std_complex", "y")
        .close(")")
        .open("{");
    src.new_line() << "std_complex x = *this;";
    src.new_line() << "real = x.real * y.real - x.imag * y.imag;";
    src.new_line() << "imag = x.real * y.imag + x.imag * y.real;";
    src.new_line() << "return *this;";
    src.close("}");

    src.new_line() << "friend";
    src.function("std_complex", "operator+")
        .open("(")
        .parameter("std_complex", "x")
        .parameter("std_complex", "y")
        .close(")")
        .open("{")
        .new_line() << "return x += y;";
    src.close("}");

    src.new_line() << "friend";
    src.function("std_complex", "operator-")
        .open("(")
        .parameter("std_complex", "x")
        .parameter("std_complex", "y")
        .close(")")
        .open("{")
        .new_line() << "return x -= y;";
    src.close("}");

    src.new_line() << "friend";
    src.function("std_complex", "operator*")
        .open("(")
        .parameter("std_complex", "x")
        .parameter("std_complex", "y")
        .close(")")
        .open("{")
        .new_line() << "return x *= y;";
    src.close("}");

    src.new_line() << "friend";
    src.function("std_complex", "operator*")
        .open("(")
        .parameter("T", "a")
        .parameter("std_complex", "x")
        .close(")")
        .open("{")
        .new_line() << "return x *= a;";
    src.close("}");
    src.close("};");

    src.new_line() << "typedef std_complex<double> std_complex_double;";
    src.new_line() << "typedef std_complex<float> std_complex_float;";

    vex::push_program_header(q, src.str());
}

} // namespace backend
} // namespace amgcl

namespace vex {

template <typename T>
struct is_cl_native< std::complex<T> > : std::true_type {};

template <typename T>
struct type_name_impl< std::complex<T> >
{
    static std::string get() {
        std::ostringstream s;
        s << "std_complex_" << type_name<T>();
        return s.str();
    }
};

template <typename T>
struct cl_scalar_of< std::complex<T> > {
    typedef T type;
};

} // namespace vex

#endif
