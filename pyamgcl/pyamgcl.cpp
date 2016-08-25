#include <vector>
#include <string>
#include <sstream>
#include <cstring>

#include <boost/range/iterator_range.hpp>
#include <boost/property_tree/ptree.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <amgcl/runtime.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

namespace amgcl {
#ifdef AMGCL_PROFILING
profiler<> prof;
#endif
}

namespace py = pybind11;

//---------------------------------------------------------------------------
boost::property_tree::ptree make_ptree(const py::dict &args) {
    boost::property_tree::ptree prm;
    for(auto p : args) {
        std::ostringstream key, val;
        key << p.first;
        val << p.second;
        prm.put(key.str(), val.str());
    }
    return prm;
}

//---------------------------------------------------------------------------
template <typename T>
py::array_t<T> make_array(size_t n) {
    return py::array_t<T>(
            py::buffer_info(
                nullptr, sizeof(T), py::format_descriptor<T>::value, 1, {n}, {sizeof(T)}
                )
            );
}

//---------------------------------------------------------------------------
template <typename T>
boost::iterator_range<T*> make_range(py::array_t<T> a) {
    py::buffer_info i = a.request();

    amgcl::precondition(i.ndim == 1,
            "Got multidimensional array for a vector parameter");

    return boost::make_iterator_range(
            static_cast<T*>(i.ptr), static_cast<T*>(i.ptr) + i.shape[0]
            );
}

//---------------------------------------------------------------------------
struct make_solver {
    make_solver(
            py::array_t<int>    _ptr,
            py::array_t<int>    _col,
            py::array_t<double> _val,
            py::dict prm
          )
    {
        auto ptr = make_range(_ptr);
        auto col = make_range(_col);
        auto val = make_range(_val);

        n = boost::size(ptr) - 1;
        S = boost::make_shared<Solver>(boost::make_tuple(n, ptr, col, val), make_ptree(prm));
    }

    py::array_t<double> solve(py::array_t<double> rhs) const {
        auto x_array = make_array<double>(n);
        auto x = make_range(x_array);
        boost::fill(x, 0.0);

        boost::tie(iters, error) = (*S)(make_range(rhs), x);

        return x_array;
    }

    py::array_t<double> solve(
            py::array_t<int>    ptr,
            py::array_t<int>    col,
            py::array_t<double> val,
            py::array_t<double> rhs
            ) const
    {
        auto x_array = make_array<double>(n);
        auto x = make_range(x_array);
        boost::fill(x, 0.0);

        boost::tie(iters, error) = (*S)(
                boost::make_tuple(n, make_range(ptr), make_range(col), make_range(val)),
                make_range(rhs), x);

        return x_array;
    }

    int iterations() const {
        return iters;
    }

    double residual() const {
        return error;
    }

    std::string repr() const {
        std::ostringstream s;
        s << S->precond();
        return s.str();
    }

    private:
        typedef amgcl::make_solver<
            amgcl::runtime::amg<
                amgcl::backend::builtin<double>
                >,
            amgcl::runtime::iterative_solver<
                amgcl::backend::builtin<double>
                >
            > Solver;

        int n;
        boost::shared_ptr< Solver > S;

        mutable int    iters;
        mutable double error;

};

//---------------------------------------------------------------------------
struct make_preconditioner {
    make_preconditioner(
            py::array_t<int>    _ptr,
            py::array_t<int>    _col,
            py::array_t<double> _val,
            py::dict prm
          )
    {
        auto ptr = make_range(_ptr);
        auto col = make_range(_col);
        auto val = make_range(_val);

        n = boost::size(ptr) - 1;
        P = boost::make_shared<Preconditioner>(boost::tie(n, ptr, col, val), make_ptree(prm));
    }

    py::array_t<double> apply(py::array_t<double> rhs) const {
        auto x_array = make_array<double>(n);
        auto x = make_range(x_array);

        P->apply(make_range(rhs), x);

        return x_array;
    }

    std::string repr() const {
        std::ostringstream s;
        s << (*P);
        return s.str();
    }

    private:
        typedef amgcl::runtime::amg< amgcl::backend::builtin<double> > Preconditioner;

        int n;
        boost::shared_ptr<Preconditioner> P;
};

//---------------------------------------------------------------------------
PYBIND11_PLUGIN(pyamgcl_ext) {
    py::module m("pyamgcl_ext");

    py::class_<make_solver>(m, "make_solver")
        .def(py::init<
                py::array_t<int>,
                py::array_t<int>,
                py::array_t<double>,
                py::dict
                >()
            )
        .def("__repr__", &make_solver::repr)
        .def("__call__",
                (py::array_t<double> (make_solver::*)(py::array_t<double>) const) &make_solver::solve,
                "Solves the problem for the given RHS",
                py::arg("rhs")
            )
        .def("__call__",
                (py::array_t<double> (make_solver::*)(
                        py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<double>
                        ) const) &make_solver::solve,
                "Solves the problem for the given RHS",
                py::arg("ptr"), py::arg("col"), py::arg("val"), py::arg("rhs")
            )
        .def_property_readonly("iters", &make_solver::iterations)
        .def_property_readonly("error", &make_solver::residual)
        ;

    py::class_<make_preconditioner>(m, "make_preconditioner")
        .def(py::init<
                py::array_t<int>,
                py::array_t<int>,
                py::array_t<double>,
                py::dict
                >()
            )
        .def("__repr__", &make_preconditioner::repr)
        .def("__call__", &make_preconditioner::apply, "Applies preconditioner to the given vector",
                py::arg("rhs")
            )
        ;

    return m.ptr();
}
