#include <vector>
#include <string>
#include <sstream>
#include <cstring>

#include <boost/range/iterator_range.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy_boost_python.hpp"

#include <amgcl/runtime.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

namespace amgcl {
#ifdef AMGCL_PROFILING
profiler<> prof;
#endif
namespace backend {
template <>
struct is_builtin_vector< numpy_boost<double,1> > : boost::true_type {};
}
}

//---------------------------------------------------------------------------
struct params {
    void seti(const char *name, int value) {
        p.put(name, value);
    }

    void setf(const char *name, float value) {
        p.put(name, value);
    }

    std::string str() const {
        std::ostringstream buf;
        write_json(buf, p);
        return buf.str();
    }

    std::string repr() const {
        return "amgcl params: " + str();
    }

    boost::property_tree::ptree p;
};

//---------------------------------------------------------------------------
params make_params(boost::python::tuple args, boost::python::dict kwargs) {
    params p;

    using namespace boost::python;

    for(stl_input_iterator<tuple> arg(kwargs.items()), end; arg != end; ++arg) {
        const char *name = extract<const char*>((*arg)[0]);
        const char *type = extract<const char*>((*arg)[1].attr("__class__").attr("__name__"));

        if (strcmp(type, "int") == 0)
            p.seti(name, extract<int>((*arg)[1]));
        else
            p.setf(name, extract<float>((*arg)[1]));
    }

    return p;
}

//---------------------------------------------------------------------------
struct make_solver {
    make_solver(
            amgcl::runtime::coarsening::type coarsening,
            amgcl::runtime::relaxation::type relaxation,
            amgcl::runtime::solver::type     solver,
            const params    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1),
          S(coarsening, relaxation, solver, boost::tie(n, ptr, col, val), prm.p)
    {
    }

    PyObject* solve(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        BOOST_FOREACH(double &v, x) v = 0;

        cnv = S(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    int iterations() const {
        return boost::get<0>(cnv);
    }

    double residual() const {
        return boost::get<1>(cnv);
    }

    std::string str() const {
        std::ostringstream buf;
        buf << S.amg();
        return buf.str();
    }

    std::string repr() const {
        return "amgcl: " + str();
    }

    private:
        int n;
        amgcl::runtime::make_solver< amgcl::backend::builtin<double> > S;

        mutable boost::tuple<int, double> cnv;
};

BOOST_PYTHON_MODULE(pyamgcl)
{
    using namespace boost::python;

    class_<params>("params")
        .def("__setitem__", &params::seti)
        .def("__setitem__", &params::setf)
        .def("__str__",     &params::str)
        .def("__repr__",    &params::repr)
        ;

    def("make_params", raw_function(make_params));

    enum_<amgcl::runtime::coarsening::type>("coarsening")
        .value("ruge_stuben",          amgcl::runtime::coarsening::ruge_stuben)
        .value("aggregation",          amgcl::runtime::coarsening::aggregation)
        .value("smoothed_aggregation", amgcl::runtime::coarsening::smoothed_aggregation)
        .value("smoothed_aggr_emin",   amgcl::runtime::coarsening::smoothed_aggr_emin)
        ;

    enum_<amgcl::runtime::relaxation::type>("relaxation")
        .value("damped_jacobi", amgcl::runtime::relaxation::damped_jacobi)
        .value("gauss_seidel",  amgcl::runtime::relaxation::gauss_seidel)
        .value("chebyshev",     amgcl::runtime::relaxation::chebyshev)
        .value("spai0",         amgcl::runtime::relaxation::spai0)
        .value("ilu0",          amgcl::runtime::relaxation::ilu0)
        ;

    enum_<amgcl::runtime::solver::type>("solver_type")
        .value("cg",        amgcl::runtime::solver::cg)
        .value("bicgstab",  amgcl::runtime::solver::bicgstab)
        .value("bicgstabl", amgcl::runtime::solver::bicgstabl)
        .value("gmres",     amgcl::runtime::solver::gmres)
        ;

    import_array();
    numpy_boost_python_register_type<int,    1>();
    numpy_boost_python_register_type<double, 1>();

    class_<make_solver, boost::noncopyable>("make_solver",
            init<
                amgcl::runtime::coarsening::type,
                amgcl::runtime::relaxation::type,
                amgcl::runtime::solver::type,
                const params&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
                >())
        .def("__call__",   &make_solver::solve)
        .def("__str__",    &make_solver::str)
        .def("__repr__",   &make_solver::repr)
        .def("iterations", &make_solver::iterations)
        .def("residual",   &make_solver::residual)
        ;
}
