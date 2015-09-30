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
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/preconditioner/simple.hpp>
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
boost::property_tree::ptree make_ptree(const boost::python::dict &args) {
    using namespace boost::python;
    boost::property_tree::ptree p;

    for(stl_input_iterator<tuple> arg(args.items()), end; arg != end; ++arg) {
        const char *name = extract<const char*>((*arg)[0]);
        const char *type = extract<const char*>((*arg)[1].attr("__class__").attr("__name__"));

        if (strcmp(type, "int") == 0)
            p.put(name, extract<int>((*arg)[1]));
        else
            p.put(name, extract<float>((*arg)[1]));
    }

    return p;
}

//---------------------------------------------------------------------------
struct make_solver {
    make_solver(
            amgcl::runtime::coarsening::type coarsening,
            amgcl::runtime::relaxation::type relaxation,
            amgcl::runtime::solver::type     solver,
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1)
    {
        boost::property_tree::ptree pt = make_ptree(prm);
        pt.put("amg.coarsening.type", coarsening);
        pt.put("amg.relaxation.type", relaxation);
        pt.put("solver.type",         solver);

        S = boost::make_shared<Solver>(boost::tie(n, ptr, col, val), pt);
    }

    PyObject* solve(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        BOOST_FOREACH(double &v, x) v = 0;

        cnv = (*S)(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    PyObject* solve(
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val,
            const numpy_boost<double, 1> &rhs
            ) const
    {
        numpy_boost<double, 1> x(&n);
        BOOST_FOREACH(double &v, x) v = 0;

        cnv = (*S)(boost::tie(n, ptr, col, val), rhs, x);

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

    std::string repr() const {
        std::ostringstream buf;
        buf << "pyamgcl solver\n" << S->amg();
        return buf.str();
    }

    private:
        typedef amgcl::runtime::make_solver< amgcl::backend::builtin<double> > Solver;
        int n;
        boost::shared_ptr< Solver > S;

        mutable boost::tuple<int, double> cnv;

};

//---------------------------------------------------------------------------
struct make_preconditioner {
    make_preconditioner(
            amgcl::runtime::coarsening::type coarsening,
            amgcl::runtime::relaxation::type relaxation,
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1)
    {
        boost::property_tree::ptree pt = make_ptree(prm);
        pt.put("coarsening.type", coarsening);
        pt.put("relaxation.type", relaxation);

        P = boost::make_shared<Preconditioner>(boost::tie(n, ptr, col, val), pt);
    }

    PyObject* apply(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        P->apply(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    std::string repr() const {
        std::ostringstream buf;
        buf << "pyamgcl preconditioner\n" << (*P);
        return buf.str();
    }

    private:
        int n;
        typedef amgcl::runtime::amg< amgcl::backend::builtin<double> > Preconditioner;
        boost::shared_ptr<Preconditioner> P;
};

//---------------------------------------------------------------------------
struct make_cpr {
    make_cpr(
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val,
            const numpy_boost<int,    1> &pm
          )
        : n(ptr.num_elements() - 1),
          P(boost::tie(n, ptr, col, val), pmask(n, pm.data()), make_ptree(prm))
    { }

    PyObject* apply(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        P.apply(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    private:
        struct pmask {
            std::vector<int> pm;

            pmask(int n, const int *pm) : pm(pm, pm + n) {}

            bool operator()(size_t i) const {
                return pm[i];
            }
        };

        int n;

        amgcl::preconditioner::cpr<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            > P;
};

//---------------------------------------------------------------------------
struct make_simple {
    make_simple(
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val,
            const numpy_boost<int,    1> &pm
          )
        : n(ptr.num_elements() - 1),
          P(boost::tie(n, ptr, col, val), pmask(n, pm.data()), make_ptree(prm))
    { }

    PyObject* apply(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        P.apply(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    private:
        struct pmask {
            std::vector<int> pm;

            pmask(int n, const int *pm) : pm(pm, pm + n) {}

            bool operator()(size_t i) const {
                return pm[i];
            }
        };

        int n;

        amgcl::preconditioner::simple<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            > P;
};

#if PY_MAJOR_VERSION >= 3
void*
#else
void
#endif
call_import_array() {
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

//---------------------------------------------------------------------------
BOOST_PYTHON_MODULE(pyamgcl_ext)
{
    using namespace boost::python;
    docstring_options docopts(true, true, false);

    enum_<amgcl::runtime::coarsening::type>("coarsening", "coarsening kinds")
        .value("ruge_stuben",          amgcl::runtime::coarsening::ruge_stuben)
        .value("aggregation",          amgcl::runtime::coarsening::aggregation)
        .value("smoothed_aggregation", amgcl::runtime::coarsening::smoothed_aggregation)
        .value("smoothed_aggr_emin",   amgcl::runtime::coarsening::smoothed_aggr_emin)
        ;

    enum_<amgcl::runtime::relaxation::type>("relaxation", "relaxation schemes")
        .value("damped_jacobi",            amgcl::runtime::relaxation::damped_jacobi)
        .value("gauss_seidel",             amgcl::runtime::relaxation::gauss_seidel)
        .value("multicolor_gauss_seidel",  amgcl::runtime::relaxation::multicolor_gauss_seidel)
        .value("chebyshev",                amgcl::runtime::relaxation::chebyshev)
        .value("spai0",                    amgcl::runtime::relaxation::spai0)
        .value("ilu0",                     amgcl::runtime::relaxation::ilu0)
        ;

    enum_<amgcl::runtime::solver::type>("solver_type", "iterative solvers")
        .value("cg",        amgcl::runtime::solver::cg)
        .value("bicgstab",  amgcl::runtime::solver::bicgstab)
        .value("bicgstabl", amgcl::runtime::solver::bicgstabl)
        .value("gmres",     amgcl::runtime::solver::gmres)
        ;

    call_import_array();

    numpy_boost_python_register_type<int,    1>();
    numpy_boost_python_register_type<double, 1>();

    PyObject* (make_solver::*s1)(
            const numpy_boost<double, 1>&
            ) const = &make_solver::solve;

    PyObject* (make_solver::*s2)(
            const numpy_boost<int,    1>&,
            const numpy_boost<int,    1>&,
            const numpy_boost<double, 1>&,
            const numpy_boost<double, 1>&
            ) const = &make_solver::solve;

    class_<make_solver, boost::noncopyable>(
            "make_solver",
            "Creates iterative solver preconditioned by AMG",
            init<
                amgcl::runtime::coarsening::type,
                amgcl::runtime::relaxation::type,
                amgcl::runtime::solver::type,
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
            >(
                args(
                    "coarsening",
                    "relaxation",
                    "iterative_solver",
                    "params",
                    "indptr",
                    "indices",
                    "values"
                    ),
                "Creates iterative solver preconditioned by AMG"
             )
            )
        .def("__repr__",   &make_solver::repr)
        .def("__call__",   s1, args("rhs"),
                "Solves the problem for the given RHS")
        .def("__call__",   s2, args("ptr", "col", "val", "rhs"),
                "Solves the problem for the given matrix and the RHS")
        .def("iterations", &make_solver::iterations,
                "Returns iterations made during last solve")
        .def("residual",   &make_solver::residual,
                "Returns relative error achieved during last solve")
        ;

    class_<make_preconditioner, boost::noncopyable>(
            "make_preconditioner",
            "Creates AMG hierarchy to be used as a preconditioner",
            init<
                amgcl::runtime::coarsening::type,
                amgcl::runtime::relaxation::type,
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
            >(
                args(
                    "coarsening",
                    "relaxation",
                    "params",
                    "indptr",
                    "indices",
                    "values"
                    ),
                "Creates AMG hierarchy to be used as a preconditioner"
             )
            )
        .def("__repr__",   &make_preconditioner::repr)
        .def("__call__",   &make_preconditioner::apply,
                "Apply preconditioner to the given vector")
        ;

    class_<make_cpr, boost::noncopyable>(
            "make_cpr",
            "Creates CPR preconditioner",
            init<
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&,
                const numpy_boost<int,    1>&
            >(
                args(
                    "params",
                    "indptr",
                    "indices",
                    "values",
                    "pmask"
                    ),
                "Creates CPR preconditioner"
             )
            )
        .def("__call__",   &make_cpr::apply,
                "Apply preconditioner to the given vector")
        ;

    class_<make_simple, boost::noncopyable>(
            "make_simple",
            "Creates SIMPLE preconditioner",
            init<
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&,
                const numpy_boost<int,    1>&
            >(
                args(
                    "params",
                    "indptr",
                    "indices",
                    "values",
                    "pmask"
                    ),
                "Creates SIMPLE preconditioner"
             )
            )
        .def("__call__",   &make_simple::apply,
                "Apply preconditioner to the given vector")
        ;
}
