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
#include <amgcl/make_solver.hpp>
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
        const char *key = extract<const char*>((*arg)[0]);
        const char *val = extract<const char*>((*arg)[1]);
        p.put(key, val);
    }

    return p;
}

//---------------------------------------------------------------------------
struct make_solver {
    make_solver(
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1),
          S(boost::make_shared<Solver>(boost::tie(n, ptr, col, val), make_ptree(prm)))
    { }

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
        buf << S->precond();
        return buf.str();
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

        mutable boost::tuple<int, double> cnv;

};

//---------------------------------------------------------------------------
struct make_preconditioner {
    make_preconditioner(
            const boost::python::dict    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1),
          P(boost::make_shared<Preconditioner>(boost::tie(n, ptr, col, val), make_ptree(prm)))
    { }

    PyObject* apply(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        P->apply(rhs, x);

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    std::string repr() const {
        std::ostringstream buf;
        buf << (*P);
        return buf.str();
    }

    private:
        int n;
        typedef amgcl::runtime::amg< amgcl::backend::builtin<double> > Preconditioner;
        boost::shared_ptr<Preconditioner> P;
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
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
            >(
                args(
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
        .add_property("iters", &make_solver::iterations,
                "Returns iterations made during last solve")
        .add_property("error", &make_solver::residual,
                "Returns relative error achieved during last solve")
        ;

    class_<make_preconditioner, boost::noncopyable>(
            "make_preconditioner",
            "Creates AMG hierarchy to be used as a preconditioner",
            init<
                const dict&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
            >(
                args(
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
}
