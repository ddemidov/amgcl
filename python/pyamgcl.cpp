#include <vector>
#include <string>
#include <sstream>
#include <cstring>
#include <amgcl.h>

#include <boost/shared_ptr.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/stl_iterator.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy_boost_python.hpp"


//---------------------------------------------------------------------------
struct params {
    params() : h( amgcl_params_create(), amgcl_params_destroy ) {}

    void seti(const char *name, int value) {
        amgcl_params_seti(h.get(), name, value);
    }

    void setf(const char *name, float value) {
        amgcl_params_setf(h.get(), name, value);
    }

    std::string str() const {
        using boost::property_tree::ptree;
        ptree *p = static_cast<ptree*>(h.get());
        std::ostringstream buf;
        write_json(buf, *p);
        return buf.str();
    }

    std::string repr() const {
        return "amgcl params: " + str();
    }

    boost::shared_ptr<void> h;
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
struct solver {
    solver(
            amgclBackend    backend,
            amgclCoarsening coarsening,
            amgclRelaxation relaxation,
            amgclSolver     solver_type,
            const params    &prm,
            const numpy_boost<int,    1> &ptr,
            const numpy_boost<int,    1> &col,
            const numpy_boost<double, 1> &val
          )
        : n(ptr.num_elements() - 1),
          hs(
                amgcl_solver_create(backend, solver_type, prm.h.get(), n),
                amgcl_solver_destroy
           )
    {
        hp.reset(
                amgcl_precond_create(
                    backend, coarsening, relaxation, prm.h.get(),
                    n, ptr.data(), col.data(), val.data()
                    ),
                amgcl_precond_destroy
               );
    }

    PyObject* solve(const numpy_boost<double, 1> &rhs) const {
        numpy_boost<double, 1> x(&n);
        std::fill_n(x.data(), n, 0);
        amgcl_solver_solve(hs.get(), hp.get(), rhs.data(), x.data());

        PyObject *result = x.py_ptr();
        Py_INCREF(result);
        return result;
    }

    int n;
    boost::shared_ptr<void> hs;
    boost::shared_ptr<void> hp;
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

    enum_<amgclBackend>("backend")
        .value("builtin",   amgclBackendBuiltin)
        .value("block_crs", amgclBackendBlockCRS)
        ;

    enum_<amgclCoarsening>("coarsening")
        .value("ruge_stuben",          amgclCoarseningRugeStuben)
        .value("aggregation",          amgclCoarseningAggregation)
        .value("smoothed_aggregation", amgclCoarseningSmoothedAggregation)
        .value("smoothed_aggr_emin",   amgclCoarseningSmoothedAggrEMin)
        ;

    enum_<amgclRelaxation>("relaxation")
        .value("damped_jacobi", amgclRelaxationDampedJacobi)
        .value("gauss_seidel",  amgclRelaxationGaussSeidel)
        .value("chebyshev",     amgclRelaxationChebyshev)
        .value("spai0",         amgclRelaxationSPAI0)
        .value("ilu0",          amgclRelaxationILU0)
        ;

    enum_<amgclSolver>("solver_type")
        .value("cg",        amgclSolverCG)
        .value("bicgstab",  amgclSolverBiCGStab)
        .value("bicgstabl", amgclSolverBiCGStabL)
        .value("gmres",     amgclSolverGMRES)
        ;

    import_array();
    numpy_boost_python_register_type<int,    1>();
    numpy_boost_python_register_type<double, 1>();

    class_<solver>("solver",
            init<
                amgclBackend,
                amgclCoarsening,
                amgclRelaxation,
                amgclSolver,
                const params&,
                const numpy_boost<int,    1>&,
                const numpy_boost<int,    1>&,
                const numpy_boost<double, 1>&
                >())
        .def("__call__", &solver::solve)
        ;
}
