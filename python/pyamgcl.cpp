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
struct precond {
    precond(
            amgclBackend    backend,
            amgclCoarsening coarsening,
            amgclRelaxation relaxation,
            const params    &prm,
            const boost::python::numeric::array &ptr,
            const boost::python::numeric::array &col,
            const boost::python::numeric::array &val
           )
    {
        using namespace boost::python;

        int n = extract<int>(ptr.attr("size")) - 1;
        std::vector<int> t_ptr(n + 1);

        for(int i = 0; i <= n; ++i) t_ptr[i] = extract<int>(ptr[i]);

        int nnz = t_ptr[n];

        std::vector<int>    t_col(nnz);
        std::vector<double> t_val(nnz);

        for(int i = 0; i < nnz; ++i) t_col[i] = extract<int>   (col[i]);
        for(int i = 0; i < nnz; ++i) t_val[i] = extract<double>(val[i]);

        h.reset(
                amgcl_precond_create(
                    backend, coarsening, relaxation, prm.h.get(),
                    n,
                    t_ptr.data(),
                    t_col.data(),
                    t_val.data()
                    ),
                amgcl_precond_destroy
               );
    }

    void apply(
            boost::python::numeric::array const &rhs,
            boost::python::numeric::array       &x
            ) const
    {
        using namespace boost::python;

        int n = extract<int>(rhs.attr("size"));

        std::vector<double> t_rhs(n);
        std::vector<double> t_x(n);

        for(int i = 0; i < n; ++i) {
            t_rhs[i] = extract<double>(rhs[i]);
            t_x[i]   = extract<double>(x[i]);
        }

        amgcl_precond_apply(h.get(), t_rhs.data(), t_x.data());

        for(int i = 0; i < n; ++i) x[i] = t_x[i];
    }

    boost::shared_ptr<void> h;
};

//---------------------------------------------------------------------------
struct solver {
    solver(
            amgclBackend backend,
            amgclSolver  solver_type,
            const params &prm,
            int n
          )
        : h(
                amgcl_solver_create(backend, solver_type, prm.h.get(), n),
                amgcl_solver_destroy
           )
    {}

    void solve(
            const precond &P,
            boost::python::numeric::array const &rhs,
            boost::python::numeric::array       &x
            ) const
    {
        using namespace boost::python;

        int n = extract<int>(rhs.attr("size"));

        std::vector<double> t_rhs(n);
        std::vector<double> t_x(n);

        for(int i = 0; i < n; ++i) {
            t_rhs[i] = extract<double>(rhs[i]);
            t_x[i]   = extract<double>(x[i]);
        }

        amgcl_solver_solve(h.get(), P.h.get(), t_rhs.data(), t_x.data());

        for(int i = 0; i < n; ++i) x[i] = t_x[i];
    }

    boost::shared_ptr<void> h;
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

    numeric::array::set_module_and_type("numpy", "ndarray");

    class_<precond>("precond",
            init<
                amgclBackend,
                amgclCoarsening,
                amgclRelaxation,
                const params&,
                const numeric::array&,
                const numeric::array&,
                const numeric::array&
                >())
        .def("apply", &precond::apply)
        ;

    def("make_params", raw_function(make_params));

    class_<solver>("solver",
            init<
                amgclBackend,
                amgclSolver,
                const params&,
                int
                >()
            )
        .def("solve", &solver::solve)
        ;
}
