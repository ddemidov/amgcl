#include <vector>
#include <string>
#include <sstream>
#include <cstring>

#include <boost/range/iterator_range.hpp>
#include <boost/property_tree/ptree.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <amgcl/solver/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
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
    for(auto p : args)
        prm.put(static_cast<std::string>(py::str(p.first)),
                static_cast<std::string>(py::str(p.second)));
    return prm;
}

//---------------------------------------------------------------------------
template <typename T>
py::array_t<T> make_array(size_t n, T *ptr = nullptr) {
    return py::array_t<T>(
            py::buffer_info(
                ptr, sizeof(T), py::format_descriptor<T>::value, 1, {n}, {sizeof(T)}
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
struct precond {
    typedef amgcl::backend::builtin<double> backend_type;
    typedef backend_type::matrix matrix;
    typedef backend_type::vector vector;
    typedef boost::iterator_range<double*> range;

    virtual void apply(const vector& rhs, vector &x) const = 0;
    virtual const matrix& system_matrix() const = 0;
    virtual std::string repr() const = 0;

    const precond& matvec() const { return *this; }

    py::array_t<double> call(py::array_t<double> rhs) const {
        vector x(rhs.size(), true);
        vector f(rhs.data(), rhs.data() + rhs.size());
        this->apply(f, x);
        return make_array(x.size(), x.data());
    }
};

//---------------------------------------------------------------------------
class solver
    : public amgcl::runtime::solver::wrapper<amgcl::backend::builtin<double> >
{
    public:
        solver(const precond &P, py::dict prm)
            : S(amgcl::backend::rows(P.system_matrix()), make_ptree(prm)), P(P)
        {}

        py::array_t<double> solve(
                py::array_t<int>    _ptr,
                py::array_t<int>    _col,
                py::array_t<double> _val,
                py::array_t<double> _rhs
                ) const
        {
            auto ptr = make_range(_ptr);
            auto col = make_range(_col);
            auto val = make_range(_val);
            auto rhs = make_range(_rhs);

            size_t n = boost::size(rhs);
            std::vector<double> x(n, 0.0);

            std::tie(iters, error) = (*this)(
                    std::make_tuple(n, ptr, col, val), P,
                    std::vector<double>(boost::begin(rhs), boost::end(rhs)), x
                    );

            return make_array(x.size(), x.data());
        }

        py::array_t<double> solve(py::array_t<double> _rhs) const {
            auto rhs = make_range(_rhs);
            std::vector<double> x(boost::size(rhs), 0.0);
            std::tie(iters, error) = (*this)(
                    P, std::vector<double>(boost::begin(rhs), boost::end(rhs)), x);
            return make_array(x.size(), x.data());
        }

        int iterations() const {
            return iters;
        }

        double residual() const {
            return error;
        }

    private:
        typedef amgcl::runtime::solver::wrapper<amgcl::backend::builtin<double> > S;

        const precond &P;

        mutable int    iters;
        mutable double error;
};

//---------------------------------------------------------------------------
template <class Precond>
class amg_precond: public precond
{
    public:
        amg_precond(
            py::array_t<int>    _ptr,
            py::array_t<int>    _col,
            py::array_t<double> _val,
            py::dict prm
           )
        {
            auto ptr = make_range(_ptr);
            auto col = make_range(_col);
            auto val = make_range(_val);

            P = std::make_shared<Precond>(
                    std::make_tuple(boost::size(ptr) - 1, ptr, col, val),
                    make_ptree(prm)
                    );
        }

        void apply(const precond::vector& rhs, precond::vector &x) const {
            P->apply(rhs, x);
        }

        const matrix& system_matrix() const {
            return P->system_matrix();
        }

        std::string repr() const {
            std::ostringstream s;
            s << *P;
            return s.str();
        }

    private:
        std::shared_ptr<Precond> P;
};

//---------------------------------------------------------------------------
PYBIND11_MODULE(pyamgcl_ext, m) {
    py::class_<precond> Precond(m, "precond");
    Precond
        .def("__repr__", &precond::repr)
        .def("__call__", &precond::call, "Applies preconditioner to the given vector", py::arg("rhs"))
        .def_property_readonly("matvec", &precond::matvec);
        ;

    typedef amgcl::backend::builtin<double> Backend;

    typedef amg_precond<amgcl::runtime::preconditioner<Backend>> AMG;
    py::class_<AMG>(m, "amgcl", Precond)
        .def(py::init<py::array_t<int>, py::array_t<int>, py::array_t<double>, py::dict>());

    py::class_<solver>(m, "solver")
        .def(py::init<
                const precond&,
                py::dict
                >()
            )
        .def("__call__", (py::array_t<double> (solver::*)(py::array_t<double>) const) &solver::solve)
        .def("__call__", (py::array_t<double> (solver::*)(
                        py::array_t<int>, py::array_t<int>, py::array_t<double>, py::array_t<double>) const
                    ) &solver::solve)
        .def_property_readonly("iters", &solver::iterations)
        .def_property_readonly("error", &solver::residual)
        ;
}
