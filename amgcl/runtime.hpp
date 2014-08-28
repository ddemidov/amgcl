#ifndef AMGCL_RUNTIME_HPP
#define AMGCL_RUNTIME_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   amgcl/runtime.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Runtime-configurable wrappers around amgcl classes.
 */

#include <iostream>
#include <stdexcept>

#include <boost/type_traits.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/noncopyable.hpp>

#include <amgcl/amgcl.hpp>

#include <amgcl/backend/interface.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/pointwise_aggregates.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>


namespace amgcl {
namespace runtime {

namespace coarsening {
enum type {
    ruge_stuben,
    aggregation,
    smoothed_aggregation,
    smoothed_aggr_emin
};

std::ostream& operator<<(std::ostream &os, type c) {
    switch (c) {
        case ruge_stuben:
            return os << "ruge_stuben";
        case aggregation:
            return os << "aggregation";
        case smoothed_aggregation:
            return os << "smoothed_aggregation";
        case smoothed_aggr_emin:
            return os << "smoothed_aggr_emin";
        default:
            return os << "???";
    }
}

std::istream& operator>>(std::istream &in, type &c)
{
    namespace po = boost::program_options;

    std::string val;
    in >> val;

    if (val == "ruge_stuben")
        c = ruge_stuben;
    else if (val == "aggregation")
        c = aggregation;
    else if (val == "smoothed_aggregation")
        c = smoothed_aggregation;
    else if (val == "smoothed_aggr_emin")
        c = smoothed_aggr_emin;
    else
        throw std::invalid_argument("Invalid coarsening value");

    return in;
}

} // namespace coarsening

namespace relaxation {
enum type {
    gauss_seidel,
    ilu0,
    damped_jacobi,
    spai0,
    chebyshev
};

std::ostream& operator<<(std::ostream &os, type r)
{
    switch (r) {
        case gauss_seidel:
            return os << "gauss_seidel";
        case ilu0:
            return os << "ilu0";
        case damped_jacobi:
            return os << "damped_jacobi";
        case spai0:
            return os << "spai0";
        case chebyshev:
            return os << "chebyshev";
        default:
            return os << "???";
    }
}

std::istream& operator>>(std::istream &in, type &r)
{
    namespace po = boost::program_options;

    std::string val;
    in >> val;

    if (val == "gauss_seidel")
        r = gauss_seidel;
    else if (val == "ilu0")
        r = ilu0;
    else if (val == "damped_jacobi")
        r = damped_jacobi;
    else if (val == "spai0")
        r = spai0;
    else if (val == "chebyshev")
        r = chebyshev;
    else
        throw std::invalid_argument("Invalid relaxation value");

    return in;
}

} // namespace relaxation

namespace solver {
enum type {
    cg,
    bicgstab,
    bicgstabl,
    gmres
};

std::ostream& operator<<(std::ostream &os, type s)
{
    switch (s) {
        case cg:
            return os << "cg";
        case bicgstab:
            return os << "bicgstab";
        case bicgstabl:
            return os << "bicgstabl";
        case gmres:
            return os << "gmres";
        default:
            return os << "???";
    }
}

std::istream& operator>>(std::istream &in, type &s)
{
    namespace po = boost::program_options;

    std::string val;
    in >> val;

    if (val == "cg")
        s = cg;
    else if (val == "bicgstab")
        s = bicgstab;
    else if (val == "bicgstabl")
        s = bicgstabl;
    else if (val == "gmres")
        s = gmres;
    else
        throw std::invalid_argument("Invalid solver value");

    return in;
}

} // namespace solver

namespace detail {

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    template <class> class Relaxation,
    class Func
    >
inline void process_amg(const Func &func) {
    typedef amgcl::amg<Backend, Coarsening, Relaxation> AMG;
    func.template process<AMG>();
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Coarsening,
    class Func
    >
inline void process_amg(
        runtime::relaxation::type relaxation,
        const Func &func
        )
{
    switch (relaxation) {
        case runtime::relaxation::gauss_seidel:
            process_amg<
                Backend,
                Coarsening,
                amgcl::relaxation::gauss_seidel
                >(func);
            break;
        case runtime::relaxation::ilu0:
            process_amg<
                Backend,
                Coarsening,
                amgcl::relaxation::ilu0
                >(func);
            break;
        case runtime::relaxation::damped_jacobi:
            process_amg<
                Backend,
                Coarsening,
                amgcl::relaxation::damped_jacobi
                >(func);
            break;
        case runtime::relaxation::spai0:
            process_amg<
                Backend,
                Coarsening,
                amgcl::relaxation::spai0
                >(func);
            break;
        case runtime::relaxation::chebyshev:
            process_amg<
                Backend,
                Coarsening,
                amgcl::relaxation::chebyshev
                >(func);
            break;
    }
}

//---------------------------------------------------------------------------
template <
    class Backend,
    class Func
    >
inline void process_amg(
        runtime::coarsening::type coarsening,
        runtime::relaxation::type relaxation,
        const Func &func
        )
{
    switch (coarsening) {
        case runtime::coarsening::ruge_stuben:
            process_amg<
                Backend,
                amgcl::coarsening::ruge_stuben
                >(relaxation, func);
            break;
        case runtime::coarsening::aggregation:
            process_amg<
                Backend,
                amgcl::coarsening::aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
        case runtime::coarsening::smoothed_aggregation:
            process_amg<
                Backend,
                amgcl::coarsening::smoothed_aggregation<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
        case runtime::coarsening::smoothed_aggr_emin:
            process_amg<
                Backend,
                amgcl::coarsening::smoothed_aggr_emin<
                    amgcl::coarsening::pointwise_aggregates
                    >
                >(relaxation, func);
            break;
    }
}

template <class Matrix>
struct amg_create {
    typedef boost::property_tree::ptree params;

    void* &handle;

    const Matrix &A;
    const params &p;

    amg_create(void* &handle, const Matrix &A, const params &p)
        : handle(handle), A(A), p(p) {}

    template <class AMG>
    void process() const {
        handle = static_cast<void*>( new AMG(A, p) );
    }
};

struct amg_destroy {
    void *handle;

    amg_destroy(void *handle) : handle(handle) {}

    template <class AMG>
    void process() const {
        delete static_cast<AMG*>(handle);
    }
};

template <class Vec1, class Vec2>
struct amg_cycle {
    void       *handle;

    Vec1 const &rhs;
    Vec2       &x;

    amg_cycle(void *handle, const Vec1 &rhs, Vec2 &x)
        : handle(handle), rhs(rhs), x(x) {}

    template <class AMG>
    void process() const {
        static_cast<AMG*>(handle)->cycle(rhs, x);
    }
};

template <class Vec1, class Vec2>
struct amg_apply {
    void       *handle;

    Vec1 const &rhs;
    Vec2       &x;

    amg_apply(void *handle, const Vec1 &rhs, Vec2 &x)
        : handle(handle), rhs(rhs), x(x) {}

    template <class AMG>
    void process() const {
        static_cast<AMG*>(handle)->apply(rhs, x);
    }
};

template <class Matrix>
struct amg_top_matrix {
    void * handle;
    mutable const Matrix * matrix;

    amg_top_matrix(void * handle) : handle(handle), matrix(0) {}

    template <class AMG>
    void process() const {
        matrix = &(static_cast<AMG*>(handle)->top_matrix());
    }
};

struct amg_print {
    void * handle;
    std::ostream &os;

    amg_print(void * handle, std::ostream &os) : handle(handle), os(os) {}

    template <class AMG>
    void process() const {
        os << *static_cast<AMG*>(handle);
    }
};

} // namespace detail

template <class Backend>
class amg : boost::noncopyable {
    public:
        typedef Backend backend_type;

        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix     matrix;
        typedef typename Backend::vector     vector;

        typedef boost::property_tree::ptree params;

        template <class Matrix>
        amg(
                runtime::coarsening::type coarsening,
                runtime::relaxation::type relaxation,
                const Matrix &A,
                const params &prm = params()
           ) : coarsening(coarsening), relaxation(relaxation), handle(0)
        {
            runtime::detail::process_amg<Backend>(
                    coarsening, relaxation,
                    runtime::detail::amg_create<Matrix>(handle, A, prm)
                    );
        }

        ~amg() {
            runtime::detail::process_amg<Backend>(
                    coarsening, relaxation,
                    runtime::detail::amg_destroy(handle)
                    );
        }

        template <class Vec1, class Vec2>
        void cycle(const Vec1 &rhs, Vec2 &x) const {
            runtime::detail::process_amg<Backend>(
                    coarsening, relaxation,
                    runtime::detail::amg_cycle<Vec1, Vec2>(handle, rhs, x)
                    );
        }

        template <class Vec1, class Vec2>
        void apply(const Vec1 &rhs, Vec2 &x) const {
            runtime::detail::process_amg<Backend>(
                    coarsening, relaxation,
                    runtime::detail::amg_apply<Vec1, Vec2>(handle, rhs, x)
                    );
        }

        const matrix& top_matrix() const {
            runtime::detail::amg_top_matrix<matrix> top(handle);
            runtime::detail::process_amg<Backend>(
                    coarsening, relaxation, top
                    );
            return *top.matrix;
        }

        friend std::ostream& operator<<(std::ostream &os, const amg &a)
        {
            runtime::detail::process_amg<Backend>(
                    a.coarsening, a.relaxation,
                    runtime::detail::amg_print(a.handle, os)
                    );
            return os;
        }
    private:
        const runtime::coarsening::type coarsening;
        const runtime::relaxation::type relaxation;

        void *handle;
};

namespace detail {

template <
    class Backend,
    class Func
    >
inline void process_solver(
        runtime::solver::type solver,
        const Func &func
        )
{
    switch (solver) {
        case runtime::solver::cg:
            {
                typedef amgcl::solver::cg<Backend> Solver;
                func.template process<Solver>();
            }
            break;
        case runtime::solver::bicgstab:
            {
                typedef amgcl::solver::bicgstab<Backend> Solver;
                func.template process<Solver>();
            }
            break;
        case runtime::solver::bicgstabl:
            {
                typedef amgcl::solver::bicgstabl<Backend> Solver;
                func.template process<Solver>();
            }
            break;
        case runtime::solver::gmres:
            {
                typedef amgcl::solver::gmres<Backend> Solver;
                func.template process<Solver>();
            }
            break;
    }
}

struct solver_create {
    typedef boost::property_tree::ptree params;

    void * &handle;
    const params &prm;
    size_t n;

    solver_create(void * &handle, const params &prm, size_t n)
        : handle(handle), prm(prm), n(n) {}

    template <class Solver>
    void process() const {
        handle = static_cast<void*>( new Solver(n, prm,
                    prm.get_child("backend", amgcl::detail::empty_ptree()))
                );
    }
};

struct solver_destroy {
    void * handle;

    solver_destroy(void * handle) : handle(handle) {}

    template <class Solver>
    void process() const {
        delete static_cast<Solver*>(handle);
    }
};

template <
    class Backend,
    class Matrix,
    class Vec1,
    class Vec2
    >
struct solver_solve {
    typedef typename Backend::value_type value_type;

    void * handle;

    runtime::amg<Backend> const &P;

    Matrix const &A;
    Vec1   const &rhs;
    Vec2         &x;

    size_t     &iters;
    value_type &resid;

    solver_solve(void * handle, const runtime::amg<Backend> &P,
            const Matrix &A, const Vec1 &rhs, Vec2 &x, size_t &iters,
            value_type &resid
            )
        : handle(handle), P(P), A(A), rhs(rhs), x(x),
          iters(iters), resid(resid)
    {}

    template <class Solver>
    void process() const {
        boost::tie(iters, resid) = static_cast<Solver*>(handle)->operator()(
                A, P, rhs, x);
    }
};

} // namespace detail

template <class Backend>
class make_solver : boost::noncopyable {
    public:
        typedef typename Backend::value_type value_type;
        typedef boost::property_tree::ptree params;

        template <class Matrix>
        make_solver(
                runtime::coarsening::type coarsening,
                runtime::relaxation::type relaxation,
                runtime::solver::type     solver,
                const Matrix &A,
                const params &prm = params()
                )
            : P(coarsening, relaxation, A, prm), solver(solver), handle(0)
        {
            runtime::detail::process_solver<Backend>(
                    solver,
                    runtime::detail::solver_create(
                        handle, prm, amgcl::backend::rows(A)
                        )
                    );
        }

        ~make_solver() {
            runtime::detail::process_solver<Backend>(
                    solver,
                    runtime::detail::solver_destroy(handle)
                    );
        }

        template <class Matrix, class Vec1, class Vec2>
        boost::tuple<size_t, value_type> operator()(
                Matrix  const &A,
                Vec1    const &rhs,
                Vec2          &x
                ) const
        {
            size_t     iters = 0;
            value_type resid = 0;

            runtime::detail::process_solver<Backend>(
                    solver,
                    runtime::detail::solver_solve<Backend, Matrix, Vec1, Vec2>(
                        handle, P, A, rhs, x, iters, resid)
                    );

            return boost::make_tuple(iters, resid);
        }

        template <class Vec1, class Vec2>
        boost::tuple<size_t, value_type> operator()(
                Vec1    const &rhs,
                Vec2          &x
                ) const
        {
            return (*this)(P.top_matrix(), rhs, x);
        }

        const runtime::amg<Backend>& amg() const {
            return P;
        }
    private:
        runtime::amg<Backend>        P;
        const runtime::solver::type  solver;
        void                        *handle;
};

} // namespace runtime
} // namespace amgcl


#endif
