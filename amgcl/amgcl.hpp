#ifndef AMGCL_AMGCL_HPP
#define AMGCL_AMGCL_HPP

/*
The MIT License

Copyright (c) 2012 Denis Demidov <ddemidov@ksu.ru>

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
 * \file   amgcl.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Generic algebraic multigrid framework.
 */

/**
\mainpage amgcl Generic algebraic multigrid framework.

This is a simple and generic AMG hierarchy builder (and a work in progress).
May be used as a standalone solver or as a preconditioner. CG and BiCGStab
iterative solvers are provided. Solvers from <a
href="http://viennacl.sourceforge.net">ViennaCL</a> are supported as well.

<a href="https://github.com/ddemidov/vexcl">VexCL</a>, <a
href="http://viennacl.sourceforge.net">ViennaCL</a>, or <a
href="http://eigen.tuxfamily.org">Eigen</a> matrix/vector
containers may be used with built-in and ViennaCL's solvers. See
<a href="https://github.com/ddemidov/amgcl/blob/master/examples/vexcl.cpp">examples/vexcl.cpp</a>,
<a href="https://github.com/ddemidov/amgcl/blob/master/examples/viennacl.cpp">examples/viennacl.cpp</a> and
<a href="https://github.com/ddemidov/amgcl/blob/master/examples/eigen.cpp">examples/eigen.cpp</a> for respective examples.

\section setup AMG hierarchy building

Constructor of amgcl::solver<> object builds the multigrid hierarchy based on
algebraic information contained in the system matrix:

\code
// amgcl::sparse::matrix<double, int> A;
// or
// amgcl::sparse::matrix_map<double, int> A;
amgcl::solver<
    double,                 // Scalar type
    int,                    // Index type of the matrix
    amgcl::interp::classic, // Interpolation kind
    amgcl::level::cpu       // Where to store the hierarchy
> amg(A);
\endcode

Currently supported interpolation schemes are amgcl::interp::classic and
amgcl::interp::aggregation<amgcl::aggr::plain>. The aggregation scheme uses
less memory and is set up faster than classic interpolation, but its
convergence rate is slower. It is well suited for VexCL or ViennaCL containers,
where solution phase is accelerated by the OpenCL technology and, therefore,
the cost of the setup phase is much more important.

\code
amgcl::solver<
    double, int,
    amgcl::interp::aggregation<amgcl::aggr::plain>,
    amgcl::level::vexcl
> amg(A);
\endcode

\section solution Solution

Once the hierarchy is constructed, it may be repeatedly used to solve the
linear system for different right-hand sides:

\code
// std::vector<double> rhs, x;

auto conv = amg.solve(rhs, x);

std::cout << "Iterations: " << std::get<0>(conv) << std::endl
          << "Error:      " << std::get<1>(conv) << std::endl;
\endcode

Using the AMG as a preconditioner with a Krylov subspace method like conjugate
gradients works even better:
\code
// Eigen::VectorXd rhs, x;

auto conv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag());
\endcode

Types of right-hand side and solution vectors should be compatible with the
level type used for construction of the AMG hierarchy. For example,
if amgcl::level::vexcl is used as a storage backend, then vex::SpMat<> and
vex::vector<> types have to be used when solving:

\code
// vex::SpMat<double,int> Agpu;
// vex::vector<double> rhs, x;

auto conv = amgcl::solve(Agpu, rhs, amg, x, amgcl::cg_tag());
\endcode

\section install Installation

The library is header-only, so there is nothing to compile or link to. You just
need to copy amgcl folder somewhere and tell your compiler to scan it for
include files.
*/

#include <tuple>
#include <list>

#include <amgcl/spmat.hpp>
#include <amgcl/interp_classic.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/profiler.hpp>

/// Primary namespace for the library.
namespace amgcl {

/// Interpolation-related types and functions.
namespace interp {

/// Galerkin operator.
struct galerkin_operator {
    template <class spmat, class Params>
    static spmat apply(const spmat &R, const spmat &A, const spmat &P,
            const Params &prm)
    {
        return sparse::prod(sparse::prod(R, A), P);
    }
};

/// Returns coarse level construction scheme for a given interpolation scheme.
/**
 * By default, Galerkin operator is used to construct coarse level from system
 * matrix, restriction and prolongation operators:
 * \f[A^H = R A^h P.\f] Usually, \f$R = P^T.\f$
 *
 * \param Interpolation interpolation scheme.
 */
template <class Interpolation>
struct coarse_operator {
    typedef galerkin_operator type;
};

} // namespace interp

/// Algebraic multigrid method.
/**
 * \param value_t  Type for matrix entries.
 * \param index_t  Type for matrix indices. Should be signed integral type.
 * \param interp_t Interpolation scheme.  Possible choices:
 *                 amgcl::interp::classic and amgcl::interp::aggregation.
 * \param level_t  Class for storing the hierarchy level structure. Possible
 *                 choices: amgcl::level::cpu, amgcl::level::vexcl,
 *                 amgcl::level::ViennaCL<>.
 */
template <
    typename value_t  = double,
    typename index_t  = long long,
    typename interp_t = interp::classic,
    typename level_t  = level::cpu
    >
class solver {
    private:
        typedef sparse::matrix<value_t, index_t> matrix;
        typedef typename level_t::template instance<value_t, index_t> level_type;

    public:
        /// Parameters for AMG components.
        struct params {
            unsigned coarse_enough; ///< When level is coarse enough to be solved directly.

            typename interp_t::params interp; ///< Interpolation parameters.
            typename level_t::params  level;  ///< Level/Solution parameters.

            params() : coarse_enough(300) { }
        };

        /// Constructs the AMG hierarchy from the system matrix.
        /** 
         * The input matrix is copied here and may be freed afterwards.
         *
         * \param A   The system matrix. Should be convertible to
         *            amgcl::sparse::matrix<>.
         * \param prm Parameters controlling the setup and solution phases.
         *
         * \sa amgcl::sparse::map()
         */
        template <typename spmat>
        solver(const spmat &A, const params &prm = params()) : prm(prm)
        {
            static_assert(std::is_signed<index_t>::value,
                    "Matrix index type should be signed");

            build_level(std::move(matrix(A)), prm);
        }

        /// The AMG hierarchy is used as a standalone solver.
        /** 
         * The vector types should be compatible with level_t:
         *
         * -# Any type with operator[] should work on a CPU.
         * -# vex::vector<value_t> should be used with level::vexcl.
         * -# viennacl::vector<value_t> should be used with level::ViennaCL.
         *
         * \param rhs Right-hand side.
         * \param x   Solution. Contains an initial approximation on input, and
         *            the approximated solution on output.
         */
        template <class vector1, class vector2>
        std::tuple< int, value_t > solve(const vector1 &rhs, vector2 &x) const {
            int     iter = 0;
            value_t res  = 2 * prm.level.tol;

            for(; res > prm.level.tol && iter < prm.level.maxiter; ++iter) {
                apply(rhs, x);
                res = hier.front().resid(rhs, x);
            }

            return std::make_tuple(iter, res);
        }

        /// Performs single multigrid cycle.
        /**
         * Is intended to be used as a preconditioner with iterative methods.
         *
         * The vector types should be compatible with level_t:
         *
         * -# Any type with operator[] should work on a CPU.
         * -# vex::vector<value_t> should be used with level::vexcl.
         * -# viennacl::vector<value_t> should be used with level::ViennaCL.
         *
         * \param rhs Right-hand side.
         * \param x   Solution. Contains an initial approximation on input, and
         *            the approximated solution on output.
         */
        template <class vector1, class vector2>
        void apply(const vector1 &rhs, vector2 &x) const {
            level_type::cycle(hier.begin(), hier.end(), prm.level, rhs, x);
        }

    private:
        void build_level(matrix &&A, const params &prm, unsigned nlevel = 0)
        {
#ifdef AMGCL_PROFILING
            std::cout << A.rows << std::endl;
#endif
            if (A.rows <= prm.coarse_enough) {
                hier.emplace_back(std::move(A), std::move(sparse::inverse(A)), prm.level, nlevel);
            } else {
                TIC("interp");
                matrix P = interp_t::interp(A, prm.interp);
                TOC("interp");

                TIC("transp");
                matrix R = sparse::transpose(P);
                TOC("transp");

                TIC("coarse operator");
                matrix a = interp::coarse_operator<interp_t>::type::apply(
                        R, A, P, prm.interp);
                TOC("coarse operator");

                hier.emplace_back(std::move(A), std::move(P), std::move(R), prm.level, nlevel);

                build_level(std::move(a), prm, nlevel + 1);
            }
        }

        params prm;
        std::list< level_type > hier;
};

} // namespace amgcl

#endif
