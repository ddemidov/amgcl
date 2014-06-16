#ifndef AMGCL_MPI_DEFLATED_CG_HPP
#define AMGCL_MPI_DEFLATED_CG_HPP

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
 * \file   amgcl/mpi/deflated_cg.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed conjugate gradients method with subdomain deflation.
 */

#include <vector>
#include <map>

#include <boost/range/numeric.hpp>
#include <boost/multi_array.hpp>
#include <boost/mpi.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/mpi/dist_crs.hpp>

namespace amgcl {
namespace mpi {

/// Distributed conjugate gradients method with subdomain deflation.
/**
 * \sa \cite Frank2001
 */
template <typename real>
class deflated_cg {
    public:
        template <class PtrRange, class ColRange, class ValRange>
        deflated_cg(
                MPI_Comm mpi_comm,
                long nrows,
                const PtrRange &ptr,
                const ColRange &col,
                const ValRange &val
                )
            : comm(mpi_comm, boost::mpi::comm_attach), nrows(nrows),
              A(mpi_comm, nrows, ptr, col, val),
              r(nrows), s(nrows), p(nrows), q(nrows)
        {}

        void operator()(
                const std::vector<real> &rhs,
                std::vector<real> &x) const
        {
            amgcl::backend::copy(rhs, r);
            A.mul(-1, x, 1, r);

            real rho1 = 0, rho2 = 0;
            real norm_of_rhs = norm(rhs);

            if (norm_of_rhs == 0) {
                amgcl::backend::clear(x);
                return;
            }

            amgcl::amg<
                amgcl::backend::builtin<double>,
                amgcl::coarsening::smoothed_aggregation<
                    amgcl::coarsening::plain_aggregates
                    >,
                amgcl::relaxation::spai0
                > amg( A.local() );

            real res;
            for(long iter = 0; (res = norm(r) / norm_of_rhs) > 1e-6 && iter < 1000; ++iter) {
                if (comm.rank() == 0)
                    std::cout << iter << ": " << std::scientific << res << "\r" << std::flush;

                amg(r, s);

                rho2 = rho1;
                rho1 = inner_product(r, s);

                if (iter)
                    amgcl::backend::axpby(1, s, rho1 / rho2, p);
                else
                    amgcl::backend::copy(s, p);

                A.mul(1, p, 0, q);

                real alpha = rho1 / inner_product(q, p);

                amgcl::backend::axpby( alpha, p, 1, x);
                amgcl::backend::axpby(-alpha, q, 1, r);
            }

            if (comm.rank() == 0) std::cout << std::endl;
        }

    private:
        boost::mpi::communicator comm;
        long nrows;

        amgcl::mpi::dist_crs<real> A;
        mutable std::vector<real> r, s, p, q;

        real inner_product(
                const std::vector<real> &x,
                const std::vector<real> &y) const
        {
            real lsum = amgcl::backend::inner_product(x, y);
            real gsum;

            all_reduce( comm, lsum, gsum, std::plus<real>() );

            return gsum;
        }

        real norm(const std::vector<real> &x) const {
            return sqrt(inner_product(x, x));
        }
};


} // namespace mpi
} // namespace amgcl

#endif
