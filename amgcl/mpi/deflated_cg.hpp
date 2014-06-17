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

#include <boost/range/algorithm.hpp>
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
        template <class Matrix>
        deflated_cg(MPI_Comm mpi_comm, const Matrix &Astrip)
            : comm(mpi_comm, boost::mpi::comm_attach), nrows(backend::rows(Astrip)),
              A(mpi_comm, Astrip),
              r(nrows), s(nrows), p(nrows), q(nrows), d(comm.size()),
              E(boost::extents[comm.size()][comm.size()]),
              amg( A.local() )
        {
            typedef typename backend::row_iterator<Matrix>::type row_iterator;

            std::vector<long> domain(comm.size());
            all_gather(comm, nrows, domain.data());
            boost::partial_sum(domain, domain.begin());

            // Local contribution to E = transp(Z) A Z:
            std::vector<real> e(comm.size(), 0);

            // Compute local contribution to E = (Z^t A Z),
            for(long i = 0; i < nrows; ++i) {
                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    long d = boost::upper_bound(domain, a.col()) - domain.begin();
                    e[d] += a.value();
                }
            }

            // Exchange rows of E.
            all_gather( comm, e.data(), comm.size(), E.data() );

            // Invert E.
            detail::gaussj(comm.size(), E.data());
        }

        template <class VectorRHS, class VectorX>
        void operator()(const VectorRHS &rhs, VectorX &x) const {
            amgcl::backend::copy(rhs, r);
            A.mul(-1, x, 1, r);
            real P = premul_with_P(r);

            for(long i = 0; i < nrows; ++i) x[i] += P;

            amgcl::backend::copy(rhs, r);
            A.mul(-1, x, 1, r);

            amg(r, s);
            A.mul(1, s, 0, q);

            P = premul_with_P(q);
            for(long i = 0; i < nrows; ++i) p[i] = s[i] - P;

            real rho1 = inner_product(r, s);
            real rho2 = 0;

            for(long iter = 0; fabs(rho1) > 1e-6 && iter < 100; ++iter) {

                A.mul(1, p, 0, q);
                real alpha = rho1 / inner_product(q, p);

                amgcl::backend::axpby( alpha, p, 1, x);
                amgcl::backend::axpby(-alpha, q, 1, r);

                amg(r, s);

                rho2 = rho1;
                rho1 = inner_product(r, s);

                real beta = rho1 / rho2;

                A.mul(1, s, 0, q);
                P = premul_with_P(q);
                for(long i = 0; i < nrows; ++i) p[i] = s[i] + beta * p[i] - P;

                if (comm.rank() == 0) std::cout << iter << ": " << std::scientific << fabs(rho1) << std::endl;
            }
        }

    private:
        boost::mpi::communicator comm;
        long nrows;

        amgcl::mpi::dist_crs<real> A;
        mutable std::vector<real> r, s, p, q, d;

        boost::multi_array<real, 2> E;

        amgcl::amg<
            amgcl::backend::builtin<double>,
            amgcl::coarsening::smoothed_aggregation<
                amgcl::coarsening::plain_aggregates
                >,
            amgcl::relaxation::spai0
            > amg;

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

        template<class VectorX>
        real premul_with_P(VectorX &x) const {
            real sum = backend::sum(x);
            all_gather(comm, sum, d.data());

            sum = 0;
            for(long j = 0; j < comm.size(); ++j)
                sum += E[comm.rank()][j] * d[j];
            return sum;
        }
};


} // namespace mpi
} // namespace amgcl

#endif
