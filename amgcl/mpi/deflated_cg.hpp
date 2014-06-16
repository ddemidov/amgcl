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
#include <amgcl/coarsening/plain_aggregates.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
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
              r(nrows), s(nrows), p(nrows), q(nrows),
              df(comm.size()), dx(comm.size()),
              E(boost::extents[comm.size()][comm.size()]),
              amg( A.local() )
        {
            typedef typename backend::row_iterator<Matrix>::type row_iterator;

            std::vector<long> domain(comm.size());
            all_gather(comm, nrows, domain.data());
            boost::partial_sum(domain, domain.begin());

            // Local contribution to E = transp(Z) A Z:
            std::vector<real> e(comm.size(), 0);

            // Compute:
            // 1. local contribution to E = (Z^t A Z),
            // 2. Sparsity pattern of matrix AZ.
            AZ.nrows = nrows;
            AZ.ncols = comm.size();
            AZ.ptr.resize(nrows + 1, 0);

            std::vector<long> marker(comm.size(), -1);
            for(long i = 0; i < nrows; ++i) {
                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    long c = a.col();
                    long d = boost::upper_bound(domain, c) - domain.begin();
                    real v = a.value();

                    e[d] += v;

                    if (marker[d] != i) {
                        marker[d] = i;
                        ++( AZ.ptr[i + 1] );
                    }
                }
            }

            // Exchange rows of E.
            all_gather( comm, e.data(), comm.size(), E.data() );

            // Invert E.
            detail::gaussj(comm.size(), E.data());

            // Finish construction of AZ
            boost::partial_sum(AZ.ptr, AZ.ptr.begin());
            AZ.col.resize( AZ.ptr.back() );
            AZ.val.resize( AZ.ptr.back() );
            boost::fill(marker, -1);

            for(long i = 0; i < nrows; ++i) {
                long row_beg = AZ.ptr[i];
                long row_end = row_beg;

                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    long c = a.col();
                    long d = boost::upper_bound(domain, c) - domain.begin();
                    real v = a.value();

                    if (marker[d] < row_beg) {
                        marker[d] = row_end;
                        AZ.col[row_end] = d;
                        AZ.val[row_end] = v;
                        ++row_end;
                    } else {
                        AZ.val[marker[d]] += v;
                    }
                }
            }

            AZt = backend::transpose(AZ);
        }

        template <class VectorRHS, class VectorX>
        void operator()(const VectorRHS &rhs, VectorX &x) const {
            amgcl::backend::copy(rhs, r);
            A.mul(-1, x, 1, r);
            premul_with_P(r);

            amg(r, s);
            backend::copy(s, p);

            real rho1 = inner_product(r, s);
            real rho2;

            for(long iter = 0; fabs(rho1) > 1e-6 && iter < 500; ++iter) {
                if (comm.rank() == 0)
                    std::cout
                        << iter << ": "
                        << std::scientific << fabs(rho1)
                        << "\r" << std::flush;

                A.mul(1, p, 0, q);
                premul_with_P(q);

                real alpha = rho1 / inner_product(q, p);

                amgcl::backend::axpby( alpha, p, 1, x);
                amgcl::backend::axpby(-alpha, q, 1, r);

                amg(r, s);

                rho2 = rho1;
                rho1 = inner_product(r, s);

                amgcl::backend::axpby(1, s, rho1 / rho2, p);
            }

            if (comm.rank() == 0) std::cout << std::endl;

            postprocess(rhs, x);
        }

    private:
        boost::mpi::communicator comm;
        long nrows;

        amgcl::mpi::dist_crs<real> A;
        mutable std::vector<real> r, s, p, q, df, dx;

        boost::multi_array<real, 2> E;
        backend::crs<real,long> AZ, AZt;

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
        void premul_with_P(VectorX &x) const {
            real sum_x = backend::sum(x);
            all_gather(comm, sum_x, df.data());

            for(long i = 0; i < comm.size(); ++i) {
                real sum = 0;
                for(long j = 0; j < comm.size(); ++j)
                    sum += E[i][j] * df[j];
                dx[i] = sum;
            }

            backend::spmv(-1, AZ, dx, 1, x);
        }

        template <class VectorRHS, class VectorX>
        void postprocess(const VectorRHS &rhs, VectorX &x) const {
            real sum_f = backend::sum(rhs);
            all_gather(comm, sum_f, df.data());

            real ef = 0;
            for(long j = 0; j < comm.size(); ++j)
                ef += E[comm.rank()][j] * df[j];

            backend::spmv(1, AZt, x, 0, dx);

            all_reduce( comm, dx.data(), comm.size(), df.data(), std::plus<real>() );

            real ex = 0;
            for(long j = 0; j < comm.size(); ++j)
                ex += E[j][comm.rank()] * df[j];

            for(long i = 0; i < nrows; ++i) x[i] += ef - ex;
        }
};


} // namespace mpi
} // namespace amgcl

#endif
