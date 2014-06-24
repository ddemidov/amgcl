#ifndef AMGCL_MPI_DEFLATION_HPP
#define AMGCL_MPI_DEFLATION_HPP

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
 * \file   amgcl/mpi/deflatedion.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Subdomain deflation utilities.
 */

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/numeric.hpp>
#include <boost/multi_array.hpp>
#include <boost/mpi.hpp>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/detail/inverse.hpp>

namespace amgcl {

/// Algorithms and structures for distributed computing.
namespace mpi {

namespace detail {
struct mpi_inner_product {
    boost::mpi::communicator comm;

    mpi_inner_product(MPI_Comm comm) : comm(comm, boost::mpi::comm_attach) {}

    template <class Vec1, class Vec2>
    typename backend::value_type<Vec1>::type
    operator()(const Vec1 &x, const Vec2 &y) const {
        typedef typename backend::value_type<Vec1>::type value_type;

        value_type lsum = backend::inner_product(x, y);
        value_type gsum;

        all_reduce( comm, lsum, gsum, std::plus<value_type>() );

        return gsum;
    }
};

} // namespace detail

/// Constant deflation vectors.
struct constant_deflation {
    int dim() const { return 1; }
    int operator()(long row, int j) const { return 1; }
};

/// Distributed solver with subdomain deflation.
/**
 * \sa \cite Frank2001
 */
template <
    class                         Backend,
    class                         Coarsening,
    template <class> class        Relax,
    template <class, class> class IterativeSolver
    >
class subdomain_deflation {
    public:
        typedef amg<
            Backend, Coarsening, Relax
            > AMG;

        typedef IterativeSolver<
            Backend, detail::mpi_inner_product
            > Solver;

        typedef
            typename AMG::params
            AMG_params;

        typedef
            typename Solver::params
            Solver_params;

        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix     matrix;
        typedef typename Backend::vector     vector;

        template <class Matrix, class DeflationVectors>
        subdomain_deflation(
                MPI_Comm mpi_comm,
                const Matrix &Astrip,
                const DeflationVectors &def_vec,
                const AMG_params    &amg_params    = AMG_params(),
                const Solver_params &solver_params = Solver_params()
                )
        : comm(mpi_comm, boost::mpi::comm_attach),
          nrows(backend::rows(Astrip)), nz(comm.size() * def_vec.dim()),
          df( nz ), dx( nz ),
          E( boost::extents[nz][nz] ),
          q( Backend::create_vector(nrows, amg_params.backend) ),
          dd( Backend::create_vector(nz, amg_params.backend) ),
          Z( def_vec.dim() )
        {
            typedef typename backend::row_iterator<Matrix>::type row_iterator;
            typedef backend::crs<value_type, long> build_matrix;

            boost::shared_ptr<build_matrix> aloc = boost::make_shared<build_matrix>();
            boost::shared_ptr<build_matrix> arem = boost::make_shared<build_matrix>();
            boost::shared_ptr<build_matrix> az   = boost::make_shared<build_matrix>();

            // Get sizes of each domain in comm.
            std::vector<long> domain(comm.size() + 1, 0);
            all_gather(comm, nrows, &domain[1]);
            boost::partial_sum(domain, domain.begin());
            long chunk_start = domain[comm.rank()];

            // Fill deflation vectors.
            {
                std::vector<value_type> z(nrows);
                for(int j = 0; j < def_vec.dim(); ++j) {
                    for(long i = 0; i < nrows; ++i)
                        z[i] = def_vec(i + chunk_start, j);
                    Z[j] = Backend::copy_vector(z, amg_params.backend);
                }
            }

            // Number of nonzeros in local and remote parts of the matrix.
            long loc_nnz = 0, rem_nnz = 0;

            // Local contribution to E.
            boost::multi_array<value_type, 2> erow(boost::extents[def_vec.dim()][nz]);
            std::fill_n(erow.data(), erow.num_elements(), 0);

            // Maps remote column numbers to local ids:
            std::map<long, long> rc;
            std::map<long, long>::iterator rc_it = rc.begin();

            // First pass over Astrip rows:
            // 1. Count local and remote nonzeros,
            // 2. Build set of remote columns,
            // 3. Compute local contribution to E = (Z^t A Z),
            // 4. Build sparsity pattern of matrix AZ.
            az->nrows = nrows;
            az->ncols = nz;
            az->ptr.resize(nrows + 1, 0);

            std::vector<long> marker(nz, -1);
            for(long i = 0; i < nrows; ++i) {
                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    long       c = a.col();
                    value_type v = a.value();

                    // Domain the column belongs to
                    long d = boost::upper_bound(domain, c) - domain.begin() - 1;

                    if (d == comm.rank()) {
                        ++loc_nnz;
                    } else {
                        ++rem_nnz;
                        rc_it = rc.insert(rc_it, std::make_pair(c, 0));
                    }

                    for(long ii = 0; ii < def_vec.dim(); ++ii) {
                        for(long jj = 0; jj < def_vec.dim(); ++jj) {
                            long k = d * def_vec.dim() + jj;

                            erow[ii][k] += v * def_vec(i + chunk_start, ii) * def_vec(c, jj);

                            if (marker[k] != i) {
                                marker[k] = i;
                                ++( az->ptr[i + 1] );
                            }
                        }
                    }
                }
            }

            // Exchange rows of E.
            all_gather( comm, erow.data(), nz * def_vec.dim(), E.data() );

            // Invert E.
            amgcl::detail::inverse(nz, E.data());

            // Find out:
            // 1. How many columns do we need from each process,
            // 2. What columns do we need from them.
            //
            // Renumber remote columns while at it.
            std::vector<long> num_recv(comm.size(), 0);
            std::vector<long> recv_cols;
            recv_cols.reserve(rc.size());
            long id = 0, cur_nbr = 0;
            for(rc_it = rc.begin(); rc_it != rc.end(); ++rc_it) {
                rc_it->second = id++;
                recv_cols.push_back(rc_it->first);

                while(rc_it->first >= domain[cur_nbr + 1]) cur_nbr++;
                num_recv[cur_nbr]++;
            }

            // Second pass over Astrip rows:
            // 1. Build local and remote matrix parts.
            // 2. Build AZ matrix.
            aloc->nrows = nrows;
            aloc->ncols = nrows;
            aloc->ptr.reserve(nrows + 1);
            aloc->col.reserve(loc_nnz);
            aloc->val.reserve(loc_nnz);
            aloc->ptr.push_back(0);

            arem->nrows = nrows;
            arem->ncols = rc.size();
            arem->ptr.reserve(nrows + 1);
            arem->col.reserve(rem_nnz);
            arem->val.reserve(rem_nnz);
            arem->ptr.push_back(0);

            boost::partial_sum(az->ptr, az->ptr.begin());
            az->col.resize(az->ptr.back());
            az->val.resize(az->ptr.back());
            boost::fill(marker, -1);

            for(long i = 0; i < nrows; ++i) {
                long az_row_beg = az->ptr[i];
                long az_row_end = az_row_beg;

                for(row_iterator a = backend::row_begin(Astrip, i); a; ++a) {
                    long       c = a.col();
                    value_type v = a.value();

                    // Domain the column belongs to
                    long d = boost::upper_bound(domain, c) - domain.begin() - 1;

                    if ( d == comm.rank() ) {
                        aloc->col.push_back(c - chunk_start);
                        aloc->val.push_back(v);
                    } else {
                        arem->col.push_back(rc[c]);
                        arem->val.push_back(v);
                    }

                    for(long j = 0; j < def_vec.dim(); ++j) {
                        long k = d * def_vec.dim() + j;

                        if (marker[k] < az_row_beg) {
                            marker[k] = az_row_end;
                            az->col[az_row_end] = k;
                            az->val[az_row_end] = v * def_vec(c, j);
                            ++az_row_end;
                        } else {
                            az->val[marker[k]] += v * def_vec(c, j);
                        }
                    }
                }

                aloc->ptr.push_back(aloc->col.size());
                arem->ptr.push_back(arem->col.size());
            }

            // Set up communication pattern.
            boost::multi_array<long, 2> comm_matrix(
                    boost::extents[comm.size()][comm.size()]
                    );

            // Who sends to whom and how many
            all_gather(comm, num_recv.data(), comm.size(), comm_matrix.data());

            long snbr = 0, rnbr = 0, send_size = 0;
            for(int i = 0; i < comm.size(); ++i) {
                if (comm_matrix[comm.rank()][i]) {
                    ++rnbr;
                }

                if (comm_matrix[i][comm.rank()]) {
                    ++snbr;
                    send_size += comm_matrix[i][comm.rank()];
                }
            }

            recv.nbr.reserve(rnbr);
            recv.ptr.reserve(rnbr + 1);
            recv.val.resize(rc.size());
            recv.req.resize(rnbr);

            dv = Backend::create_vector( rc.size(), amg_params.backend );

            send.nbr.reserve(snbr);
            send.ptr.reserve(snbr + 1);
            send.val.resize(send_size);
            send.req.resize(snbr);

            std::vector<long> send_col(send_size);

            // Count how many columns to send and to receive.
            recv.ptr.push_back(0);
            send.ptr.push_back(0);
            for(int i = 0; i < comm.size(); ++i) {
                if (long nr = comm_matrix[comm.rank()][i]) {
                    recv.nbr.push_back( i );
                    recv.ptr.push_back( recv.ptr.back() + nr );
                }

                if (long ns = comm_matrix[i][comm.rank()]) {
                    send.nbr.push_back( i );
                    send.ptr.push_back( send.ptr.back() + ns );
                }
            }

            // What columns do you need from me?
            for(size_t i = 0; i < send.nbr.size(); ++i)
                send.req[i] = comm.irecv(send.nbr[i], tag_exc_vals,
                        &send_col[send.ptr[i]], comm_matrix[send.nbr[i]][comm.rank()]);

            // Here is what I need from you:
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                recv.req[i] = comm.isend(recv.nbr[i], tag_exc_vals,
                    &recv_cols[recv.ptr[i]], comm_matrix[comm.rank()][recv.nbr[i]]);

            wait_all(recv.req.begin(), recv.req.end());
            wait_all(send.req.begin(), send.req.end());

            // Shift columns to send to local numbering:
            BOOST_FOREACH(long &c, send_col) c -= chunk_start;

            // Create local AMG preconditioner.
            P = boost::make_shared<AMG>( *aloc, amg_params );

            // Create iterative solver instance.
            solve = boost::make_shared<Solver>(
                    nrows, solver_params, amg_params.backend,
                    detail::mpi_inner_product(mpi_comm)
                    );

            // Move matrices to backend.
            Arem = Backend::copy_matrix(arem, amg_params.backend);
            AZ   = Backend::copy_matrix(az,   amg_params.backend);

            // Columns gatherer. Will retrieve columns to send from backend.
            gather = boost::make_shared<typename Backend::gather>(
                    nrows, send_col, amg_params.backend);
        }

        template <class Vec1, class Vec2>
        boost::tuple<size_t, value_type>
        operator()(const Vec1 &rhs, Vec2 &x) const {
            boost::tuple<size_t, value_type> cnv = (*solve)(*this, *this, rhs, x);
            postprocess(rhs, x);
            return cnv;
        }

        template <class Vec1, class Vec2>
        void apply(const Vec1 &rhs, Vec2 &x) const {
            P->apply(rhs, x);
        }

        template <class Vec1, class Vec2>
        void mul_n_project(value_type alpha, const Vec1 &x, value_type beta, Vec2 &y) const {
            mul(alpha, x, beta, y);
            project(y);
        }

        template <class Vec1, class Vec2, class Vec3>
        void residual(const Vec1 &f, const Vec2 &x, Vec3 &r) const {
            start_exchange(x);
            backend::residual(f, P->top_matrix(), x, r);

            finish_exchange();

            if (!recv.val.empty()) {
                backend::copy_to_backend(recv.val, *dv);
                backend::spmv(-1, *Arem, *dv, 1, r);
            }

            project(r);
        }
    private:
        static const int tag_exc_cols = 1001;
        static const int tag_exc_vals = 2001;

        boost::mpi::communicator comm;
        long nrows, nz;

        boost::shared_ptr<matrix> Arem;

        boost::shared_ptr<AMG>    P;
        boost::shared_ptr<Solver> solve;

        mutable std::vector<value_type> df, dx;
        boost::multi_array<value_type, 2> E;

        boost::shared_ptr<matrix> AZ;
        boost::shared_ptr<vector> q;
        boost::shared_ptr<vector> dd;
        boost::shared_ptr<vector> dv;
        std::vector< boost::shared_ptr<vector> > Z;

        boost::shared_ptr< typename Backend::gather > gather;

        struct {
            std::vector<long> nbr;
            std::vector<long> ptr;

            mutable std::vector<value_type>          val;
            mutable std::vector<boost::mpi::request> req;
        } recv;

        struct {
            std::vector<long> nbr;
            std::vector<long> ptr;

            mutable std::vector<value_type>          val;
            mutable std::vector<boost::mpi::request> req;
        } send;

        template <class Vec1, class Vec2>
        void mul(value_type alpha, const Vec1 &x, value_type beta, Vec2 &y) const {
            start_exchange(x);
            backend::spmv(alpha, P->top_matrix(), x, beta, y);

            finish_exchange();

            if (!recv.val.empty()) {
                backend::copy_to_backend(recv.val, *dv);
                backend::spmv(alpha, *Arem, *dv, 1, y);
            }
        }

        template <class Vector>
        void project(Vector &x) const {
            boost::fill(dx, 0);
            for(long j = 0; j < Z.size(); ++j)
                dx[j] += backend::inner_product(x, *Z[j]);
            all_gather(comm, dx.data(), Z.size(), df.data());

            for(long i = 0; i < nz; ++i) {
                value_type sum = 0;
                for(long j = 0; j < nz; ++j)
                    sum += E[i][j] * df[j];
                dx[i] = sum;
            }

            backend::copy_to_backend(dx, *dd);
            backend::spmv(-1, *AZ, *dd, 1, x);
        }

        template <class Vec1, class Vec2>
        void postprocess(const Vec1 &f, Vec2 &x) const {
            mul(1, x, 0, *q);

            boost::fill(dx, 0);
            for(long j = 0; j < Z.size(); ++j)
                dx[j] += backend::inner_product(f, *Z[j])
                       - backend::inner_product(*q, *Z[j]);
            all_gather(comm, dx.data(), Z.size(), df.data());

            for(long i = 0, k = comm.rank() * Z.size(); i < Z.size(); ++i, ++k) {
                value_type sum = 0;
                for(long j = 0; j < nz; ++j)
                    sum += E[k][j] * df[j];
                dx[i] = sum;
            }

            long j = 0;
            for(; j + 1 < Z.size(); j += 2)
                backend::axpbypcz(dx[j], *Z[j], dx[j+1], *Z[j+1], 1, x);

            for(; j < Z.size(); ++j)
                backend::axpby(dx[j], *Z[j], 1, x);
        }

        template <class Vector>
        void start_exchange(const Vector &x) const {
            // Start receiving ghost values from our neighbours.
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                recv.req[i] = comm.irecv(recv.nbr[i], tag_exc_vals,
                        &recv.val[recv.ptr[i]], recv.ptr[i+1] - recv.ptr[i]);

            // Gather values to send to our neighbours.
            if (!send.val.empty()) (*gather)(x, send.val);

            // Start sending our data to neighbours.
            for(size_t i = 0; i < send.nbr.size(); ++i)
                send.req[i] = comm.isend(send.nbr[i], tag_exc_vals,
                        &send.val[send.ptr[i]], send.ptr[i+1] - send.ptr[i]);
        }

        void finish_exchange() const {
            wait_all(recv.req.begin(), recv.req.end());
            wait_all(send.req.begin(), send.req.end());
        }
};

} // namespace mpi

namespace backend {

template <
    class                         Backend,
    class                         Coarsening,
    template <class> class        Relax,
    template <class, class> class IterativeSolver,
    class Vec1,
    class Vec2
    >
struct spmv_impl<
    mpi::subdomain_deflation<Backend, Coarsening, Relax, IterativeSolver>,
    Vec1, Vec2
    >
{
    typedef mpi::subdomain_deflation<Backend, Coarsening, Relax, IterativeSolver> M;
    typedef typename Backend::value_type V;

    static void apply(V alpha, const M &A, const Vec1 &x, V beta, Vec2 &y)
    {
        A.mul_n_project(alpha, x, beta, y);
    }
};

template <
    class                         Backend,
    class                         Coarsening,
    template <class> class        Relax,
    template <class, class> class IterativeSolver,
    class Vec1,
    class Vec2,
    class Vec3
    >
struct residual_impl<
    mpi::subdomain_deflation<Backend, Coarsening, Relax, IterativeSolver>,
    Vec1, Vec2, Vec3
    >
{
    typedef mpi::subdomain_deflation<Backend, Coarsening, Relax, IterativeSolver> M;
    typedef typename Backend::value_type V;

    static void apply(const Vec1 &rhs, const M &A, const Vec2 &x, Vec3 &r) {
        A.residual(rhs, x, r);
    }
};

} // namespace backend

} // namespace amgcl

#endif
