#ifndef AMGCL_MPI_DISTRIBUTED_MATRIX_HPP
#define AMGCL_MPI_DISTRIBUTED_MATRIX_HPP

/*
The MIT License

Copyright (c) 2012-2016 Denis Demidov <dennis.demidov@gmail.com>

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

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/unordered_map.hpp>
#include <boost/range/numeric.hpp>
#include <boost/multi_array.hpp>
#include <boost/foreach.hpp>

#include <mpi.h>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>

/**
 * \file   amgcl/mpi/distributed_matrix.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed matrix implementation.
 */

namespace amgcl {
namespace mpi {

template <class Backend>
class distributed_matrix {
    public:
        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix matrix;
        typedef typename Backend::vector vector;
        typedef typename Backend::params backend_params;

        typedef backend::crs<value_type> build_matrix;

        distributed_matrix(
                MPI_Comm mpi_comm,
                const matrix &A_loc,
                boost::shared_ptr<build_matrix> a_rem,
                const backend_params &bprm = backend_params()
                )
            : comm(mpi_comm), n(backend::rows(A_loc)), A_loc(A_loc),
              comm_matrix(boost::extents[comm.size][comm.size])
        {
            // Get domain boundaries
            std::vector<ptrdiff_t> domain(comm.size + 1, 0);
            MPI_Allgather(&n, 1, datatype<ptrdiff_t>(), &domain[1], 1, datatype<ptrdiff_t>(), comm);
            boost::partial_sum(domain, domain.begin());

            ptrdiff_t loc_beg = domain[comm.rank];

            // Renumber remote columns,
            // find out how many remote values we need from each process.
            std::vector<ptrdiff_t> rem_cols = a_rem->col;  // Unique ids of remote columns
            std::vector<ptrdiff_t> num_recv(comm.size, 0); // Number of columns to receive from each process
            {
                std::sort(rem_cols.begin(), rem_cols.end());
                rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());
                ptrdiff_t ncols = a_rem->ncols = rem_cols.size();

                boost::unordered_map<ptrdiff_t, ptrdiff_t> r2l(2 * a_rem->ncols);
                ptrdiff_t cur_domain = 0;
                for(ptrdiff_t i = 0; i < ncols; ++i) {
                    r2l.insert(r2l.end(), std::make_pair(rem_cols[i], i));

                    while(rem_cols[i] >= domain[cur_domain + 1]) ++cur_domain;
                    ++num_recv[cur_domain];
                }

                for(ptrdiff_t i = 0, nnz = a_rem->ptr.back(); i < nnz; ++i)
                    a_rem->col[i] = r2l[a_rem->col[i]];
            }

            // Setup communication pattern.
            // Find out who sends to whom and how many.
            MPI_Allgather(&num_recv[0], comm.size, datatype<ptrdiff_t>(),
                    comm_matrix.data(), comm.size, datatype<ptrdiff_t>(), comm);

            ptrdiff_t snbr = 0, rnbr = 0, send_size = 0;

            for(int i = 0; i < comm.size; ++i) {
                if (comm_matrix[comm.rank][i]) {
                    ++rnbr;
                }

                if (comm_matrix[i][comm.rank]) {
                    ++snbr;
                    send_size += comm_matrix[i][comm.rank];
                }
            }

            send.nbr.reserve(snbr);
            send.val.resize(send_size);
            send.req.resize(snbr);
            send.ptr.reserve(snbr + 1);
            send.ptr.push_back(0);

            recv.nbr.reserve(rnbr);
            recv.val.resize(a_rem->ncols);
            recv.req.resize(rnbr);
            recv.ptr.reserve(rnbr + 1);
            recv.ptr.push_back(0);

            // Count how many columns to send and to receive.
            for(int i = 0; i < comm.size; ++i) {
                if (ptrdiff_t nr = comm_matrix[comm.rank][i]) {
                    recv.nbr.push_back( i );
                    recv.ptr.push_back( recv.ptr.back() + nr );
                }

                if (ptrdiff_t ns = comm_matrix[i][comm.rank]) {
                    send.nbr.push_back( i );
                    send.ptr.push_back( send.ptr.back() + ns );
                }
            }

            // What columns do you need from me?
            send.col.resize(send_size);
            for(size_t i = 0; i < send.nbr.size(); ++i)
                MPI_Irecv(&send.col[send.ptr[i]], comm_matrix[send.nbr[i]][comm.rank],
                        datatype<ptrdiff_t>(), send.nbr[i], tag_exc_cols, comm, &send.req[i]);

            // Here is what I need from you:
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                MPI_Isend(&rem_cols[recv.ptr[i]], comm_matrix[comm.rank][recv.nbr[i]],
                        datatype<ptrdiff_t>(), recv.nbr[i], tag_exc_cols, comm, &recv.req[i]);

            MPI_Waitall(recv.req.size(), &recv.req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send.req.size(), &send.req[0], MPI_STATUSES_IGNORE);

            // Shift columns to send to local numbering:
            BOOST_FOREACH(ptrdiff_t &c, send.col) c -= loc_beg;

            // Create backend structures
            A_rem    = Backend::copy_matrix(a_rem, bprm);
            rem_vals = Backend::create_vector(a_rem->ncols, bprm);
            gather   = boost::make_shared<typename Backend::gather>(n, send.col, bprm);
        }

        template <class Vec1, class Vec2>
        void mul(value_type alpha, const Vec1 &x, value_type beta, Vec2 &y) const {
            start_exchange(x);

            // Compute local part of the product.
            backend::spmv(alpha, A_loc, x, beta, y);

            // Compute remote part of the product.
            finish_exchange();

            if (!recv.val.empty()) {
                backend::copy_to_backend(recv.val, *rem_vals);
                backend::spmv(alpha, *A_rem, *rem_vals, 1, y);
            }
        }

        template <class Vec1, class Vec2, class Vec3>
        void residual(const Vec1 &f, const Vec2 &x, Vec3 &r) const {
            start_exchange(x);
            backend::residual(f, A_loc, x, r);

            finish_exchange();

            if (!recv.val.empty()) {
                backend::copy_to_backend(recv.val, *rem_vals);
                backend::spmv(-1, *A_rem, *rem_vals, 1, r);
            }
        }

        bool talks_to(int rank) const {
            return comm_matrix[comm.rank][rank] || comm_matrix[rank][comm.rank];
        }

        struct {
            std::vector<ptrdiff_t> nbr;
            std::vector<ptrdiff_t> ptr;
            std::vector<ptrdiff_t> col;

            mutable std::vector<value_type>  val;
            mutable std::vector<MPI_Request> req;
        } send;

        struct {
            std::vector<ptrdiff_t> nbr;
            std::vector<ptrdiff_t> ptr;

            mutable std::vector<value_type>  val;
            mutable std::vector<MPI_Request> req;
        } recv;

    private:
        static const int tag_exc_cols = 1001;
        static const int tag_exc_vals = 2001;

        communicator comm;
        ptrdiff_t n;
        const matrix &A_loc;

        boost::multi_array<ptrdiff_t, 2> comm_matrix;

        boost::shared_ptr<matrix> A_rem;
        boost::shared_ptr<vector> rem_vals;
        boost::shared_ptr<typename Backend::gather> gather;

        template <class Vector>
        void start_exchange(const Vector &x) const {
            // Start receiving ghost values from our neighbours.
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                MPI_Irecv(&recv.val[recv.ptr[i]], recv.ptr[i+1] - recv.ptr[i],
                        datatype<value_type>(), recv.nbr[i], tag_exc_vals, comm, &recv.req[i]);

            // Start sending our data to neighbours.
            if (!send.val.empty()) {
                (*gather)(x, send.val);

                for(size_t i = 0; i < send.nbr.size(); ++i)
                    MPI_Isend(&send.val[send.ptr[i]], send.ptr[i+1] - send.ptr[i],
                            datatype<value_type>(), send.nbr[i], tag_exc_vals, comm, &send.req[i]);
            }
        }

        void finish_exchange() const {
            MPI_Waitall(recv.req.size(), &recv.req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send.req.size(), &send.req[0], MPI_STATUSES_IGNORE);
        }
};

} // namespace mpi

namespace backend {

template <class Backend, class Alpha, class Vec1, class Beta,  class Vec2>
struct spmv_impl<Alpha, mpi::distributed_matrix<Backend>, Vec1, Beta, Vec2> {
    static void apply(Alpha alpha, const mpi::distributed_matrix<Backend> &A, const Vec1 &x, Beta beta, Vec2 &y) {
        A.mul(alpha, x, beta, y);
    }
};

template <class Backend, class Vec1, class Vec2, class Vec3>
struct residual_impl<mpi::distributed_matrix<Backend>, Vec1, Vec2, Vec3> {
    static void apply(const Vec1 &rhs, const mpi::distributed_matrix<Backend> &A, const Vec2 &x, Vec3 &r) {
        A.residual(rhs, x, r);
    }
};

} // namespace backend
} // namespace amgcl

#endif
