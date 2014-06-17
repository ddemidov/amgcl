#ifndef AMGCL_MPI_DIST_CRS_HPP
#define AMGCL_MPI_DIST_CRS_HPP

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
 * \file   amgcl/mpi/dist_crs.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed matrix in CRS format.
 */

#include <vector>
#include <map>

#include <boost/range/numeric.hpp>
#include <boost/multi_array.hpp>
#include <boost/mpi.hpp>

#include <amgcl/backend/builtin.hpp>

namespace amgcl {
namespace mpi {

/// Distributed matrix in CRS format.
template <typename real>
class dist_crs {
    public:
        typedef real value_type;

        /// Constructs distributed matrix from a slice of global matrix.
        /**
         * \param comm  MPI communicator
         * \param nrows Number of rows in our matrix slice.
         * \param ptr   Pointer to the beginning of each row in col and val.
         * \param col   Column numbers (global) of nonzero values.
         * \param col   Nonzero values.
         */
        template <class Matrix>
        dist_crs(MPI_Comm mpi_comm, const Matrix &A)
            : comm(mpi_comm, boost::mpi::comm_attach), nrows(backend::rows(A))
        {
            // Exchange chunk sizes with neighbours.
            std::vector<long> domain(comm.size() + 1, 0);
            all_gather(comm, nrows, &domain[1]);
            boost::partial_sum(domain, domain.begin());

            // Build local and remote parts of the matrix.
            // Renumber remote columns while at it.
            long my_beg = domain[comm.rank()];
            long my_end = domain[comm.rank() + 1];

            long loc_nnz = 0;
            long rem_nnz = 0;

            // Map remote column numbers to local ids:
            std::map<long, long> rc;
            std::map<long, long>::iterator rc_it = rc.begin();

            // Count local and remote nonzeros; build set of remote columns
            typedef typename backend::row_iterator<Matrix>::type row_iterator;
            for(long i = 0; i < nrows; ++i) {
                for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                    long c = a.col();

                    if (c >= my_beg && c < my_end) {
                        ++loc_nnz;
                    } else {
                        ++rem_nnz;
                        rc_it = rc.insert(rc_it, std::make_pair(c, 0));
                    }
                }
            }

            // How many columns do we need from each process:
            std::vector<long> num_recv(comm.size(), 0);

            // What columns do we need from them:
            std::vector<long> recv_cols;
            recv_cols.reserve(rc.size());

            long id = 0, cur_nbr = 0;
            for(rc_it = rc.begin(); rc_it != rc.end(); ++rc_it) {
                rc_it->second = id++;
                recv_cols.push_back(rc_it->first);

                while(rc_it->first >= domain[cur_nbr + 1]) cur_nbr++;
                num_recv[cur_nbr]++;
            }

            // Build local and remote parts of the matrix.
            Aloc.nrows = nrows;
            Aloc.ncols = nrows;
            Aloc.ptr.reserve(nrows + 1);
            Aloc.col.reserve(loc_nnz);
            Aloc.val.reserve(loc_nnz);
            Aloc.ptr.push_back(0);

            Arem.nrows = nrows;
            Arem.ncols = rc.size();
            Arem.ptr.reserve(nrows + 1);
            Arem.col.reserve(rem_nnz);
            Arem.val.reserve(rem_nnz);
            Arem.ptr.push_back(0);

            for(long i = 0; i < nrows; ++i) {
                for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                    long       c = a.col();
                    value_type v = a.value();

                    if (c >= my_beg && c < my_end) {
                        Aloc.col.push_back(c - my_beg);
                        Aloc.val.push_back(v);
                    } else {
                        Arem.col.push_back(rc[c]);
                        Arem.val.push_back(v);
                    }
                }

                Aloc.ptr.push_back(Aloc.col.size());
                Arem.ptr.push_back(Arem.col.size());
            }

            // Set up communication pattern.
            boost::multi_array<long, 2> comm_matrix(
                    boost::extents[comm.size()][comm.size()]
                    );

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

            send.nbr.reserve(snbr);
            send.ptr.reserve(snbr + 1);
            send.col.resize(send_size);
            send.val.resize(send_size);
            send.req.resize(snbr);

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

            for(size_t i = 0; i < send.nbr.size(); ++i)
                send.req[i] = comm.irecv(send.nbr[i], tag_exc_vals,
                        &send.col[send.ptr[i]], comm_matrix[send.nbr[i]][comm.rank()]);

            for(size_t i = 0; i < recv.nbr.size(); ++i)
                recv.req[i] = comm.isend(recv.nbr[i], tag_exc_vals,
                    &recv_cols[recv.ptr[i]], comm_matrix[comm.rank()][recv.nbr[i]]);

            wait_all(recv.req.begin(), recv.req.end());
            wait_all(send.req.begin(), send.req.end());

            BOOST_FOREACH(long &c, send.col) c -= my_beg;
        }

        /// Multiplies the matrix by x and puts the result into y.
        /**
         * \f$ y = \alpha A x + \beta y \f$
         */
        template <class VectorX, class VectorY>
        void mul(
                value_type alpha, VectorX const &x,
                value_type beta,  VectorY       &y
                ) const
        {
            start_exchange(x);
            backend::spmv(alpha, Aloc, x, beta, y);

            finish_exchange();
            backend::spmv(alpha, Arem, recv.val, 1, y);
        }

        const backend::crs<value_type, long>& local() const {
            return Aloc;
        }
    private:
        static const int tag_exc_cols = 1001;
        static const int tag_exc_vals = 2001;

        boost::mpi::communicator comm;

        long nrows;

        backend::crs<value_type, long> Aloc;
        backend::crs<value_type, long> Arem;

        struct {
            std::vector<long> nbr;
            std::vector<long> ptr;

            mutable std::vector<value_type>          val;
            mutable std::vector<boost::mpi::request> req;
        } recv;

        struct {
            std::vector<long> nbr;
            std::vector<long> ptr;
            std::vector<long> col;

            mutable std::vector<value_type>          val;
            mutable std::vector<boost::mpi::request> req;
        } send;

        template <class Vector>
        void start_exchange(const Vector &x) const {
            // Start receiving ghost values from our neighbours.
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                recv.req[i] = comm.irecv(recv.nbr[i], tag_exc_vals,
                        &recv.val[recv.ptr[i]], recv.ptr[i+1] - recv.ptr[i]);

            // Gather values to send to our neighbours.
            for(size_t i = 0; i < send.col.size(); ++i)
                send.val[i] = x[send.col[i]];

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
} // namespace amgcl

#endif
