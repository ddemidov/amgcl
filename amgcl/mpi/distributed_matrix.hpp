#ifndef AMGCL_MPI_DISTRIBUTED_MATRIX_HPP
#define AMGCL_MPI_DISTRIBUTED_MATRIX_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
class comm_pattern {
    public:
        typedef typename Backend::value_type value_type;
        typedef typename Backend::matrix matrix;
        typedef typename Backend::vector vector;
        typedef typename Backend::params backend_params;

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

        boost::shared_ptr<vector> x_rem;

        comm_pattern(
                MPI_Comm mpi_comm,
                ptrdiff_t n_loc_cols,
                size_t n_rem_cols, const ptrdiff_t *p_rem_cols,
                const backend_params &bprm = backend_params()
                ) : comm(mpi_comm)
        {
            AMGCL_TIC("communication pattern");
            // Get domain boundaries
            std::vector<ptrdiff_t> domain = mpi::exclusive_sum(comm, n_loc_cols);
            ptrdiff_t loc_beg = domain[comm.rank];

            // Renumber remote columns,
            // find out how many remote values we need from each process.
            std::vector<ptrdiff_t> rem_cols(p_rem_cols, p_rem_cols + n_rem_cols);

            std::sort(rem_cols.begin(), rem_cols.end());
            rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

            ptrdiff_t ncols = rem_cols.size();
            ptrdiff_t rnbr = 0, snbr = 0, send_size = 0;

            {
                std::vector<int> rcounts(comm.size, 0);
                std::vector<int> scounts(comm.size);

                // Build index for column renumbering;
                // count how many domains send us data and how much.
                idx.reserve(2 * ncols);
                for(int i = 0, d = 0, last = -1; i < ncols; ++i) {
                    idx.insert(idx.end(), std::make_pair(rem_cols[i], i));

                    while(rem_cols[i] >= domain[d + 1]) ++d;
                    ++rcounts[d];

                    if (last < d) {
                        last = d;
                        ++rnbr;
                    }
                }

                recv.val.resize(ncols);
                recv.req.resize(rnbr);

                recv.nbr.reserve(rnbr);
                recv.ptr.reserve(rnbr + 1); recv.ptr.push_back(0);

                for(int d = 0; d < comm.size; ++d) {
                    if (rcounts[d]) {
                        recv.nbr.push_back(d);
                        recv.ptr.push_back(recv.ptr.back() + rcounts[d]);
                    }
                }

                MPI_Alltoall(&rcounts[0], 1, MPI_INT, &scounts[0], 1, MPI_INT, comm);

                for(ptrdiff_t d = 0; d < comm.size; ++d) {
                    if (scounts[d]) {
                        ++snbr;
                        send_size += scounts[d];
                    }
                }

                send.col.resize(send_size);
                send.val.resize(send_size);
                send.req.resize(snbr);

                send.nbr.reserve(snbr);
                send.ptr.reserve(snbr + 1); send.ptr.push_back(0);

                for(ptrdiff_t d = 0; d < comm.size; ++d) {
                    if (scounts[d]) {
                        send.nbr.push_back(d);
                        send.ptr.push_back(send.ptr.back() + scounts[d]);
                    }
                }
            }

            // What columns do you need from me?
            for(size_t i = 0; i < send.nbr.size(); ++i)
                MPI_Irecv(&send.col[send.ptr[i]], send.ptr[i+1] - send.ptr[i],
                        datatype<ptrdiff_t>(), send.nbr[i], tag_exc_cols, comm, &send.req[i]);

            // Here is what I need from you:
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                MPI_Isend(&rem_cols[recv.ptr[i]], recv.ptr[i+1] - recv.ptr[i],
                        datatype<ptrdiff_t>(), recv.nbr[i], tag_exc_cols, comm, &recv.req[i]);

            MPI_Waitall(recv.req.size(), &recv.req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send.req.size(), &send.req[0], MPI_STATUSES_IGNORE);

            // Shift columns to send to local numbering:
            BOOST_FOREACH(ptrdiff_t &c, send.col) c -= loc_beg;

            // Create backend structures
            x_rem  = Backend::create_vector(ncols, bprm);
            gather = boost::make_shared<Gather>(n_loc_cols, send.col, bprm);
            AMGCL_TOC("communication pattern");
        }

        size_t renumber(size_t n, ptrdiff_t *col) {
            for(size_t i = 0; i < n; ++i) col[i] = idx[col[i]];
            return recv.val.size();
        }

        bool talks_to(int rank) const {
            return
                std::binary_search(send.nbr.begin(), send.nbr.end(), rank) ||
                std::binary_search(recv.nbr.begin(), recv.nbr.end(), rank);
        }

        bool needs_remote() const {
            return !recv.val.empty();
        }

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

            if (!recv.val.empty())
                backend::copy_to_backend(recv.val, *x_rem);
        }

        template <typename T>
        void exchange(const T *send_val, T *recv_val) const {
            for(size_t i = 0; i < recv.nbr.size(); ++i)
                MPI_Irecv(&recv_val[recv.ptr[i]], recv.ptr[i+1] - recv.ptr[i],
                        datatype<T>(), recv.nbr[i], tag_exc_vals, comm, &recv.req[i]);

            for(size_t i = 0; i < send.nbr.size(); ++i)
                MPI_Isend(const_cast<T*>(&send_val[send.ptr[i]]), send.ptr[i+1] - send.ptr[i],
                        datatype<T>(), send.nbr[i], tag_exc_vals, comm, &send.req[i]);

            MPI_Waitall(recv.req.size(), &recv.req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send.req.size(), &send.req[0], MPI_STATUSES_IGNORE);
        }

        communicator mpi_comm() const {
            return comm;
        }
    private:
        typedef typename Backend::gather Gather;

        static const int tag_set_comm = 1001;
        static const int tag_exc_cols = 1002;
        static const int tag_exc_vals = 1003;

        communicator comm;

        boost::unordered_map<ptrdiff_t, ptrdiff_t> idx;
        boost::shared_ptr<Gather> gather;
};

template <class Backend, class LocalMatrix = typename Backend::matrix, class RemoteMatrix = LocalMatrix>
class distributed_matrix {
    public:
        typedef typename Backend::value_type value_type;
        typedef typename Backend::params backend_params;
        typedef backend::crs<value_type> build_matrix;

        distributed_matrix(
                communicator comm,
                boost::shared_ptr<build_matrix> a_loc,
                boost::shared_ptr<build_matrix> a_rem,
                const backend_params &bprm = backend_params()
                )
            : a_loc(a_loc), a_rem(a_rem), bprm(bprm)
        {
            C = boost::make_shared<CommPattern>(comm, a_loc->ncols, a_rem->nnz, a_rem->col, bprm);
            a_rem->ncols = C->recv.val.size();
        }

        template <class Matrix>
        distributed_matrix(
                communicator comm,
                const Matrix &A,
                ptrdiff_t n_loc_cols,
                const backend_params &bprm = backend_params()
                )
            : bprm(bprm)
        {
            typedef typename backend::row_iterator<Matrix>::type row_iterator;

            ptrdiff_t n_loc_rows = backend::rows(A);

            // Get sizes of each domain in comm.
            std::vector<ptrdiff_t> domain = mpi::exclusive_sum(comm, n_loc_cols);
            ptrdiff_t loc_beg = domain[comm.rank];
            ptrdiff_t loc_end = domain[comm.rank + 1];

            // Split the matrix into local and remote parts.
            a_loc = boost::make_shared<build_matrix>();
            a_rem = boost::make_shared<build_matrix>();

            a_loc->set_size(n_loc_rows, n_loc_cols, true);
            a_rem->set_size(n_loc_rows, 0, true);

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < n_loc_rows; ++i) {
                for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                    ptrdiff_t c = a.col();

                    if (loc_beg <= c && c < loc_end)
                        ++a_loc->ptr[i + 1];
                    else
                        ++a_rem->ptr[i + 1];
                }
            }

            a_loc->set_nonzeros(a_loc->scan_row_sizes());
            a_rem->set_nonzeros(a_rem->scan_row_sizes());

#pragma omp parallel for
            for(ptrdiff_t i = 0; i < n_loc_rows; ++i) {
                ptrdiff_t loc_head = a_loc->ptr[i];
                ptrdiff_t rem_head = a_rem->ptr[i];

                for(row_iterator a = backend::row_begin(A, i); a; ++a) {
                    ptrdiff_t  c = a.col();
                    value_type v = a.value();

                    if (loc_beg <= c && c < loc_end) {
                        a_loc->col[loc_head] = c - loc_beg;
                        a_loc->val[loc_head] = v;
                        ++loc_head;
                    } else {
                        a_rem->col[rem_head] = c;
                        a_rem->val[rem_head] = v;
                        ++rem_head;
                    }
                }
            }

            C = boost::make_shared<CommPattern>(comm, n_loc_cols, a_rem->nnz, a_rem->col, bprm);
            a_rem->ncols = C->recv.val.size();
        }

        communicator comm() const {
            return C->mpi_comm();
        }

        boost::shared_ptr<build_matrix> local() const {
            return a_loc;
        }

        boost::shared_ptr<build_matrix> remote() const {
            return a_rem;
        }

        const comm_pattern<Backend>& cpat() const {
            return *C;
        }

        void set_local(boost::shared_ptr<LocalMatrix> a) {
            A_loc = a;
        }

        void move_to_backend() {
            if (!A_loc) {
                A_loc = Backend::copy_matrix(a_loc, bprm);
            }

            if (!A_rem) {
                C->renumber(a_rem->nnz, a_rem->col);
                A_rem = Backend::copy_matrix(a_rem, bprm);
            }
            
            a_loc.reset();
            a_rem.reset();
        }

        template <class Vec1, class Vec2>
        void mul(value_type alpha, const Vec1 &x, value_type beta, Vec2 &y) const {
            C->start_exchange(x);

            // Compute local part of the product.
            backend::spmv(alpha, *A_loc, x, beta, y);

            // Compute remote part of the product.
            C->finish_exchange();

            if (C->needs_remote())
                backend::spmv(alpha, *A_rem, *C->x_rem, 1, y);
        }

        template <class Vec1, class Vec2, class Vec3>
        void residual(const Vec1 &f, const Vec2 &x, Vec3 &r) const {
            C->start_exchange(x);
            backend::residual(f, *A_loc, x, r);

            C->finish_exchange();

            if (C->needs_remote())
                backend::spmv(-1, *A_rem, *C->x_rem, 1, r);
        }

        friend boost::shared_ptr<distributed_matrix>
        transpose(boost::shared_ptr<distributed_matrix> A) {
            static const int tag_exc_nnz = 2001;
            static const int tag_exc_cnt = 2002;
            static const int tag_exc_col = 2003;
            static const int tag_exc_val = 2004;

            communicator comm = A->comm();
            CommPattern &C = *(A->C);

            ptrdiff_t nrows = A->a_loc->ncols;
            ptrdiff_t ncols = A->a_loc->nrows;

            // Our transposed remote part becomes remote part of someone else,
            // and the other way around.
            boost::shared_ptr<build_matrix> t_rem;
            {
                std::vector<ptrdiff_t> tmp_col(A->a_rem->col, A->a_rem->col + A->a_rem->nnz);
                A->C->renumber(tmp_col.size(), &tmp_col[0]);

                ptrdiff_t *a_rem_col = &tmp_col[0];
                std::swap(a_rem_col, A->a_rem->col);

                t_rem = backend::transpose(*A->a_rem);

                std::swap(a_rem_col, A->a_rem->col);
            }

            // Shift to global numbering:
            std::vector<ptrdiff_t> domain = mpi::exclusive_sum(comm, ncols);
            ptrdiff_t loc_beg = domain[comm.rank];
            for(size_t i = 0; i < t_rem->nnz; ++i)
                t_rem->col[i] += loc_beg;

            // Shift from row pointers to row sizes:
            std::vector<ptrdiff_t> row_size(t_rem->nrows);
            for(size_t i = 0; i < t_rem->nrows; ++i)
                row_size[i] = t_rem->ptr[i+1] - t_rem->ptr[i];

            // Sizes of transposed remote blocks:
            std::vector<ptrdiff_t> block_ptr = exclusive_sum(comm,
                    t_rem->ptr[C.recv.ptr[comm.rank+1]] -
                    t_rem->ptr[C.recv.ptr[comm.rank]]
                    );

            // 2. Start exchange of rem_cnt, rem_col, rem_val
            std::vector<ptrdiff_t>  rem_cnt(C.send.col.size());
            std::vector<ptrdiff_t>  rem_col(block_ptr.back());
            std::vector<value_type> rem_val(block_ptr.back());

            std::vector<MPI_Request> recv_col_req(C.recv.req.size());
            std::vector<MPI_Request> recv_val_req(C.recv.req.size());
            std::vector<MPI_Request> send_col_req(C.send.req.size());
            std::vector<MPI_Request> send_val_req(C.send.req.size());

            for(size_t i = 0; i < C.send.nbr.size(); ++i) {
                MPI_Irecv(&rem_cnt[C.send.ptr[i]], C.send.ptr[i+1] - C.send.ptr[i],
                        datatype<ptrdiff_t>(), C.send.nbr[i],
                        tag_exc_cnt, comm, &C.send.req[i]);

                MPI_Irecv(&rem_col[block_ptr[i]], block_ptr[i+1] - block_ptr[i],
                        datatype<ptrdiff_t>(), C.send.nbr[i],
                        tag_exc_col, comm, &send_col_req[i]);

                MPI_Irecv(&rem_val[block_ptr[i]], block_ptr[i+1] - block_ptr[i],
                        datatype<value_type>(), C.send.nbr[i],
                        tag_exc_val, comm, &send_val_req[i]);
            }

            for(size_t i = 0; i < C.recv.nbr.size(); ++i) {
                MPI_Isend(&row_size[C.recv.ptr[i]], C.recv.ptr[i+1] - C.recv.ptr[i],
                        datatype<ptrdiff_t>(), C.recv.nbr[i],
                        tag_exc_cnt, comm, &C.recv.req[i]);

                ptrdiff_t beg = t_rem->ptr[C.recv.ptr[i]];
                ptrdiff_t end = t_rem->ptr[C.recv.ptr[i+1]];

                MPI_Isend(&t_rem->col[beg], end - beg, datatype<ptrdiff_t>(),
                        C.recv.nbr[i], tag_exc_col, comm, &recv_col_req[i]);

                MPI_Isend(&t_rem->val[beg], end - beg, datatype<value_type>(),
                        C.recv.nbr[i], tag_exc_val, comm, &recv_val_req[i]);
            }

            // 3. While rem_col and rem_val are in flight,
            //    finish exchange of rem_cnt, and
            //    start constructing our remote part:
            MPI_Waitall(C.recv.req.size(), &C.recv.req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(C.send.req.size(), &C.send.req[0], MPI_STATUSES_IGNORE);

            boost::shared_ptr<build_matrix> T_rem = boost::make_shared<build_matrix>();
            T_rem->set_size(nrows, 0, true);

            for(size_t i = 0; i < C.send.col.size(); ++i)
                T_rem->ptr[1 + C.send.col[i]] += rem_cnt[i];

            T_rem->scan_row_sizes();
            T_rem->set_nonzeros();

            // 4. Finish rem_col and rem_val exchange, and
            //    finish contruction of our remote part.
            MPI_Waitall(recv_col_req.size(), &recv_col_req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(recv_val_req.size(), &recv_val_req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send_col_req.size(), &send_col_req[0], MPI_STATUSES_IGNORE);
            MPI_Waitall(send_val_req.size(), &send_val_req[0], MPI_STATUSES_IGNORE);

            for(size_t i = 0, p = 0; i < C.send.col.size(); ++i) {
                ptrdiff_t row  = C.send.col[i];
                ptrdiff_t head = T_rem->ptr[row];

                for(ptrdiff_t j = 0; j < rem_cnt[i]; ++j, ++p, ++head) {
                    T_rem->col[head] = rem_col[p];
                    T_rem->val[head] = rem_val[p];
                }

                T_rem->ptr[row] = head;
            }

            std::rotate(T_rem->ptr, T_rem->ptr + nrows, T_rem->ptr + nrows + 1);
            T_rem->ptr[0] = 0;

            // TODO: This should work correctly, but the performance may be
            // improved by reusing A's communication pattern:
            return boost::make_shared<distributed_matrix>(comm,
                    backend::transpose(*A->a_loc), T_rem, A->bprm);
        }

    private:
        typedef comm_pattern<Backend> CommPattern;

        boost::shared_ptr<CommPattern>  C;
        boost::shared_ptr<LocalMatrix>  A_loc;
        boost::shared_ptr<RemoteMatrix> A_rem;
        boost::shared_ptr<build_matrix> a_loc, a_rem;
        backend_params bprm;

};

} // namespace mpi

namespace backend {

template <
    class Backend, class LocalMatrix, class RemoteMatrix,
    class Alpha, class Vec1, class Beta,  class Vec2
    >
struct spmv_impl<Alpha, mpi::distributed_matrix<Backend, LocalMatrix, RemoteMatrix>, Vec1, Beta, Vec2>
{
    static void apply(
            Alpha alpha,
            const mpi::distributed_matrix<Backend, LocalMatrix, RemoteMatrix> &A,
            const Vec1 &x, Beta beta, Vec2 &y)
    {
        A.mul(alpha, x, beta, y);
    }
};

template <
    class Backend, class LocalMatrix, class RemoteMatrix,
    class Vec1, class Vec2, class Vec3
    >
struct residual_impl<mpi::distributed_matrix<Backend, LocalMatrix, RemoteMatrix>, Vec1, Vec2, Vec3>
{
    static void apply(
            const Vec1 &rhs,
            const mpi::distributed_matrix<Backend, LocalMatrix, RemoteMatrix> &A,
            const Vec2 &x, Vec3 &r)
    {
        A.residual(rhs, x, r);
    }
};

} // namespace backend
} // namespace amgcl

#endif
