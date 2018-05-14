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
#include <algorithm>

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
                    while(rem_cols[i] >= domain[d + 1]) ++d;

                    ++rcounts[d];

                    if (last < d) {
                        last = d;
                        ++rnbr;
                    }

                    idx.insert(idx.end(), std::make_pair(
                                rem_cols[i], boost::make_tuple(rnbr-1,i)));
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

        int domain(ptrdiff_t col) {
            return boost::get<0>(idx[col]);
        }

        int local_index(ptrdiff_t col) {
            return boost::get<1>(idx[col]);
        }

        size_t renumber(size_t n, ptrdiff_t *col) {
            for(size_t i = 0; i < n; ++i)
                col[i] = boost::get<1>(idx[col[i]]);
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

        boost::unordered_map<ptrdiff_t, boost::tuple<int, int> > idx;
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

            n_loc_rows = a_loc->nrows;
            n_loc_cols = a_loc->ncols;
            n_loc_nonzeros = a_loc->nnz + a_rem->nnz;

            MPI_Allreduce(&n_loc_rows, &n_glob_rows, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);
            MPI_Allreduce(&n_loc_cols, &n_glob_cols, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);
            MPI_Allreduce(&n_loc_nonzeros, &n_glob_nonzeros, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);
        }

        template <class Matrix>
        distributed_matrix(
                communicator comm,
                const Matrix &A,
                ptrdiff_t n_loc_cols,
                const backend_params &bprm = backend_params()
                )
            : bprm(bprm),
              n_loc_rows(backend::rows(A)),
              n_loc_cols(n_loc_cols),
              n_loc_nonzeros(backend::nonzeros(A))
        {
            typedef typename backend::row_iterator<Matrix>::type row_iterator;

            // Get sizes of each domain in comm.
            std::vector<ptrdiff_t> domain = mpi::exclusive_sum(comm, n_loc_cols);
            ptrdiff_t loc_beg = domain[comm.rank];
            ptrdiff_t loc_end = domain[comm.rank + 1];
            n_glob_cols = domain.back();
            MPI_Allreduce(&n_loc_rows, &n_glob_rows, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);
            MPI_Allreduce(&n_loc_nonzeros, &n_glob_nonzeros, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);

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

        ptrdiff_t loc_rows() const {
            return n_loc_rows;
        }

        ptrdiff_t loc_cols() const {
            return n_loc_cols;
        }

        ptrdiff_t loc_nonzeros() const {
            return n_loc_nonzeros;
        }

        ptrdiff_t glob_rows() const {
            return n_glob_rows;
        }

        ptrdiff_t glob_cols() const {
            return n_glob_cols;
        }

        ptrdiff_t glob_nonzeros() const {
            return n_glob_nonzeros;
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

        template <class B, class L, class R>
        friend boost::shared_ptr< distributed_matrix<B,L,R> >
        transpose(const distributed_matrix<B,L,R> &A);

        template <class B, class L, class R>
        friend boost::shared_ptr< distributed_matrix<B,L,R> >
        product(const distributed_matrix<B,L,R> &a, const distributed_matrix<B,L,R> &b);

    private:
        typedef comm_pattern<Backend> CommPattern;

        boost::shared_ptr<CommPattern>  C;
        boost::shared_ptr<LocalMatrix>  A_loc;
        boost::shared_ptr<RemoteMatrix> A_rem;
        boost::shared_ptr<build_matrix> a_loc, a_rem;
        backend_params bprm;

        ptrdiff_t n_loc_rows, n_glob_rows;
        ptrdiff_t n_loc_cols, n_glob_cols;
        ptrdiff_t n_loc_nonzeros, n_glob_nonzeros;
};

template <class Backend, class Local, class Remote>
boost::shared_ptr< distributed_matrix<Backend, Local, Remote> >
transpose(const distributed_matrix<Backend, Local, Remote> &A) {
    typedef typename Backend::value_type value_type;
    typedef comm_pattern<Backend>        CommPattern;
    typedef backend::crs<value_type>     build_matrix;

    static const int tag_cnt = 2001;
    static const int tag_col = 2002;
    static const int tag_val = 2003;

    communicator comm = A.comm();
    CommPattern &C = *(A.C);

    ptrdiff_t nrows = A.loc_cols();
    ptrdiff_t ncols = A.loc_rows();

    std::vector<MPI_Request> recv_cnt_req(C.send.req.size());
    std::vector<MPI_Request> recv_col_req(C.send.req.size());
    std::vector<MPI_Request> recv_val_req(C.send.req.size());

    std::vector<MPI_Request> send_cnt_req(C.recv.req.size());
    std::vector<MPI_Request> send_col_req(C.recv.req.size());
    std::vector<MPI_Request> send_val_req(C.recv.req.size());

    // Our transposed remote part becomes remote part of someone else,
    // and the other way around.
    boost::shared_ptr<build_matrix> t_rem;
    {
        std::vector<ptrdiff_t> tmp_col(A.a_rem->col, A.a_rem->col + A.a_rem->nnz);
        C.renumber(tmp_col.size(), &tmp_col[0]);

        ptrdiff_t *a_rem_col = &tmp_col[0];
        std::swap(a_rem_col, A.a_rem->col);

        t_rem = backend::transpose(*A.a_rem);

        std::swap(a_rem_col, A.a_rem->col);
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
    // 1. Exchange rem_ptr
    std::vector<ptrdiff_t> rem_ptr(C.send.col.size() + 1); rem_ptr[0] = 0;

    for(size_t i = 0; i < C.send.nbr.size(); ++i) {
        ptrdiff_t beg = C.send.ptr[i];
        ptrdiff_t end = C.send.ptr[i + 1];

        MPI_Irecv(&rem_ptr[beg + 1], end - beg, datatype<ptrdiff_t>(),
                C.send.nbr[i], tag_cnt, comm, &recv_cnt_req[i]);
    }

    for(size_t i = 0; i < C.recv.nbr.size(); ++i) {
        ptrdiff_t beg = C.recv.ptr[i];
        ptrdiff_t end = C.recv.ptr[i + 1];

        MPI_Isend(&row_size[beg], end - beg, datatype<ptrdiff_t>(),
                C.recv.nbr[i], tag_cnt, comm, &send_cnt_req[i]);
    }

    MPI_Waitall(recv_cnt_req.size(), &recv_cnt_req[0], MPI_STATUSES_IGNORE);
    std::partial_sum(rem_ptr.begin(), rem_ptr.end(), rem_ptr.begin());

    // 2. Start exchange of rem_col, rem_val
    std::vector<ptrdiff_t>  rem_col(rem_ptr.back());
    std::vector<value_type> rem_val(rem_ptr.back());

    for(size_t i = 0; i < C.send.nbr.size(); ++i) {
        ptrdiff_t rbeg = C.send.ptr[i];
        ptrdiff_t rend = C.send.ptr[i + 1];

        ptrdiff_t cbeg = rem_ptr[rbeg];
        ptrdiff_t cend = rem_ptr[rend];

        MPI_Irecv(&rem_col[cbeg], cend - cbeg, datatype<ptrdiff_t>(),
                C.send.nbr[i], tag_col, comm, &recv_col_req[i]);

        MPI_Irecv(&rem_val[cbeg], cend - cbeg, datatype<value_type>(),
                C.send.nbr[i], tag_val, comm, &recv_val_req[i]);
    }

    for(size_t i = 0; i < C.recv.nbr.size(); ++i) {
        ptrdiff_t rbeg = C.recv.ptr[i];
        ptrdiff_t rend = C.recv.ptr[i + 1];

        ptrdiff_t cbeg = t_rem->ptr[rbeg];
        ptrdiff_t cend = t_rem->ptr[rend];

        MPI_Isend(&t_rem->col[cbeg], cend - cbeg, datatype<ptrdiff_t>(),
                C.recv.nbr[i], tag_col, comm, &send_col_req[i]);

        MPI_Isend(&t_rem->val[cbeg], cend - cbeg, datatype<value_type>(),
                C.recv.nbr[i], tag_val, comm, &send_val_req[i]);
    }

    // 3. While rem_col and rem_val are in flight,
    //    start constructing our remote part:
    boost::shared_ptr<build_matrix> T_rem = boost::make_shared<build_matrix>();
    T_rem->set_size(nrows, 0, true);

    for(size_t i = 0; i < C.send.col.size(); ++i)
        T_rem->ptr[1 + C.send.col[i]] += rem_ptr[i+1] - rem_ptr[i];

    T_rem->scan_row_sizes();
    T_rem->set_nonzeros();

    // 4. Finish rem_col and rem_val exchange, and
    //    finish contruction of our remote part.
    MPI_Waitall(recv_col_req.size(), &recv_col_req[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_val_req.size(), &recv_val_req[0], MPI_STATUSES_IGNORE);

    for(size_t i = 0; i < C.send.col.size(); ++i) {
        ptrdiff_t row  = C.send.col[i];
        ptrdiff_t head = T_rem->ptr[row];

        for(ptrdiff_t j = rem_ptr[i]; j < rem_ptr[i+1]; ++j, ++head) {
            T_rem->col[head] = rem_col[j];
            T_rem->val[head] = rem_val[j];
        }

        T_rem->ptr[row] = head;
    }

    std::rotate(T_rem->ptr, T_rem->ptr + nrows, T_rem->ptr + nrows + 1);
    T_rem->ptr[0] = 0;

    MPI_Waitall(send_cnt_req.size(), &send_cnt_req[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(send_col_req.size(), &send_col_req[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(send_val_req.size(), &send_val_req[0], MPI_STATUSES_IGNORE);

    // TODO: This should work correctly, but the performance may be
    // improved by reusing A's communication pattern:
    return boost::make_shared< distributed_matrix<Backend, Local, Remote> >(
            comm, backend::transpose(*A.a_loc), T_rem, A.bprm);
}

template <class Backend, class Local, class Remote>
boost::shared_ptr< distributed_matrix<Backend, Local, Remote> >
product(
        const distributed_matrix<Backend, Local, Remote> &A,
        const distributed_matrix<Backend, Local, Remote> &B
       )
{
    typedef typename Backend::value_type value_type;
    typedef comm_pattern<Backend>        CommPattern;
    typedef backend::crs<value_type>     build_matrix;

    static const int tag_ptr = 3001;
    static const int tag_col = 3002;
    static const int tag_val = 3003;

    communicator comm = A.comm();
    CommPattern  &Acp = *A.C;

    build_matrix &A_loc = *A.local();
    build_matrix &A_rem = *A.remote();
    build_matrix &B_loc = *B.local();
    build_matrix &B_rem = *B.remote();

    ptrdiff_t A_rows = A.loc_rows();
    ptrdiff_t B_cols = B.loc_cols();

    std::vector<ptrdiff_t> B_dom = mpi::exclusive_sum(comm, static_cast<ptrdiff_t>(B_cols));

    ptrdiff_t B_beg = B_dom[comm.rank];
    ptrdiff_t B_end = B_dom[comm.rank + 1];

    size_t nrecv = Acp.recv.nbr.size();
    size_t nsend = Acp.send.nbr.size();

    // Create blocked matrix to send to each domain
    // that needs data from us:
    std::vector<MPI_Request> send_ptr_req(nsend);
    std::vector<MPI_Request> send_col_req(nsend);
    std::vector<MPI_Request> send_val_req(nsend);

    std::vector<build_matrix> send_rows(nsend);

    for(size_t k = 0; k < nsend; ++k) {
        ptrdiff_t beg = Acp.send.ptr[k];
        ptrdiff_t end = Acp.send.ptr[k + 1];

        ptrdiff_t nr = end - beg;

        build_matrix &m = send_rows[k];
        m.set_size(nr, 0, true);

        for(ptrdiff_t i = 0, ii = beg; ii < end; ++i, ++ii) {
            ptrdiff_t r = Acp.send.col[ii];

            ptrdiff_t w =
                (B_loc.ptr[r + 1] - B_loc.ptr[r]) +
                (B_rem.ptr[r + 1] - B_rem.ptr[r]);

            m.ptr[i] = w;
            m.nnz += w;
        }

        MPI_Isend(m.ptr, m.nrows, datatype<ptrdiff_t>(),
                Acp.send.nbr[k], tag_ptr, comm, &send_ptr_req[k]);

        m.set_nonzeros(m.nnz);

        for(ptrdiff_t i = 0, ii = beg, head = 0; ii < end; ++i, ++ii) {
            ptrdiff_t r = Acp.send.col[ii];

            // Contribution of the local part:
            for(ptrdiff_t j = B_loc.ptr[r]; j < B_loc.ptr[r+1]; ++j) {
                m.col[head] = B_loc.col[j] + B_beg;
                m.val[head] = B_loc.val[j];
                ++head;
            }

            // Contribution of the remote part:
            for(ptrdiff_t j = B_rem.ptr[r]; j < B_rem.ptr[r+1]; ++j) {
                m.col[head] = B_rem.col[j];
                m.val[head] = B_rem.val[j];
                ++head;
            }
        }

        MPI_Isend(m.col, m.nnz, datatype<ptrdiff_t>(),
                Acp.send.nbr[k], tag_col, comm, &send_col_req[k]);
        MPI_Isend(m.val, m.nnz, datatype<value_type>(),
                Acp.send.nbr[k], tag_val, comm, &send_val_req[k]);
    }

    // Receive rows of B in block format from our neighbors:
    std::vector<MPI_Request> recv_ptr_req(nrecv);
    std::vector<MPI_Request> recv_col_req(nrecv);
    std::vector<MPI_Request> recv_val_req(nrecv);

    build_matrix B_nbr;
    B_nbr.set_size(Acp.recv.val.size(), 0, true);

    for(size_t k = 0; k < nrecv; ++k) {
        ptrdiff_t beg = Acp.recv.ptr[k];
        ptrdiff_t end = Acp.recv.ptr[k + 1];

        MPI_Irecv(&B_nbr.ptr[beg + 1], end - beg, datatype<ptrdiff_t>(),
                Acp.recv.nbr[k], tag_ptr, comm, &recv_ptr_req[k]);
    }

    MPI_Waitall(recv_ptr_req.size(), &recv_ptr_req[0], MPI_STATUSES_IGNORE);

    B_nbr.set_nonzeros(B_nbr.scan_row_sizes());

    for(size_t k = 0; k < nrecv; ++k) {
        ptrdiff_t rbeg = Acp.recv.ptr[k];
        ptrdiff_t rend = Acp.recv.ptr[k + 1];

        ptrdiff_t cbeg = B_nbr.ptr[rbeg];
        ptrdiff_t cend = B_nbr.ptr[rend];

        MPI_Irecv(&B_nbr.col[cbeg], cend - cbeg, datatype<ptrdiff_t>(),
                Acp.recv.nbr[k], tag_col, comm, &recv_col_req[k]);
        MPI_Irecv(&B_nbr.val[cbeg], cend - cbeg, datatype<value_type>(),
                Acp.recv.nbr[k], tag_val, comm, &recv_val_req[k]);
    }

    MPI_Waitall(recv_col_req.size(), &recv_col_req[0], MPI_STATUSES_IGNORE);

    // Build mapping from global to local column numbers in the remote part of
    // the product matrix.
    std::vector<ptrdiff_t> rem_cols(B_rem.nnz + B_nbr.nnz);

    std::copy(B_nbr.col, B_nbr.col + B_nbr.nnz,
            std::copy(B_rem.col, B_rem.col + B_rem.nnz, rem_cols.begin()));

    std::sort(rem_cols.begin(), rem_cols.end());
    rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

    ptrdiff_t n_rem_cols = 0;
    boost::unordered_map<ptrdiff_t, int> rem_idx(2 * rem_cols.size());
    BOOST_FOREACH(ptrdiff_t c, rem_cols) {
        if (c >= B_beg && c < B_end) continue;
        rem_idx[c] = n_rem_cols++;
    }

    MPI_Waitall(recv_val_req.size(), &recv_val_req[0], MPI_STATUSES_IGNORE);

    // Build the product.
    boost::shared_ptr<build_matrix> C_loc = boost::make_shared<build_matrix>();
    boost::shared_ptr<build_matrix> C_rem = boost::make_shared<build_matrix>();

    C_loc->set_size(A_rows, B_cols, false);
    C_rem->set_size(A_rows, 0,      false);

    C_loc->ptr[0] = 0;
    C_rem->ptr[0] = 0;

#pragma omp parallel
    {
        std::vector<ptrdiff_t> loc_marker(B_end - B_beg, -1);
        std::vector<ptrdiff_t> rem_marker(n_rem_cols,    -1);

#pragma omp for
        for(ptrdiff_t ia = 0; ia < A_rows; ++ia) {
            ptrdiff_t loc_cols = 0;
            ptrdiff_t rem_cols = 0;

            for(ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t ca = A_loc.col[ja];

                for(ptrdiff_t jb = B_loc.ptr[ca], eb = B_loc.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t cb = B_loc.col[jb];

                    if (loc_marker[cb] != ia) {
                        loc_marker[cb]  = ia;
                        ++loc_cols;
                    }
                }

                for(ptrdiff_t jb = B_rem.ptr[ca], eb = B_rem.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t cb = rem_idx[B_rem.col[jb]];

                    if (rem_marker[cb] != ia) {
                        rem_marker[cb]  = ia;
                        ++rem_cols;
                    }
                }
            }

            for(ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t ca = Acp.local_index(A_rem.col[ja]);

                for(ptrdiff_t jb = B_nbr.ptr[ca], eb = B_nbr.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t cb = B_nbr.col[jb];

                    if (cb >= B_beg && cb < B_end) {
                        cb -= B_beg;

                        if (loc_marker[cb] != ia) {
                            loc_marker[cb]  = ia;
                            ++loc_cols;
                        }
                    } else {
                        cb = rem_idx[cb];

                        if (rem_marker[cb] != ia) {
                            rem_marker[cb]  = ia;
                            ++rem_cols;
                        }
                    }
                }
            }

            C_loc->ptr[ia + 1] = loc_cols;
            C_rem->ptr[ia + 1] = rem_cols;
        }
    }

    C_loc->set_nonzeros(C_loc->scan_row_sizes());
    C_rem->set_nonzeros(C_rem->scan_row_sizes());

#pragma omp parallel
    {
        std::vector<ptrdiff_t> loc_marker(B_end - B_beg, -1);
        std::vector<ptrdiff_t> rem_marker(n_rem_cols,    -1);

#pragma omp for
        for(ptrdiff_t ia = 0; ia < A_rows; ++ia) {
            ptrdiff_t loc_beg = C_loc->ptr[ia];
            ptrdiff_t rem_beg = C_rem->ptr[ia];
            ptrdiff_t loc_end = loc_beg;
            ptrdiff_t rem_end = rem_beg;

            for(ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t ca  = A_loc.col[ja];
                value_type va = A_loc.val[ja];

                for(ptrdiff_t jb = B_loc.ptr[ca], eb = B_loc.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  cb = B_loc.col[jb];
                    value_type vb = B_loc.val[jb];

                    if (loc_marker[cb] < loc_beg) {
                        loc_marker[cb] = loc_end;

                        C_loc->col[loc_end] = cb;
                        C_loc->val[loc_end] = va * vb;

                        ++loc_end;
                    } else {
                        C_loc->val[loc_marker[cb]] += va * vb;
                    }
                }

                for(ptrdiff_t jb = B_rem.ptr[ca], eb = B_rem.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  gb = B_rem.col[jb];
                    ptrdiff_t  cb = rem_idx[gb];
                    value_type vb = B_rem.val[jb];

                    if (rem_marker[cb] < rem_beg) {
                        rem_marker[cb] = rem_end;

                        C_rem->col[rem_end] = gb;
                        C_rem->val[rem_end] = va * vb;

                        ++rem_end;
                    } else {
                        C_rem->val[rem_marker[cb]] += va * vb;
                    }
                }
            }

            for(ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t  ca = Acp.local_index(A_rem.col[ja]);
                value_type va = A_rem.val[ja];

                for(ptrdiff_t jb = B_nbr.ptr[ca], eb = B_nbr.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  gb = B_nbr.col[jb];
                    value_type vb = B_nbr.val[jb];

                    if (gb >= B_beg && gb < B_end) {
                        ptrdiff_t cb = gb - B_beg;

                        if (loc_marker[cb] < loc_beg) {
                            loc_marker[cb] = loc_end;

                            C_loc->col[loc_end] = cb;
                            C_loc->val[loc_end] = va * vb;

                            ++loc_end;
                        } else {
                            C_loc->val[loc_marker[cb]] += va * vb;
                        }
                    } else {
                        ptrdiff_t cb = rem_idx[gb];

                        if (rem_marker[cb] < rem_beg) {
                            rem_marker[cb] = rem_end;

                            C_rem->col[rem_end] = gb;
                            C_rem->val[rem_end] = va * vb;

                            ++rem_end;
                        } else {
                            C_rem->val[rem_marker[cb]] += va * vb;
                        }
                    }
                }
            }
        }
    }

    MPI_Waitall(send_ptr_req.size(), &send_ptr_req[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(send_col_req.size(), &send_col_req[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(send_val_req.size(), &send_val_req[0], MPI_STATUSES_IGNORE);


    // TODO: This should work correctly, but we may have enough information to
    // build C's communication pattern here and save some work:
    return boost::make_shared<distributed_matrix<Backend, Local, Remote> >(comm,
            C_loc, C_rem, A.bprm);
}

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
