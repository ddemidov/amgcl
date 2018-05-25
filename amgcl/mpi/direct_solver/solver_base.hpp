#ifndef AMGCL_MPI_DIRECT_SOLVER_SOLVER_BASE_HPP
#define AMGCL_MPI_DIRECT_SOLVER_SOLVER_BASE_HPP

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

/**
\file   amgcl/mpi/direct_solver/solver_base.hpp
\author Denis Demidov <dennis.demidov@gmail.com>
\brief  Basic functionality for distributed direct solvers.
*/

#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace direct {

template <class value_type, class Solver>
class solver_base {
    public:
        typedef typename math::scalar_of<value_type>::type scalar_type;
        typedef typename math::rhs_of<value_type>::type    rhs_type;
        typedef backend::crs<value_type> build_matrix;

        solver_base() {}

        void init(communicator comm, const build_matrix &Astrip) {
            this->comm = comm;
            n = Astrip.nrows;

            std::vector<int> domain = mpi::exclusive_sum(comm, n);
            uniform_n = true;
            for(int i = 0; i < comm.size; ++i) {
                if (domain[i+1] - domain[i] != n) {
                    uniform_n = false;
                    break;
                }
            }

            // Consolidate the matrix on a fewer processes.
            int nmasters = std::min(comm.size, solver().comm_size(domain.back()));
            int slaves_per_master = (comm.size + nmasters - 1) / nmasters;
            int group_beg = (comm.rank / slaves_per_master) * slaves_per_master;
            int group_end = std::min(group_beg + slaves_per_master, comm.size);
            int group_size = group_end - group_beg;

            group_master = group_beg;

            // Communicator for masters (used to solve the coarse problem):
            MPI_Comm_split(comm,
                    comm.rank == group_master ? 0 : MPI_UNDEFINED,
                    comm.rank, &masters_comm
                    );

            // Communicator for slaves (used to send/recv coarse data):
            MPI_Comm_split(comm, group_master, comm.rank, &slaves_comm);

            // Count rows in local chunk of the consolidated matrix.
            int nloc;
            MPI_Reduce(&n, &nloc, 1, MPI_INT, MPI_SUM, 0, slaves_comm);

            // Shift from row pointers to row widths:
            std::vector<ptrdiff_t> widths(n);
            for(ptrdiff_t i = 0; i < n; ++i)
                widths[i] = Astrip.ptr[i+1] - Astrip.ptr[i];

            // Consolidate the matrix on group masters
            if (comm.rank == group_master) {
                build_matrix A;
                A.set_size(nloc, domain.back(), false);
                A.ptr[0] = 0;
                cons_f.resize(A.nrows);
                cons_x.resize(A.nrows);

                count.resize(group_size);
                displ.resize(group_size);

                if (uniform_n) {
                    MPI_Gather(&widths[0], n, datatype<ptrdiff_t>(),
                            &A.ptr[1], n, datatype<ptrdiff_t>(), 0, slaves_comm);
                } else {

                    for(int i = 0, j = group_beg; j < group_end; ++i, ++j) {
                        count[i] = domain[j+1] - domain[j];
                        displ[i] = i ? displ[i-1] + count[i-1] : 0;
                    }

                    MPI_Gatherv(&widths[0], n, datatype<ptrdiff_t>(),
                            &A.ptr[1], &count[0], &displ[0], datatype<ptrdiff_t>(),
                            0, slaves_comm);
                }

                A.set_nonzeros(A.scan_row_sizes());

                std::vector<int> nnz_count(group_size);
                std::vector<int> nnz_displ(group_size);

                for(int i = 0, j = group_beg, d0 = domain[group_beg]; j < group_end; ++i, ++j) {
                    nnz_count[i] = A.ptr[domain[j+1] - d0] - A.ptr[domain[j] - d0];
                    nnz_displ[i] = i ? nnz_displ[i-1] + nnz_count[i-1] : 0;
                }

                MPI_Gatherv(&Astrip.col[0], Astrip.nnz, datatype<ptrdiff_t>(),
                        &A.col[0], &nnz_count[0], &nnz_displ[0], datatype<ptrdiff_t>(),
                        0, slaves_comm);

                MPI_Gatherv(&Astrip.val[0], Astrip.nnz, datatype<value_type>(),
                        &A.val[0], &nnz_count[0], &nnz_displ[0], datatype<value_type>(),
                        0, slaves_comm);

                solver().init(masters_comm, A);
            } else {
                if (uniform_n) {
                    MPI_Gather(&widths[0], n, datatype<ptrdiff_t>(),
                            NULL, n, datatype<ptrdiff_t>(), 0, slaves_comm);
                } else {
                    MPI_Gatherv(&widths[0], n, datatype<ptrdiff_t>(),
                            NULL, NULL, NULL, datatype<ptrdiff_t>(),
                            0, slaves_comm);
                }

                MPI_Gatherv(&Astrip.col[0], Astrip.nnz, datatype<ptrdiff_t>(),
                        NULL, NULL, NULL, datatype<ptrdiff_t>(),
                        0, slaves_comm);

                MPI_Gatherv(&Astrip.val[0], Astrip.nnz, datatype<value_type>(),
                        NULL, NULL, NULL, datatype<value_type>(),
                        0, slaves_comm);
            }
        }

        template <class B>
        void init(communicator comm, const distributed_matrix<B> &A) {
            const build_matrix &A_loc = *A.local();
            const build_matrix &A_rem = *A.remote();

            build_matrix a;

            a.set_size(A.loc_rows(), A.glob_cols(), false);
            a.set_nonzeros(A_loc.nnz + A_rem.nnz);
            a.ptr[0] = 0;

            for(size_t i = 0, head = 0; i < A_loc.nrows; ++i) {
                ptrdiff_t shift = A.loc_col_shift();

                for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
                    a.col[head] = A_loc.col[j] + shift;
                    a.val[head] = A_loc.val[j];
                    ++head;
                }

                for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
                    a.col[head] = A_rem.col[j];
                    a.val[head] = A_rem.val[j];
                    ++head;
                }

                a.ptr[i+1] = head;
            }

            init(comm, a);
        }

        virtual ~solver_base() {
            if (masters_comm != MPI_COMM_NULL) MPI_Comm_free(&masters_comm);
            if (slaves_comm  != MPI_COMM_NULL) MPI_Comm_free(&slaves_comm);
        }

        Solver& solver() {
            return *static_cast<Solver*>(this);
        }

        const Solver& solver() const {
            return *static_cast<const Solver*>(this);
        }

        template <class VecF, class VecX>
        void operator()(const VecF &f, VecX &x) const {
            static const MPI_Datatype T = datatype<rhs_type>();

            if (comm.rank == group_master) {
                if (uniform_n) {
                    MPI_Gather(const_cast<rhs_type*>(&f[0]), n, T,
                            &cons_f[0], n, T, 0, slaves_comm);
                } else {
                    MPI_Gatherv(const_cast<rhs_type*>(&f[0]), n, T, &cons_f[0],
                            const_cast<int*>(&count[0]), const_cast<int*>(&displ[0]),
                            T, 0, slaves_comm);
                }

                solver().solve(cons_f, cons_x);

                if (uniform_n) {
                    MPI_Scatter(&cons_x[0], n, T, &x[0], n, T, 0, slaves_comm);
                } else {
                    MPI_Scatterv(&cons_x[0],
                            const_cast<int*>(&count[0]), const_cast<int*>(&displ[0]),
                            T, &x[0], n, T, 0, slaves_comm);
                }
            } else {
                if (uniform_n) {
                    MPI_Gather(const_cast<rhs_type*>(&f[0]), n, T, NULL, n, T, 0, slaves_comm);
                    MPI_Scatter(NULL, n, T, &x[0], n, T, 0, slaves_comm);
                } else {
                    MPI_Gatherv(const_cast<rhs_type*>(&f[0]), n, T, NULL, NULL, NULL, T, 0, slaves_comm);
                    MPI_Scatterv(NULL, NULL, NULL, T, &x[0], n, T, 0, slaves_comm);
                }
            }
        }
    private:

        communicator comm;
        int          n;
        bool         uniform_n;
        int          group_master;
        MPI_Comm     masters_comm, slaves_comm;
        std::vector<int> count, displ;
        mutable std::vector<rhs_type> cons_f, cons_x;
};

} // namespace direct
} // namespace mpi
} // namespace amgcl

#endif
