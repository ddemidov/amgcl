#include <iostream>
#include <vector>

#include <boost/scope_exit.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

namespace amgcl {
    profiler<> prof;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator comm(MPI_COMM_WORLD);

    int n = 16;
    int chunk_len = (n + comm.size - 1) / comm.size;
    int chunk_beg = std::min(n, chunk_len * comm.rank);
    int chunk_end = std::min(n, chunk_len * (comm.rank + 1));
    int chunk = chunk_end - chunk_beg;

    std::vector<int>    ptr; ptr.reserve(chunk + 1); ptr.push_back(0);
    std::vector<int>    col; col.reserve(chunk * 4);
    std::vector<double> val; val.reserve(chunk * 4);

    for(int i = 0, j = chunk_beg; i < chunk; ++i, ++j) {
        if (j > 0) {
            col.push_back(j - 1);
            val.push_back(-1);
        }

        col.push_back(j);
        val.push_back(2);

        if (j+1 < n) {
            col.push_back(j+1);
            val.push_back(-1);
        }

        if (j+5 < n) {
            col.push_back(j+5);
            val.push_back(-0.1);
        }

        ptr.push_back(col.size());
    }

    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::mpi::distributed_matrix<Backend> Matrix; 

    Matrix A(comm, boost::tie(chunk, ptr, col, val), chunk);

    {
        std::ostringstream fname;
        fname << "A_loc_" << comm.rank << ".mtx";
        amgcl::io::mm_write(fname.str(), *A.local());
    }

    {
        std::ostringstream fname;
        fname << "A_rem_" << comm.rank << ".mtx";
        amgcl::io::mm_write(fname.str(), *A.remote());
    }

    boost::shared_ptr<Matrix> B = transpose(A);

    {
        std::ostringstream fname;
        fname << "B_loc_" << comm.rank << ".mtx";
        amgcl::io::mm_write(fname.str(), *B->local());
    }

    {
        std::ostringstream fname;
        fname << "B_rem_" << comm.rank << ".mtx";
        amgcl::io::mm_write(fname.str(), *B->remote());
    }

}
