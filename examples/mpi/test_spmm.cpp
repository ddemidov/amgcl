#include <iostream>
#include <vector>

#include <boost/scope_exit.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/io/mm.hpp>

void assemble(
        int n, int beg, int end,
        std::vector<int>    &ptr,
        std::vector<int>    &col,
        std::vector<double> &val
        )
{
    int chunk = end - beg;

    ptr.clear(); ptr.reserve(chunk + 1); ptr.push_back(0);
    col.clear(); col.reserve(chunk * 4);
    val.clear(); val.reserve(chunk * 4);

    for(int j = beg, i = 0; j < end; ++j, ++i) {
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

    std::vector<int> chunks(comm.size);
    MPI_Allgather(&chunk, 1, MPI_INT, &chunks[0], 1, MPI_INT, comm);
    std::vector<int> displ(comm.size, 0);
    for(int i = 1; i < comm.size; ++i)
        displ[i] = displ[i-1] + chunks[i-1];

    std::vector<int>    ptr;
    std::vector<int>    col;
    std::vector<double> val;
    std::vector<double> x(chunk);
    std::vector<double> y(chunk);

    assemble(n, chunk_beg, chunk_end, ptr, col, val);

    for(int i = 0; i < chunk; ++i) x[i] = drand48();

    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::mpi::distributed_matrix<Backend> Matrix; 

    boost::shared_ptr<Matrix> A = boost::make_shared<Matrix>(comm,
            boost::tie(chunk, ptr, col, val), chunk);

    boost::shared_ptr<Matrix> B = amgcl::mpi::product(A, A);
    B->move_to_backend();

    amgcl::backend::spmv(1, *B, x, 0, y);

    std::vector<double> X(n), R(n);
    MPI_Gatherv(&x[0], chunk, MPI_DOUBLE, &X[0], &chunks[0], &displ[0], MPI_DOUBLE, 0, comm);
    MPI_Gatherv(&y[0], chunk, MPI_DOUBLE, &R[0], &chunks[0], &displ[0], MPI_DOUBLE, 0, comm);

    if (comm.rank == 0) {
        std::vector<double> Y(n);
        assemble(n, 0, n, ptr, col, val);

        amgcl::backend::crs<double> A( boost::tie(n, ptr, col, val) );
        amgcl::backend::spmv(1, *amgcl::backend::product(A, A), X, 0, Y);

        double s = 0;
        for(int i = 0; i < n; ++i) {
            double d = R[i] - Y[i];
            s += d * d;
        }
        std::cout << "Error: " << s << std::endl;
    }
}
