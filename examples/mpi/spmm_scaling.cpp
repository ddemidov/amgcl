#include <iostream>
#include <vector>

#include <boost/scope_exit.hpp>
#include <boost/program_options.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>
#include <amgcl/profiler.hpp>

#include "domain_partition.hpp"

namespace amgcl {
    profiler<> prof;
}

struct renumbering {
    const domain_partition<3> &part;
    const std::vector<ptrdiff_t> &dom;

    renumbering(
            const domain_partition<3> &p,
            const std::vector<ptrdiff_t> &d
            ) : part(p), dom(d)
    {}

    ptrdiff_t operator()(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) const {
        boost::array<ptrdiff_t, 3> p = {{i, j, k}};
        std::pair<int,ptrdiff_t> v = part.index(p);
        return dom[v.first] + v.second;
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    // Read configuration from command line
    ptrdiff_t n = 128;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "size,n",
         po::value<ptrdiff_t>(&n)->default_value(n),
         "domain size"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    boost::array<ptrdiff_t, 3> lo = { {0,   0,   0  } };
    boost::array<ptrdiff_t, 3> hi = { {n-1, n-1, n-1} };

    using amgcl::prof;

    prof.tic("partition");
    domain_partition<3> part(lo, hi, world.size);
    ptrdiff_t chunk = part.size( world.rank );

    std::vector<ptrdiff_t> domain = amgcl::mpi::exclusive_sum(world, chunk);

    lo = part.domain(world.rank).min_corner();
    hi = part.domain(world.rank).max_corner();

    renumbering renum(part, domain);
    prof.toc("partition");

    prof.tic("assemble");
    std::vector<ptrdiff_t> ptr;
    std::vector<ptrdiff_t> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    ptr.reserve(chunk + 1);
    col.reserve(chunk * 7);
    val.reserve(chunk * 7);

    ptr.push_back(0);

    const double h2i  = (n - 1) * (n - 1);

    for(ptrdiff_t k = lo[2]; k <= hi[2]; ++k) {
        for(ptrdiff_t j = lo[1]; j <= hi[1]; ++j) {
            for(ptrdiff_t i = lo[0]; i <= hi[0]; ++i) {
                if (k > 0)  {
                    col.push_back(renum(i,j,k-1));
                    val.push_back(-h2i);
                }

                if (j > 0)  {
                    col.push_back(renum(i,j-1,k));
                    val.push_back(-h2i);
                }

                if (i > 0) {
                    col.push_back(renum(i-1,j,k));
                    val.push_back(-h2i);
                }

                col.push_back(renum(i,j,k));
                val.push_back(6 * h2i);

                if (i + 1 < n) {
                    col.push_back(renum(i+1,j,k));
                    val.push_back(-h2i);
                }

                if (j + 1 < n) {
                    col.push_back(renum(i,j+1,k));
                    val.push_back(-h2i);
                }

                if (k + 1 < n) {
                    col.push_back(renum(i,j,k+1));
                    val.push_back(-h2i);
                }

                ptr.push_back( col.size() );
            }
        }
    }
    prof.toc("assemble");

    typedef amgcl::backend::builtin<double>         Backend;
    typedef amgcl::mpi::distributed_matrix<Backend> Matrix;

    prof.tic("create distributed version");
    Matrix A(world, boost::tie(chunk, ptr, col, val), chunk);
    prof.toc("create distributed version");

    prof.tic("distributed product");
    boost::shared_ptr<Matrix> B = amgcl::mpi::product(A, A);
    prof.toc("distributed product");

    if (world.rank == 0) {
        if (world.size == 1) {
            typedef amgcl::backend::crs<double> matrix;
            matrix A(boost::tie(chunk, ptr, col, val));
            prof.tic("openmp product");
            boost::shared_ptr<matrix> B = amgcl::backend::product(A, A);
            prof.toc("openmp product");
        }

        std::cout << prof << std::endl;
    }
}
