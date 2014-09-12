#include <iostream>
#include <iterator>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <boost/scope_exit.hpp>
#include <boost/range/algorithm.hpp>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

extern "C" {
#include <metis.h>
}

#include <amgcl/mpi/runtime.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/profiler.hpp>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> EigenMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1>          EigenVector;

namespace amgcl {
    profiler<> prof;
}

//---------------------------------------------------------------------------
inline size_t alignup(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

//---------------------------------------------------------------------------
void pointwise_graph(
        int n, int block_size, const int *ptr,  const int *col,
        std::vector<int> &pptr,
        std::vector<int> &pcol
        )
{
    int np = n / block_size;

    assert(np * block_size == n);

    // Create pointwise matrix
    std::vector<int> ptr1(np + 1, 0);
    std::vector<int> marker(np, -1);
    for(int ip = 0, i = 0; ip < np; ++ip) {
        for(int k = 0; k < block_size; ++k, ++i) {
            for(int j = ptr[i]; j < ptr[i+1]; ++j) {
                int cp = col[j] / block_size;
                if (marker[cp] != ip) {
                    marker[cp] = ip;
                    ++ptr1[ip+1];
                }
            }
        }
    }

    boost::partial_sum(ptr1, ptr1.begin());
    boost::fill(marker, -1);

    std::vector<int> col1(ptr1.back());

    for(int ip = 0, i = 0; ip < np; ++ip) {
        int row_beg = ptr1[ip];
        int row_end = row_beg;

        for(int k = 0; k < block_size; ++k, ++i) {
            for(int j = ptr[i]; j < ptr[i+1]; ++j) {
                int cp = col[j] / block_size;

                if (marker[cp] < row_beg) {
                    marker[cp] = row_end;
                    col1[row_end++] = cp;
                }
            }
        }
    }


    // Transpose pointwise matrix
    int nnz = ptr1.back();

    std::vector<int> ptr2(np + 1, 0);
    std::vector<int> col2(nnz);

    for(int i = 0; i < nnz; ++i)
        ++( ptr2[ col1[i] + 1 ] );

    boost::partial_sum(ptr2, ptr2.begin());

    for(int i = 0; i < np; ++i)
        for(int j = ptr1[i]; j < ptr1[i+1]; ++j)
            col2[ptr2[col1[j]]++] = i;

    std::rotate(ptr2.begin(), ptr2.end() - 1, ptr2.end());
    ptr2.front() = 0;

    // Merge both matrices.
    boost::fill(marker, -1);
    pptr.resize(np + 1, 0);

    for(int i = 0; i < np; ++i) {
        for(int j = ptr1[i]; j < ptr1[i+1]; ++j) {
            int c = col1[j];
            if (marker[c] != i) {
                marker[c] = i;
                ++pptr[i + 1];
            }
        }

        for(int j = ptr2[i]; j < ptr2[i+1]; ++j) {
            int c = col2[j];
            if (marker[c] != i) {
                marker[c] = i;
                ++pptr[i + 1];
            }
        }
    }

    boost::partial_sum(pptr, pptr.begin());
    boost::fill(marker, -1);

    pcol.resize(pptr.back());

    for(int i = 0; i < np; ++i) {
        int row_beg = pptr[i];
        int row_end = row_beg;

        for(int j = ptr1[i]; j < ptr1[i+1]; ++j) {
            int c = col1[j];

            if (marker[c] < row_beg) {
                marker[c] = row_end;
                pcol[row_end++] = c;
            }
        }

        for(int j = ptr2[i]; j < ptr2[i+1]; ++j) {
            int c = col2[j];

            if (marker[c] < row_beg) {
                marker[c] = row_end;
                pcol[row_end++] = c;
            }
        }
    }
}

//---------------------------------------------------------------------------
std::vector<int> partition(
        int npart,
        const std::vector<int> &ptr,
        const std::vector<int> &col
        )
{
    int nrows = ptr.size() - 1;

    std::vector<int> part(nrows);

    if (npart == 1) {
        boost::fill(part, 0);
    } else {
        int wgtflag = 0;
        int numflag = 0;
        int options = 0;
        int edgecut;

        METIS_PartGraphKway(
                &nrows,
                const_cast<int*>(ptr.data()),
                const_cast<int*>(col.data()),
                NULL,
                NULL,
                &wgtflag,
                &numflag,
                &npart,
                &options,
                &edgecut,
                part.data()
                );
    }

    return part;
}

//---------------------------------------------------------------------------
std::vector<int> read_problem(
        const amgcl::mpi::communicator &world,
        const std::string &A_file,
        const std::string &b_file,
        int block_size,
        std::vector<int> &lptr,
        std::vector<int> &lcol,
        std::vector<double>    &lval,
        std::vector<double>    &lrhs
        )
{
    using amgcl::precondition;
    using amgcl::prof;

    prof.tic("read problem");
    EigenMatrix A;
    EigenVector b;

    amgcl::precondition(
            Eigen::loadMarket(A, A_file),
            "Failed to load matrix file (" + A_file + ")"
            );

    amgcl::precondition(
            Eigen::loadMarketVector(b, b_file),
            "Failed to load RHS file (" + b_file + ")"
            );

    const int    *ptr = A.outerIndexPtr();
    const int    *col = A.innerIndexPtr();
    const double *val = A.valuePtr();
    prof.toc("read problem");

    prof.tic("Pointwise graph");
    std::vector<int> pptr, pcol;
    pointwise_graph(A.rows(), block_size, ptr, col, pptr, pcol);
    prof.toc("Pointwise graph");

    prof.tic("Partition");
    std::vector<int> part = partition(world.size, pptr, pcol);
    prof.toc("Partition");

    prof.tic("Reorder");
    std::vector<int> dist(world.size + 1, 0);
    BOOST_FOREACH(int p, part) dist[p + 1] += block_size;
    boost::partial_sum(dist, dist.begin());

    std::vector<int> order(A.rows());
    std::vector<int> inv_order(A.rows());
    for(int i = 0; i < A.rows(); ++i) {
        int p = part[i / block_size];
        int j = dist[p]++;
        order[i] = j;
        inv_order[j] = i;
    }
    std::rotate(dist.begin(), dist.end() - 1, dist.end());
    dist.front() = 0;
    prof.toc("Reorder");

    prof.tic("Local matrix");
    int chunk = dist[world.rank + 1] - dist[world.rank];

    lptr.clear();
    lcol.clear();
    lval.clear();
    lrhs.clear();

    lptr.reserve(chunk + 1);
    lptr.push_back(0);
    for(int i = 0, j = dist[world.rank]; i < chunk; ++i, ++j) {
        int p = inv_order[j];
        lptr.push_back( lptr.back() - ptr[p] + ptr[p + 1] );
    }

    lcol.reserve(lptr.back());
    lval.reserve(lptr.back());
    lrhs.reserve(chunk);

    for(int i = 0, j = dist[world.rank]; i < chunk; ++i, ++j) {
        int p = inv_order[j];

        for(int k = ptr[p]; k < ptr[p+1]; ++k) {
            lcol.push_back( order[col[k]] );
            lval.push_back( val[k] );
        }

        lrhs.push_back( b[p] );
    }
    prof.toc("Local matrix");

    return dist;
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    BOOST_SCOPE_EXIT(void) {
        MPI_Finalize();
    } BOOST_SCOPE_EXIT_END

    amgcl::mpi::communicator world(MPI_COMM_WORLD);

    if (world.rank == 0)
        std::cout << "World size: " << world.size << std::endl;

    // Read configuration from command line
    amgcl::runtime::coarsening::type    coarsening       = amgcl::runtime::coarsening::smoothed_aggregation;
    amgcl::runtime::relaxation::type    relaxation       = amgcl::runtime::relaxation::spai0;
    amgcl::runtime::solver::type        iterative_solver = amgcl::runtime::solver::bicgstabl;
    amgcl::runtime::direct_solver::type direct_solver    = amgcl::runtime::direct_solver::skyline_lu;
    std::string parameter_file;
    std::string A_file   = "A.mm";
    std::string rhs_file = "rhs.mm";
    std::string out_file;

    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "show help")
        (
         "coarsening,c",
         po::value<amgcl::runtime::coarsening::type>(&coarsening)->default_value(coarsening),
         "ruge_stuben, aggregation, smoothed_aggregation, smoothed_aggr_emin"
        )
        (
         "relaxation,r",
         po::value<amgcl::runtime::relaxation::type>(&relaxation)->default_value(relaxation),
         "gauss_seidel, ilu0, damped_jacobi, spai0, chebyshev"
        )
        (
         "iter_solver,i",
         po::value<amgcl::runtime::solver::type>(&iterative_solver)->default_value(iterative_solver),
         "cg, bicgstab, bicgstabl, gmres"
        )
        (
         "dir_solver,d",
         po::value<amgcl::runtime::direct_solver::type>(&direct_solver)->default_value(direct_solver),
         "skyline_lu"
#ifdef AMGCL_HAVE_PASTIX
         ", pastix"
#endif
        )
        (
         "params,p",
         po::value<std::string>(&parameter_file),
         "parameter file in json format"
        )
        (
         "matrix,A",
         po::value<std::string>(&A_file)->default_value(A_file),
         "The system matrix in MatrixMarket format"
        )
        (
         "rhs,b",
         po::value<std::string>(&rhs_file)->default_value(rhs_file),
         "The right-hand side in MatrixMarket format"
        )
        (
         "output,o",
         po::value<std::string>(&out_file),
         "The output file (saved in MatrixMarket format)"
        )
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    boost::property_tree::ptree prm;
    if (vm.count("params")) read_json(parameter_file, prm);

    using amgcl::prof;

    int block_size = prm.get("amg.coarsening.aggr.block_size", 1);

    prof.tic("read problem");
    std::vector<int> ptr;
    std::vector<int> col;
    std::vector<double>    val;
    std::vector<double>    rhs;

    std::vector<int> domain = read_problem(
            world, A_file, rhs_file, block_size, ptr, col, val, rhs
            );

    int chunk = domain[world.rank + 1] - domain[world.rank];
    prof.toc("read problem");

    prof.tic("setup");
    typedef
        amgcl::runtime::mpi::subdomain_deflation<
            amgcl::backend::builtin<double>
            >
        SDD;

    SDD solve(
            coarsening, relaxation, iterative_solver, direct_solver,
            world, boost::tie(chunk, ptr, col, val),
            amgcl::mpi::constant_deflation(block_size), prm
            );
    double tm_setup = prof.toc("setup");

    std::vector<double> x(chunk, 0);

    prof.tic("solve");
    size_t iters;
    double resid;
    boost::tie(iters, resid) = solve(rhs, x);
    double tm_solve = prof.toc("solve");

    if (vm.count("output")) {
        prof.tic("save");
        for(int r = 0; r < world.size; ++r) {
            if (r == world.rank) {
                std::ofstream f(out_file.c_str(), std::ios::app);

                if (world.rank == 0)
                    f << domain.back() << " 1\n";

                std::ostream_iterator<double> oi(f, "\n");
                boost::copy(x, oi);
            }
            MPI_Barrier(world);
        }
        prof.toc("save");
    }

    if (world.rank == 0) {
        std::cout
            << "Iterations: " << iters << std::endl
            << "Error:      " << resid << std::endl
            << std::endl
            << prof << std::endl;

#ifdef _OPENMP
        int nt = omp_get_max_threads();
#else
        int nt = 1;
#endif
        std::ostringstream log_name;
        log_name << "log_" << domain.back() << "_" << nt << "_" << world.size << ".txt";
        std::ofstream log(log_name.str().c_str(), std::ios::app);
        log << domain.back() << "\t" << nt << "\t" << world.size
            << "\t" << tm_setup << "\t" << tm_solve
            << "\t" << iters << "\t" << std::endl;
    }

}
