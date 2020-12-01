#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include <amgcl/backend/builtin.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/coarsening/rigid_body_modes.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/deflated_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/reorder.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

#include <amgcl/profiler.hpp>

namespace amgcl { profiler<> prof; }
using amgcl::prof;
using amgcl::precondition;

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    using amgcl::prof;
    using std::vector;
    using std::string;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("prm-file,P",
         po::value<string>(),
         "Parameter file in json format. "
        )
        (
         "prm,p",
         po::value< vector<string> >()->multitoken(),
         "Parameters specified as name=value pairs. "
         "May be provided multiple times. Examples:\n"
         "  -p solver.tol=1e-3\n"
         "  -p precond.coarse_enough=300"
        )
        ("matrix,A",
         po::value<string>()->required(),
         "System matrix in the MatrixMarket format."
        )
        (
         "rhs,f",
         po::value<string>(),
         "The RHS vector in the MatrixMarket format. "
         "When omitted, a vector of ones is used by default. "
         "Should only be provided together with a system matrix. "
        )
        (
         "defvec,D",
         po::value<string>(),
         "The near null-space vectors in the MatrixMarket format. "
        )
        (
         "coords,C",
         po::value<string>(),
         "Coordinate matrix where number of rows corresponds to the number of grid nodes "
         "and the number of columns corresponds to the problem dimensionality (2 or 3). "
         "Will be used to construct near null-space vectors as rigid body modes. "
        )
        (
         "binary,B",
         po::bool_switch()->default_value(false),
         "When specified, treat input files as binary instead of as MatrixMarket. "
         "It is assumed the files were converted to binary format with mm2bin utility. "
        )
        (
         "single-level,1",
         po::bool_switch()->default_value(false),
         "When specified, the AMG hierarchy is not constructed. "
         "Instead, the problem is solved using a single-level smoother as preconditioner. "
        )
        (
         "output,o",
         po::value<string>(),
         "Output file. Will be saved in the MatrixMarket format. "
         "When omitted, the solution is not saved. "
        )
        ;

    po::positional_options_description p;
    p.add("prm", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    for (int i = 0; i < argc; ++i) {
        if (i) std::cout << " ";
        std::cout << argv[i];
    }
    std::cout << std::endl;

    boost::property_tree::ptree prm;
    if (vm.count("prm-file")) {
        read_json(vm["prm-file"].as<string>(), prm);
    }

    if (vm.count("prm")) {
        for(const string &v : vm["prm"].as<vector<string> >()) {
            amgcl::put(prm, v);
        }
    }

    if (!vm.count("defvec") && !vm.count("coords")) {
        std::cerr << "Either defvec or coords should be given" << std::endl;
        return 1;
    }

    ptrdiff_t rows, nv;
    vector<ptrdiff_t> ptr, col;
    vector<double> val, rhs, z;

    {
        auto t = prof.scoped_tic("reading");

        string Afile  = vm["matrix"].as<string>();
        bool   binary = vm["binary"].as<bool>();

        if (binary) {
            io::read_crs(Afile, rows, ptr, col, val);
        } else {
            ptrdiff_t cols;
            std::tie(rows, cols) = io::mm_reader(Afile)(ptr, col, val);
            precondition(rows == cols, "Non-square system matrix");
        }

        if (vm.count("rhs")) {
            string bfile = vm["rhs"].as<string>();

            ptrdiff_t n, m;

            if (binary) {
                io::read_dense(bfile, n, m, rhs);
            } else {
                std::tie(n, m) = io::mm_reader(bfile)(rhs);
            }

            precondition(n == rows && m == 1, "The RHS vector has wrong size");
        } else {
            rhs.resize(rows, 1.0);
        }

        if (vm.count("defvec")) {
            string nfile = vm["defvec"].as<string>();
            std::vector<double> N;

            ptrdiff_t m;

            if (binary) {
                io::read_dense(nfile, m, nv, N);
            } else {
                std::tie(m, nv) = io::mm_reader(nfile)(N);
            }

            precondition(m == rows, "Deflation vectors have wrong size");

            z.resize(N.size());
            for(ptrdiff_t i = 0; i < rows; ++i)
                for(ptrdiff_t j = 0; j < nv; ++j)
                    z[i + j * rows] = N[i * nv + j];
        } else if (vm.count("coords")) {
            string cfile = vm["coords"].as<string>();
            std::vector<double> coo;

            ptrdiff_t m, ndim;

            if (binary) {
                io::read_dense(cfile, m, ndim, coo);
            } else {
                std::tie(m, ndim) = io::mm_reader(cfile)(coo);
            }

            precondition(m * ndim == rows && (ndim == 2 || ndim == 3), "Coordinate matrix has wrong size");

            nv = amgcl::coarsening::rigid_body_modes(ndim, coo, z, /*transpose = */true);
        }

        prm.put("nvec", nv);
        prm.put("vec",  z.data());
    }

    std::vector<double> x(rows, 0);

    int    iters;
    double error;

    if (vm["single-level"].as<bool>())
        prm.put("precond.class", "relaxation");

    typedef amgcl::backend::builtin<double> Backend;
    typedef amgcl::deflated_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::runtime::solver::wrapper<Backend>
        > Solver;

    auto A = std::tie(rows, ptr, col, val);

    prof.tic("setup");
    Solver solve(A, prm);
    prof.toc("setup");

    prof.tic("solve");
    std::tie(iters, error) = solve(rhs, x);
    prof.toc("solve");

    if (vm.count("output")) {
        auto t = prof.scoped_tic("write");
        amgcl::io::mm_write(vm["output"].as<string>(), x.data(), x.size());
    }

    std::vector<double> r(rows);
    amgcl::backend::residual(rhs, A, x, r);

    std::cout << "Iterations: " << iters << std::endl
              << "Error:      " << error << std::endl
              << "True error: " << sqrt(amgcl::backend::inner_product(r, r)) / sqrt(amgcl::backend::inner_product(rhs, rhs))
              << prof << std::endl;
}
