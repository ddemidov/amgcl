#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>

#include <vexcl/vexcl.hpp>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#define AMGCL_PROFILING

#include <amgcl/amgcl.hpp>
#include <amgcl/aggr_plain.hpp>
#include <amgcl/interp_aggr.hpp>
#include <amgcl/interp_smoothed_aggr.hpp>
#include <amgcl/interp_sa_emin.hpp>
#include <amgcl/interp_classic.hpp>
#include <amgcl/operations_vexcl.hpp>
#include <amgcl/operations_eigen.hpp>
#include <amgcl/level_cpu.hpp>
#include <amgcl/level_vexcl.hpp>
#include <amgcl/cg.hpp>
#include <amgcl/bicgstab.hpp>
#include <amgcl/gmres.hpp>
#include <amgcl/profiler.hpp>

#include "read.hpp"

typedef double real;
typedef Eigen::Matrix<real, Eigen::Dynamic, 1> EigenVector;

namespace po = boost::program_options;

namespace amgcl {
    profiler<> prof("utest");
}
using amgcl::prof;

enum interp_t {
    classic              = 1,
    aggregation          = 2,
    smoothed_aggregation = 3,
    sa_emin              = 4
};

enum level_t {
    vexcl_lvl = 1,
    cpu_lvl   = 2
};

enum solver_t {
    cg         = 1,
    bicg       = 2,
    gmres      = 3,
    standalone = 4
};

#define GMRES_M 30

enum relax_t {
    damped_jacobi = 1,
    spai0         = 2,
    gauss_seidel  = 3,
    ilu0          = 4
};

struct options {
    int         solver;
    std::string pfile;

    unsigned coarse_enough;
    amgcl::level::params lp;
};

//---------------------------------------------------------------------------
template <class AMG, class spmat, class vector>
void solve(
        const AMG          &amg,
        const spmat        &A,
        const vector       &rhs,
        vector             &x,
        const options      &op
        )
{
    const int n = rhs.size();

    std::pair<int,real> cnv;
    prof.tic("solve");
    switch (static_cast<solver_t>(op.solver)) {
        case cg:
            cnv = amgcl::solve(A, rhs, amg, x, amgcl::cg_tag(op.lp.maxiter, op.lp.tol));
            break;
        case bicg:
            cnv = amgcl::solve(A, rhs, amg, x, amgcl::bicg_tag(op.lp.maxiter, op.lp.tol));
            break;
        case gmres:
            cnv = amgcl::solve(A, rhs, amg, x, amgcl::gmres_tag(GMRES_M, op.lp.maxiter, op.lp.tol));
            break;
        case standalone:
            cnv = amg.solve(rhs, x);
            break;
        default:
            throw std::invalid_argument("Unsupported iterative solver");
    }
    prof.toc("solve");

    std::cout << "Iterations: " << std::get<0>(cnv) << std::endl
              << "Error:      " << std::get<1>(cnv) << std::endl
              << std::endl;
}

//---------------------------------------------------------------------------
template <class Interp, amgcl::relax::scheme Relax, class spmat, class vector>
void run_cpu_test(const spmat &A, const vector &rhs, const options &op) {
    typedef amgcl::solver<real, int, Interp, amgcl::level::cpu<Relax> > AMG;

    typename AMG::params prm;

    prm.coarse_enough = op.coarse_enough;

    prm.level.npre   = op.lp.npre;
    prm.level.npost  = op.lp.npost;
    prm.level.ncycle = op.lp.ncycle;
    prm.level.kcycle = op.lp.kcycle;
    prm.level.tol    = op.lp.tol;
    prm.level.maxiter= op.lp.maxiter;

    EigenVector x = EigenVector::Zero(rhs.size());

    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    Eigen::MappedSparseMatrix<real, Eigen::RowMajor, int> Amap(
            amgcl::sparse::matrix_rows(A),
            amgcl::sparse::matrix_cols(A),
            amgcl::sparse::matrix_nonzeros(A),
            const_cast<int* >(amgcl::sparse::matrix_outer_index(A)),
            const_cast<int* >(amgcl::sparse::matrix_inner_index(A)),
            const_cast<real*>(amgcl::sparse::matrix_values(A))
            );

    solve(amg, Amap, rhs, x, op);
}

//---------------------------------------------------------------------------
template <class Interp, amgcl::relax::scheme Relax, class spmat, class vector>
void run_vexcl_test(const spmat &A, const vector &rhs, const options &op) {
    typedef amgcl::solver<real, int, Interp, amgcl::level::vexcl<Relax> > AMG;

    prof.tic("OpenCL initialization");
    vex::Context ctx( vex::Filter::Env && vex::Filter::DoublePrecision );
    prof.toc("OpenCL initialization");

    if (!ctx.size()) throw std::runtime_error("No available compute devices");
    std::cout << ctx << std::endl;

    typename AMG::params prm;

    prm.coarse_enough = op.coarse_enough;

    prm.level.npre   = op.lp.npre;
    prm.level.npost  = op.lp.npost;
    prm.level.ncycle = op.lp.ncycle;
    prm.level.kcycle = op.lp.kcycle;
    prm.level.tol    = op.lp.tol;
    prm.level.maxiter= op.lp.maxiter;


    prof.tic("setup");
    AMG amg(A, prm);
    prof.toc("setup");

    std::cout << amg << std::endl;

    vex::vector<real> f(ctx.queue(), rhs.size(), rhs.data());
    vex::vector<real> x(ctx.queue(), rhs.size());
    x = 0;

    solve(amg, amg.top_matrix(), f, x, op);
}

//---------------------------------------------------------------------------
template <class interp, class spmat, class vector>
void run_vexcl_test(int relax, const spmat &A, const vector &rhs, const options &op) {
    switch (static_cast<relax_t>(relax)) {
        case damped_jacobi:
            run_vexcl_test<interp, amgcl::relax::damped_jacobi>(A, rhs, op);
            break;
        case spai0:
            run_vexcl_test<interp, amgcl::relax::spai0>(A, rhs, op);
            break;
        default:
            throw std::invalid_argument("Unsupported relaxation scheme for vexcl level");
    }
}

//---------------------------------------------------------------------------
template <class interp, class spmat, class vector>
void run_cpu_test(int relax, const spmat &A, const vector &rhs, const options &op) {
    switch (static_cast<relax_t>(relax)) {
        case damped_jacobi:
            run_cpu_test<interp, amgcl::relax::damped_jacobi>(A, rhs, op);
            break;
        case spai0:
            run_cpu_test<interp, amgcl::relax::spai0>(A, rhs, op);
            break;
        case gauss_seidel:
            run_cpu_test<interp, amgcl::relax::gauss_seidel>(A, rhs, op);
            break;
        case ilu0:
            run_cpu_test<interp, amgcl::relax::ilu0>(A, rhs, op);
            break;
        default:
            throw std::invalid_argument("Unsupported relaxation scheme");
    }
}

//---------------------------------------------------------------------------
template <class interp, class spmat, class vector>
void run_test(int level, int relax, const spmat &A, const vector &rhs, const options &op) {
    switch(static_cast<level_t>(level)) {
        case vexcl_lvl:
            run_vexcl_test<interp>(relax, A, rhs, op);
            break;
        case cpu_lvl:
            run_cpu_test<interp>(relax, A, rhs, op);
            break;
        default:
            throw std::invalid_argument("Unsupported backend");
    }
}

//---------------------------------------------------------------------------
template <class spmat, class vector>
void run_test(int interp, int level, int relax, const spmat &A, const vector &rhs, const options &op) {
    switch(static_cast<interp_t>(interp)) {
        case classic:
            run_test< amgcl::interp::classic >(level, relax, A, rhs, op);
            break;
        case aggregation:
            run_test< amgcl::interp::aggregation< amgcl::aggr::plain > >(level, relax, A, rhs, op);
            break;
        case smoothed_aggregation:
            run_test< amgcl::interp::smoothed_aggregation< amgcl::aggr::plain > >(level, relax, A, rhs, op);
            break;
        case sa_emin:
            run_test< amgcl::interp::sa_emin< amgcl::aggr::plain > >(level, relax, A, rhs, op);
            break;
        default:
            throw std::invalid_argument("Unsupported interpolation scheme");
    }
}

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int interp;
    int level;
    int relax;

    options op;

    po::options_description desc("Possible options");

    desc.add_options()
        ("help", "Show help")
        ("interp", po::value<int>(&interp)->default_value(smoothed_aggregation),
            "Interpolation: classic(1), aggregation(2), "
            "smoothed_aggregation (3), smoothed aggregation with energy minimization (4)"
            )
        ("level", po::value<int>(&level)->default_value(vexcl_lvl),
            "Backend: vexcl(1), cpu(2)"
            )
        ("solver", po::value<int>(&op.solver)->default_value(cg),
            "Iterative solver: cg(1), bicgstab(2), gmres(3), standalone(4)")
        ("relax", po::value<int>(&relax)->default_value(spai0),
            "Iterative solver: damped jacobi(1), spai0(2), gauss-seidel (3), ilu0(4)")

        ("problem",
            po::value<std::string>(&op.pfile)->default_value("problem.dat"),
            "Problem file"
            )

        ("coarse_enough", po::value<unsigned>(&op.coarse_enough)->default_value(300))

        ("npre",   po::value<unsigned>(&op.lp.npre  )->default_value(op.lp.npre))
        ("npost",  po::value<unsigned>(&op.lp.npost )->default_value(op.lp.npost))
        ("ncycle", po::value<unsigned>(&op.lp.ncycle)->default_value(op.lp.ncycle))
        ("kcycle", po::value<unsigned>(&op.lp.kcycle)->default_value(op.lp.kcycle))
        ("tol",    po::value<double  >(&op.lp.tol   )->default_value(op.lp.tol))
        ("maxiter",po::value<unsigned>(&op.lp.maxiter)->default_value(op.lp.maxiter))
        ;

    po::positional_options_description pdesc;
    pdesc.add("problem", -1);


    po::variables_map vm;
    po::store(
            po::command_line_parser(argc, argv).
                options(desc).
                positional(pdesc).
                run(),
            vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    prof.tic("Read problem");
    std::vector<int>  row;
    std::vector<int>  col;
    std::vector<real> val;
    EigenVector       rhs;
    int n = read_problem(op.pfile, row, col, val, rhs);
    prof.toc("Read problem");

    auto A = amgcl::sparse::map(n, n, row.data(), col.data(), val.data());

    run_test(interp, level, relax, A, rhs, op);

    std::cout << prof << std::endl;
}
