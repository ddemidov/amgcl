#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <amgcl/util.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/io/mm.hpp>
#include <amgcl/io/binary.hpp>

int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    namespace io = amgcl::io;

    using amgcl::precondition;

    po::options_description desc("Options");

    desc.add_options()
        ("help,h", "Show this help.")
        ("dense,d", po::bool_switch()->default_value(false),
         "Matrix is dense.")
        ("input,i", po::value<std::string>()->required(),
         "Input binary file.")
        ("output,o", po::value<std::string>()->required(),
         "Ouput matrix in the MatrixMarket format.")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    po::notify(vm);

    if (vm["dense"].as<bool>()) {
        size_t n, m;
        std::vector<double> v;

        io::read_dense(vm["input"].as<std::string>(), n, m, v);
        io::mm_write(vm["output"].as<std::string>(), v.data(), n, m);

        std::cout
            << "Wrote " << n << " by " << m << " dense matrix"
            << std::endl;
    } else {
        size_t n;
        std::vector<ptrdiff_t> ptr, col;
        std::vector<double> val;

        io::read_crs(vm["input"].as<std::string>(), n, ptr, col, val);
        io::mm_write(vm["output"].as<std::string>(), std::tie(n, ptr, col, val));

        std::cout
            << "Wrote " << n << " by " << n << " sparse matrix, "
            << ptr.back() << " nonzeros" << std::endl;
    }
}
