#ifndef READ_H
#define READ_H

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

template <class RHS>
inline int read_problem(const std::string &fname,
        std::vector<int>    &row,
        std::vector<int>    &col,
        std::vector<double> &val,
        RHS &rhs
        )
{
    std::cout << "Reading \"" << fname << "\"..." << std::endl;
    std::ifstream f(fname.c_str(), std::ios::binary);
    if (!f) throw std::invalid_argument("Failed to open problem file");

    int n;

    f.read((char*)&n, sizeof(int));

    row.resize(n + 1);
    f.read((char*)row.data(), row.size() * sizeof(int));

    col.resize(row.back());
    val.resize(row.back());
    rhs.resize(n);

    f.read((char*)&col[0], col.size() * sizeof(int));
    f.read((char*)&val[0], val.size() * sizeof(double));
    f.read((char*)&rhs[0], rhs.size() * sizeof(double));

    std::cout << "Done\n" << std::endl;

    return n;
}

template <class RHS, typename value_type>
inline int read_problem(const std::string &fname,
        std::vector<int>        &row,
        std::vector<int>        &col,
        std::vector<value_type> &val,
        RHS &rhs
        )
{
    std::cout << "Reading \"" << fname << "\"..." << std::endl;
    std::ifstream f(fname.c_str(), std::ios::binary);
    if (!f) throw std::invalid_argument("Failed to open problem file");

    int n;

    f.read((char*)&n, sizeof(int));

    row.resize(n + 1);
    f.read((char*)row.data(), row.size() * sizeof(int));

    col.resize(row.back());
    val.resize(row.back());
    rhs.resize(n);

    f.read((char*)col.data(), col.size() * sizeof(int));

    for(size_t i = 0, nnz = row.back(); i < nnz; ++i) {
        double v;
        f.read((char*)&v, sizeof(double));
        val[i] = v;
    }
    for(size_t i = 0; i < n; ++i) {
        double v;
        f.read((char*)&v, sizeof(double));
        rhs[i] = v;
    }

    std::cout << "Done\n" << std::endl;

    return n;
}

#endif
