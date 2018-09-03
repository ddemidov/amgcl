#ifndef TESTS_SAMPLE_PROBLEM_HPP
#define TESTS_SAMPLE_PROBLEM_HPP

#include <complex>
#include <type_traits>
#include <cstddef>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/util.hpp>

// Generates matrix for a Poisson problem in a unit square.
template <typename ValueType, typename IndexType, typename RhsType>
int sample_problem_2d(
        ptrdiff_t               n,
        std::vector<ValueType>  &val,
        std::vector<IndexType>  &col,
        std::vector<IndexType>  &ptr,
        std::vector<RhsType>    &rhs,
        double anisotropy = 1.0
        )
{
    ptrdiff_t n2  = n * n;

    ptr.clear();
    col.clear();
    val.clear();
    rhs.clear();

    ptr.reserve(n2 + 1);
    col.reserve(n2 * 5);
    val.reserve(n2 * 5);
    rhs.reserve(n2);

    ValueType one = amgcl::math::identity<ValueType>();

    double hx = 1;
    double hy = hx * anisotropy;

    ptr.push_back(0);
    for(ptrdiff_t j = 0, idx = 0; j < n; ++j) {
        for (ptrdiff_t i = 0; i < n; ++i, ++idx) {
            if (j > 0) {
                col.push_back(idx - n);
                val.push_back(-1.0/(hy * hy) * one);
            }

            if (i > 0) {
                col.push_back(idx - 1);
                val.push_back(-1.0/(hx * hx) * one);
            }

            col.push_back(idx);
            val.push_back((2 / (hx * hx) + 2 / (hy * hy)) * one);

            if (i + 1 < n) {
                col.push_back(idx + 1);
                val.push_back(-1.0/(hx * hx) * one);
            }

            if (j + 1 < n) {
                col.push_back(idx + n);
                val.push_back(-1.0/(hy * hy) * one);
            }

            rhs.push_back( amgcl::math::constant<RhsType>(1.0) );
            ptr.push_back( static_cast<IndexType>(col.size()) );
        }
    }

    return n2;
}

// Generates matrix for a Poisson problem in a unit cube.
template <typename ValueType, typename IndexType, typename RhsType>
int sample_problem_3d(
        ptrdiff_t               n,
        std::vector<ValueType>  &val,
        std::vector<IndexType>  &col,
        std::vector<IndexType>  &ptr,
        std::vector<RhsType>    &rhs,
        double anisotropy = 1.0
        )
{
    ptrdiff_t n3  = n * n * n;

    ptr.clear();
    col.clear();
    val.clear();
    rhs.clear();

    ptr.reserve(n3 + 1);
    col.reserve(n3 * 7);
    val.reserve(n3 * 7);
    rhs.reserve(n3);

    ValueType one = amgcl::math::identity<ValueType>();

    double hx = 1;
    double hy = hx * anisotropy;
    double hz = hy * anisotropy;

    ptr.push_back(0);
    for(ptrdiff_t k = 0, idx = 0; k < n; ++k) {
        for(ptrdiff_t j = 0; j < n; ++j) {
            for (ptrdiff_t i = 0; i < n; ++i, ++idx) {
                if (k > 0) {
                    col.push_back(idx - n * n);
                    val.push_back(-1.0/(hz * hz) * one);
                }

                if (j > 0) {
                    col.push_back(idx - n);
                    val.push_back(-1.0/(hy * hy) * one);
                }

                if (i > 0) {
                    col.push_back(idx - 1);
                    val.push_back(-1.0/(hx * hx) * one);
                }

                col.push_back(idx);
                val.push_back((2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz)) * one);

                if (i + 1 < n) {
                    col.push_back(idx + 1);
                    val.push_back(-1.0/(hx * hx) * one);
                }

                if (j + 1 < n) {
                    col.push_back(idx + n);
                    val.push_back(-1.0/(hy * hy) * one);
                }

                if (k + 1 < n) {
                    col.push_back(idx + n * n);
                    val.push_back(-1.0/(hz * hz) * one);
                }

                rhs.push_back( amgcl::math::constant<RhsType>(1.0) );
                ptr.push_back( static_cast<IndexType>(col.size()) );
            }
        }
    }

    return n3;
}

// Generates matrix for a Poisson problem
template <typename ValueType, typename IndexType, typename RhsType>
int sample_problem(
        ptrdiff_t               n,
        std::vector<ValueType>  &val,
        std::vector<IndexType>  &col,
        std::vector<IndexType>  &ptr,
        std::vector<RhsType>    &rhs,
        double anisotropy = 1.0,
        int dim = 3
        )
{
    if (dim == 3)
        return sample_problem_3d(n, val, col, ptr, rhs, anisotropy);

    if (dim == 2)
        return sample_problem_2d(n, val, col, ptr, rhs, anisotropy);

    amgcl::precondition(false, "Unsupported problem dimensionality");
    return 0;
}
#endif
