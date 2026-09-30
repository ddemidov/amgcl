#define BOOST_TEST_MODULE TestDeterministicReductions
#include <boost/test/unit_test.hpp>

#include <amgcl/backend/builtin.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Built with AMGCL_DETERMINISTIC_REDUCTIONS defined (see tests/CMakeLists.txt).

namespace {
    std::vector<double> make_random_vector(size_t n) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> rnd(-1.0, 1.0);

        std::vector<double> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = rnd(gen);
        }

        return v;
    }

    // Random matrix with sorted, unique columns in each row (spgemm_rmerge
    // requires this).
    amgcl::backend::crs<double, ptrdiff_t, ptrdiff_t> make_random_matrix(ptrdiff_t n, int nnz_per_row) {
        std::mt19937 gen(7);
        std::uniform_real_distribution<double> rnd(0.1, 1.0);

        amgcl::backend::crs<double, ptrdiff_t, ptrdiff_t> A;
        A.set_size(n, n);
        A.ptr[0] = 0;

        for (ptrdiff_t i = 0; i < n; ++i) {
            A.ptr[i + 1] = A.ptr[i] + nnz_per_row;
        }

        A.set_nonzeros();

        std::vector<ptrdiff_t> all_cols(static_cast<size_t>(n));
        std::iota(all_cols.begin(), all_cols.end(), ptrdiff_t{0});

        for (ptrdiff_t i = 0; i < n; ++i) {
            std::vector<ptrdiff_t> cols = all_cols;
            std::shuffle(cols.begin(), cols.end(), gen);
            cols.resize(static_cast<size_t>(nnz_per_row));
            std::sort(cols.begin(), cols.end());

            ptrdiff_t j = A.ptr[i];
            for (ptrdiff_t col : cols) {
                A.col[j] = col;
                A.val[j] = rnd(gen);
                ++j;
            }
        }

        return A;
    }

    bool crs_equal(
        const amgcl::backend::crs<double, ptrdiff_t, ptrdiff_t> &a,
        const amgcl::backend::crs<double, ptrdiff_t, ptrdiff_t> &b)
    {
        if (a.nrows != b.nrows || a.ptr[a.nrows] != b.ptr[b.nrows]) return false;

        for (ptrdiff_t i = 0; i <= a.nrows; ++i) {
            if (a.ptr[i] != b.ptr[i]) return false;
        }

        for (ptrdiff_t i = 0; i < a.ptr[a.nrows]; ++i) {
            if (a.col[i] != b.col[i] || a.val[i] != b.val[i]) return false;
        }

        return true;
    }

    double sequential_dot(const std::vector<double> &x, const std::vector<double> &y) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

#ifdef _OPENMP
    struct thread_count_scope {
        int previous;
        explicit thread_count_scope(int n) : previous(omp_get_max_threads()) { omp_set_num_threads(n); }
        ~thread_count_scope() { omp_set_num_threads(previous); }
    };

    template <class Function>
    auto with_threads(int n, Function function) -> decltype(function()) {
        thread_count_scope scope(n);
        return function();
    }
#endif
}

BOOST_AUTO_TEST_CASE(inner_product_is_thread_count_invariant)
{
    // Sizes below, at and above the 4096-element block size.
    for (size_t n : {1, 100, 4096, 4097, 12345, 50000}) {
        auto x = make_random_vector(n);
        auto y = make_random_vector(n);

        auto dot = [&]() { return amgcl::backend::inner_product(x, y); };

#ifdef _OPENMP
        double at_one_thread = with_threads(1, dot);

        BOOST_CHECK_EQUAL(at_one_thread, with_threads(4, dot));
        BOOST_CHECK_EQUAL(at_one_thread, with_threads(8, dot));
        BOOST_CHECK_CLOSE(at_one_thread, sequential_dot(x, y), 1e-6);
#else
        BOOST_CHECK_CLOSE(dot(), sequential_dot(x, y), 1e-6);
#endif
    }
}

BOOST_AUTO_TEST_CASE(product_is_thread_count_invariant)
{
    auto A = make_random_matrix(500, 8);
    auto B = make_random_matrix(500, 8);

    auto multiply = [&]() { return amgcl::backend::product(A, B); };

#ifdef _OPENMP
    // By default product() switches algorithm above 16 threads.
    BOOST_CHECK(crs_equal(*with_threads(1, multiply), *with_threads(24, multiply)));
#else
    BOOST_CHECK(crs_equal(*multiply(), *multiply()));
#endif
}
