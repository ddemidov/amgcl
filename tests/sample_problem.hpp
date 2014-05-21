#ifndef TESTS_SAMPLE_PROBLEM_HPP
#define TESTS_SAMPLE_PROBLEM_HPP

// Generates matrix for poisson problem in a unit cube.
template <typename real, class RHS>
int sample_problem(int n,
        std::vector<real> &val,
        std::vector<int>  &col,
        std::vector<int>  &ptr,
        RHS &rhs
        )
{
    int  n3  = n * n * n;
    real h2i = (n - 1) * (n - 1);

    ptr.clear();
    col.clear();
    val.clear();
    rhs.resize(n3);

    ptr.reserve(n3 + 1);
    col.reserve(n3 * 7);
    val.reserve(n3 * 7);

    ptr.push_back(0);
    for(int k = 0, idx = 0; k < n; ++k) {
        for(int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i, ++idx) {
                if (
                        i == 0 || i == n - 1 ||
                        j == 0 || j == n - 1 ||
                        k == 0 || k == n - 1
                   )
                {
                    col.push_back(idx);
                    val.push_back(1);

                    rhs[idx] = 0;
                } else {
                    col.push_back(idx - n * n);
                    val.push_back(-h2i);

                    col.push_back(idx - n);
                    val.push_back(-h2i);

                    col.push_back(idx - 1);
                    val.push_back(-h2i);

                    col.push_back(idx);
                    val.push_back(6 * h2i);

                    col.push_back(idx + 1);
                    val.push_back(-h2i);

                    col.push_back(idx + n);
                    val.push_back(-h2i);

                    col.push_back(idx + n * n);
                    val.push_back(-h2i);

                    rhs[idx] = 1;
                }

                ptr.push_back(col.size());
            }
        }
    }

    return n3;
}

#endif
