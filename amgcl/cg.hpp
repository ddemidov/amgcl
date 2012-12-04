#ifndef AMGCL_CG_HPP
#define AMGCL_CG_HPP

#include <tuple>

namespace amg {

struct cg_tag {
    int maxiter;
    double tol;

    cg_tag(int maxiter = 100, double tol = 1e-8)
        : maxiter(maxiter), tol(tol)
    {}
};

template <class matrix, class vector, class precond>
std::tuple< int, typename value_type<vector>::type >
solve(const matrix &A, const vector &rhs, precond &P, vector &x, cg_tag prm = cg_tag())
{
    typedef typename value_type<vector>::type value_t;

    const auto n = x.size();

    vector r(n), s(n), p(n), q(n);
    r = rhs - A * x;

    value_t rho1 = 0, rho2 = 0;
    value_t norm_of_rhs = norm(rhs);

    if (norm_of_rhs == 0) {
        clear(x);
        return std::make_tuple(0, norm_of_rhs);
    }

    int     iter;
    value_t res;
    for(
            iter = 0;
            (res = norm(r) / norm_of_rhs) > prm.tol && iter < prm.maxiter;
            ++iter
       )
    {
        clear(s);
        P.apply(r, s);

        rho2 = rho1;
        rho1 = inner_prod(r, s);

        if (iter)
            p = s + (rho1 / rho2) * p;
        else
            p = s;

        q = A * p;

        value_t alpha = rho1 / inner_prod(q, p);

        x += alpha * p;
        r -= alpha * q;
    }

    return std::make_tuple(iter, res);
}

} // namespace amg

#endif
