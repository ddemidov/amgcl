#ifndef AMGCL_BICGSTAB_HPP
#define AMGCL_BICGSTAB_HPP

#include <tuple>
#include <stdexcept>

namespace amg {

struct bicg_tag {
    int maxiter;
    double tol;

    bicg_tag(int maxiter = 100, double tol = 1e-8)
        : maxiter(maxiter), tol(tol)
    {}
};

template <class matrix, class vector, class precond>
std::tuple< int, typename value_type<vector>::type >
solve(const matrix &A, const vector &rhs, precond &P, vector &x, bicg_tag prm = bicg_tag())
{
    typedef typename value_type<vector>::type value_t;

    const auto n = x.size();

    vector r (n);
    vector p (n);
    vector v (n);
    vector s (n);
    vector t (n);
    vector rh(n);
    vector ph(n);
    vector sh(n);

    rh = r = rhs - A * x;

    value_t rho1  = 0, rho2  = 0;
    value_t alpha = 0, omega = 0;

    value_t norm_of_rhs = norm(rhs);

    int     iter;
    value_t res = 2 * prm.tol;
    for(iter = 0; res > prm.tol && iter < prm.maxiter; ++iter) {
        rho2 = rho1;
        rho1 = inner_prod(r, rh);

        if (fabs(rho1) < 1e-32)
            throw std::logic_error("Zero rho in BiCGStab");

        if (iter)
            p = r + ((rho1 * alpha) / (rho2 * omega)) * (p - omega * v);
        else
            p = r;

        clear(ph);
        P.apply(p, ph);

        v = A * ph;

        alpha = rho1 / inner_prod(rh, v);

        s = r - alpha * v;

        if ((res = norm(s) / norm_of_rhs) < prm.tol) {
            x += alpha * ph;
        } else {
            clear(sh);
            P.apply(s, sh);

            t = A * sh;

            omega = inner_prod(t, s) / inner_prod(t, t);

            if (fabs(omega) < 1e-32)
                throw std::logic_error("Zero omega in BiCGStab");

            x += alpha * ph + omega * sh;
            r = s - omega * t;

            res = norm(r) / norm_of_rhs;
        }
    }

    return std::make_tuple(iter, res);
}

} // namespace amg

#endif
