#ifndef AMGCL_CG_HPP
#define AMGCL_CG_HPP

#include <vexcl/vexcl.hpp>

namespace amg {

template <typename T>
T inner_prod(const vex::vector<T> &x, const vex::vector<T> &y) {
    static vex::Reductor<T, vex::SUM> sum(vex::StaticContext<>::get().queue());
    return sum(x * y);
}

template <typename T>
T norm(const vex::vector<T> &x) {
    return sqrt( inner_prod(x, x) );
}

template <class matrix, class vector, class precond>
void cg(const matrix &A, const vector &f, precond &P,
        vector &x)
{
    typedef typename vector::value_type value_t;

    const auto n = x.size();

    vector r(n), s(n), p(n), q(n);
    r = f - A * x;

    value_t rho1 = 0, rho2 = 0;

    int     iter;
    value_t res;
    for(iter = 0; iter < 11 && (res = norm(r)) > 1e-8; ++iter) {
        s = 0;
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

    std::cout << iter << " " << res << std::endl;
}

} // namespace amg

#endif
