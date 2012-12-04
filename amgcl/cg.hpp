#ifndef AMGCL_CG_HPP
#define AMGCL_CG_HPP

namespace amg {

template <class matrix, class vector, class precond>
void cg(const matrix &A, const vector &f, precond &P,
        vector &x)
{
    typedef typename value_type<vector>::type value_t;

    const auto n = x.size();

    vector r(n), s(n), p(n), q(n);
    r = f - A * x;

    value_t rho1 = 0, rho2 = 0;

    int     iter;
    value_t res;
    for(iter = 0; iter < 11 && (res = norm(r)) > 1e-8; ++iter) {
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

    std::cout << iter << " " << res << std::endl;
}

} // namespace amg

#endif
