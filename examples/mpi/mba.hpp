#ifndef MBA_MBA_HPP
#define MBA_MBA_HPP

/*
The MIT License

Copyright (c) 2015 Denis Demidov <dennis.demidov@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   mba/mba.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Multilevel B-spline interpolation.
 */

#include <iostream>
#include <iomanip>
#include <map>
#include <list>
#include <utility>
#include <algorithm>

#include <boost/container/flat_map.hpp>
#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/function.hpp>

namespace mba {
namespace detail {

template <size_t N, size_t M>
struct power : boost::integral_constant<size_t, N * power<N, M-1>::value> {};

template <size_t N>
struct power<N, 0> : boost::integral_constant<size_t, 1> {};

/// N-dimensional grid iterator (nested loop with variable depth).
template <unsigned NDim>
class grid_iterator {
    public:
        typedef boost::array<size_t, NDim> index;

        explicit grid_iterator(const boost::array<size_t, NDim> &dims)
            : N(dims), idx(0)
        {
            std::fill(i.begin(), i.end(), 0);
            done = (i == N);
        }

        explicit grid_iterator(size_t dim) : idx(0) {
            std::fill(N.begin(), N.end(), dim);
            std::fill(i.begin(), i.end(), 0);
            done = (0 == dim);
        }

        size_t operator[](size_t d) const {
            return i[d];
        }

        const index& operator*() const {
            return i;
        }

        size_t position() const {
            return idx;
        }

        grid_iterator& operator++() {
            done = true;
            for(size_t d = NDim; d--; ) {
                if (++i[d] < N[d]) {
                    done = false;
                    break;
                }
                i[d] = 0;
            }

            ++idx;

            return *this;
        }

        operator bool() const { return !done; }

    private:
        index N, i;
        bool  done;
        size_t idx;
};

template <typename T, size_t N>
boost::array<T, N> operator+(boost::array<T, N> a, const boost::array<T, N> &b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
    return a;
}

template <typename T, size_t N>
boost::array<T, N> operator-(boost::array<T, N> a, T b) {
    std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::minus<T>(), b));
    return a;
}

template <typename T, size_t N>
boost::array<T, N> operator*(boost::array<T, N> a, T b) {
    std::transform(a.begin(), a.end(), a.begin(), std::bind2nd(std::multiplies<T>(), b));
    return a;
}

// Value of k-th B-Spline basic function at t.
inline double Bspline(size_t k, double t) {
    assert(0 <= t && t < 1);
    assert(k < 4);

    switch (k) {
        case 0:
            return (t * (t * (-t + 3) - 3) + 1) / 6;
        case 1:
            return (t * t * (3 * t - 6) + 4) / 6;
        case 2:
            return (t * (t * (-3 * t + 3) + 3) + 1) / 6;
        case 3:
            return t * t * t / 6;
        default:
            return 0;
    }
}

// Checks if p is between lo and hi
template <typename T, size_t N>
bool boxed(const boost::array<T,N> &lo, const boost::array<T,N> &p, const boost::array<T,N> &hi) {
    for(unsigned i = 0; i < N; ++i) {
        if (p[i] < lo[i] || p[i] > hi[i]) return false;
    }
    return true;
}

inline double safe_divide(double a, double b) {
    return b == 0.0 ? 0.0 : a / b;
}

template <unsigned NDim>
class control_lattice {
    public:
        typedef boost::array<size_t, NDim> index;
        typedef boost::array<double, NDim> point;

        virtual ~control_lattice() {}

        virtual double operator()(const point &p) const = 0;

        virtual void report(std::ostream&) const = 0;

        template <class CooIter, class ValIter>
        double residual(CooIter coo_begin, CooIter coo_end, ValIter val_begin) const {
            double res = 0.0;

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for(; p != coo_end; ++p, ++v) {
                (*v) -= (*this)(*p);
                res = std::max(res, std::abs(*v));
            }

            return res;
        }
};

template <unsigned NDim>
class initial_approximation : public control_lattice<NDim> {
    public:
        typedef typename control_lattice<NDim>::point point;

        initial_approximation(boost::function<double(const point&)> f)
            : f(f) {}

        double operator()(const point &p) const {
            return f(p);
        }

        void report(std::ostream &os) const {
            os << "initial approximation";
        }
    private:
        boost::function<double(const point&)> f;
};

template <unsigned NDim>
class control_lattice_dense : public control_lattice<NDim> {
    public:
        typedef typename control_lattice<NDim>::index index;
        typedef typename control_lattice<NDim>::point point;

        template <class CooIter, class ValIter>
        control_lattice_dense(
                const point &coo_min, const point &coo_max, index grid_size,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin
                ) : cmin(coo_min), cmax(coo_max), grid(grid_size)
        {
            for(unsigned i = 0; i < NDim; ++i) {
                hinv[i] = (grid[i] - 1) / (cmax[i] - cmin[i]);
                cmin[i] -= 1 / hinv[i];
                grid[i] += 2;
            }

            boost::multi_array<double, NDim> delta(grid);
            boost::multi_array<double, NDim> omega(grid);

            std::fill(delta.data(), delta.data() + delta.num_elements(), 0.0);
            std::fill(omega.data(), omega.data() + omega.num_elements(), 0.0);

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for(; p != coo_end; ++p, ++v) {
                if (!boxed(coo_min, *p, coo_max)) continue;

                index i;
                point s;

                for(unsigned d = 0; d < NDim; ++d) {
                    double u = ((*p)[d] - cmin[d]) * hinv[d];
                    i[d] = floor(u) - 1;
                    s[d] = u - floor(u);
                }

                boost::array< double, power<4, NDim>::value > w;
                double sum_w2 = 0.0;

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double prod = 1.0;
                    for(unsigned k = 0; k < NDim; ++k) prod *= Bspline(d[k], s[k]);

                    w[d.position()] = prod;
                    sum_w2 += prod * prod;
                }

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double w1  = w[d.position()];
                    double w2  = w1 * w1;
                    double phi = (*v) * w1 / sum_w2;

                    index j = i + (*d);

                    delta(j) += w2 * phi;
                    omega(j) += w2;
                }
            }

            phi.resize(grid);

            std::transform(
                    delta.data(), delta.data() + delta.num_elements(),
                    omega.data(), phi.data(), safe_divide
                    );
        }

        double operator()(const point &p) const {
            index i;
            point s;

            for(unsigned d = 0; d < NDim; ++d) {
                double u = (p[d] - cmin[d]) * hinv[d];
                i[d] = floor(u) - 1;
                s[d] = u - floor(u);
            }

            double f = 0;

            for(grid_iterator<NDim> d(4); d; ++d) {
                double w = 1.0;
                for(unsigned k = 0; k < NDim; ++k) w *= Bspline(d[k], s[k]);

                f += w * phi(i + (*d));
            }

            return f;
        }

        void report(std::ostream &os) const {
            boost::io::ios_all_saver stream_state(os);

            os << "dense  [" << grid[0];
            for(unsigned i = 1; i < NDim; ++i)
                os << ", " << grid[i];
            os << "] (" << phi.num_elements() * sizeof(double) << " bytes)";
        }

        void append_refined(const control_lattice_dense &r) {
            static const boost::array<double, 5> s = {
                0.125, 0.500, 0.750, 0.500, 0.125
            };

            for(grid_iterator<NDim> i(r.grid); i; ++i) {
                double f = r.phi(*i);

                if (f == 0.0) continue;

                for(grid_iterator<NDim> d(5); d; ++d) {
                    index j;
                    bool skip = false;
                    for(unsigned k = 0; k < NDim; ++k) {
                        j[k] = 2 * i[k] + d[k] - 3;
                        if (j[k] >= grid[k]) {
                            skip = true;
                            break;
                        }
                    }

                    if (skip) continue;

                    double c = 1.0;
                    for(unsigned k = 0; k < NDim; ++k) c *= s[d[k]];

                    phi(j) += f * c;
                }
            }
        }

        double fill_ratio() const {
            size_t total    = phi.num_elements();
            size_t nonzeros = total - std::count(phi.data(), phi.data() + total, 0.0);

            return static_cast<double>(nonzeros) / total;
        }

    private:
        point cmin, cmax, hinv;
        index grid;

        boost::multi_array<double, NDim> phi;

};

template <unsigned NDim>
class control_lattice_sparse : public control_lattice<NDim> {
    public:
        typedef typename control_lattice<NDim>::index index;
        typedef typename control_lattice<NDim>::point point;

        template <class CooIter, class ValIter>
        control_lattice_sparse(
                const point &coo_min, const point &coo_max, index grid_size,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin
                ) : cmin(coo_min), cmax(coo_max), grid(grid_size)
        {
            for(unsigned i = 0; i < NDim; ++i) {
                hinv[i] = (grid[i] - 1) / (cmax[i] - cmin[i]);
                cmin[i] -= 1 / hinv[i];
                grid[i] += 2;
            }

            std::map<index, two_doubles> dw;

            CooIter p = coo_begin;
            ValIter v = val_begin;

            for(; p != coo_end; ++p, ++v) {
                if (!boxed(coo_min, *p, coo_max)) continue;

                index i;
                point s;

                for(unsigned d = 0; d < NDim; ++d) {
                    double u = ((*p)[d] - cmin[d]) * hinv[d];
                    i[d] = floor(u) - 1;
                    s[d] = u - floor(u);
                }

                boost::array< double, power<4, NDim>::value > w;
                double sum_w2 = 0.0;

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double prod = 1.0;
                    for(unsigned k = 0; k < NDim; ++k) prod *= Bspline(d[k], s[k]);

                    w[d.position()] = prod;
                    sum_w2 += prod * prod;
                }

                for(grid_iterator<NDim> d(4); d; ++d) {
                    double w1  = w[d.position()];
                    double w2  = w1 * w1;
                    double phi = (*v) * w1 / sum_w2;

                    two_doubles delta_omega = {w2 * phi, w2};

                    append(dw[i + (*d)], delta_omega);
                }
            }

            phi.insert(//boost::container::ordered_unique_range,
                    boost::make_transform_iterator(dw.begin(), delta_over_omega),
                    boost::make_transform_iterator(dw.end(),   delta_over_omega)
                    );
        }

        double operator()(const point &p) const {
            index i;
            point s;

            for(unsigned d = 0; d < NDim; ++d) {
                double u = (p[d] - cmin[d]) * hinv[d];
                i[d] = floor(u) - 1;
                s[d] = u - floor(u);
            }

            double f = 0;

            for(grid_iterator<NDim> d(4); d; ++d) {
                double w = 1.0;
                for(unsigned k = 0; k < NDim; ++k) w *= Bspline(d[k], s[k]);

                f += w * get_phi(i + (*d));
            }

            return f;
        }

        void report(std::ostream &os) const {
            boost::io::ios_all_saver stream_state(os);

            size_t grid_size = grid[0];

            os << "sparse [" << grid[0];
            for(unsigned i = 1; i < NDim; ++i) {
                os << ", " << grid[i];
                grid_size *= grid[i];
            }

            size_t bytes = phi.size() * sizeof(std::pair<index, double>);
            size_t dense_bytes = grid_size * sizeof(double);

            double compression = static_cast<double>(bytes) / dense_bytes;
            os << "] (" << bytes << " bytes, compression: "
                << std::fixed << std::setprecision(2) << compression << ")";
        }
    private:
        point cmin, cmax, hinv;
        index grid;

        typedef boost::container::flat_map<index, double> sparse_grid;
        sparse_grid phi;

        typedef boost::array<double, 2> two_doubles;

        static std::pair<index, double> delta_over_omega(const std::pair<index, two_doubles> &dw) {
            return std::make_pair(dw.first, safe_divide(dw.second[0], dw.second[1]));
        }

        static void append(two_doubles &a, const two_doubles &b) {
            std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<double>());
        }

        double get_phi(const index &i) const {
            typename sparse_grid::const_iterator c = phi.find(i);
            return c == phi.end() ? 0.0 : c->second;
        }
};

} // namespace detail

template <unsigned NDim>
class linear_approximation {
    public:
        typedef typename detail::control_lattice<NDim>::point point;

        template <class CooIter, class ValIter>
        linear_approximation(CooIter coo_begin, CooIter coo_end, ValIter val_begin)
        {
            namespace ublas = boost::numeric::ublas;

            size_t n = std::distance(coo_begin, coo_end);

            if (n <= NDim) {
                // Not enough points to get a unique plane
                std::fill(C.begin(), C.end(), 0.0);
                C[NDim] = std::accumulate(val_begin, val_begin + n, 0.0) / n;
                return;
            }

            ublas::matrix<double> A(NDim+1, NDim+1); A.clear();
            ublas::vector<double> f(NDim+1);         f.clear();

            CooIter p = coo_begin;
            ValIter v = val_begin;

            double sum_val = 0.0;

            // Solve least-squares problem to get approximation with a plane.
            for(; p != coo_end; ++p, ++v, ++n) {
                boost::array<double, NDim+1> x;
                std::copy(p->begin(), p->end(), boost::begin(x));
                x[NDim] = 1.0;

                for(unsigned i = 0; i <= NDim; ++i) {
                    for(unsigned j = 0; j <= NDim; ++j) {
                        A(i,j) += x[i] * x[j];
                    }
                    f(i) += x[i] * (*v);
                }

                sum_val += (*v);
            }

            ublas::permutation_matrix<size_t> pm(NDim+1);
            ublas::lu_factorize(A, pm);

            bool singular = false;
            for(unsigned i = 0; i <= NDim; ++i) {
                if (A(i,i) == 0.0) {
                    singular = true;
                    break;
                }
            }

            if (singular) {
                std::fill(C.begin(), C.end(), 0.0);
                C[NDim] = sum_val / n;
            } else {
                ublas::lu_substitute(A, pm, f);
                for(unsigned i = 0; i <= NDim; ++i) C[i] = f(i);
            }
        }

        double operator()(const point &p) const {
            double f = C[NDim];

            for(unsigned i = 0; i < NDim; ++i)
                f += C[i] * p[i];

            return f;
        }
    private:
        boost::array<double, NDim+1> C;
};

template <unsigned NDim>
class MBA {
    public:
        typedef boost::array<size_t, NDim> index;
        typedef boost::array<double, NDim> point;

        template <class CooIter, class ValIter>
        MBA(
                const point &coo_min, const point &coo_max, index grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels = 8, double tol = 1e-8, double min_fill = 0.5,
                boost::function<double(point)> initial = boost::function<double(point)>()
           )
        {
            init(
                    coo_min, coo_max, grid,
                    coo_begin, coo_end, val_begin,
                    max_levels, tol, min_fill, initial
                );
        }

        template <class CooRange, class ValRange>
        MBA(
                const point &coo_min, const point &coo_max, index grid,
                CooRange coo, ValRange val,
                unsigned max_levels = 8, double tol = 1e-8, double min_fill = 0.5,
                boost::function<double(point)> initial = boost::function<double(point)>()
           )
        {
            init(
                    coo_min, coo_max, grid,
                    boost::begin(coo), boost::end(coo), boost::begin(val),
                    max_levels, tol, min_fill, initial
                );
        }

        double operator()(const point &p) const {
            double f = 0.0;

            BOOST_FOREACH(const boost::shared_ptr<lattice> &psi, cl) {
                f += (*psi)(p);
            }

            return f;
        }

        friend std::ostream& operator<<(std::ostream &os, const MBA &h) {
            size_t level = 0;
            BOOST_FOREACH(const boost::shared_ptr<lattice> &psi, h.cl) {
                os << "level " << ++level << ": ";
                psi->report(os);
                os << std::endl;
            }
            return os;
        }

    private:
        typedef detail::control_lattice<NDim>        lattice;
        typedef detail::initial_approximation<NDim>  initial_approximation;
        typedef detail::control_lattice_dense<NDim>  dense_lattice;
        typedef detail::control_lattice_sparse<NDim> sparse_lattice;


        std::list< boost::shared_ptr<lattice> > cl;

        template <class CooIter, class ValIter>
        void init(
                const point &cmin, const point &cmax, index grid,
                CooIter coo_begin, CooIter coo_end, ValIter val_begin,
                unsigned max_levels, double tol, double min_fill,
                boost::function<double(point)> initial
                )
        {
            using namespace mba::detail;

            const ptrdiff_t n = std::distance(coo_begin, coo_end);
            std::vector<double> val(val_begin, val_begin + n);

            double res, eps = 0.0;
            for(ptrdiff_t i = 0; i < n; ++i)
                eps = std::max(eps, std::abs(val[i]));
            eps *= tol;

            if (initial) {
                // Start with the given approximation.
                cl.push_back(boost::make_shared<initial_approximation>(initial));
                res = cl.back()->residual(coo_begin, coo_end, val.begin());
                if (res <= eps) return;
            }

            size_t lev = 1;
            // Create dense head of the hierarchy.
            {
                boost::shared_ptr<dense_lattice> psi = boost::make_shared<dense_lattice>(
                        cmin, cmax, grid, coo_begin, coo_end, val.begin());

                res = psi->residual(coo_begin, coo_end, val.begin());
                double fill = psi->fill_ratio();

                for(; (lev < max_levels) && (res > eps) && (fill > min_fill); ++lev) {
                    grid = grid * 2ul - 1ul;

                    boost::shared_ptr<dense_lattice> f = boost::make_shared<dense_lattice>(
                            cmin, cmax, grid, coo_begin, coo_end, val.begin());

                    res = f->residual(coo_begin, coo_end, val.begin());
                    fill = f->fill_ratio();

                    f->append_refined(*psi);
                    psi.swap(f);
                }

                cl.push_back(psi);
            }

            // Create sparse tail of the hierrchy.
            for(; (lev < max_levels) && (res > eps); ++lev) {
                grid = grid * 2ul - 1ul;

                cl.push_back(boost::make_shared<sparse_lattice>(
                        cmin, cmax, grid, coo_begin, coo_end, val.begin()));

                res = cl.back()->residual(coo_begin, coo_end, val.begin());
            }
        }
};

} // namespace mba

#endif
