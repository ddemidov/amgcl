#ifndef AMGCL_COARSENING_SMOOTHED_AGGREGATION_HPP
#define AMGCL_COARSENING_SMOOTHED_AGGREGATION_HPP

/*
The MIT License

Copyright (c) 2012-2014 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/coarsening/smoothed_aggregation.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Smoothed aggregation coarsening scheme.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/numeric.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/coarsening/detail/tentative.hpp>
#include <amgcl/util.hpp>

namespace amgcl {
namespace coarsening {

namespace detail {

template <class Base>
struct filtered_matrix {
    typedef typename backend::value_type<Base>::type value_type;

    typedef value_type val_type;
    typedef ptrdiff_t  col_type;
    typedef ptrdiff_t  ptr_type;

    const Base &base;
    float omega;
    const std::vector<char> &strong;

    class row_iterator {
        public:
            row_iterator(
                    const col_type * col,
                    const col_type * end,
                    const val_type * val,
                    const char     * str,
                    col_type row,
                    float    omega,
                    val_type dia
                    )
                : m_col(col), m_end(end), m_val(val), m_str(str),
                  m_row(row), m_omega(omega), m_dia(dia)
            {}

            operator bool() const {
                return m_col < m_end;
            }

            row_iterator& operator++() {
                do {
                    ++m_col;
                    ++m_val;
                    ++m_str;
                } while(m_col < m_end && !(*m_str));

                return *this;
            }

            col_type col() const {
                return *m_col;
            }

            val_type value() const {
                if (m_col[0] == m_dia)
                    return 1 - m_omega;
                else
                    return -m_omega * m_dia * m_val[0];
            }

        private:
            const col_type * m_col;
            const col_type * m_end;
            const val_type * m_val;
            const char     * m_str;

            col_type m_row;
            float    m_omega;
            val_type m_dia;
    };

    filtered_matrix(
            const Base &base,
            float omega,
            const std::vector<char> &strong
            ) : base(base), omega(omega), strong(strong)
    {}

    size_t rows() const {
        return backend::rows(base);
    }

    row_iterator row_begin(size_t row) const {
        ptr_type b = base.ptr[row];
        ptr_type e = base.ptr[row + 1];

        const col_type *col = &base.col[b], *c = col;
        const col_type *end = &base.col[e];
        const val_type *val = &base.val[b], *v = val;
        const char     *str = &strong[b],   *s = str;

        // Diagonal of the filtered matrix is the original matrix
        // diagonal minus its weak connections.
        val_type dia = 0;
        for(; c < end; ++c, ++v, ++s) {
            if (static_cast<size_t>(*c) == row)
                dia += *v;
            else if ( !(*s) )
                dia -= *v;
        }

        return row_iterator(col, end, val, str, row, omega, 1 / dia);
    }
};

} // namespace detail
} // namespace coarsening

namespace backend {

template <class Base>
struct rows_impl< coarsening::detail::filtered_matrix<Base> > {
    typedef coarsening::detail::filtered_matrix<Base> Matrix;

    static size_t get(const Matrix &A) {
        return A.rows();
    }
};

template <class Base>
struct row_iterator< coarsening::detail::filtered_matrix<Base> > {
    typedef
        typename coarsening::detail::filtered_matrix<Base>::row_iterator
        type;
};

template <class Base>
struct row_begin_impl< coarsening::detail::filtered_matrix<Base> > {
    typedef coarsening::detail::filtered_matrix<Base> Matrix;

    static typename row_iterator<Matrix>::type
    get(const Matrix &matrix, size_t row) {
        return matrix.row_begin(row);
    }
};

} // namespace backend

namespace coarsening {

/// Smoothed aggregation coarsening.
/**
 * \param Aggregates \ref aggregates formation.
 * \ingroup coarsening
 * \sa \cite Vanek1996
 */
template <class Aggregates>
struct smoothed_aggregation {
    /// Coarsening parameters
    struct params {
        /// Aggregation parameters.
        typename Aggregates::params aggr;

        /// Relaxation factor \f$\omega\f$.
        /**
         * Piecewise constant prolongation \f$\tilde P\f$ from non-smoothed
         * aggregation is improved by a smoothing to get the final prolongation
         * matrix \f$P\f$. Simple Jacobi smoother is used here, giving the
         * prolongation matrix
         * \f[P = \left( I - \omega D^{-1} A^F \right) \tilde P.\f]
         * Here \f$A^F = (a_{ij}^F)\f$ is the filtered matrix given by
         * \f[
         * a_{ij}^F =
         * \begin{cases}
         * a_{ij} \quad \text{if} \; j \in N_i\\
         * 0 \quad \text{otherwise}
         * \end{cases}, \quad \text{if}\; i \neq j,
         * \quad a_{ii}^F = a_{ii} - \sum\limits_{j=1,j\neq i}^n
         * \left(a_{ij} - a_{ij}^F \right),
         * \f]
         * where \f$N_i\f$ is the set of variables, strongly coupled to
         * variable \f$i\f$, and \f$D\f$ denotes the diagonal of \f$A^F\f$.
         */
        float relax;

        /// Number of vectors in problem's null-space.
        int Bcols;

        /// The vectors in problem's null-space.
        /**
         * The 2D matrix B is stored row-wise in a continuous vector. Here row
         * corresponds to a degree of freedom in the original problem, and
         * column corresponds to a vector in the problem's null-space.
         */
        std::vector<double> B;

        params() : relax(0.666f), Bcols(0) { }

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_CHILD(p, aggr)
            , AMGCL_PARAMS_IMPORT_VALUE(p, relax)
            , AMGCL_PARAMS_IMPORT_VALUE(p, Bcols)
        {
            double *b = 0;
            size_t Brows = 0;

            b     = p.get("B",     b);
            Brows = p.get("Brows", Brows);

            if (b) {
                precondition(Bcols > 0,
                        "Error in aggregation parameters: "
                        "B is set, but Bcols is not"
                        );

                precondition(Brows > 0,
                        "Error in aggregation parameters: "
                        "B is set, but Brows is not"
                        );

                B.assign(b, b + Brows * Bcols);
            }
        }
    };

    /// \copydoc amgcl::coarsening::aggregation::transfer_operators
    template <class Matrix>
    static boost::tuple< boost::shared_ptr<Matrix>, boost::shared_ptr<Matrix> >
    transfer_operators(const Matrix &A, params &prm)
    {
        typedef typename backend::value_type<Matrix>::type Val;

        const size_t n = rows(A);

        TIC("aggregates");
        Aggregates aggr(A, prm.aggr);
        prm.aggr.eps_strong *= 0.5;
        TOC("aggregates");

        TIC("interpolation");
        boost::shared_ptr<Matrix> P_tent;

        if (prm.Bcols > 0) {
            precondition(!prm.B.empty(),
                    "Error in aggregation parameters: "
                    "Bcols > 0, but B is empty"
                    );

            boost::tie(P_tent, prm.B) = detail::tentative_prolongation<Matrix>(
                    n, aggr.count, aggr.id, prm.Bcols, prm.B
                    );
        } else {
            P_tent = detail::tentative_prolongation<Matrix>(n, aggr.count, aggr.id);
        }

        boost::shared_ptr<Matrix> P = boost::make_shared<Matrix>();
        *P = product(
                detail::filtered_matrix<Matrix>(A, prm.relax, aggr.strong_connection),
                *P_tent
                );
        TOC("interpolation");

        boost::shared_ptr<Matrix> R = boost::make_shared<Matrix>();
        *R = transpose(*P);

        return boost::make_tuple(P, R);
    }

    /// \copydoc amgcl::coarsening::aggregation::coarse_operator
    template <class Matrix>
    static boost::shared_ptr<Matrix>
    coarse_operator(
            const Matrix &A,
            const Matrix &P,
            const Matrix &R,
            const params&
            )
    {
        return detail::galerkin(A, P, R);
    }
};

} // namespace coarsening
} // namespace amgcl

#endif
