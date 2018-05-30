#ifndef AMGCL_MPI_COARSENING_CONN_STRENGTH_HPP
#define AMGCL_MPI_COARSENING_CONN_STRENGTH_HPP

/*
The MIT License

Copyright (c) 2012-2018 Denis Demidov <dennis.demidov@gmail.com>

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
 * \file   amgcl/mpi/coarsening/conn_strength.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Strength of connection for the distributed_matrix.
 */

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend>
boost::shared_ptr< distributed_matrix<Backend> >
squared_interface(const distributed_matrix<Backend> &A) {
    typedef typename Backend::value_type value_type;
    typedef backend::crs<value_type> build_matrix;
    const comm_pattern<Backend> &C = A.cpat();

    build_matrix &A_loc = *A.local();
    build_matrix &A_rem = *A.remote();

    ptrdiff_t A_rows = A.loc_rows();

    ptrdiff_t A_beg = A.loc_col_shift();
    ptrdiff_t A_end = A_beg + A_rows;

    boost::shared_ptr<build_matrix> a_nbr = remote_rows(C, A);
    build_matrix &A_nbr = *a_nbr;

    // Build mapping from global to local column numbers in the remote part of
    // the square matrix.
    std::vector<ptrdiff_t> rem_cols(A_rem.nnz + A_nbr.nnz);

    std::copy(A_nbr.col, A_nbr.col + A_nbr.nnz,
            std::copy(A_rem.col, A_rem.col + A_rem.nnz, rem_cols.begin()));

    std::sort(rem_cols.begin(), rem_cols.end());
    rem_cols.erase(std::unique(rem_cols.begin(), rem_cols.end()), rem_cols.end());

    ptrdiff_t n_rem_cols = 0;
    boost::unordered_map<ptrdiff_t, int> rem_idx(2 * rem_cols.size());
    BOOST_FOREACH(ptrdiff_t c, rem_cols) {
        if (c >= A_beg && c < A_end) continue;
        rem_idx[c] = n_rem_cols++;
    }

    // Build the product.
    boost::shared_ptr<build_matrix> s_loc = boost::make_shared<build_matrix>();
    boost::shared_ptr<build_matrix> s_rem = boost::make_shared<build_matrix>();

    build_matrix &S_loc = *s_loc;
    build_matrix &S_rem = *s_rem;

    S_loc.set_size(A_rows, A_rows, false);
    S_rem.set_size(A_rows, 0,      false);

    S_loc.ptr[0] = 0;
    S_rem.ptr[0] = 0;

    AMGCL_TIC("analyze");
#pragma omp parallel
    {
        std::vector<ptrdiff_t> loc_marker(A_rows,     -1);
        std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

#pragma omp for
        for(ptrdiff_t ia = 0; ia < A_rows; ++ia) {
            ptrdiff_t loc_cols = 0;
            ptrdiff_t rem_cols = 0;

            for(ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t  ca = C.local_index(A_rem.col[ja]);

                for(ptrdiff_t jb = A_nbr.ptr[ca], eb = A_nbr.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  cb = A_nbr.col[jb];

                    if (cb >= A_beg && cb < A_end) {
                        cb -= A_beg;

                        if (loc_marker[cb] != ia) {
                            loc_marker[cb]  = ia;
                            ++loc_cols;
                        }
                    } else {
                        cb = rem_idx[cb];

                        if (rem_marker[cb] != ia) {
                            rem_marker[cb]  = ia;
                            ++rem_cols;
                        }
                    }
                }
            }

            for(ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t  ca = A_loc.col[ja];

                for(ptrdiff_t jb = A_rem.ptr[ca], eb = A_rem.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  cb = rem_idx[A_rem.col[jb]];

                    if (rem_marker[cb] != ia) {
                        rem_marker[cb]  = ia;
                        ++rem_cols;
                    }
                }

            }

            if (rem_cols) {
                for(ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
                    ptrdiff_t  ca = A_loc.col[ja];

                    for(ptrdiff_t jb = A_loc.ptr[ca], eb = A_loc.ptr[ca+1]; jb < eb; ++jb) {
                        ptrdiff_t  cb = A_loc.col[jb];

                        if (loc_marker[cb] != ia) {
                            loc_marker[cb]  = ia;
                            ++loc_cols;
                        }
                    }

                }
            }

            S_rem.ptr[ia + 1] = rem_cols;
            S_loc.ptr[ia + 1] = rem_cols ? loc_cols : 0;
        }
    }
    AMGCL_TOC("analyze");

    S_loc.set_nonzeros(S_loc.scan_row_sizes(), false);
    S_rem.set_nonzeros(S_rem.scan_row_sizes(), false);

    AMGCL_TIC("compute");
#pragma omp parallel
    {
        std::vector<ptrdiff_t> loc_marker(A_rows,     -1);
        std::vector<ptrdiff_t> rem_marker(n_rem_cols, -1);

#pragma omp for
        for(ptrdiff_t ia = 0; ia < A_rows; ++ia) {
            ptrdiff_t loc_beg = S_loc.ptr[ia];
            ptrdiff_t rem_beg = S_rem.ptr[ia];
            ptrdiff_t loc_end = loc_beg;
            ptrdiff_t rem_end = rem_beg;

            if (rem_beg == S_rem.ptr[ia+1]) continue;

            for(ptrdiff_t ja = A_loc.ptr[ia], ea = A_loc.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t  ca = A_loc.col[ja];

                for(ptrdiff_t jb = A_loc.ptr[ca], eb = A_loc.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  cb = A_loc.col[jb];

                    if (loc_marker[cb] < loc_beg) {
                        loc_marker[cb] = loc_end;
                        S_loc.col[loc_end] = cb;
                        ++loc_end;
                    }
                }

                for(ptrdiff_t jb = A_rem.ptr[ca], eb = A_rem.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  gb = A_rem.col[jb];
                    ptrdiff_t  cb = rem_idx[gb];

                    if (rem_marker[cb] < rem_beg) {
                        rem_marker[cb] = rem_end;
                        S_rem.col[rem_end] = gb;
                        ++rem_end;
                    }
                }
            }

            for(ptrdiff_t ja = A_rem.ptr[ia], ea = A_rem.ptr[ia + 1]; ja < ea; ++ja) {
                ptrdiff_t  ca = C.local_index(A_rem.col[ja]);

                for(ptrdiff_t jb = A_nbr.ptr[ca], eb = A_nbr.ptr[ca+1]; jb < eb; ++jb) {
                    ptrdiff_t  gb = A_nbr.col[jb];

                    if (gb >= A_beg && gb < A_end) {
                        ptrdiff_t cb = gb - A_beg;

                        if (loc_marker[cb] < loc_beg) {
                            loc_marker[cb] = loc_end;
                            S_loc.col[loc_end] = cb;
                            ++loc_end;
                        }
                    } else {
                        ptrdiff_t cb = rem_idx[gb];

                        if (rem_marker[cb] < rem_beg) {
                            rem_marker[cb] = rem_end;
                            S_rem.col[rem_end] = gb;
                            ++rem_end;
                        }
                    }
                }
            }
        }
    }
    AMGCL_TOC("compute");

    return boost::make_shared<distributed_matrix<Backend> >(
            A.comm(), s_loc, s_rem);
}

template <class Backend>
boost::shared_ptr< distributed_matrix< backend::builtin<char> > >
conn_strength(const distributed_matrix<Backend> &A, float eps_strong) {
    typedef typename Backend::value_type value_type;
    typedef typename math::scalar_of<value_type>::type scalar_type;
    typedef backend::crs<value_type> build_matrix;
    typedef backend::crs<char> bool_matrix;

    AMGCL_TIC("conn_strength");
    ptrdiff_t n = A.loc_rows();

    const build_matrix &A_loc = *A.local();
    const build_matrix &A_rem = *A.remote();
    const comm_pattern<Backend> &C = A.cpat();

    scalar_type eps_squared = eps_strong * eps_strong;

    boost::shared_ptr< backend::numa_vector<value_type> > d = backend::diagonal(A_loc);
    backend::numa_vector<value_type> &D = *d;

    std::vector<value_type> D_loc(C.send.count());
    std::vector<value_type> D_rem(C.recv.count());

    for(size_t i = 0, nv = C.send.count(); i < nv; ++i)
        D_loc[i] = D[C.send.col[i]];

    C.exchange(&D_loc[0], &D_rem[0]);

    boost::shared_ptr<bool_matrix> s_loc = boost::make_shared<bool_matrix>();
    boost::shared_ptr<bool_matrix> s_rem = boost::make_shared<bool_matrix>();

    bool_matrix &S_loc = *s_loc;
    bool_matrix &S_rem = *s_rem;

    S_loc.set_size(n, n, true);
    S_rem.set_size(n, 0, true);

    S_loc.val = new char[A_loc.nnz];
    S_rem.val = new char[A_rem.nnz];

#pragma omp parallel for
    for(ptrdiff_t i = 0; i < n; ++i) {
        value_type eps_dia_i = eps_squared * D[i];

        for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
            ptrdiff_t  c = A_loc.col[j];
            value_type v = A_loc.val[j];

            if ((S_loc.val[j] = (c == i || (eps_dia_i * D[c] < v * v))))
                ++S_loc.ptr[i + 1];
        }

        for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
            ptrdiff_t  c = C.local_index(A_rem.col[j]);
            value_type v = A_rem.val[j];

            if ((S_rem.val[j] = (eps_dia_i * D_rem[c] < v * v)))
                ++S_rem.ptr[i + 1];
        }
    }

    S_loc.nnz = S_loc.scan_row_sizes();
    S_rem.nnz = S_rem.scan_row_sizes();

    S_loc.col = new ptrdiff_t[S_loc.nnz];
    S_rem.col = new ptrdiff_t[S_rem.nnz];

#pragma omp parallel for
    for(ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t loc_head = S_loc.ptr[i];
        ptrdiff_t rem_head = S_rem.ptr[i];

        for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j)
            if (S_loc.val[j]) S_loc.col[loc_head++] = A_loc.col[j];

        for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j)
            if (S_rem.val[j]) S_rem.col[rem_head++] = A_rem.col[j];
    }
    AMGCL_TOC("conn_strength");

    return boost::make_shared< distributed_matrix< backend::builtin<char> > >(
            A.comm(), s_loc, s_rem);
}

} // namespace coarsening
} // namespace mpi
} // namespace amgcl


#endif
