#ifndef AMGCL_MPI_REPARTITION_DUMMY_HPP
#define AMGCL_MPI_REPARTITION_DUMMY_HPP

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
 * \file   amgcl/mpi/repartition/dummy.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Dummy repartitioner (merges consecutive domains together).
 */

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <amgcl/backend/interface.hpp>
#include <amgcl/value_type/interface.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace repartition {

template <class Backend>
struct dummy {
    typedef typename Backend::value_type value_type;
    typedef distributed_matrix<Backend>  matrix;

    struct params {
        bool      enable;
        ptrdiff_t min_per_proc;
        int       shrink_ratio;

        params() :
            enable(false), min_per_proc(10000), shrink_ratio(8)
        {}

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, enable),
              AMGCL_PARAMS_IMPORT_VALUE(p, min_per_proc),
              AMGCL_PARAMS_IMPORT_VALUE(p, shrink_ratio)
        {
            AMGCL_PARAMS_CHECK(p, (enable)(min_per_proc)(shrink_ratio));
        }

        void get(
                boost::property_tree::ptree &p,
                const std::string &path = ""
                ) const
        {
            AMGCL_PARAMS_EXPORT_VALUE(p, path, enable);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, min_per_proc);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, shrink_ratio);
        }
    } prm;

    dummy(const params &prm = params()) : prm(prm) {}

    bool is_needed(const matrix &A) const {
        if (!prm.enable) return false;

        communicator comm = A.comm();
        ptrdiff_t n = A.loc_rows();
        std::vector<ptrdiff_t> row_dom = mpi::exclusive_sum(comm, n);

        int non_empty = 0;
        ptrdiff_t min_n = std::numeric_limits<ptrdiff_t>::max();
        for(int i = 0; i < comm.size; ++i) {
            ptrdiff_t m = row_dom[i+1] - row_dom[i];
            if (m) {
                min_n = std::min(min_n, m);
                ++non_empty;
            }
        }

        return (non_empty > 1) && (min_n <= prm.min_per_proc);
    }

    boost::shared_ptr<matrix> operator()(const matrix &A) const {
        typedef backend::crs<value_type> build_matrix;

        communicator comm = A.comm();
        ptrdiff_t nrows = A.loc_rows();

        std::vector<ptrdiff_t> row_dom = mpi::exclusive_sum(comm, nrows);
        std::vector<ptrdiff_t> col_dom(comm.size + 1);

        for(int i = 0; i <= comm.size; ++i)
            col_dom[i] = row_dom[std::min<int>(i * prm.shrink_ratio, comm.size)];

        int old_domains = 0;
        int new_domains = 0;

        for(int i = 0; i < comm.size; ++i) {
            if (row_dom[i+1] > row_dom[i]) ++old_domains;
            if (col_dom[i+1] > col_dom[i]) ++new_domains;
        }

        if (comm.rank == 0) {
            std::cout << "Repartitioning " << old_domains << " -> " << new_domains << std::endl;
        }

        ptrdiff_t row_beg = row_dom[comm.rank];
        ptrdiff_t col_beg = col_dom[comm.rank];
        ptrdiff_t col_end = col_dom[comm.rank + 1];
        ptrdiff_t ncols   = col_end - col_beg;

        boost::shared_ptr<build_matrix> i_loc = boost::make_shared<build_matrix>();
        boost::shared_ptr<build_matrix> i_rem = boost::make_shared<build_matrix>();

        build_matrix &I_loc = *i_loc;
        build_matrix &I_rem = *i_rem;

        I_loc.set_size(nrows, ncols, false);
        I_rem.set_size(nrows, 0, false);

        I_loc.ptr[0] = 0;
        I_rem.ptr[0] = 0;

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < nrows; ++i) {
            ptrdiff_t j = i + row_beg;
            if (col_beg <= j && j < col_end) {
                I_loc.ptr[i+1] = 1;
                I_rem.ptr[i+1] = 0;
            } else {
                I_loc.ptr[i+1] = 0;
                I_rem.ptr[i+1] = 1;
            }
        }

        I_loc.set_nonzeros(I_loc.scan_row_sizes());
        I_rem.set_nonzeros(I_rem.scan_row_sizes());

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < nrows; ++i) {
            ptrdiff_t j = i + row_beg;
            if (col_beg <= j && j < col_end) {
                ptrdiff_t k = I_loc.ptr[i];
                I_loc.col[k] = j - col_beg;
                I_loc.val[k] = math::identity<value_type>();
            } else {
                ptrdiff_t k = I_rem.ptr[i];
                I_rem.col[k] = j;
                I_rem.val[k] = math::identity<value_type>();
            }
        }

        return boost::make_shared<matrix>(comm, i_loc, i_rem, A.backend_prm());
    }

};


} // namespace repartition
} // namespace mpi
} // namespace amgcl

#endif
