#ifndef AMGCL_MPI_COARSENING_PMIS_HPP
#define AMGCL_MPI_COARSENING_PMIS_HPP

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
 * \file   amgcl/mpi/coarsening/pmis.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed PMIS aggregation.
 */

#include <boost/tuple/tuple.hpp>
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
class pmis {
    public:
        typedef typename Backend::value_type value_type;
        typedef typename math::scalar_of<value_type>::type scalar_type;
        typedef distributed_matrix<Backend> matrix;
        typedef backend::crs<value_type> build_matrix;

        struct params {
            // Strong connectivity threshold
            scalar_type eps_strong;

            // Block size for non-scalar problems.
            unsigned    block_size;

            params() : eps_strong(0.08), block_size(1) { }

            params(const boost::property_tree::ptree &p)
                : AMGCL_PARAMS_IMPORT_VALUE(p, eps_strong),
                  AMGCL_PARAMS_IMPORT_VALUE(p, block_size)
            {
                AMGCL_PARAMS_CHECK(p, (eps_strong)(block_size));
            }

            void get(boost::property_tree::ptree &p, const std::string &path) const {
                AMGCL_PARAMS_EXPORT_VALUE(p, path, eps_strong);
                AMGCL_PARAMS_EXPORT_VALUE(p, path, block_size);
            }
        };

        pmis(const matrix &A, const params &prm = params()) {
            const build_matrix &A_loc = *A.local();
            const build_matrix &A_rem = *A.remote();
        }

        boost::shared_ptr<matrix> tentative_prolongation() const {
        }

        bool loc_strong(ptrdiff_t j) const {
        }

        bool rem_strong(ptrdiff_t j) const {
        }
    private:
};


#endif
