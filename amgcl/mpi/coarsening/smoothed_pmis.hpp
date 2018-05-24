#ifndef AMGCL_MPI_COARSENING_SMOOTHED_PMIS_HPP
#define AMGCL_MPI_COARSENING_SMOOTHED_PMIS_HPP

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
 * \file   amgcl/mpi/coarsening/smoothed_pmis.hpp
 * \author Denis Demidov <dennis.demidov@gmail.com>
 * \brief  Distributed memory smoothed PMIS coarsening scheme.
 */

#include <boost/tuple/tuple.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/foreach.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/util.hpp>
#include <amgcl/coarsening/detail/galerkin.hpp>
#include <amgcl/mpi/util.hpp>
#include <amgcl/mpi/distributed_matrix.hpp>

namespace amgcl {
namespace mpi {
namespace coarsening {

template <class Backend>
struct smoothed_pmis {
    typedef typename Backend::value_type value_type;
    typedef typename math::scalar_of<value_type>::type scalar_type;
    typedef backend::crs<value_type> build_matrix;

    struct params {
        /// Strong connectivity threshold
        scalar_type eps_strong;

        /// Relaxation factor.
        scalar_type relax;

        // Use power iterations to estimate the matrix spectral radius.
        // This usually improves convergence rate and results in faster solves,
        // but costs some time during setup.
        bool estimate_spectral_radius;

        // Number of power iterations to apply for the spectral radius
        // estimation.
        int power_iters;

        params()
            : eps_strong(0.08), relax(1.0f),
              estimate_spectral_radius(false), power_iters(5)
        { }

        params(const boost::property_tree::ptree &p)
            : AMGCL_PARAMS_IMPORT_VALUE(p, eps_strong),
              AMGCL_PARAMS_IMPORT_VALUE(p, relax),
              AMGCL_PARAMS_IMPORT_VALUE(p, estimate_spectral_radius),
              AMGCL_PARAMS_IMPORT_VALUE(p, power_iters)
        {
            AMGCL_PARAMS_CHECK(p, (eps_strong)(relax)(estimate_spectral_radius)(power_iters));
        }

        void get(boost::property_tree::ptree &p, const std::string &path) const {
            AMGCL_PARAMS_EXPORT_VALUE(p, path, eps_strong);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, relax);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, estimate_spectral_radius);
            AMGCL_PARAMS_EXPORT_VALUE(p, path, power_iters);
        }
    } prm;

    smoothed_pmis(const params &prm = params()) : prm(prm) {}

    template <class LM, class RM>
    boost::tuple<
        boost::shared_ptr< distributed_matrix<Backend, LM, RM> >,
        boost::shared_ptr< distributed_matrix<Backend, LM, RM> >
        >
    transfer_operators(const distributed_matrix<Backend, LM, RM> &A) {
        typedef distributed_matrix<Backend, LM, RM> DM;

        communicator comm = A.comm();

        scalar_type eps_squared = prm.eps_strong * prm.eps_strong;
        prm.eps_strong *= 0.5;

        scalar_type omega = prm.relax;
        if (prm.estimate_spectral_radius) {
            omega *= static_cast<scalar_type>(4.0/3) / spectral_radius(A, prm.power_iters);
        } else {
            omega *= static_cast<scalar_type>(2.0/3);
        }

        // 1. Create filtered matrix
        AMGCL_TIC("filtered matrix");
        ptrdiff_t n = A.loc_rows();

        const build_matrix &A_loc = *A.local();
        const build_matrix &A_rem = *A.remote();
        const comm_pattern<Backend> &Ap = A.cpat();

        boost::shared_ptr<build_matrix> af_loc = boost::make_shared<build_matrix>();
        boost::shared_ptr<build_matrix> af_rem = boost::make_shared<build_matrix>();

        build_matrix &Af_loc = *af_loc;
        build_matrix &Af_rem = *af_rem;

        Af_loc.set_size(n, n, true);
        Af_rem.set_size(n, 0, true);

        boost::shared_ptr< backend::numa_vector<value_type> > D = backend::diagonal(A_loc);
        backend::numa_vector<value_type> Df(n, false);

        std::vector<value_type> D_loc(Ap.send.count());
        std::vector<value_type> D_rem(Ap.recv.count());

        for(size_t i = 0, nv = Ap.send.count(); i < nv; ++i)
            D_loc[i] = (*D)[Ap.send.col[i]];

        Ap.exchange(&D_loc[0], &D_rem[0]);

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            value_type dia_i = (*D)[i];
            value_type dia_f = dia_i;
            value_type eps_dia_i = eps_squared * dia_i;

            for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
                ptrdiff_t  c = A_loc.col[j];
                value_type v = A_loc.val[j];

                if (c == i || (eps_dia_i * (*D)[c] < v * v)) {
                    ++Af_loc.ptr[i+1];
                } else {
                    dia_f += v;
                }
            }

            for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
                ptrdiff_t  c = Ap.local_index(A_rem.col[j]);
                value_type v = A_rem.val[j];

                if (eps_dia_i * D_rem[c] < v * v) {
                    ++Af_rem.ptr[i+1];
                } else {
                    dia_f += v;
                }
            }

            Df[i] = dia_f;
        }

        Af_loc.set_nonzeros(Af_loc.scan_row_sizes());
        Af_rem.set_nonzeros(Af_rem.scan_row_sizes());

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            value_type dia_f = -omega * math::inverse(Df[i]);
            value_type eps_dia_i = eps_squared * (*D)[i];
            ptrdiff_t loc_head = Af_loc.ptr[i];
            ptrdiff_t rem_head = Af_rem.ptr[i];

            for(ptrdiff_t j = A_loc.ptr[i], e = A_loc.ptr[i+1]; j < e; ++j) {
                ptrdiff_t  c = A_loc.col[j];
                value_type v = A_loc.val[j];

                if (c == i) {
                    Af_loc.col[loc_head] = c;
                    Af_loc.val[loc_head] = (1 - omega) * math::identity<value_type>();
                    ++loc_head;
                } else if(eps_dia_i * (*D)[c] < v * v) {
                    Af_loc.col[loc_head] = c;
                    Af_loc.val[loc_head] = dia_f * v;
                    ++loc_head;
                }
            }

            for(ptrdiff_t j = A_rem.ptr[i], e = A_rem.ptr[i+1]; j < e; ++j) {
                ptrdiff_t  c = A_rem.col[j];
                ptrdiff_t  k = Ap.local_index(c);
                value_type v = A_rem.val[j];

                if (eps_dia_i * D_rem[k] < v * v) {
                    Af_rem.col[rem_head] = c;
                    Af_rem.val[rem_head] = dia_f * v;
                    ++rem_head;
                }
            }
        }

        boost::shared_ptr<DM> Af = boost::make_shared<DM>(comm, af_loc, af_rem, A.backend_prm());
        AMGCL_TOC("filtered matrix");

        AMGCL_TIC("tentative prolongation");
        boost::shared_ptr<DM> P_tent = tentative_prolongation(*Af);
        AMGCL_TOC("tentative prolongation");

        // 5. Smooth tentative prolongation with the filtered matrix.
        AMGCL_TIC("smoothing");
        boost::shared_ptr<DM> P = product(*Af, *P_tent);
        AMGCL_TOC("smoothing");

        return boost::make_tuple(P, transpose(*P));
    }

    template <class LM, class RM>
    boost::shared_ptr< distributed_matrix<Backend, LM, RM> >
    tentative_prolongation(const distributed_matrix<Backend, LM, RM> &Af) {
        typedef distributed_matrix<Backend, LM, RM> DM;

        static const int tag_exc_cnt = 4001;
        static const int tag_exc_pts = 4002;

        const build_matrix &Af_loc = *Af.local();
        const build_matrix &Af_rem = *Af.remote();

        ptrdiff_t n = Af_loc.nrows;

        communicator comm = Af.comm();

        // 1. Get symbolic square of the filtered matrix.
        AMGCL_TIC("symbolic square");
        boost::shared_ptr<DM> S = symb_product(Af, Af);
        const build_matrix &S_loc = *S->local();
        const build_matrix &S_rem = *S->remote();
        const comm_pattern<Backend> &Sp = S->cpat();
        AMGCL_TOC("symbolic square");

        // 2. Apply PMIS algorithm to the symbolic square.
        AMGCL_TIC("PMIS");
        const ptrdiff_t undone   = -2;
        const ptrdiff_t deleted  = -1;

        ptrdiff_t n_undone = 0;
        std::vector<int>       loc_owner(n, -1);
        std::vector<ptrdiff_t> loc_state(n, undone);
        std::vector<ptrdiff_t> rem_state(Sp.recv.count(), undone);
        std::vector<ptrdiff_t> send_state(Sp.send.count());

        // Remove lonely nodes.
#pragma omp parallel for reduction(+:n_undone)
        for(ptrdiff_t i = 0; i < n; ++i) {
            ptrdiff_t wl = Af_loc.ptr[i+1] - Af_loc.ptr[i];
            ptrdiff_t wr = S_rem.ptr[i+1] - S_rem.ptr[i];

            if (wl + wr == 1) {
                loc_state[i] = deleted;
                ++n_undone;
            } else {
                loc_state[i] = undone;
            }
        }

        n_undone = n - n_undone;

        // Exchange state
        for(ptrdiff_t i = 0, m = Sp.send.count(); i < m; ++i)
            send_state[i] = loc_state[Sp.send.col[i]];
        Sp.exchange(&send_state[0], &rem_state[0]);

        std::vector< std::vector<ptrdiff_t> > send_pts(Sp.recv.nbr.size());
        std::vector<ptrdiff_t> recv_pts;

        std::vector<MPI_Request> send_cnt_req(Sp.recv.nbr.size());
        std::vector<MPI_Request> send_pts_req(Sp.recv.nbr.size());

        ptrdiff_t naggr = 0;

        std::vector<ptrdiff_t> nbr;

        while(true) {
            for(size_t i = 0; i < Sp.recv.nbr.size(); ++i)
                send_pts[i].clear();

            if (n_undone) {
                for(ptrdiff_t i = 0; i < n; ++i) {
                    if (loc_state[i] != undone) continue;

                    if (S_rem.ptr[i+1] > S_rem.ptr[i]) {
                        // Boundary points
                        bool selectable = true;
                        for(ptrdiff_t j = S_rem.ptr[i], e = S_rem.ptr[i+1]; j < e; ++j) {
                            int d,c;
                            boost::tie(d,c) = Sp.remote_info(S_rem.col[j]);

                            if (rem_state[c] == undone && Sp.recv.nbr[d] > comm.rank) {
                                selectable = false;
                                break;
                            }
                        }

                        if (!selectable) continue;

                        ptrdiff_t id = naggr++;
                        loc_owner[i] = comm.rank;
                        loc_state[i] = id;
                        --n_undone;

                        // Af gives immediate neighbors
                        for(ptrdiff_t j = Af_loc.ptr[i], e = Af_loc.ptr[i+1]; j < e; ++j) {
                            ptrdiff_t c = Af_loc.col[j];
                            if (c != i) {
                                if (loc_state[c] == undone) --n_undone;
                                loc_owner[c] = comm.rank;
                                loc_state[c] = id;
                            }
                        }

                        for(ptrdiff_t j = Af_rem.ptr[i], e = Af_rem.ptr[i+1]; j < e; ++j) {
                            ptrdiff_t c = Af_rem.col[j];
                            int d,k;
                            boost::tie(d,k) = Sp.remote_info(c);

                            rem_state[k] = id;
                            send_pts[d].push_back(c);
                            send_pts[d].push_back(id);
                        }

                        // S gives removed neighbors
                        for(ptrdiff_t j = S_loc.ptr[i], e = S_loc.ptr[i+1]; j < e; ++j) {
                            ptrdiff_t c = S_loc.col[j];
                            if (c != i && loc_state[c] == undone) {
                                loc_owner[c] = comm.rank;
                                loc_state[c] = id;
                                --n_undone;
                            }
                        }

                        for(ptrdiff_t j = S_rem.ptr[i], e = S_rem.ptr[i+1]; j < e; ++j) {
                            ptrdiff_t c = S_rem.col[j];
                            int d,k;
                            boost::tie(d,k) = Sp.remote_info(c);

                            if (rem_state[k] == undone) {
                                rem_state[k] = id;
                                send_pts[d].push_back(c);
                                send_pts[d].push_back(id);
                            }
                        }
                    } else {
                        // Inner points
                        ptrdiff_t id = naggr++;
                        loc_owner[i] = comm.rank;
                        loc_state[i] = id;
                        --n_undone;

                        nbr.clear();

                        for(ptrdiff_t j = Af_loc.ptr[i], e = Af_loc.ptr[i+1]; j < e; ++j) {
                            ptrdiff_t c = Af_loc.col[j];
                            nbr.push_back(c);

                            if (c != i) {
                                if (loc_state[c] == undone) --n_undone;
                                loc_owner[c] = comm.rank;
                                loc_state[c] = id;
                            }
                        }

                        BOOST_FOREACH(ptrdiff_t k, nbr) {
                            for(ptrdiff_t j = Af_loc.ptr[k], e = Af_loc.ptr[k+1]; j < e; ++j) {
                                ptrdiff_t c = Af_loc.col[j];
                                if (c != k && loc_state[c] == undone) {
                                    loc_owner[c] = comm.rank;
                                    loc_state[c] = id;
                                    --n_undone;
                                }
                            }
                        }
                    }
                }
            }

            for(size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
                int npts = send_pts[i].size();
                MPI_Isend(&npts, 1, MPI_INT, Sp.recv.nbr[i], tag_exc_cnt, comm, &send_cnt_req[i]);

                if (!npts) continue;
                MPI_Isend(&send_pts[i][0], npts, datatype<ptrdiff_t>(), Sp.recv.nbr[i], tag_exc_pts, comm, &send_pts_req[i]);
            }

            for(size_t i = 0; i < Sp.send.nbr.size(); ++i) {
                int npts;
                MPI_Recv(&npts, 1, MPI_INT, Sp.send.nbr[i], tag_exc_cnt, comm, MPI_STATUS_IGNORE);

                if (!npts) continue;
                recv_pts.resize(npts);
                MPI_Recv(&recv_pts[0], npts, datatype<ptrdiff_t>(), Sp.send.nbr[i], tag_exc_pts, comm, MPI_STATUS_IGNORE);

                for(int k = 0; k < npts; k += 2) {
                    ptrdiff_t c  = recv_pts[k] - Sp.loc_col_shift();
                    ptrdiff_t id = recv_pts[k+1];

                    if (loc_state[c] == undone) --n_undone;

                    loc_owner[c] = Sp.send.nbr[i];
                    loc_state[c] = id;
                }
            }

            for(size_t i = 0; i < Sp.recv.nbr.size(); ++i) {
                int npts = send_pts[i].size();
                MPI_Wait(&send_cnt_req[i], MPI_STATUS_IGNORE);
                if (!npts) continue;
                MPI_Wait(&send_pts_req[i], MPI_STATUS_IGNORE);
            }


            for(ptrdiff_t i = 0, m = Sp.send.count(); i < m; ++i)
                send_state[i] = loc_state[Sp.send.col[i]];
            Sp.exchange(&send_state[0], &rem_state[0]);

            ptrdiff_t glob_undone;
            MPI_Allreduce(&n_undone, &glob_undone, 1, datatype<ptrdiff_t>(), MPI_SUM, comm);

            if (glob_undone == 0) {
                break;
            }
        }
        AMGCL_TOC("PMIS");

        // 3. Form tentative prolongation operator.
        boost::shared_ptr<build_matrix> p_loc = boost::make_shared<build_matrix>();
        boost::shared_ptr<build_matrix> p_rem = boost::make_shared<build_matrix>();

        build_matrix &P_loc = *p_loc;
        build_matrix &P_rem = *p_rem;

        std::vector<ptrdiff_t> aggr_dom = exclusive_sum(comm, naggr);
        P_loc.set_size(n, naggr, true);
        P_rem.set_size(n, 0, true);

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            if (loc_state[i] == deleted) continue;

            if (loc_owner[i] == comm.rank) {
                ++P_loc.ptr[i+1];
            } else {
                ++P_rem.ptr[i+1];
            }
        }

        P_loc.set_nonzeros(P_loc.scan_row_sizes());
        P_rem.set_nonzeros(P_rem.scan_row_sizes());

#pragma omp parallel for
        for(ptrdiff_t i = 0; i < n; ++i) {
            ptrdiff_t s = loc_state[i];
            if (s == deleted) continue;

            int d = loc_owner[i];
            if (d == comm.rank) {
                P_loc.col[P_loc.ptr[i]] = s;
                P_loc.val[P_loc.ptr[i]] = math::identity<value_type>();
            } else {
                P_rem.col[P_rem.ptr[i]] = s + aggr_dom[d];
                P_rem.val[P_rem.ptr[i]] = math::identity<value_type>();
            }
        }

        return boost::make_shared<DM>(comm, p_loc, p_rem, Af.backend_prm());
    }

    template <class LM, class RM>
    boost::shared_ptr< distributed_matrix<Backend, LM, RM> >
    coarse_operator(
            const distributed_matrix<Backend, LM, RM> &A,
            const distributed_matrix<Backend, LM, RM> &P,
            const distributed_matrix<Backend, LM, RM> &R
            ) const
    {
        return amgcl::coarsening::detail::galerkin(A, P, R);
    }
};

} // namespace coarsening
} // namespace mpi
} // namespace amgcl

#endif
