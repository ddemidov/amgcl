In this section we test performance of the library on a shared memory system.
We also compare the results with PETSC_ and `Trilinos ML`_ distributed memory
libraries and CUSP_ GPGPU library.  The tests were performed on a dual socket
system with two Intel Xeon E5-2640 v3 CPUs. The system also had an NVIDIA Tesla
K80 GPU installed, which was used for testing the GPU based versions.

3D Poisson problem
^^^^^^^^^^^^^^^^^^

The Poisson problem is dicretized with the finite difference method on a
uniform mesh, and the resulting linear system contained 3375000 unknowns and
23490000 nonzeros.

The figure below presents the multicore scalability of the problem. Here
AMGCL uses the ``builtin`` OpenMP backend, while PETSC and Trilinos use MPI for
parallelization. We also show results for the CUDA backend of AMGCL library
compared with the CUSP library. All libraries use the Conjugate Gradient
iterative solver preconditioned with a smoothed aggregation AMG. Trilinos and
PETSC use default options for smoothers (symmetric Gauss-Seidel and damped
Jacobi accordingly) on each level of the hierarchy, AMGCL uses SPAI0, and CUSP
uses Gauss-Seidel smoother.

.. plot::

    from pylab import *
    rc('font', size=12)

    names = dict(
        total='Total time',
        setup='Setup time',
        solve='Solve time',
        iters='Iterations'
        )

    handles = []

    figure(figsize=(10,8))
    gs = GridSpec(2,2)

    def set_ticks(ax, t):
        ax.set_xticks(t)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)

    for test in ('amgcl', 'petsc', 'trilinos'):
        data = loadtxt('smem_data/poisson/{}.txt'.format(test), delimiter=' ', dtype=dict(
            names=('np', 'size', 'iters', 'setup', 'solve'),
            formats=('i4', 'i4', 'i4', 'f8', 'f8'))
        )

        np = data['np']
        cores = unique(sorted(data['np']))
        setup = empty(cores.shape)
        solve = empty(cores.shape)
        iters = empty(cores.shape)

        for i,n in enumerate(cores):
            setup[i] = median(data[np==n]['setup'])
            solve[i] = median(data[np==n]['solve'])
            iters[i] = median(data[np==n]['iters'])

        total = setup + solve

        for r,dset in enumerate(('total', 'setup', 'solve', 'iters')):
            subplot(gs[r])

            if dset == 'iters':
                h = semilogx(cores, eval(dset), marker='o')
                ylim([5,25])
                yticks([5,10,15,20,25])
            else:
                h = loglog(cores, eval(dset), marker='o')
                #ylim([1e0, 1e2])
            set_ticks(gca(), [1, 2, 4, 8, 16])
            ylabel(names[dset])

            if r == 0: handles.append(h[0])
            if r >= 2: xlabel('Cores/MPI processes')


    for test in ('amgcl-cuda', 'cusp'):
        data = loadtxt('smem_data/poisson/{}.txt'.format(test), delimiter=' ', dtype=dict(
            names=('np', 'size', 'iters', 'setup', 'solve'),
            formats=('i4', 'i4', 'i4', 'f8', 'f8')))

        total = ones_like(cores) * median(data['setup'] + data['solve'])
        setup = ones_like(cores) * median(data['setup'])
        solve = ones_like(cores) * median(data['solve'])
        iters = ones_like(cores) * median(data['iters'])

        for r,dset in enumerate(('total', 'setup', 'solve', 'iters')):
            subplot(gs[r])
            h = plot(cores, eval(dset), '--')
            if r == 0: handles.append(h[0])

    tight_layout()

    figlegend(handles, ['AMGCL', 'PETSC', 'Trilinos', 'AMGCL/CUDA', 'CUSP'],
            ncol=5, loc='lower center')
    gcf().suptitle('3D Poisson problem')
    gcf().subplots_adjust(top=0.93, bottom=0.17)

    show()

The CPU-based results show that AMGCL performs on par with Trilinos, and both
of the libraries outperform PETSC by a large margin. Also, AMGCL is able to
setup the solver about 20–100% faster than Trilinos, and 4–7 times faster than
PETSC. This is probably due to the fact that both Trilinos and PETSC target
distributed memory machines and hence need to do some complicated bookkeeping
under the hood.  PETSC shows better scalability than both Trilinos and AMGCL,
which scale in a similar fashion.

On the GPU, AMGCL performs slightly better than CUSP. If we consider the
solution time (without setup), then both libraries are able to outperform
CPU-based versions by a factor of 3-4. The total solution time of AMGCL with
CUDA backend is only 30% better than that of either AMGCL with OpenMP backend
or Trilinos ML. This is due to the fact that the setup step in AMGCL is always
performed on the CPU and in case of the CUDA backend has an additional overhead
of moving the constructed hierarchy into the GPU memory.

3D Navier-Stokes problem
^^^^^^^^^^^^^^^^^^^^^^^^

The system matrix resulting from the problem discretization has block structure
with blocks of 4-by-4 elements, and contains 713456 unknowns and 41277920
nonzeros. 

There are at least two ways to solve the system. First, one can treat the
system as a monolythic one, and provide some minimal help to the preconditioner
in form of near null space vectors. Second option is to employ the knowledge
about the problem structure, and to combine separate preconditioners for
individual fields (in this particular case, for pressure and velocity). In case
of AMGCL both options were tested, where the monolythic system was solved with
static 4x4 matrices as value type, and the field-split approach was implemented
using the ``schur_pressure_correction`` preconditioner.  Trilinos ML only
provides the first option; PETSC implement both options, but we only show
results for the second, superior option here. CUSP library does not provide
field-split preconditioner and does not allow to specify near null space
vectors, so it was not tested for this problem.

The figure below shows multicore scalability results for the Navier-Stokes
problem.  Lines labelled with 'block' correspond to the cases when the problem
is treated as a monolythic system, and 'split' results correspond to the
field-split approach.

.. plot::

    from pylab import *
    rc('font', size=12)

    dset_names = dict(
        total='Total time',
        setup='Setup time',
        solve='Solve time',
        iters='Iterations'
        )

    handles = []

    figure(figsize=(10,8))
    gs = GridSpec(2,2)

    def set_ticks(ax, t):
        ax.set_xticks(t)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)

    for test in ('amgcl', 'amgcl-schur', 'petsc', 'trilinos'):
        data = loadtxt('smem_data/nstokes/{}.txt'.format(test), delimiter=' ', dtype=dict(
            names=('np', 'size', 'iters', 'setup', 'solve'),
            formats=('i4', 'i4', 'i4', 'f8', 'f8'))
        )

        np = data['np']
        cores = unique(sorted(data['np']))
        setup = empty(cores.shape)
        solve = empty(cores.shape)
        iters = empty(cores.shape)

        for i,n in enumerate(cores):
            setup[i] = median(data[np==n]['setup'])
            solve[i] = median(data[np==n]['solve'])
            iters[i] = median(data[np==n]['iters'])

        total = setup + solve

        for r,dset in enumerate(('total', 'setup', 'solve', 'iters')):
            subplot(gs[r])

            if dset == 'iters':
                h = semilogx(cores, eval(dset), marker='o')
            else:
                h = loglog(cores, eval(dset), marker='o')
            set_ticks(gca(), [1, 2, 4, 8, 16])
            ylabel(dset_names[dset])

            if r == 0: handles.append(h[0])
            if r >= 2: xlabel('Cores/MPI processes')


    for test in ('amgcl-vexcl-cuda', 'amgcl-schur-cuda'):
        data = loadtxt('smem_data/nstokes/{}.txt'.format(test), delimiter=' ', dtype=dict(
            names=('np', 'size', 'iters', 'setup', 'solve'),
            formats=('i4', 'i4', 'i4', 'f8', 'f8')))

        total = ones_like(cores) * median(data['setup'] + data['solve'])
        setup = ones_like(cores) * median(data['setup'])
        solve = ones_like(cores) * median(data['solve'])
        iters = ones_like(cores) * median(data['iters'])

        for r,dset in enumerate(('total', 'setup', 'solve', 'iters')):
            subplot(gs[r])
            h = plot(cores, eval(dset), '--')
            if r == 0: handles.append(h[0])

    tight_layout()

    figlegend(handles, [
        'AMGCL (block)', 'AMGCL (split)', 'PETSC (split)', 'Trilinos (block)',
        'AMGCL (block, VexCL)', 'AMGCL (split, CUDA)'
        ],
        ncol=3, loc='lower center')
    gcf().suptitle('3D Navier-Stokes problem')
    gcf().subplots_adjust(top=0.93, bottom=0.15)

    show()

.. _PETSC: https://www.mcs.anl.gov/petsc/
.. _`Trilinos ML`: https://trilinos.org/packages/ml/
.. _CUSP: https://github.com/cusplibrary/cusplibrary
