Support for distributed memory systems in AMGCL is implemented using the
subdomin deflation method. Here we demonstrate performance and scalability of
the approach on the example of a Poisson problem and a Navier-Stokes problem in
a three dimensional space. To provide a reference, we compare performance of
the AMGCL library with that of the well-known `Trilinos ML`_ package.  The
benchmarks were run on MareNostrum 4 and PizDaint clusters which we gained
access to via PRACE program (project 2010PA4058). The MareNostrum 4 cluster has
3456 compute nodes, each equipped with two 24 core Intel Xeon Platinum 8160
CPUs, and 96 GB of RAM. The peak performance of the cluster is 6.2 Petaflops.
The PizDaint cluster has 5320 hybrid compute nodes, where each node has one 12
core Intel Xeon E5-2690 v3 CPU with 64 GB RAM and one NVIDIA Tesla P100 GPU
with 16 GB RAM. The peak performance of the PizDaint cluster is 25.3 Petaflops.

3D Poisson problem
^^^^^^^^^^^^^^^^^^

The AMGCL implementation uses a BiCGStab(2) iterative solver preconditioned
with subdomain deflation. Smoothed aggregation AMG is used as the local
preconditioner.  The Trilinos implementation uses CG solver preconditioned with
smoothed aggregation AMG with default settings.

The figure below shows weak scaling of the solution on the MareNostrum 4
cluster. Here the problem size is chosen to be proportional to the number of
CPU cores with about :math:`100^3` unknowns per core. The rows in the figure
from top to bottom show total computation time, time spent on constructing the
preconditioner, solution time, and the number of iterations. The AMGCL library
results are labelled 'OMP=n', where n=1,4,12,24 corresponds to the number of
OpenMP threads controlled by each MPI process. The Trilinos library uses
single-threaded MPI processes. The Trilinos data is only available for up to
768 MPI processes, because the library runs out of memory for larger
configurations. The AMGCL data points for 19200 cores with 'OMP=1' are missing
for the same reason. AMGCL plots in the left and the right columns correspond
to the linear deflation and the constant deflation correspondingly

.. plot::

    from pylab import *
    rc('font', size=12)

    def load_data(fname):
        return loadtxt(fname, dtype={
            'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
            'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
            })

    tri = loadtxt('dmem_data/mn4_trilinos_weak.dat', dtype={
        'names'   : ('mpi', 'size', 'iters', 'setup', 'solve'),
        'formats' : ('i4', 'i8', 'i4', 'f8', 'f8')
        })

    def set_ticks(ax, t):
        ax.set_xticks(t)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)
        setp(ax.get_xticklabels(), fontsize=10)
        setp(ax.get_yticklabels(), fontsize=10)

    figure(figsize=(10,12))
    gs = GridSpec(4,2)
    handles = []

    for k,fname in enumerate(('dmem_data/mn4_linear_weak.dat', 'dmem_data/mn4_const_weak.dat')):
        data = load_data(fname)
        for omp in sorted(unique(data['omp'])):
            if omp == 48: continue

            d = data[data['omp'] == omp]
            c = unique(d['mpi'] * omp)
            m = unique(d['mpi'])

            setup = array([min(d[d['mpi']==i]['setup']) for i in m])
            solve = array([min(d[d['mpi']==i]['solve']) for i in m])
            iters = array([min(d[d['mpi']==i]['iters']) for i in m])
            total = setup + solve

            subplot(gs[0,k])
            h = loglog(c, total, '.-')
            ylim([1e1, 2e2])
            if k == 0: handles.append(h[0])

            subplot(gs[1,k])
            loglog(c, setup, '.-')
            ylim([1e0, 100])

            subplot(gs[2,k])
            loglog(c, solve, '.-')
            ylim([1e1, 2e2])

            subplot(gs[3,k])
            semilogx(c, iters, '.-')
            ylim([0, 200])

        subplot(gs[3,k])
        xlabel('Number of cores (MPI * OMP)')

    for i in range(4):
        for j in range(2):
            set_ticks(subplot(gs[i,j]), [48 * 2**i for i in range(8)] + [19200])

    subplot(gs[0,0])
    h = plot(tri['mpi'], tri['setup'] + tri['solve'], '.-')
    title('Linear deflation')
    ylabel('Total time')

    subplot(gs[0,1])
    title('Constant deflation')

    subplot(gs[1,0])
    plot(tri['mpi'], tri['setup'], '.-')
    ylabel('Setup time')

    subplot(gs[2,0])
    plot(tri['mpi'], tri['solve'], '.-')
    ylabel('Solve time')

    subplot(gs[3,0])
    ylabel('Iterations')
    plot(tri['mpi'], tri['iters'], '.-')

    tight_layout()

    figlegend(handles + h,
           ['OMP={}'.format(i) for i in (1, 4, 12, 24)] + ['Trilinos'],
           ncol=3, loc='lower center')
    gcf().suptitle('Weak scaling of the Poisson problem on the MareNostrum 4 cluster')
    gcf().subplots_adjust(top=0.93, bottom=0.12)

    show()

In the case of ideal scaling the timing plots on this figure would be strictly
horizontal. This is not the case here: instead, we see that AMGCL looses about
6-8% efficiency whenever number of cores doubles. This, however, is much better
than we managed to obtain for the Trilinos library, which looses about 36% on
each step.

If we look at the AMGCL results for the linear deflation alone, we can see that
the ‘OMP=1’ line stops scaling properly at 1536 cores, and ‘OMP=4’ looses
scalability at 6144 cores. We refer to the following table for the explanation.

+-------+---------------------+-------------------------------+------------+
| Cores | Setup               | Solve                         | Iterations |
+-------+-------+-------------+--------+------------+---------+------------+
|       | Total | Factorize E | Total  | RHS for E  | Solve E |            |
+=======+=======+=============+========+============+=========+============+
| - **Linear deflation, OMP=1**                                            |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  3.33 |        0.04 |  49.35 |       0.82 |    0.08 |         76 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  5.12 |        1.09 |  52.13 |       1.83 |    0.80 |         76 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 | 20.39 |       15.42 |  79.23 |      31.81 |    4.30 |         54 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Constant deflation, OMP=1**                                          |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  2.88 |        0.00 |  58.52 |       0.73 |    0.01 |         98 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  3.80 |        0.02 |  74.42 |       2.51 |    0.10 |        118 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  5.31 |        0.24 | 130.76 |      63.52 |    0.52 |         90 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Linear deflation, OMP=4**                                            |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  3.86 |        0.00 |  49.90 |       0.15 |    0.01 |         74 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  6.68 |        0.05 |  64.91 |       0.66 |    0.13 |         96 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  7.36 |        0.76 |  60.74 |       2.87 |    0.79 |         82 |
+-------+-------+-------------+--------+------------+---------+------------+
| 19200 | 59.72 |       51.11 | 105.96 |      30.86 |    9.54 |         84 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Constant deflation, OMP=4**                                          |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  3.97 |        0.00 |  65.11 |       0.30 |    0.00 |        104 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  6.73 |        0.00 |  76.44 |       1.01 |    0.01 |        122 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  7.57 |        0.02 | 100.39 |       4.30 |    0.10 |        148 |
+-------+-------+-------------+--------+------------+---------+------------+
| 19200 | 10.08 |        0.74 | 125.41 |      48.67 |    0.83 |        106 |
+-------+-------+-------------+--------+------------+---------+------------+

The table presents the profiling data for the solution of the Poisson problem
on the MareNostrum 4 cluster. The first two columns show time spent on the
setup of the preconditioner and the solution of the problem; the third column
shows the number of iterations required for convergence. The 'Setup' and the
'Solve' columns are further split into subcolumns detailing time required for
factorization and solution of the coarse system.  It is apparent from the table
that weak scalability is affected by two factors. First, factorization of the
coarse (deflated) matrix starts to dominate the setup phase as the number of
subdomains (or MPI processes) grows, since we use a sparse direct solver for
the coarse problem. Second factor is the solution of the coarse problem, which
in our experiments is dominated by communication; namely, most of the coarse
solve time is spent on gathering the deflated problem right-hand side for
solution on the master MPI process.

The constant deflation scales better since the deflation matrix is four times
smaller than for a corresponding linear deflation case. Hence, the setup time
is not affected that much by factorization of the coarse problem. The
communication bottleneck is still present though, as is apparent from the table
above.

The advantage of the linear deflation is that it results in a better
approximation of the problem on a coarse scale and hence needs less iterations
for convergence and performs slightly better within it’s scalability limits,
but the constant deflation eventually outperforms linear deflation as the scale
grows.

Next figure shows weak scaling of the Poisson problem on the PizDaint cluster.
The problem size here is chosen so that each node owns about :math:`200^3`
unknowns. We only show the results of the AMGCL library on this cluster to
compare performance of the OpenMP and CUDA backends. Intel Xeon E5-2690 v3 CPU
is used with the OpenMP backend, and NVIDIA Tesla P100 GPU is used with the
CUDA backend on each compute node. The scaling behavior is similar to the
MareNostrum 4 cluster.  We can see that the CUDA backend is about 9 times
faster than OpenMP during solution phase and 4 times faster overall. The
discrepancy is explained by the fact that the setup phase in AMGCL is always
performed on the CPU, and in the case of CUDA backend it has the additional
overhead of moving the generated hierarchy into the GPU memory.

.. plot::

    from pylab import *
    rc('font', size=12)

    def load_data(fname):
        return loadtxt(fname, dtype={
            'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
            'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
            })

    def set_ticks(ax, t):
        ax.set_xscale('log')
        ax.set_xticks(t[0::2])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)

    figure(figsize=(10,12))
    gs = GridSpec(4,2)
    handles = []

    for k,fname in (
            (0, 'dmem_data/daint_gpu_linear_weak.dat'),
            (0, 'dmem_data/daint_cpu_linear_weak.dat'),
            (1, 'dmem_data/daint_gpu_const_weak.dat'),
            (1, 'dmem_data/daint_cpu_const_weak.dat'),
            ):
        d = load_data(fname)
        m = unique(d['mpi'])

        setup = array([min(d[d['mpi']==i]['setup']) for i in m])
        solve = array([min(d[d['mpi']==i]['solve']) for i in m])
        iters = array([min(d[d['mpi']==i]['iters']) for i in m])
        total = setup + solve

        ax = subplot(gs[0,k])
        h = loglog(m, total, '.-')
        ylim([1e0,200])
        set_ticks(ax, m)
        if k == 0: handles.append(h[0])

        ax = subplot(gs[1,k])
        loglog(m, setup, '.-')
        ylim([1e0,20])
        set_ticks(ax, m)

        ax = subplot(gs[2,k])
        loglog(m, solve, '.-')
        ylim([1e0,200])
        set_ticks(ax, m)

        ax = subplot(gs[3,k])
        semilogx(m, iters, '.-')
        ylim([40,160])
        set_ticks(ax, m)
        xlabel('Compute nodes')

    subplot(gs[0,0])
    title('Linear deflation')
    ylabel('Total time')

    subplot(gs[0,1])
    title('Constant deflation')

    subplot(gs[1,0])
    ylabel('Setup time')

    subplot(gs[2,0])
    ylabel('Solve time')

    subplot(gs[3,0])
    ylabel('Iterations')

    tight_layout()

    figlegend(handles, ('GPU', 'CPU (OMP=12)'), ncol=2, loc='lower center')
    gcf().suptitle('Weak scaling of the Poisson problem on PizDaint cluster')
    gcf().subplots_adjust(top=0.93, bottom=0.1)

    show()


The figure below shows strong scaling results for the MareNostrum 4 cluster.
The problem size is fixed to :math:`512^3` unknowns and ideally the compute
time should decrease as we increase the number of CPU cores. The case of ideal
scaling is depicted for reference on the plots with thin gray dotted lines.

.. plot::

    from pylab import *
    rc('font',   size=12)

    def load_data(fname):
        return loadtxt(fname, dtype={
            'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
            'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
            })

    tri = loadtxt('dmem_data/mn4_trilinos_strong.dat', dtype={
        'names'   : ('mpi', 'size', 'iters', 'setup', 'solve'),
        'formats' : ('i4', 'i8', 'i4', 'f8', 'f8')
        })

    def set_ticks(ax, t):
        ax.set_xticks(t)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)

    figure(figsize=(10,12))
    gs = GridSpec(4,2)
    handles = []

    for k,fname in enumerate(('dmem_data/mn4_linear_strong.dat', 'dmem_data/mn4_const_strong.dat')):
        data = load_data(fname)
        for omp in sorted(unique(data['omp'])):
            if omp == 48: continue

            d = data[data['omp'] == omp]
            c = unique(d['mpi'] * omp)
            m = unique(d['mpi'])

            setup = array([min(d[d['mpi']==i]['setup']) for i in m])
            solve = array([min(d[d['mpi']==i]['solve']) for i in m])
            iters = array([min(d[d['mpi']==i]['iters']) for i in m])
            total = setup + solve

            ax = subplot(gs[0,k])
            h = loglog(c, total, '.-')
            ylim([1e0, 1.5e2])
            set_ticks(ax, c)
            if k == 0: handles.append(h[0])
            ideal = total[0] * c[0] / c
            if omp == 4:
                hi = plot(c,ideal,'k:', zorder=1, linewidth=1, alpha=0.5)

            ax = subplot(gs[1,k])
            loglog(c, setup, '.-')
            ylim([1e-1, 1e2])
            ideal = setup[0] * c[0] / c
            if omp == 12:
                plot(c,ideal,'k:', zorder=1, linewidth=1, alpha=0.5)
            set_ticks(ax, c)

            ax = subplot(gs[2,k])
            loglog(c, solve, '.-')
            ideal = solve[0] * c[0] / c
            if omp == 4:
                plot(c,ideal,'k:', zorder=1, linewidth=1, alpha=0.5)
            ylim([1e0, 1e2])
            set_ticks(ax, c)

            ax = subplot(gs[3,k])
            semilogx(c, iters, '.-')
            ylim([0,110])
            set_ticks(ax, c)

        subplot(gs[3,k])
        xlabel('Number of cores (MPI * OMP)')

    subplot(gs[0,0])
    h = plot(tri['mpi'][1:], tri['setup'][1:] + tri['solve'][1:], '.-')
    title('Linear deflation')
    ylabel('Total time')

    subplot(gs[0,1])
    title('Constant deflation')

    subplot(gs[1,0])
    plot(tri['mpi'][1:], tri['setup'][1:], '.-')
    ylabel('Setup time')

    subplot(gs[2,0])
    plot(tri['mpi'][1:], tri['solve'][1:], '.-')
    ylabel('Solve time')

    subplot(gs[3,0])
    plot(tri['mpi'][1:], tri['iters'][1:], '.-')
    ylabel('Iterations')

    tight_layout()

    figlegend(handles + [h[0], hi[0]], ['OMP={}'.format(i) for i in (1, 4, 12, 24)]
            + ['Trilinos', 'Ideal scaling'],
           ncol=3, loc='lower center')
    gcf().suptitle('Strong scaling of the Poisson problem on the MareNostrum 4 cluster')
    gcf().subplots_adjust(top=0.93, bottom=0.12)

    show()


Here AMGCL scales much better than Trilinos, and is close to ideal for both
kinds of deflation. As in the weak scaling case, we see a drop in scalability
at about 1536 cores for ‘OMP=1’, but unlike before, the drop is also observable
for the constant deflation case. This is explained by the fact that work size
per each subdomain becomes too small to cover both setup and communication
costs.

The profiling data for the strong scaling case is shown in the following table,
and it is apparent that the same factorization and coarse solve communication
bottlenecks as in the weak scaling scenario come into play. Unfortunately, we
were not able to obtain detailed profiling info for the constant deflation, but
it should be obvious that in this case communication is the main limiting
factor, as the coarse problem factorization costs much less due to reduced size
of the deflated space.

+-------+---------------------+-------------------------------+------------+
| Cores | Setup               | Solve                         | Iterations |
+-------+-------+-------------+--------+------------+---------+------------+
|       | Total | Factorize E | Total  | RHS for E  | Solve E |            |
+=======+=======+=============+========+============+=========+============+
| - **Linear deflation, OMP=1**                                            |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  1.01 |        0.03 |  14.77 |       1.04 |    0.07 |         64 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  1.16 |        0.76 |   5.15 |       0.71 |    0.48 |         50 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 | 17.43 |       15.58 |  40.93 |      34.23 |    2.72 |         34 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Constant deflation, OMP=1**                                          |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  1.22 |             |  16.16 |            |         |         76 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  0.55 |             |  12.92 |            |         |         72 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  3.20 |             |  48.91 |            |         |         46 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Linear deflation, OMP=4**                                            |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  1.34 |        0.00 |  14.38 |       0.13 |    0.01 |         62 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  0.77 |        0.03 |   4.66 |       0.40 |    0.08 |         68 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  0.98 |        0.76 |   3.24 |       0.78 |    0.48 |         50 |
+-------+-------+-------------+--------+------------+---------+------------+
| - **Constant deflation, OMP=4**                                          |
+-------+-------+-------------+--------+------------+---------+------------+
|   384 |  2.75 |             |  18.05 |            |         |         80 |
+-------+-------+-------------+--------+------------+---------+------------+
|  1536 |  0.55 |             |   4.63 |            |         |         76 |
+-------+-------+-------------+--------+------------+---------+------------+
|  6144 |  0.21 |             |   3.83 |            |         |         66 |
+-------+-------+-------------+--------+------------+---------+------------+

Next figure shows strong scaling AMGCL results for OpenMP and CUDA backends on
the PizDaint cluster. The problem size here is :math:`256^3` unknowns. The
scalability curves show similar trends as on the MareNostrum 4 cluster, but the
GPU scaling is a bit further from ideal due to higher overheads required for
managing the GPU and transferring the communication data between the GPU and
CPU memories. As in the weak scaling case, the GPU backend is about 9 times
faster than the CPU backend during solution phase, and about 3 times faster
overall.

.. plot::

    from pylab import *
    rc('font',   size=12)

    def load_data(fname):
        return loadtxt(fname, dtype={
            'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
            'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
            })

    figure(figsize=(10,12))
    gs = GridSpec(4,2)
    handles = []

    marker = dict(GPU='o', CPU='d')

    for k,backend,fname in (
            (0, 'GPU', 'dmem_data/daint_gpu_linear_strong.dat'),
            (0, 'CPU', 'dmem_data/daint_cpu_linear_strong.dat'),
            (1, 'GPU', 'dmem_data/daint_gpu_const_strong.dat'),
            (1, 'CPU', 'dmem_data/daint_cpu_const_strong.dat'),
            ):
        d = load_data(fname)
        m = unique(d['mpi'])

        setup = array([min(d[d['mpi']==i]['setup']) for i in m])
        solve = array([min(d[d['mpi']==i]['solve']) for i in m])
        iters = array([min(d[d['mpi']==i]['iters']) for i in m])
        total = setup + solve

        ax = subplot(gs[0,k])
        h = loglog(m, total, '.-')
        ylim([1e-1,1e2])
        if backend == 'CPU':
            ideal = total[0] * m[0] / m
            hi = plot(m, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        if k == 0: handles.append(h[0])

        ax = subplot(gs[1,k])
        loglog(m, setup, '.-')
        if backend == 'CPU':
            ideal = setup[0] * m[0] / m
            plot(m, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        ylim([1e-2,1e2])

        ax = subplot(gs[2,k])
        loglog(m, solve, '.-')
        if backend == 'CPU':
            ideal = solve[0] * m[0] / m
            plot(m, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        ylim([1e-1,1e2])

        ax = subplot(gs[3,k])
        semilogx(m, iters, '.-')
        ylim([20, 80])
        xlabel('Compute nodes')

    for k in range(4):
        for j in range(2):
            ax = subplot(gs[k,j])
            ax.set_xticks([2**i for i in (1, 3, 5, 7, 9, 11)])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_tick_params(which='minor', size=0)
            ax.get_xaxis().set_tick_params(which='minor', width=0)

    subplot(gs[0,0])
    title('Linear deflation')
    ylabel('Total time')

    subplot(gs[0,1])
    title('Constant deflation')

    subplot(gs[1,0])
    ylabel('Setup time')

    subplot(gs[2,0])
    ylabel('Solve time')

    subplot(gs[3,0])
    ylabel('Iterations')

    tight_layout()

    figlegend(handles + [hi[0]], ('GPU', 'CPU (OMP=12)', 'Ideal scaling'),
            ncol=3, loc='lower center')
    gcf().suptitle('Strong scaling of the Poisson problem on PizDaint cluster')
    gcf().subplots_adjust(top=0.93, bottom=0.1)

    show()

An interesting observation is that convergence of the method improves with
grow- ing number of MPI processes. In other words, the number of iterations
required to reach the desired tolerance decreases with as the number of
subdomains grows, since the deflated system is able to describe the main
problem better and better. This is especially apparent from the strong
scalability results, where the problem size remains fixed, but is also
observable in the weak scaling case for 'OMP=1'.

3D Navier-Stokes problem
^^^^^^^^^^^^^^^^^^^^^^^^

The system matrix in these tests contains 4773588 unknowns and 281089456
nonzeros. AMGCL library uses field-split approach with the
``mpi::schur_pressure_correction`` preconditioner. Trilinos ML does not provide
field-split type preconditioners, and uses the nonsymmetric smoothed
aggregation variant (NSSA) applied to the monolithic problem.  Default NSSA
parameters were employed in the tests.

The next figure shows scalability results for the Navier-Stokes problem on the
MareNostrum 4 cluster. Since we are solving a fixed-size problem, this is
essentially a strong scalability test.

.. plot::

    from pylab import *
    rc('font',   size=12)

    def load_data(fname):
        return loadtxt(fname, dtype={
            'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
            'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
            })

    def set_ticks(ax, t):
        ax.set_xticks(t)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)

    figure(figsize=(10,7))
    gs = GridSpec(2,2)
    handles = []

    #--- Schur PC ---
    data = loadtxt('dmem_data/mn4_schur.dat', dtype={
        'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
        'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
        })

    for omp in sorted(unique(data['omp'])):
        if omp == 48: continue

        d = data[data['omp'] == omp]
        c = unique(d['mpi'] * omp)
        m = unique(d['mpi'])

        setup = array([min(d[d['mpi']==i]['setup']) for i in m])
        solve = array([min(d[d['mpi']==i]['solve']) for i in m])
        iters = array([min(d[d['mpi']==i]['iters']) for i in m])
        total = setup + solve

        subplot(gs[0,0])
        h = loglog(c, total, '.-')
        ylim([1e0, 5e2])
        if omp==24:
            ideal = total[0] * c[0] / c
            hi = plot(c, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        handles.append(h[0])

        subplot(gs[0,1])
        loglog(c, setup, '.-')
        if omp==24:
            ideal = setup[0] * c[0] / c
            plot(c, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        ylim([5e-2, 5e2])

        subplot(gs[1,0])
        loglog(c, solve, '.-')
        if omp==24:
            ideal = solve[0] * c[0] / c
            plot(c, ideal, 'k:', zorder=1, linewidth=1, alpha=0.5)
        ylim([1e0, 5e2])

        subplot(gs[1,1])
        semilogx(c, iters, '.-')
        ylim([0,110])

    #--- Trilinos ---
    d = loadtxt('dmem_data/mn4_ns_trilinos.txt', dtype={
            'names'   : ('mpi', 'size', 'iters', 'setup', 'solve'),
            'formats' : ('i4', 'i8', 'i4', 'f8', 'f8')
            })

    m = d['mpi']

    setup = d['setup']
    solve = d['solve']
    iters = d['iters']
    total = setup + solve

    ax = subplot(gs[0,0])
    h = loglog(m, total, '.-')
    handles.append(h[0])

    ax = subplot(gs[0,1])
    loglog(m, setup, '.-')

    ax = subplot(gs[1,0])
    loglog(m, solve, '.-')

    ax = subplot(gs[1,1])
    semilogx(m, iters, '.-')

    for i in range(2):
        for j in range(2):
            set_ticks(subplot(gs[i,j]), [96 * 2**k for k in range(7)])

    subplot(gs[0,0])
    ylabel('Total time')

    subplot(gs[0,1])
    ylabel('Setup time')

    subplot(gs[1,0])
    ylabel('Solve time')
    xlabel('Number of cores (MPI * OMP)')

    subplot(gs[1,1])
    ylabel('Iterations')
    xlabel('Number of cores (MPI * OMP)')

    tight_layout()

    figlegend(handles + [hi[0]], ['OMP={}'.format(i) for i in (1, 4, 12, 24)] +
            ['Trilinos', 'Ideal scaling'],
           ncol=3, loc='lower center')
    gcf().suptitle('Strong scaling of the Navier-Stokes problem on MareNostrum 4 cluster')
    gcf().subplots_adjust(top=0.93, bottom=0.2)

    show()


Both AMGCL and ML preconditioners deliver a very flat number of iterations with
growing number of MPI processes.  As expected, the field-split preconditioner
pays off and performs better than the monolithic approach in the solution of
the problem.  Overall the AMGCL implementation shows a decent, although less
than optimal parallel scalability.  This is not unexpected since the problem
size quickly becomes too little to justify the use of more parallel resources
(note that at 192 processes, less than 25000 unknowns are assigned to each MPI
subdomain).  Unsurprisingly, in this context the use of OpenMP within each
domain pays off and allows delivering a greater level of scalability.

.. _`Trilinos ML`: https://trilinos.org/packages/ml/
