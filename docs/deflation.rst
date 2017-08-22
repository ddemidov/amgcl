Subdomain deflation
-------------------

The capability to solve large and sparse systems of equations is a cornerstone
of modern numerical methods, sparse linear systems of equations being
ubiquitous  in engineering and physics.  Direct techniques, despite their
attractiveness, simply become not viable beyond a certain size, typically of
the order of the few millions of unknowns, due to their intrinsic memory
requirements and shear computational cost.

Hence, preconditioned iterative methods become the only feasible alternative in
addressing such large scale problems. In practice it is customary to blend the
capability of multigrid and Krylov techniques so to preserve some of the
advantages of each. A number of successful (distributed memory) libraries exist
implementing different flavors of such blending.  Even though such
implementation were proven to be weakly scalable, their strong scalability
remains questionable.  This fact motivated the renowned interest in Domain
Decomposition (DD) methods as well as the rise of a new
class of approaches named *deflation techniques*.

Essentially all of the recent efforts in this area take on the idea of
constructing a multi level decomposition of the solution space, understood as a
very aggressive coarsening of the original problem. The use of such coarser
basis is proved sufficient to guarantee good weak scaling properties, while
implying a minimal additional computational complexity and thus  good strong
scaling properties.  An additional advantage of such approaches is the ease in
combining them *in a modular way* with local preconditioners.  AMGCL allows to
combine the subdomain deflation approach with any of the shared-memory
preconditioners implemented in the library, thus providing good weak and strong
scalability properties.

The figures below demonstrate scalability of the subdomain deflation approach
combined with AMG used as a local preconditioner. We solve the classical
3D Poisson problem in a unit cube :math:`\Omega=[0,1]^3`:

.. math::

    -\Delta u = 1, \; u \in \Omega \quad u = 0, \; u \in \partial \Omega

The first figure shows weak scaling for the problem in case of linear deflation
Here the problem size grows proportionally with number of MPI processes used to
solve the problem, so that a subdomain on each node corresponds to about
:math:`64^3` unknowns. The top two subplots show the setup and the solution
cost of the algorithm (in wall-clock time, seconds).  The bottom plot shows the
number of iterations required to achieve convergence.  It is clear from the
figure that the setup time constitutes only a small fraction of the total cost
of the method, and hence the most time is spent on actually solving the
problem.

We can see that the scalability of the setup phase begins to noticeably suffer
at about 5000 MPI processes.  The main reason here is that the direct solver
used to solve the coarse deflated problem becomes more and more expensive to
setup. One possible solution to this problem would be to use more scalable
direct solver. Another possibility is to reduce the size of the coarse problem
by employing parallelism available within each of the compute nodes.

.. plot::

    from pylab import *

    data = loadtxt('scaling/weak_3d_cd_64.dat', dtype={
        'names'   : ('size', 'np', 'setup', 'solve', 'factorize', 'pastix', 'iters'),
        'formats' : ('i8', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4')
        })

    figure(figsize=(12,8))

    subplot(3,1,1)
    plot(data['np'], data['setup'], '.-')
    ylim([0,5])
    title('Setup (s)')
    xlabel('MPI processes')
    legend()

    subplot(3,1,2)
    plot(data['np'], data['solve'], '.-')
    ylim([0,15])
    title('Solve (s)')
    xlabel('MPI processes')
    legend()

    subplot(3,1,3)
    plot(data['np'], data['iters'], '.-')
    title('Iterations')
    xlabel('MPI processes')
    legend()

    tight_layout()

The next figure shows the results of the second approach. Here we study how the
problem scales with increasing number of OpenMP threads withing each node,
while allocating single MPI process per node. We can observe good scalability
for up to 24 OpenMP threads, and even though going from 24 to 48 threads does
not yield immediate improvement in solution time, it should allow us to gain
advantage at extreme scale, where the size of the coarse system will be a
limiting factor.

.. plot::

    from pylab import *

    data = loadtxt('scaling/weak_3d_cd_100_mpi.txt', dtype={
        'names'   : ('size', 'omp', 'mpi', 'setup', 'solve', 'iters'),
        'formats' : ('i8', 'i4', 'i4', 'f8', 'f8', 'i4')
        })

    figure(figsize=(12,8))

    for n in unique(data['omp']):
        d = data[data['omp']==n]
        m = unique(d['mpi'])
        t1 = [min(d[d['mpi']==i]['setup']) for i in m]
        t2 = [min(d[d['mpi']==i]['solve']) for i in m]
        it = [min(d[d['mpi']==i]['iters']) for i in m]
        
        subplot(3,1,1); semilogy(m, t1, '.-', label='OMP={}'.format(n))
        subplot(3,1,2); semilogy(m, t2, '.-', label='OMP={}'.format(n))
        subplot(3,1,3); plot(m, it, '.-', label='OMP={}'.format(n))

    subplot(3,1,1); title('Setup (s)');  xlabel('MPI processes'); legend()
    subplot(3,1,2); title('Solve (s)');  xlabel('MPI processes'); legend()
    subplot(3,1,3); title('Iterations'); xlabel('MPI processes'); legend()

    tight_layout()
