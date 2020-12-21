Preconditioners
===============

Aside from the AMG, AMGCL implements preconditioners for some common problem
types. For example, there is a Schur complement pressure correction
preconditioner for Navie-Stokes type problems, or CPR preconditioner for
reservoir simulations. Also, it is possible to use single level relaxation
method as a preconditioner.

General preconditioners
^^^^^^^^^^^^^^^^^^^^^^^

These preconditioners do not take the origin or structure of matrix into
account, and may be useful both on their own, as well as building blocks for
the composite preconditioners.

AMG
---

.. cpp:class:: template <class Backend, template <class> class Coarsening, template <class> class Relax> \
               amgcl::amg

   .. rubric:: Include ``<amgcl/amg.hpp>``

   AMG is one the most effective methods for the solution of large sparse
   unstructured systems of equations, arising, for example, from discretization
   of PDEs on unstructured grids [TrOS01]_. The method may be used
   as a black-box solver for various computational problems, since it does not
   require any information about the underlying geometry.
   
   The three template parameters allow the user to select the exact components
   of the method:
   
   - The :doc:`Backend <backends>` to transfer the constructed hierarchy to,
   - The :doc:`Coarsening <coarsening>` strategy for the hierarchy construction, and
   - The :doc:`Relaxation <relaxation>` scheme (smoother to use during the solution phase).

   .. cpp:type:: typename Backend::params backend_params

      The backend parameters

   .. cpp:type:: typename Backend::value_type value_type

      The value type of the system matrix

   .. cpp:type:: typename amgcl::math::scalar_of<value_type>::type scalar_type

      The scalar type corresponding to the value type. For example, when the
      value type is ``std::complex<double>``, then the scalar type is
      ``double``.

   .. cpp:type:: Coarsening<Backend> coarsening_type

      The coarsening class instantiated for the specified backend

   .. cpp:type:: Relax<Backend> relax_type

      The relaxation class instantiated for the specified backend

   .. cpp:class:: params 

      The AMG parameters. The coarsening and the relaxation parameters are
      encapsulated as part of the AMG parameters.

      .. cpp:type:: typename coarsening_type::params coarsening_params
         
         The type of the coarsening parameters

      .. cpp:type:: typename relax_type::params relax_params

         The type of the relaxation parameters

      .. cpp:member:: coarsening_params coarsening

         The coarsening parameters

      .. cpp:member:: relax_params relax

         The relaxation parameters

      .. cpp:member:: unsigned coarse_enough = Backend::direct_solver::coarse_enough()

         Specifies when a hierarchy level is coarse enough to be solved
         directly. If the number of variables at the level is lower than this
         threshold, then the hierarchy construction is stopped and the linear
         system is solved directly at this level. The default value comes from
         the direct solver class defined as part of the backend. 
         
      .. cpp:member:: bool direct_coarse = true

         Use direct solver at the coarsest level.  When set, the coarsest level
         is solved with a direct solver.  Otherwise a smoother is used as a
         solver. This may be useful when the system is singular, but is still
         possible to solve with an iterative solver.

      .. cpp:member:: unsigned max_levels = std::numeric_limits<unsigned>::max()

         The maximum number of levels.  If this number is reached even when the
         size of the last level is greater that ``coarse_enough``, then the
         hierarchy construction is stopped. The coarsest level will not be
         solved directly, but will use a smoother.

      .. cpp:member:: unsigned npre = 1

         The number of pre-relaxations.

      .. cpp:member:: unsigned npost = 1

         The number of post-relaxations.

      .. cpp:member:: unsigned ncycle = 1

         The shape of AMG cycle (1 for V-cycle, 2 for W-cycle, etc).

      .. cpp:member:: unsigned pre_cycles = 1

         The number of cycles to make as part of preconditioning.

Single-level relaxation
-----------------------

.. cpp:class:: template <class Backend, template <class> class Relax> \
               amgcl::relaxation::as_preconditioner

   .. rubric:: Include ``<amgcl/relaxation/as_preconditioner.hpp>``

   Allows to use a :doc:`relaxation <relaxation>` method as a standalone
   preconditioner.

   .. cpp:type:: Relax<backend> smoother;

      The relaxation class instantiated for the specified backend

   .. cpp:type:: typename smoother::params params

      The relaxation params are inherited as the parameters for the
      preconditioner

Dummy
-----

.. cpp:class:: template <class Backend> \
               amgcl::preconditioner::dummy

   .. rubric:: Include ``<amgcl/preconditioner/dummy.hpp>``

   The dummy preconditioner, equivalent to an identity matrix. May be used to
   test the convergence of unpreconditioned iterative solvers.

   .. cpp:class:: params

      There are no parameters

Composite preconditioners
^^^^^^^^^^^^^^^^^^^^^^^^^

The preconditioners in this section take the into account the block structure
of the system and properties of the individual blocks. Most often the
preconditioners are used for the solution of saddle point or Stokes-like
systems, where the system matrix may be represented in the following form:

.. math::
   :label: saddle_point_eq

    \begin{pmatrix}
        A & B_1^T \\
        B_2 & C
    \end{pmatrix}
    \begin{pmatrix}
        u \\ p
    \end{pmatrix}
    =
    \begin{pmatrix}
        b_u \\ b_p
    \end{pmatrix}

CPR
---

.. cpp:class:: template <class PPrecond, class SPrecond> \
               amgcl::preconditioner::cpr

   .. rubric:: Include ``<amgcl/preconditioner/cpr.hpp>``

   The Constrained Pressure Residual (CPR) preconditioner [Stue07]_. The CPR
   preconditioners are based on the idea that coupled system solutions are
   mainly determined by the solution of their elliptic components (i.e.,
   pressure). Thus, the procedure consists of extracting and accurately solving
   pressure subsystems. Residuals associated with this solution are corrected
   with an additional preconditioning step that recovers part of the global
   information contained in the original system.

   The template parameters ``PPrecond`` and ``SPrecond`` for the CPR
   preconditioner specify which preconditioner to use with the pressure
   subblock (the :math:`C` matrix in :eq:`saddle_point_eq`), and with the
   complete system.

   The system matrix should be ordered by grid nodes, so that the pressure and
   suturation/concentration unknowns belonging to the same grid node are
   compactly located together in the vector of unknowns. The pressure should be
   the first unknown in the block of unknowns associated with a grid node.

   .. cpp:class:: params

      The CPR preconditioner parameters

      .. cpp:type:: typename SPrecond::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename PPrecond::params pprecond_params

         The type of the pressure preconditioner parameters

      .. cpp:type:: typename SPrecond::params sprecond_params

         The type of the global preconditioner parameters

      .. cpp:member:: pprecond_params pprecond

         The pressure preconditioner parameters

      .. cpp:member:: sprecond_params sprecond

         The global preconditioner parameters

      .. cpp:member:: int block_size = 2
      
         The number of unknowns associated with each grid node. The default
         value is 2 when the system matrix has scalar value type. Otherwise,
         the block size of the system matrix value type is used.

      .. cpp:member:: size_t active_rows = 0

         When non-zero, only unknowns below this number are considered to be
         pressure. May be used when a system matrix contains unstructured tail
         block (for example, the unknowns associated with wells).

CPR (DRS)
---------

.. cpp:class:: template <class PPrecond, class SPrecond> \
               amgcl::preconditioner::cpr_drs

   .. rubric:: Include ``<amgcl/preconditioner/cpr.hpp>``

   The Constrained Pressure Residual (CPR) preconditioner with weighted dynamic
   row sum (WDRS) [Grie14]_, [BrCC15]_.

   The template parameters ``PPrecond`` and ``SPrecond`` for the CPR WDRS
   preconditioner specify which preconditioner to use with the pressure
   subblock (the :math:`C` matrix in :eq:`saddle_point_eq`), and with the
   complete system.

   The system matrix should be ordered by grid nodes, so that the pressure and
   suturation/concentration unknowns belonging to the same grid node are
   compactly located together in the vector of unknowns. The pressure should be
   the first unknown in the block of unknowns associated with a grid node.

   .. cpp:class:: params

      The CPR preconditioner parameters

      .. cpp:type:: typename SPrecond::value_type value_type

         The value type of the system matrix

      .. cpp:type:: typename PPrecond::params pprecond_params

         The type of the pressure preconditioner parameters

      .. cpp:type:: typename SPrecond::params sprecond_params

         The type of the global preconditioner parameters

      .. cpp:member:: pprecond_params pprecond

         The pressure preconditioner parameters

      .. cpp:member:: sprecond_params sprecond

         The global preconditioner parameters

      .. cpp:member:: int block_size = 2
      
         The number of unknowns associated with each grid node. The default
         value is 2 when the system matrix has scalar value type. Otherwise,
         the block size of the system matrix value type is used.

      .. cpp:member:: size_t active_rows = 0

         When non-zero, only unknowns below this number are considered to be
         pressure. May be used when a system matrix contains unstructured tail
         block (for example, the unknowns associated with wells).

      .. cpp:member:: double eps_dd = 0.2

         Controls the severity of the violation of diagonal dominance. See
         [Grie14]_ for more details.

      .. cpp:member:: double eps_ps = 0.02

         Controls the pressure/saturation coupling. See [Grie14]_ for more
         details.

      .. cpp:member:: std::vector<double> weights

         The weights for the weighted DRS method. See [BrCC15]_ for more
         details.

Schur Pressure Correction
-------------------------

.. cpp:class:: template <class USolver, class PSolver> \
               amgcl::preconditioner::schur_pressure_correction

   .. rubric:: Include ``<amgcl/preconditioner/schur_pressure_correction.hpp>``

   The system :eq:`saddle_point_eq` may be rewritten as

   .. math::

      \begin{pmatrix}
          A & B_1^T \\
          0 & S
      \end{pmatrix}
      \begin{pmatrix}
          u \\ p
      \end{pmatrix}
      =
      \begin{pmatrix}
          b_u \\ b_p - B_2 A^{-1} b_u
      \end{pmatrix}

   where :math:`S = C - B_2 A^{-1} B_1^T` is the Schur complement. The Schur
   complement pressure correction preconditioner uses this representation and
   an approximation to the Schur complement matrix in order to decouple the
   pressure and the velocity parts of the system [ElHS08]_.

   The two template parameters for the method, ``USolver`` and ``PSolver``,
   specify the :doc:`preconditioned solvers <coupled_solvers>` for the
   :math:`A` and :math:`S` blocks.

   .. cpp:class:: params

      The parameters for the Schur pressure correction preconditioner

      .. cpp:type:: typename USolver::params usolver_params

         The type of the USolver parameters

      .. cpp:type:: typename PSolver::params psolver_params

         The type of the PSolver parameters

      .. cpp:member:: usolver_params usolver

         The USolver parameters

      .. cpp:member:: psolver_params psolver

         The PSolver parameters

      .. cpp:member:: std::vector<char> pmask

         The indicator vector, containing 1 for pressure unknowns, and 0
         otherwise.

      .. cpp:member:: int type = 1

         The variant of the block preconditioner to use.

         - When ``type = 1``:

           .. math::

              \begin{aligned}
              S p &= b_p - B_2 A^{-1} b_u \\
              A u &= b_u - B_1^T p
              \end{aligned}

         - When ``type = 2``:

           .. math::

              \begin{aligned}
              S p &= b_p \\
              A u &= b_u - B_1^T p
              \end{aligned}

      .. cpp:member:: bool approx_schur = false

         When set, approximate :math:`A^{-1}` as :math:`\mathrm{diag}(A)^{-1}`
         during computation of the matrix-less Schur complement when solving
         the :math:`Sp=b_p` system. Otherwise, the full solve using ``USolver``
         is used.

      .. cpp:member:: int adjust_p = 1

         Adjust the matrix used to construct the preconditioner for the Schur
         complement system.

         - When ``adjust_p = 0``, use the unmodified :math:`C` matrix;
         - When ``adjust_p = 1``, use :math:`C - \mathrm{diag}(B_2 \mathrm{diag}(A)^{-1} B_1^T)`;
         - When ``adjust_p = 2``, use :math:`C - B_2 \mathrm{diag}(A)^{-1} B_1^T`.

      .. cpp:member:: bool simplec_dia = true

         When set, use :math:`\frac{1}{\sum_j \|A_{i,j}\|}` instead of
         :math:`\mathrm{diag}(A)^{-1}` as the approximation for :math:`A^{-1}`
         (similar to the SIMPLEC algorithm).

      .. cpp:member:: int verbose = 0

         - When ``verbose >= 1``, show the number of iterations and the relative
           residual achieved after each nested solve.
         - When ``verbose >= 2``, save the :math:`A` and :math:`C` submatrices as
           ``Kuu.mtx`` and ``Kpp.mtx``.
