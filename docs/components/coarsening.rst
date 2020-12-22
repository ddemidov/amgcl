Coarsening Strategies
=====================

A coarsening strategy defines various options for creating coarse systems in
the AMG hierarchy. A coarsening strategy takes the system matrix :math:`A` at
the current level, and returns prolongation operator :math:`P` and the
corresponding restriction operator :math:`R`.

Ruge-Stuben
-----------

.. cpp:class:: template <class Backend> \
               amgcl::coarsening::ruge_stuben

   .. rubric:: Include ``<amgcl/coarsening/ruge_stuben>``

   The classic Ruge-Stuben coarsening with direct interpolation [Stue99]_.

   .. cpp:class:: params

      .. cpp:member:: float eps_strong = 0.25

         Parameter :math:`\varepsilon_{str}` defining strong couplings.
         
         Variable :math:`i` is defined to be strongly negatively coupled to
         another variable, :math:`j`, if

         .. math::

            -a_{ij} \geq \varepsilon_{str}\max\limits_{a_{ik}<0}|a_{ik}|\quad
            \text{with fixed} \quad 0 < \varepsilon_{str} < 1.

         In practice, a value of :math:`\varepsilon_{str}=0.25` is usually
         taken.
      

      .. cpp:member:: bool do_trunc = true

         Prolongation operator truncation.
         Interpolation operators, and, hence coarse operators may increase
         substantially towards coarser levels. Without truncation, this may
         become too costly. Truncation ignores all interpolatory connections
         which are smaller (in absolute value) than the largest one by a
         factor of :math:`\varepsilon_{tr}`. The remaining weights are rescaled
         so that the total sum remains unchanged. In practice, a value of
         :math:`\varepsilon_{tr}=0.2` is usually taken.

      .. cpp:member:: float eps_trunc = 0.2

         Truncation parameter :math:`\varepsilon_{tr}`.

Aggregation-based coarsening
----------------------------

The aggregation-base class of coarsening methods split the nodes at the fine
grid into disjoint sets of nodes, the so-called aggregates that act as nodes on
the coarse grid. The prolongation operators are then built by first
constructing a tentative prolongator using the knowledge of zero energy modes
of the principal part of the differential operator with natural boundary
conditions (e.g., near null-space vectors, or rigid body modes for elasticity).
In case of smoothed aggregation the prolongation operator is then smoothed by a
carefully selected iteration.

All of the aggregation based methods take the aggregation and the nullspace
parameters:

.. cpp:class:: amgcl::coarsening::pointwise_aggregates

   Pointwise aggregation. When the system matrix has a block structure, it is
   converted to a poinwise matrix (single value per block), and the aggregates
   are created for this reduced matrix instead.

   .. cpp:class:: params

      The aggregation parameters.

      .. cpp:member:: float eps_strong = 0.08

         Parameter :math:`\varepsilon_{strong}` defining strong couplings.
         Connectivity is defined in a symmetric way, that is, two variables
         :math:`i` and :math:`j` are considered to be connected to each other
         if :math:`\frac{a_{ij}^2}{a_{ii}a_{jj}} > \varepsilon_{strong}` with
         fixed :math:`0 < \varepsilon_{strong} < 1`.

      .. cpp:member:: int block_size = 1

         The block size in case the system matrix has a block structure.

.. cpp:class:: amgcl::coarsening::nullspace_params

   The nullspace parameters.

   .. cpp:member int cols = 0

      The number of near nullspace vectors.

   .. cpp:member:: std::vector<double> B;

      The near nullspace vectors. The vectors are represented as columns of a
      2D matrix stored in row-major order.
      
Aggregation
^^^^^^^^^^^

.. cpp:class:: template <class Backend> \
               amgcl::coarsening::aggregation

   .. rubric:: Include ``<amgcl/coarsening/aggregation.hpp>``

   The non-smoothed aggregation coarsening [Stue99]_.

   .. cpp:class:: params

      The aggregation coarsening parameters

      .. cpp:member:: amgcl::coarsening::pointwise_aggregates::params aggr;

         The aggregation parameters

      .. cpp:member:: amgcl::coarsening::nullspace_params nullspace

         The near nullspace parameters
      
      .. cpp:member:: float over_interp = 1.5

         Over-interpolation factor :math:`\alpha` [Stue99]_.  In case of
         aggregation coarsening, coarse-grid correction of smooth error, and by
         this the overall convergence, can often be substantially improved by
         using "over-interpolation", that is, by multiplying the actual
         correction (corresponding to piecewise constant interpolation) by some
         factor :math:`\alpha > 1`. Equivalently, this means that the
         coarse-level Galerkin operator is re-scaled by :math:`1/\alpha`:

         .. math::
         
            I_h^HA_hI_H^h \to \frac{1}{\alpha}I_h^HA_hI_H^h.

Smoothed Aggregation
^^^^^^^^^^^^^^^^^^^^
.. cpp:class:: template <class Backend> \
               amgcl::coarsening::smoothed_aggregation

   .. rubric:: Include ``<amgcl/coarsening/smoothed_aggregation.hpp>``

   The smoothed aggregation coarsening [VaMB96]_.

   .. cpp:class:: params

      The smoothed aggregation coarsening parameters

      .. cpp:member:: amgcl::coarsening::pointwise_aggregates::params aggr;

         The aggregation parameters

      .. cpp:member:: amgcl::coarsening::nullspace_params nullspace

         The near nullspace parameters
      
      .. cpp:member:: float relax = 1.0

         The relaxation factor :math:`r`.
         Used as a scaling for the damping factor :math:`\omega`.
         When ``estimate_spectral_radius`` is set, then

         .. math::
            
            \omega = r * (4/3) / \rho.

         where :math:`\rho` is the spectral radius of the system matrix.
         Otherwise

         .. math::
            
            \omega = r * (2/3).
        
         The tentative prolongation :math:`\tilde P` from the non-smoothed
         aggregation is improved by smoothing to get the final prolongation
         matrix :math:`P`. Simple Jacobi smoother is used here, giving the
         prolongation matrix

         .. math::

            P = \left( I - \omega D^{-1} A^F \right) \tilde P.

         Here :math:`A^F = (a_{ij}^F)` is the filtered matrix given by

         .. math::
         
            \begin{aligned}
            a_{ij}^F &= \begin{cases}
            a_{ij} \quad \text{if} \; j \in N_i\\
            0 \quad \text{otherwise}
            \end{cases}, \quad \text{if}\; i \neq j, \\
            \quad a_{ii}^F &= a_{ii} - \sum\limits_{j=1,j\neq i}^n
            \left(a_{ij} - a_{ij}^F \right),
            \end{aligned}

         where :math:`N_i` is the set of variables strongly coupled to
         variable :math:`i`, and :math:`D` denotes the diagonal of :math:`A^F`.

      .. cpp:member:: bool estimate_spectral_radius = false

         Estimate the matrix spectral radius.  This usually improves the
         convergence rate and results in faster solves, but costs some time
         during setup.

      .. cpp:member:: int power_iters = 0

         The number of power iterations to apply for the spectral radius
         estimation. Use Gershgorin disk theorem when ``power_iters = 0``.

Smoothed Aggregation with Energy Minimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. cpp:class:: template <class Backend> \
               amgcl::coarsening::smoothed_aggr_emin

   .. rubric:: Include ``<amgcl/coarsening/smoothed_aggr_emin.hpp>``

   The smoothed aggregation with energy minimization coarsening [SaTu08]_.

   .. cpp:class:: params

      The smoothed aggregation with energy minimization coarsening parameters

      .. cpp:member:: amgcl::coarsening::pointwise_aggregates::params aggr;

         The aggregation parameters

      .. cpp:member:: amgcl::coarsening::nullspace_params nullspace

         The near nullspace parameters
