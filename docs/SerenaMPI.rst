Structural problem (MPI version)
--------------------------------

In this section we look at how to use the MPI version of the AMGCL solver for
the Serena_ system. We have already determined in the :doc:`Serena` section
that the system is best solved with the block-valued backend, and needs to be
scaled so that it has the unit diagonal.

.. _Serena: https://sparse.tamu.edu/Janna/Serena

.. literalinclude:: ../tutorial/2.Serena/serena_mpi.cpp
   :caption: The MPI solution of the Serena problem
   :language: cpp
   :linenos:

