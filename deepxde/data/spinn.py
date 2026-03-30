"""Data class for Separable Physics-Informed Neural Networks (SPINN).

Provides ``SPINNPDE`` — a Data class analogous to ``TripleCartesianProd`` in the
DeepONet family.  Interior (PDE) collocation points and boundary-condition points
are stored as *tuples of 1-D coordinate arrays* (one per spatial dimension).
The SPINN network receives these tuples directly, so no changes to ``model.py``
are required.

Boundary edges of a rectangle are represented by collapsing one dimension to a
single value, e.g. the left edge ``x = x_min`` of a 2-D domain is encoded as
``(np.array([[0.]]), y_array)``.
"""

import numpy as np

from .data import Data
from .. import config
from ..utils import run_if_all_none


class SPINNPDE(Data):
    """PDE solver using a Separable PINN.

    The user provides the collocation coordinates directly as a tuple of
    1-D arrays (one per spatial dimension).

    Args:
        pde_points: Tuple/list of 1-D numpy arrays ``(x1, x2, ..., xd)``
            where ``xi.shape == (Ni, 1)``.  These are the interior
            collocation points for PDE losses.
        pde: PDE residual function ``pde(inputs, outputs)`` → residual
            tensor(s).
        bcs: A single ``SPINNPointSetBC`` / ``SPINNPointSetOperatorBC`` or
            a list of them.  Use ``[]`` for no boundary conditions.
        solution: Optional reference solution for metrics.  Should accept
            a full-grid array of shape ``(N1*N2*...*Nd, d)`` and return
            the solution values.
        test_points: Optional tuple/list of 1-D arrays for testing.
            ``None`` → reuse ``pde_points``.

    Attributes:
        train_x: Tuple of 1-D coordinate arrays for PDE training.
        train_y: Reference solution on the Cartesian grid, or ``None``.
    """

    def __init__(
        self,
        pde_points,
        pde,
        bcs,
        solution=None,
        test_points=None,
    ):
        self.pde_points = tuple(pde_points)
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]
        self.soln = solution
        self.test_points = tuple(test_points) if test_points is not None else None

        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.train_next_batch()
        self.test()

    # ------------------------------------------------------------------
    # Data interface
    # ------------------------------------------------------------------

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        """Compute PDE + BC losses.

        ``outputs`` is the network prediction on the *interior* grid
        (already evaluated by ``model.py``).  For each BC, we re-evaluate
        the network via ``aux[0]`` (the ``outputs_fn`` callable provided
        by the JAX compile path) on the BC's own point set.
        """
        outputs_fn = aux[0] if aux else None

        # -- PDE residual losses --
        f = []
        if self.pde is not None:
            f = self.pde(inputs, (outputs, outputs_fn))
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss_fn, (list, tuple)):
            loss_fn = [loss_fn] * (len(f) + len(self.bcs))
        elif len(loss_fn) != len(f) + len(self.bcs):
            raise ValueError(
                f"There are {len(f) + len(self.bcs)} errors, "
                f"but {len(loss_fn)} losses."
            )

        from .. import backend as bkd

        losses = [
            loss_fn[i](bkd.zeros_like(error), error)
            for i, error in enumerate(f)
        ]

        # -- Boundary-condition losses --
        for j, bc in enumerate(self.bcs):
            bc_inputs = bc.points  # tuple of 1-D arrays
            bc_outputs = outputs_fn(bc_inputs)
            error = bc.error(bc_inputs, bc_outputs)
            losses.append(
                loss_fn[len(f) + j](bkd.zeros_like(error), error)
            )

        return losses

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x = self.pde_points
        self.train_y = self._eval_solution(self.train_x)
        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.test_points is None:
            self.test_x = self.train_x
        else:
            self.test_x = self.test_points
        self.test_y = self._eval_solution(self.test_x)
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self, new_points):
        """Replace interior collocation points and retrigger data setup."""
        self.pde_points = tuple(new_points)
        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.train_next_batch()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eval_solution(self, coords):
        """Evaluate the reference solution on a Cartesian grid."""
        if self.soln is None:
            return None
        grids = np.meshgrid(*[c.ravel() for c in coords], indexing="ij")
        full_grid = np.column_stack([g.ravel() for g in grids])
        return self.soln(full_grid)


# ======================================================================
# Boundary-condition helpers for SPINN
# ======================================================================


class SPINNPointSetBC:
    """Dirichlet boundary condition for SPINN.

    Args:
        points: Tuple of 1-D coordinate arrays, one per dimension.
            Collapse one dimension to a single value for boundary edges
            (e.g. ``(np.array([[0.]]), y_array)`` for the left edge).
        values: Target values on the Cartesian product of ``points``,
            flattened in row-major order.  Shape ``(N, 1)`` or scalar.
        component (int): Output component this BC applies to.
    """

    def __init__(self, points, values, component=0):
        from .. import backend as bkd

        self.points = points
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.component = component

    def error(self, inputs, outputs):
        return outputs[:, self.component : self.component + 1] - self.values


class SPINNPointSetOperatorBC:
    """Operator boundary condition for SPINN.

    Args:
        points: Tuple of 1-D coordinate arrays (same format as
            ``SPINNPointSetBC``).
        values: Target values for the operator.
        func: ``func(inputs, outputs)`` → tensor of shape ``(N, 1)``.
    """

    def __init__(self, points, values, func):
        from .. import backend as bkd

        self.points = points
        self.values = bkd.as_tensor(values, dtype=config.real(bkd.lib))
        self.func = func

    def error(self, inputs, outputs):
        return self.func(inputs, outputs) - self.values
