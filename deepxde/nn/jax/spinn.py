"""Separable Physics-Informed Neural Networks (SPINN) for JAX.

This code is adapted from the original SPINN paper:
- paper: https://arxiv.org/abs/2306.15969
- code: https://github.com/stnamjef/SPINN
"""

from collections.abc import Mapping
from typing import Any, Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers


# ── Body networks ────────────────────────────────────────────────────
# Each body network operates coordinate-wise: it maps a single 1-D
# coordinate input (N, 1) to an output (N, rank * out_dim).
# To add a new variant, define a nn.Module subclass and register it
# in BODY_NETWORK_DICT.
# ─────────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """Standard dense MLP body network.

    Args:
        layer_sizes: ``[1, h1, ..., hK, rank * out_dim]``.
            First element must be 1 (coordinate-wise input).
        activation: Activation name or callable (applied to hidden layers).
        kernel_initializer: Weight initializer name or callable.
    """

    layer_sizes: Sequence[int]
    activation: Any = "tanh"
    kernel_initializer: Any = "Glorot uniform"

    @nn.compact
    def __call__(self, x):
        act = activations.get(self.activation)
        init = initializers.get(self.kernel_initializer)
        for fs in self.layer_sizes[1:-1]:
            x = act(nn.Dense(fs, kernel_init=init)(x))
        x = nn.Dense(self.layer_sizes[-1], kernel_init=init)(x)
        return x


class ModifiedMLP(nn.Module):
    """Modified MLP with multiplicative gating (U/V highway).

    Uses encoder gates U, V and a highway loop to improve expressiveness.

    Args:
        layer_sizes: ``[1, h1, ..., hK, rank * out_dim]``.
        activation: Activation name or callable.
        kernel_initializer: Weight initializer name or callable.
    """

    layer_sizes: Sequence[int]
    activation: Any = "tanh"
    kernel_initializer: Any = "Glorot uniform"

    @nn.compact
    def __call__(self, x):
        act = activations.get(self.activation)
        init = initializers.get(self.kernel_initializer)
        hidden = self.layer_sizes[1]  # first hidden width sets the gate size
        U = act(nn.Dense(hidden, kernel_init=init)(x))
        V = act(nn.Dense(hidden, kernel_init=init)(x))
        H = act(nn.Dense(hidden, kernel_init=init)(x))
        for fs in self.layer_sizes[1:-1]:
            Z = act(nn.Dense(fs, kernel_init=init)(H))
            H = (jnp.ones_like(Z) - Z) * U + Z * V
        x = nn.Dense(self.layer_sizes[-1], kernel_init=init)(H)
        return x


# Registry: maps string identifiers to body-network classes.
BODY_NETWORK_DICT = {
    "mlp": MLP,
    "modified_mlp": ModifiedMLP,
}


def _get_body_network(identifier):
    """Resolve a body-network specification into a ``nn.Module`` instance.

    Args:
        identifier: One of:
            - ``dict``/mapping: ``{name: kwargs}`` where *name* is a key in
              ``BODY_NETWORK_DICT`` and *kwargs* are forwarded to the
              constructor, e.g.
              ``{"mlp": {"layer_sizes": [1, 32, 32, 160]}}``.
            - ``nn.Module``: a pre-built Flax module used as-is.

    Returns:
        A ``nn.Module`` instance ready to be used as a SPINN body network.
    """
    if isinstance(identifier, nn.Module):
        return identifier

    if isinstance(identifier, Mapping):
        if len(identifier) != 1:
            raise ValueError(
                "body_network dict must have exactly one key (the variant name)."
            )
        name, kwargs = next(iter(identifier.items()))
        if name not in BODY_NETWORK_DICT:
            raise ValueError(
                f"Unknown body network '{name}'. "
                f"Available: {list(BODY_NETWORK_DICT)}"
            )
        return BODY_NETWORK_DICT[name](**dict(kwargs))

    raise TypeError(
        f"Cannot interpret body_network: {identifier!r}. "
        "Pass a dict (e.g. {{'mlp': {{...}}}}) or an nn.Module instance."
    )


# ── SPINN ─────────────────────────────────────────────────────────────


class SPINN(NN):
    """Separable Physics-Informed Neural Network.

    SPINN takes factorized 1-D coordinate arrays as input and computes the
    Cartesian product internally via outer products of per-axis network outputs.

    Input: tuple/list of arrays ``(x1, x2, ..., xd)`` where ``xi`` has shape
        ``(Ni, 1)``.
    Output: array of shape ``(N1 * N2 * ... * Nd, out_dim)``.

    Args:
        body_network: Specification of the per-coordinate sub-network:
            - ``dict``: ``{name: kwargs}`` selects a registered variant and
              passes constructor kwargs.  ``layer_sizes`` must start with 1
              (coordinate-wise) and end with ``rank * out_dim``.
            - ``nn.Module``: a custom Flax module mapping
              ``(N, 1) -> (N, rank * out_dim)``.
        in_dim: Number of spatial dimensions (= length of the input
            coordinate list, must be >= 2).
        rank: Rank of the tensor decomposition.
        out_dim: Number of output components.  Each body network outputs
            ``rank * out_dim`` features; the final output is assembled by
            slicing along the output-component axis.

    References:
        `Cho et al. Separable Physics-Informed Neural Networks. NeurIPS, 2023.
        <https://arxiv.org/abs/2306.15969>`_
    """

    body_network: Any  # dict | nn.Module
    in_dim: int
    rank: int
    out_dim: int

    params: Any = None
    regularization: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        if self.in_dim < 2:
            raise ValueError("SPINN requires in_dim >= 2.")
        # One independent sub-network per spatial coordinate.
        # Re-resolving per axis ensures Flax assigns independent parameters.
        self._bodies = [
            _get_body_network(self.body_network) for _ in range(self.in_dim)
        ]

    @nn.compact
    def __call__(self, inputs, training=False):
        # inputs: list/tuple of 1-D coordinate arrays, one per dimension
        flat_inputs = inputs[0].ndim == 0
        if flat_inputs:
            inputs = [xi.reshape(-1, 1) for xi in inputs]

        if self._input_transform is not None:
            inputs_features = self._input_transform(inputs)
        else:
            inputs_features = inputs

        # Normalize to list form
        if isinstance(inputs_features, (list, tuple)):
            list_inputs = list(inputs_features)
        else:
            raise ValueError("Inputs must be a list or tuple of 1-D coordinate arrays.")

        # Dispatch to dimension-specific tensor product
        if self.in_dim == 2:
            outputs = self._forward_2d(list_inputs)
        elif self.in_dim == 3:
            outputs = self._forward_3d(list_inputs)
        else:
            outputs = self._forward_nd(list_inputs)

        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)

        return outputs.reshape(-1) if flat_inputs else outputs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_bodies(self, inputs):
        """Run each coordinate through its own body sub-network."""
        outputs = []
        for body, X in zip(self._bodies, inputs):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            outputs.append(body(X))
        return outputs

    def _forward_2d(self, inputs):
        flat_inputs = inputs[0].ndim == 1
        if flat_inputs:
            inputs = [xi.reshape(-1, 1) for xi in inputs]
        outputs = self._run_bodies(inputs)
        r = self.rank

        # Outer product per output component over the rank axis
        pred = []
        for i in range(self.out_dim):
            s = slice(r * i, r * (i + 1))
            pred.append(jnp.dot(outputs[0][:, s], outputs[1][:, s].T).reshape(-1))

        if len(pred) == 1:
            return pred[0].reshape(-1) if flat_inputs else pred[0].reshape(-1, 1)
        return (
            jnp.stack(pred, axis=-1).reshape(-1)
            if flat_inputs
            else jnp.stack(pred, axis=-1)
        )

    def _forward_3d(self, inputs):
        flat_inputs = inputs[0].ndim == 1
        if flat_inputs:
            inputs = [xi.reshape(-1, 1) for xi in inputs]
        outputs = self._run_bodies(inputs)
        outputs = [jnp.transpose(o, (1, 0)) for o in outputs]
        r = self.rank

        pred = []
        for i in range(self.out_dim):
            s = slice(r * i, r * (i + 1))
            xy = jnp.einsum("fx,fy->fxy", outputs[0][s], outputs[1][s])
            pred.append(jnp.einsum("fxy,fz->xyz", xy, outputs[2][s]).ravel())

        if len(pred) == 1:
            return pred[0].reshape(-1) if flat_inputs else pred[0].reshape(-1, 1)
        return (
            jnp.stack(pred, axis=-1).reshape(-1)
            if flat_inputs
            else jnp.stack(pred, axis=-1)
        )

    def _forward_nd(self, inputs):
        """General n-dimensional SPINN via repeated einsum."""
        outputs = self._run_bodies(inputs)
        outputs = [jnp.transpose(o, (1, 0)) for o in outputs]
        dim = len(inputs)

        a, b = "za", "zb"
        c = "zab"
        pred = jnp.einsum(f"{a},{b}->{c}", outputs[0], outputs[1])
        for i in range(dim - 2):
            a = c
            b = f"z{chr(97 + i + 2)}"
            c = c + chr(97 + i + 2)
            if i == dim - 3:
                c = c[1:]  # drop leading 'z' on last contraction
            pred = jnp.einsum(f"{a},{b}->{c}", pred, outputs[i + 2])
        return pred.ravel().reshape(-1, 1)
