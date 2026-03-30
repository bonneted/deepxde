"""Separable Physics-Informed Neural Networks (SPINN) for JAX.

This code is adapted from the original SPINN paper:
- paper: https://arxiv.org/abs/2306.15969
- code: https://github.com/stnamjef/SPINN
"""

from typing import Any, Callable

import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers


class SPINN(NN):
    """Separable Physics-Informed Neural Network.

    SPINN takes factorized 1D coordinate arrays as input and computes the
    Cartesian product internally via outer products of per-axis network outputs.

    Input: tuple/list of arrays ``(x1, x2, ..., xd)`` where ``xi`` has shape
        ``(Ni, 1)``.
    Output: array of shape ``(N1 * N2 * ... * Nd, d_out)``.

    Args:
        layer_sizes: ``[d_in, hidden1, ..., hiddenK, rank, d_out]``.
            ``d_in`` is the number of spatial dimensions. ``rank`` is the
            rank of the tensor decomposition. ``d_out`` is the number of
            output components.
        activation: Activation function name or list of names.
        kernel_initializer: Weight initializer name.
        mlp: Network variant — ``"mlp"`` (standard) or ``"modified_mlp"``.

    References:
        `Cho et al. Separable Physics-Informed Neural Networks. NeurIPS, 2023.
        <https://arxiv.org/abs/2306.15969>`_
    """

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any
    mlp: str = "mlp"

    params: Any = None
    regularization: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        self.in_dim = self.layer_sizes[0]
        self.r = self.layer_sizes[-2]  # rank
        self.out_dim = self.layer_sizes[-1]
        self.init = initializers.get(self.kernel_initializer)
        self.features = self.layer_sizes[1:-2]

        if isinstance(self.activation, (list, tuple)):
            if len(self.layer_sizes) - 1 != len(self.activation):
                raise ValueError(
                    "Number of activation functions does not match "
                    "the number of layers."
                )
            self._activation = list(map(activations.get, self.activation))
        else:
            self._activation = [activations.get(self.activation)] * (
                len(self.layer_sizes) - 1
            )

    @nn.compact
    def __call__(self, inputs, training=False):
        # inputs: tuple/list of arrays, one per dimension
        flat_inputs = inputs[0].ndim == 0
        if flat_inputs:
            inputs = [xi.reshape(-1, 1) for xi in inputs]

        if self._input_transform is not None:
            inputs_features = self._input_transform(inputs)
        else:
            inputs_features = inputs

        if isinstance(inputs_features, (list, tuple)):
            list_inputs = list(inputs_features)
        else:
            list_inputs = []
            for i in range(self.in_dim):
                if inputs_features.ndim == 1:
                    list_inputs.append(inputs_features[i : i + 1])
                else:
                    list_inputs.append(inputs_features[:, i : i + 1])

        if self.in_dim < 2:
            raise ValueError("SPINN requires input dimension >= 2.")

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
    # Internal forward methods
    # ------------------------------------------------------------------

    def _body_networks(self, inputs):
        """Run each coordinate through its own sub-network."""
        outputs = []
        for X in inputs:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if self.mlp == "mlp":
                for i, fs in enumerate(self.features):
                    X = nn.Dense(fs, kernel_init=self.init)(X)
                    X = self._activation[i](X)
                X = nn.Dense(self.r * self.out_dim, kernel_init=self.init)(X)
            else:  # modified_mlp
                U = jnp.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                V = jnp.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                H = jnp.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                for fs in self.features:
                    Z = nn.Dense(fs, kernel_init=self.init)(H)
                    Z = jnp.tanh(Z)
                    H = (jnp.ones_like(Z) - Z) * U + Z * V
                X = nn.Dense(self.r * self.out_dim, kernel_init=self.init)(H)
            outputs.append(X)
        return outputs

    def _forward_2d(self, inputs):
        flat_inputs = inputs[0].ndim == 1
        if flat_inputs:
            inputs = [xi.reshape(-1, 1) for xi in inputs]
        outputs = self._body_networks(inputs)

        pred = []
        for i in range(self.out_dim):
            s = slice(self.r * i, self.r * (i + 1))
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
        outputs = self._body_networks(inputs)
        outputs = [jnp.transpose(o, (1, 0)) for o in outputs]

        pred = []
        for i in range(self.out_dim):
            s = slice(self.r * i, self.r * (i + 1))
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
        outputs = self._body_networks(inputs)
        outputs = [jnp.transpose(o, (1, 0)) for o in outputs]
        dim = len(inputs)

        # Build einsum chain: (z,a) x (z,b) -> (z,a,b) x (z,c) -> ...
        a, b = "za", "zb"
        c = "zab"
        pred = jnp.einsum(f"{a},{b}->{c}", outputs[0], outputs[1])
        for i in range(dim - 2):
            a = c
            b = f"z{chr(97 + i + 2)}"
            c = c + chr(97 + i + 2)
            if i == dim - 3:
                c = c[1:]  # remove leading 'z' on last contraction
            pred = jnp.einsum(f"{a},{b}->{c}", pred, outputs[i + 2])
        return pred.ravel().reshape(-1, 1)
