from typing import Any, Callable, List, Sequence, Union

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers


class FNN(NN):
    """Fully-connected neural network."""

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any
    bias_free: bool = False  # If True, create bias-free dense layers.

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        # TODO: implement get regularizer
        if isinstance(self.activation, list):
            if not (len(self.layer_sizes) - 1) == len(self.activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self._activation = list(map(activations.get, self.activation))
        else:
            self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)

        # Choose the bias initializer only if biases are used.
        bias_initializer = None if self.bias_free else jax.nn.initializers.zeros

        self.denses = [
            nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=bias_initializer,
                use_bias=not self.bias_free,
            )
            for unit in self.layer_sizes[1:]
        ]

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for j, linear in enumerate(self.denses[:-1]):
            x = (
                self._activation[j](linear(x))
                if isinstance(self._activation, list)
                else self._activation(linear(x))
            )
        x = self.denses[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


class PFNN(NN):
    """Parallel fully-connected network that uses independent sub-networks for each
    network output.

    Args:
        layer_sizes: A nested list that defines the architecture of the neural network
            (how the layers are connected). If `layer_sizes[i]` is an int, it represents
            one layer shared by all the outputs; if `layer_sizes[i]` is a list, it
            represents `len(layer_sizes[i])` sub-layers, each of which is exclusively
            used by one output. Every layer_sizes[i] list must have the same length
            (= number of subnetworks). If the last element of `layer_sizes` is an int
            preceded by a list, it must be equal to the number of subnetworks: all
            subnetworks have an output size of 1 and are then concatenated. If the last
            element is a list, it specifies the output size for each subnetwork before
            concatenation.
    """

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        if len(self.layer_sizes) <= 1:
            raise ValueError("must specify input and output sizes")
        if not isinstance(self.layer_sizes[0], int):
            raise ValueError("input size must be integer")

        list_layer = [
            layer_size
            for layer_size in self.layer_sizes
            if isinstance(layer_size, (list, tuple))
        ]
        if not list_layer:  # if there is only one subnetwork (=FNN)
            raise ValueError(
                "no list in layer_sizes, use FNN instead of PFNN for single subnetwork"
            )
        n_subnetworks = len(list_layer[0])
        if not all(len(sublist) == n_subnetworks for sublist in list_layer):
            raise ValueError(
                "all layer_size lists must have the same length(=number of subnetworks)"
            )
        if (
            isinstance(self.layer_sizes[-1], int)
            and n_subnetworks != self.layer_sizes[-1]
            and isinstance(self.layer_sizes[-2], (list, tuple))
        ):
            raise ValueError(
                "if the last element of layer_sizes is an int preceded by a list, it must be equal to the number of subnetworks"
            )

        self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        def make_dense(unit):
            return nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )

        denses = [
            (
                make_dense(unit)
                if isinstance(unit, int)
                else [make_dense(unit[j]) for j in range(n_subnetworks)]
            )
            for unit in self.layer_sizes[1:-1]
        ]

        if isinstance(self.layer_sizes[-1], int):
            if isinstance(self.layer_sizes[-2], (list, tuple)):
                # if output layer size is an int and the previous layer size is a list,
                # the output size must be equal to the number of subnetworks:
                # all subnetworks have an output size of 1 and are then concatenated
                denses.append([make_dense(1) for _ in range(self.layer_sizes[-1])])
            else:
                denses.append(make_dense(self.layer_sizes[-1]))
        else:
            # if the output layer size is a list, it specifies the output size for each subnetwork before concatenation
            denses.append([make_dense(unit) for unit in self.layer_sizes[-1]])

        self.denses = denses  # can't assign directly to self.denses because linen list attributes are converted to tuple
        # see https://github.com/google/flax/issues/524

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for layer in self.denses[:-1]:
            if isinstance(layer, (list, tuple)):
                if isinstance(x, list):
                    x = [self._activation(dense(x_)) for dense, x_ in zip(layer, x)]
                else:
                    x = [self._activation(dense(x)) for dense in layer]
            else:
                if isinstance(x, list):
                    x = jnp.concatenate(x, axis=0 if x[0].ndim == 1 else 1)
                x = self._activation(layer(x))

        # output layers
        if isinstance(x, list):
            x = jnp.concatenate(
                [f(x_) for f, x_ in zip(self.denses[-1], x)],
                axis=0 if x[0].ndim == 1 else 1,
            )
        else:
            x = self.denses[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    
class MixNN(NN):
    """
    A neural network container that applies a list of different neural networks
    (e.g., FNN or PFNN instances) to a corresponding list of inputs.

    Each network in the provided list is applied to the corresponding item
    in the input list during the forward pass.
    """
    # The list of networks should be passed during initialization.
    # Flax will automatically assign it as an attribute.
    networks: Sequence[Union[FNN, PFNN]]
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        """
        Validates the provided networks list.

        In Flax, `setup` is primarily for *creating* submodules based on configuration
        passed at initialization (like layer_sizes). Here, the submodules (networks)
        are *already created* and passed in directly. So, setup primarily serves
        as a validation step.
        """
        if not isinstance(self.networks, (list, tuple)):
             raise TypeError(f"Expected 'networks' to be a list or tuple, but got {type(self.networks)}")
        if not self.networks:
            raise ValueError("'networks' list cannot be empty.")
        for i, net in enumerate(self.networks):
            if not isinstance(net, (FNN, PFNN)):
                # You might want to make this check more robust depending on
                # the exact base class or expected types.
                raise TypeError(
                    f"Network at index {i} is not an instance of FNN or PFNN. "
                    f"Got type: {type(net)}"
                )
        # No new layers need to be created *within* MixNN itself.

    def __call__(self, inputs: List[Any], training: bool = False) -> List[Any]:
        """
        Applies each network to its corresponding input.

        Args:
            inputs: A list of input tensors/arrays. The length of this list
                    must match the number of networks provided during initialization.
                    Each element `inputs[i]` is passed to `self.networks[i]`.
            training: A boolean indicating if the model is in training mode. This
                      flag is passed down to the individual networks.

        Returns:
            A list of outputs, where each element `outputs[i]` is the result of
            applying `self.networks[i]` to `inputs[i]`.
        """
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(f"Inputs to MixNN must be a list or tuple, got {type(inputs)}.")
        if len(inputs) != len(self.networks):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) must match the number of "
                f"networks ({len(self.networks)})."
            )

        # 1. Apply MixNN's global input transform (operates on the list)
        current_inputs = self._input_transform(inputs) if self._input_transform is not None else inputs

        if len(current_inputs) != len(self.networks):
             raise ValueError(f"MixNN input_transform must return a list of the same length ({len(self.networks)}). Got {len(current_inputs)}")


        outputs = []
        # 2. Iterate through networks and apply them to corresponding processed inputs
        for i, net in enumerate(self.networks):
            # The network 'net' will apply its *own* internal input/output transforms
            # using current_inputs[i] as its starting point.
            output = net(current_inputs[i], training=training)
            outputs.append(output)

        # 3. Apply MixNN's global output transform (operates on the list)
        final_outputs = self._output_transform(inputs, outputs) if self._output_transform is not None else outputs

        return final_outputs