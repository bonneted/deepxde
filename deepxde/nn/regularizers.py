from .. import backend as bkd
from ..backend import backend_name, tf, jax


def get(identifier):
    # TODO: other backends
    if identifier is None:
        return None
    name, scales = identifier[0], identifier[1:]
    if backend_name == "tensorflow":
        return (
            tf.keras.regularizers.l1(l=scales[0])
            if name == "l1"
            else tf.keras.regularizers.l2(l=scales[0])
            if name == "l2"
            else tf.keras.regularizers.l1_l2(l1=scales[0], l2=scales[1])
            if name == "l1+l2"
            else None
        )
    elif backend_name == "jax":
        if name == "l1":
            return lambda params: scales[0] * jax.numpy.sum(
                jax.numpy.concatenate([jax.numpy.abs(w).flatten() for w in params])
            )
        if name == "l2":
            return lambda params: scales[0] * jax.numpy.sum(
                jax.numpy.concatenate([jax.numpy.square(w).flatten() for w in params])
            )
        if name == "l1+l2":
            return lambda params: scales[0] * jax.numpy.sum(
                jax.numpy.concatenate([jax.numpy.abs(w).flatten() for w in params])
            ) + scales[1] * jax.numpy.sum(
                jax.numpy.concatenate([jax.numpy.square(w).flatten() for w in params])
            )
        return None
