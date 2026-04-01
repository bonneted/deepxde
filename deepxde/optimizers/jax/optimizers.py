__all__ = ["get", "is_external_optimizer", "apply_updates"]

import jax
import optax
from optax import schedules


apply_updates = optax.apply_updates


def is_external_optimizer(optimizer):
    # TODO: add external optimizers
    return False


def get(optimizer, learning_rate=None, decay=None):
    """Retrieves an optax Optimizer instance."""
    if isinstance(optimizer, optax._src.base.GradientTransformation):
        return optimizer
    if is_external_optimizer(optimizer):
        raise NotImplementedError(f"{optimizer} to be implemented for backend jax.")

    if learning_rate is None:
        raise ValueError("No learning rate for {}.".format(optimizer))

    lr_schedule = _get_learningrate(learning_rate, decay)
    if optimizer == "adam":
        return optax.adam(lr_schedule)
    if optimizer == "rmsprop":
        return optax.rmsprop(lr_schedule)
    if optimizer == "sgd":
        return optax.sgd(lr_schedule)

    raise NotImplementedError(f"{optimizer} to be implemented for backend jax.")


def _get_learningrate(lr, decay):
    if decay is None:
        return lr
    if callable(decay):
        return decay

    schedule_map = {
        "linear": schedules.linear_schedule,
        "cosine": schedules.cosine_decay_schedule,
        "exponential": schedules.exponential_decay,
        "warmup_cosine": schedules.warmup_cosine_decay_schedule,
        "warmup_exponential": schedules.warmup_exponential_decay_schedule,
    }

    # Preferred format: decay = ("schedule_name", {"kwarg": value, ...})
    # Legacy tuple formats are still supported below.
    if isinstance(decay, tuple) and len(decay) == 2 and isinstance(decay[1], dict):
        name, kwargs = decay
    elif isinstance(decay, tuple) and len(decay) == 3 and decay[0] == "linear":
        name, kwargs = "linear", {
            "init_value": lr,
            "end_value": decay[1],
            "transition_steps": decay[2],
        }
    elif isinstance(decay, tuple) and len(decay) == 3 and decay[0] == "cosine":
        name, kwargs = "cosine", {
            "init_value": lr,
            "decay_steps": decay[1],
            "alpha": decay[2],
        }
    elif isinstance(decay, tuple) and len(decay) == 3 and decay[0] == "exponential":
        name, kwargs = "exponential", {
            "init_value": lr,
            "transition_steps": decay[1],
            "decay_rate": decay[2],
        }
    else:
        raise ValueError(
            "For JAX, use decay=(name, kwargs) with Optax schedule kwargs, "
            "or a legacy tuple for linear/cosine/exponential. See "
            "https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html"
        )

    # Keep a simple default: if schedule accepts init_value, use lr unless provided.
    if isinstance(kwargs, dict) and "init_value" not in kwargs:
        kwargs = {"init_value": lr, **kwargs}

    try:
        return schedule_map[name](**kwargs)
    except KeyError as exc:
        raise NotImplementedError(
            f"Unknown JAX decay schedule '{name}'. Supported schedules: "
            f"{list(schedule_map.keys())}. See "
            "https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html"
        ) from exc
    except TypeError as exc:
        raise ValueError(
            f"Invalid kwargs for JAX decay schedule '{name}': {kwargs}. See "
            "https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html"
        ) from exc
