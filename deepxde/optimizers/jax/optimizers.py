__all__ = ["get", "is_external_optimizer", "apply_updates"]

import jax
import optax
import warnings


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

    # Use decay[1] as init value for all decay schedules
    lr_init = decay[1] if len(decay) > 1 else lr

    # Add optax learning rate schedules (decay[0] = type, decay[1] = init_value)
    decay_type = decay[0].lower()
    if decay_type == "inverse time":
        # optax.inverse_time_decay(init_value, decay_rate, transition_begin, staircase)
        # decay[2]: decay_rate (optional), decay[3]: transition_begin (optional), decay[4]: staircase (optional)
        decay_rate = decay[2] if len(decay) > 2 else 0.1
        transition_begin = decay[3] if len(decay) > 3 else 0
        staircase = decay[4] if len(decay) > 4 else False
        return optax.inverse_time_decay(lr_init, decay_rate, transition_begin, staircase)
    elif decay_type == "exponential":
        # optax.exponential_decay(init_value, transition_steps, decay_rate, staircase)
        # decay[2]: transition_steps (optional), decay[3]: decay_rate (optional), decay[4]: staircase (optional)
        transition_steps = decay[2] if len(decay) > 2 else 1000
        decay_rate = decay[3] if len(decay) > 3 else 0.99
        staircase = decay[4] if len(decay) > 4 else False
        return optax.exponential_decay(lr_init, transition_steps, decay_rate, staircase)
    elif decay_type == "warmup exponential":
        # optax.warmup_exponential_decay_schedule(init_value, peak_value, warmup_steps, transition_steps, decay_rate, staircase)
        # decay[2]: peak_value (optional), decay[3]: warmup_steps (optional), decay[4]: transition_steps (optional),
        # decay[5]: decay_rate (optional), decay[6]: staircase (optional)
        peak_value = decay[2] if len(decay) > 2 else lr_init * 10
        warmup_steps = decay[3] if len(decay) > 3 else 1000
        transition_steps = decay[4] if len(decay) > 4 else 1000
        decay_rate = decay[5] if len(decay) > 5 else 0.9
        staircase = decay[6] if len(decay) > 6 else False
        return optax.warmup_exponential_decay_schedule(
            init_value=lr_init,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
    elif decay_type == "cosine":
        # optax.cosine_decay_schedule(init_value, decay_steps)
        # decay[2]: decay_steps (optional)
        decay_steps = decay[2] if len(decay) > 2 else 10000
        alpha = decay[3] if len(decay) > 3 else 0.0 #end value
        return optax.cosine_decay_schedule(lr_init, decay_steps, alpha)
    elif decay_type == "warmup cosine":
        # optax.warmup_cosine_decay_schedule(init_value, peak_value, warmup_steps, decay_steps, end_value)
        # decay[2]: peak_value (optional), decay[3]: warmup_steps (optional), decay[4]: decay_steps (optional), decay[5]: end_value (optional)
        peak_value = decay[2] if len(decay) > 2 else lr_init * 10
        warmup_steps = decay[3] if len(decay) > 3 else 1000
        decay_steps = decay[4] if len(decay) > 4 else 10000
        end_value = decay[5] if len(decay) > 5 else 0.0
        return optax.warmup_cosine_decay_schedule(
            init_value=lr_init,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )
    else:
        raise NotImplementedError(
            f"{decay[0]} learning rate decay to be implemented for backend jax."
        )
