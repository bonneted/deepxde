"""Quick JAX LR decay comparison using Poisson 1D exact-BC setup.

Run with:
  DDE_BACKEND=jax /home/r2d2/miniconda3/envs/jax/bin/python examples/pinn_forward/jax_lr_decay_quick_compare.py
"""

import time

import deepxde as dde
import jax
import numpy as np

sin = jax.numpy.sin


geom = dde.geometry.Interval(0, np.pi)


def pde(x, y):
    dy_xx, _ = dde.grad.hessian(y, x)
    summation = sum(i * sin(i * x) for i in range(1, 5))
    return -dy_xx - summation - 8 * sin(8 * x)


def func(x):
    summation = sum(np.sin(i * x) / i for i in range(1, 5))
    return x + summation + np.sin(8 * x) / 8


def output_transform(x, y):
    return x * (np.pi - x) * y + x


def build_model():
    data = dde.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=400)
    net = dde.nn.FNN([1, 50, 50, 50, 1], "tanh", "Glorot uniform")
    net.apply_output_transform(output_transform)
    return dde.Model(data, net)


def run_case(name, decay, lr, iterations):
    model = build_model()
    model.compile("adam", lr=lr, decay=decay, metrics=["l2 relative error"], verbose=0)

    start = time.time()
    _, train_state = model.train(iterations=iterations, verbose=0)
    wall = time.time() - start

    return {
        "name": name,
        "final_train_loss": float(np.sum(train_state.loss_train)),
        "final_test_loss": float(np.sum(train_state.loss_test)),
        "l2_rel": float(train_state.metrics_test[0]),
        "seconds": wall,
    }


def main():
    lr = 1e-3
    iterations = 2000

    cases = [
        ("linear", ("linear", {"end_value": lr * 0.1, "transition_steps": iterations})),
        ("cosine", ("cosine", {"decay_steps": iterations, "alpha": 0.1})),
        (
            "exponential",
            ("exponential", {"transition_steps": max(1, iterations // 10), "decay_rate": 0.95}),
        ),
        (
            "warmup_cosine",
            (
                "warmup_cosine",
                {
                    "peak_value": lr,
                    "warmup_steps": max(1, iterations // 10),
                    "decay_steps": iterations,
                    "end_value": lr * 0.1,
                },
            ),
        ),
        (
            "warmup_exponential",
            (
                "warmup_exponential",
                {
                    "peak_value": lr,
                    "warmup_steps": max(1, iterations // 10),
                    "transition_steps": max(1, iterations // 10),
                    "decay_rate": 0.95,
                    "end_value": lr * 0.1,
                },
            ),
        ),
    ]

    results = []
    print(f"Backend: {dde.backend.backend_name}, lr={lr}, iterations={iterations}")
    for name, decay in cases:
        print(f"Running {name}...")
        results.append(run_case(name, decay, lr, iterations))

    print("\n=== Quick comparison (lower is better) ===")
    print(f"{'decay':20s} {'train_loss':>12s} {'test_loss':>12s} {'l2_rel':>12s} {'sec':>8s}")
    for r in sorted(results, key=lambda x: x["l2_rel"]):
        print(
            f"{r['name']:20s} {r['final_train_loss']:12.4e} {r['final_test_loss']:12.4e} "
            f"{r['l2_rel']:12.4e} {r['seconds']:8.2f}"
        )


if __name__ == "__main__":
    if dde.backend.backend_name != "jax":
        raise RuntimeError("Please run with DDE_BACKEND=jax.")
    main()
