"""Backend supported: jax (SPINN), pytorch/jax/paddle (PINN).

Comparing Separable PINN (SPINN) and  standard PINN on the linear elasticity problem from
https://doi.org/10.1016/j.cma.2021.113741.
SPINN reference:
- paper: https://arxiv.org/abs/2306.15969
- code: https://github.com/stnamjef/SPINN
"""

import sys
import time

import deepxde as dde
import numpy as np

net_type = sys.argv[1].lower() if len(sys.argv) > 1 else "spinn"
assert net_type in ("spinn", "pinn"), "Choose 'spinn' or 'pinn'"

jax = None
jnp = None

if net_type == "spinn":
    if dde.backend.backend_name != "jax":
        raise ValueError("SPINN requires DDE_BACKEND=jax")
    dde.config.set_default_autodiff("forward")
    import jax
    import jax.numpy as jnp

lmbd = 1.0
mu = 0.5
Q = 4.0

sin = dde.backend.sin
cos = dde.backend.cos
stack = dde.backend.stack

geom = dde.geometry.Rectangle([0, 0], [1, 1])


def func(x):
    ux = np.cos(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    uy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4

    E_xx = -2 * np.pi * np.sin(2 * np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
    E_yy = np.sin(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 3
    E_xy = 0.5 * (
        np.pi * np.cos(2 * np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
        + np.pi * np.cos(np.pi * x[:, 0:1]) * Q * x[:, 1:2] ** 4 / 4
    )

    Sxx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    Syy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    Sxy = 2 * E_xy * mu
    return np.hstack((ux, uy, Sxx, Syy, Sxy))


def hard_BC(x, f):
    if net_type == "spinn":
        assert jnp is not None
        x1 = jnp.atleast_1d(x[0].squeeze())
        x2 = jnp.atleast_1d(x[1].squeeze())
        x1g, x2g = jnp.meshgrid(x1, x2, indexing="ij")
        x = jnp.stack((x1g.ravel(), x2g.ravel()), axis=-1)
    Ux = f[:, 0] * x[:, 1] * (1 - x[:, 1])
    Uy = f[:, 1] * x[:, 0] * (1 - x[:, 0]) * x[:, 1]
    Sxx = f[:, 2] * x[:, 0] * (1 - x[:, 0])
    Syy = f[:, 3] * (1 - x[:, 1]) + (lmbd + 2 * mu) * Q * sin(np.pi * x[:, 0])
    Sxy = f[:, 4]
    return stack((Ux, Uy, Sxx, Syy, Sxy), axis=1)


def fx(x):
    return (
        -lmbd
        * (
            4 * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - mu
        * (
            np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
            - Q * x[:, 1:2] ** 3 * np.pi * cos(np.pi * x[:, 0:1])
        )
        - 8 * mu * np.pi**2 * cos(2 * np.pi * x[:, 0:1]) * sin(np.pi * x[:, 1:2])
    )


def fy(x):
    return (
        lmbd
        * (
            3 * Q * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
            - 2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
        )
        - mu
        * (
            2 * np.pi**2 * cos(np.pi * x[:, 1:2]) * sin(2 * np.pi * x[:, 0:1])
            + (Q * x[:, 1:2] ** 4 * np.pi**2 * sin(np.pi * x[:, 0:1])) / 4
        )
        + 6 * Q * mu * x[:, 1:2] ** 2 * sin(np.pi * x[:, 0:1])
    )


def jacobian_pinn(f, x, i, j):
    j_val = dde.grad.jacobian(f, x, i=i, j=j)
    return j_val[0] if dde.backend.backend_name == "jax" else j_val


def jacobian_spinn(f, x):
    assert jax is not None and jnp is not None
    x1 = jnp.atleast_1d(x[0].squeeze()).reshape(-1, 1)
    x2 = jnp.atleast_1d(x[1].squeeze()).reshape(-1, 1)
    y, y_fn = f

    v1 = jnp.ones_like(x1)
    v2 = jnp.ones_like(x2)
    # SPINN is factorized by coordinate, so two 1D JVPs are the most efficient way
    # to get d/dx and d/dy without building a full reverse-mode Jacobian.
    J_x1 = jax.jvp(lambda x1_: y_fn((x1_, x2)), (x1,), (v1,))[1]
    J_x2 = jax.jvp(lambda x2_: y_fn((x1, x2_)), (x2,), (v2,))[1]

    x1g, x2g = jnp.meshgrid(x1.ravel(), x2.ravel(), indexing="ij")
    x_full = jnp.stack((x1g.ravel(), x2g.ravel()), axis=-1)
    return x_full, y, jnp.stack((J_x1, J_x2), axis=2)


def pde(x, f):
    if net_type == "spinn":
        x_coord, y, J = jacobian_spinn(f, x)

        def jac(i, j):
            return J[:, i : i + 1, j]

    else:
        x_coord = x
        y = f[0] if dde.backend.backend_name == "jax" else f

        def jac(i, j):
            return jacobian_pinn(f, x, i=i, j=j)

    E_xx = jac(0, 0)
    E_yy = jac(1, 1)
    E_xy = 0.5 * (jac(0, 1) + jac(1, 0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = jac(2, 0)
    Syy_y = jac(3, 1)
    Sxy_x = jac(4, 0)
    Sxy_y = jac(4, 1)

    momentum_x = Sxx_x + Sxy_y - fx(x_coord)
    momentum_y = Sxy_x + Syy_y - fy(x_coord)
    stress_x = S_xx - y[:, 2:3]
    stress_y = S_yy - y[:, 3:4]
    stress_xy = S_xy - y[:, 4:5]
    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


if net_type == "spinn":
    N = 50
    pde_points = [np.linspace(0, 1, N, dtype=np.float64).reshape(-1, 1)] * 2
    data = dde.data.SPINNPDE(pde_points=pde_points, pde=pde, bcs=[], solution=func)
    net = dde.nn.SPINN([2, 32, 32, 32, 32, 5], "tanh", "Glorot uniform")
else:
    data = dde.data.PDE(geom, pde, [], num_domain=50**2, solution=func, num_test=10000)
    net = dde.nn.PFNN([2, [36] * 5, [36] * 5, [36] * 5, 5], "tanh", "Glorot uniform")

net.apply_output_transform(hard_BC)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
t0 = time.time()
losshistory, train_state = model.train(iterations=5000, display_every=1000)
t_elapsed = time.time() - t0
print(
    f"\n[{net_type.upper()}] Best L2 relative error: {train_state.best_metrics} | Training time: {t_elapsed:.2f} seconds"
)
