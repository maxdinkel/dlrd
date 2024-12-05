import functools

import jax.numpy as jnp
import jax
import numpy as np

from queens.models import SimulationModel

jax.config.update("jax_enable_x64", True)

num_nodes_1d = 21
num_nodes = num_nodes_1d ** 2

nodes_x = np.linspace(0.0, 1.0, num_nodes_1d)
meshgrid = np.meshgrid(nodes_x, nodes_x)
nodes = np.vstack([meshgrid[0].ravel(), meshgrid[1].ravel()]).T

connectivity = []
for i in range(num_nodes_1d - 1):
    for j in range(num_nodes_1d - 1):
        node_0 = i * num_nodes_1d + j
        connectivity.append([node_0, node_0 + 1, node_0 + 1 + num_nodes_1d, node_0 + num_nodes_1d])
connectivity = np.array(connectivity)

np.random.seed(2)
dists = nodes.reshape(-1, 1, 2) - nodes.reshape(1, -1, 2)
lengthscale = 0.2
signal_std = 1.0
dist_norm = np.sum(dists ** 2 / (2 * lengthscale ** 2), axis=2)
cov = signal_std ** 2 * np.exp(-dist_norm)
eig_val, eig_vec = np.linalg.eigh(cov)
eig_val = np.flip(eig_val)
eig_vec = np.flip(eig_vec, axis=1)
num_components = int(np.argmax((np.cumsum(eig_val) / np.sum(eig_val)) > 0.99)) + 1
weighted_eig_vec = eig_vec[:, :num_components] * np.sqrt(eig_val[:num_components])
print(f'Num components: {num_components}, {(np.cumsum(eig_val) / np.sum(eig_val))[num_components]}')

source_term = 10


def shape_fun(xi):
    return 0.25 * jnp.array([
        (1.0 - xi[0]) * (1.0 - xi[1]),
        (1.0 + xi[0]) * (1.0 - xi[1]),
        (1.0 + xi[0]) * (1.0 + xi[1]),
        (1.0 - xi[0]) * (1.0 + xi[1])
    ])


def grad_shape_fun(xi):
    return 0.25 * jnp.array([
        [-(1.0 - xi[1]), (1.0 - xi[1]), (1.0 + xi[1]), -(1.0 + xi[1])],
        [-(1.0 - xi[0]), -(1.0 + xi[0]), (1.0 + xi[0]), (1.0 - xi[0])]
    ])


x_ele = nodes[connectivity, :]


# -------rhs--------
gauss_point = np.zeros(2)
dshape_dxi = grad_shape_fun(gauss_point)
jacobian = np.tensordot(x_ele, dshape_dxi, axes=(1, 1))
shape = shape_fun(gauss_point)
rhs_ele = 4 * source_term * shape.reshape(1, -1) * jnp.linalg.det(jacobian).reshape(-1, 1)

rhs = np.zeros(num_nodes)
for i in range(len(connectivity)):
    rhs[connectivity[i]] += rhs_ele[i]
# --------------------


gauss_points = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) / np.sqrt(3)
dshape_dxi = grad_shape_fun(gauss_points.T).T.swapaxes(1, 2)
jacobian = np.tensordot(x_ele, dshape_dxi, axes=(1, 2)).swapaxes(1, 2)
dshape_dx = np.sum(
    np.linalg.inv(jacobian)[:, :, :, :, np.newaxis] * dshape_dxi[np.newaxis, :, np.newaxis, :, :],
    axis=3
)
dshape_dx_dshape_dx_det_jacobian = jnp.sum(
    dshape_dx[:, :, :, jnp.newaxis, :] * dshape_dx[:, :, :, :, jnp.newaxis], axis=2
) * np.linalg.det(jacobian)[:, jnp.newaxis, :, jnp.newaxis]

indices = np.where((nodes[:, 0] != 0.0) & (nodes[:, 0] != 1.0) &
                   (nodes[:, 1] != 0.0) & (nodes[:, 1] != 1.0))[0]

connectivity = jnp.array(connectivity)


def reconstruct_diffusivity(coefficients):
    return jnp.exp(jnp.dot(coefficients, weighted_eig_vec.T))


def diffusion_model(coefficients):
    diffusivity = reconstruct_diffusivity(coefficients)
    kappa_ele = diffusivity[connectivity]

    kappa = jnp.dot(kappa_ele, shape_fun(gauss_points.T))

    k_ele = jnp.sum(dshape_dx_dshape_dx_det_jacobian * kappa[:, :, jnp.newaxis, jnp.newaxis], axis=1)

    k = jnp.zeros((num_nodes, num_nodes))
    for m in range(len(connectivity)):
        k = k.at[tuple(jnp.meshgrid(connectivity[m], connectivity[m]))].add(k_ele[m])

    u = jnp.linalg.solve(k[indices][:, indices], rhs[indices])

    return u


def value_and_jacfwd(inputs):
    pushfwd = functools.partial(jax.jvp, diffusion_model, (inputs,))
    basis = jnp.eye(inputs.size, dtype=inputs.dtype)
    output, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return output, jac


jitted_model_function = jax.jit(value_and_jacfwd)


class DiffusionModel(SimulationModel):
    def __init__(self):
        self.response = None

    def evaluate(self, samples):
        solution, gradient = jax.vmap(jitted_model_function)(samples)
        self.response = {'result': np.array(solution),
                         'gradient': np.array(gradient).swapaxes(1, 2)}

        return self.response
