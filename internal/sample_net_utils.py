# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from flax import linen as nn
import gin
from internal import configs
from internal import coord
import jax
import jax.numpy as jnp

gin.config.external_configurable(coord.contract, module="coord")
gin.config.external_configurable(coord.contract_cube, module="coord")
gin.config.external_configurable(coord.contract_projective, module="coord")
gin.config.external_configurable(coord.inv_contract, module="coord")
gin.config.external_configurable(coord.inv_contract_cube, module="coord")
gin.config.external_configurable(coord.inv_contract_projective, module="coord")


def normalize(x):
  """Normalization helper function."""
  return x / jnp.linalg.norm(x, axis=-1, keepdims=True)


def posenc(points, degree):
  """Positional encoding mapping for 3D input points."""
  sh = list(points.shape[:-1])
  points = points[Ellipsis, None] * 2.0 ** jnp.arange(degree)
  points = jnp.concatenate([jnp.cos(points), jnp.sin(points)], -1)
  return points.reshape(sh + [-1])


def ease_activation(window_iters, act, val=1.0):
  def new_act(cur_iter, x):
    if window_iters > 0:
      w = jnp.clip(cur_iter / window_iters, 0.0, 1.0)
      return (1 - w) * val + w * act(x)
    else:
      return act(x)

  return new_act


@gin.configurable
class SampleNetwork(nn.Module):
  config: Any = None  # A Config class, must be set upon construction.
  aabb: Tuple[int, int, int, int, int, int] = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
  window_frac: int = 0
  num_views: int = 1
  use_viewdirs: bool = True
  use_time: bool = False
  mlp_width: int = 256
  mlp_depth: int = 4
  contract_fn: Callable[Ellipsis, Any] = lambda x: x
  inv_contract_fn: Callable[Ellipsis, Any] = lambda x: x

  def normalize_inputs(self, points, origins, viewdirs):
    aabb_min, aabb_max = jnp.split(jnp.array(self.aabb), 2, axis=-1)

    points = (points - aabb_min) / (aabb_max - aabb_min)
    points = points * 4.0 - 2.0

    origins = (origins - aabb_min) / (aabb_max - aabb_min)
    origins = origins * 4.0 - 2.0

    rays_norm = jnp.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = normalize(viewdirs / (aabb_max - aabb_min)) * rays_norm

    return points, origins, viewdirs

  def unnormalize_points(self, points):
    aabb_min, aabb_max = jnp.split(jnp.array(self.aabb), 2, axis=-1)

    points = points / 4.0 + 0.5
    points = points * (aabb_max - aabb_min) + aabb_min

    return points

  def unnormalize_viewdirs(self, viewdirs):
    aabb_min, aabb_max = jnp.split(jnp.array(self.aabb), 2, axis=-1)

    rays_norm = jnp.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = normalize(viewdirs * (aabb_max - aabb_min)) * rays_norm

    return viewdirs

  def get_network_input(self, points, viewdirs, times):
    net_input = posenc(points, 4)

    if self.use_viewdirs:
      net_input = jnp.concatenate([net_input, posenc(viewdirs, 2)], axis=-1)

    if self.use_time:
      net_input = jnp.concatenate([net_input, posenc(times, 6)], axis=-1)

    return jax.lax.stop_gradient(net_input)

  @nn.compact
  def __call__(
      self,
      train_frac,
      points_uncontract,
      origins_uncontract,
      viewdirs,
      t_idx
  ):
    # Create Outputs, Activations, MLP
    outputs = {
        "z_vals": 1,
        "point_offset": 3,
        "sigma": 1,
        "point_sigma": 1,
    }

    activations = [
        lambda train_frac, x: jax.nn.tanh(x * 0.25) * 0.125,
        lambda train_frac, x: jax.nn.tanh(x * 1.0) * 0.25,
        ease_activation(
            self.window_frac, lambda x: jax.nn.sigmoid(x + 3.0), 1.0
        ),
        ease_activation(
            self.window_frac, lambda x: jax.nn.sigmoid(x + 3.0), 1.0
        ),
    ]

    output_names = list(outputs.keys())
    output_shapes = [outputs[k] for k in outputs.keys()]
    output_slices = [sum(output_shapes[:k]) for k in range(len(output_shapes))][
        1:
    ]
    output_channels = sum(output_shapes)

    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, "he_uniform")()
    )

    # Times
    times = (t_idx / self.num_views) * 2 - 1

    # Normalize to bounding box
    points_uncontract_original = points_uncontract
    points_uncontract, origins_uncontract, viewdirs = self.normalize_inputs(
        points_uncontract, origins_uncontract, viewdirs
    )

    # Reshape
    points_uncontract = points_uncontract.reshape(
        -1, points_uncontract.shape[-1]
    )
    origins_uncontract = origins_uncontract.reshape(
        -1, origins_uncontract.shape[-1]
    )
    viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])

    # Get distances
    rays_norm = jnp.linalg.norm(viewdirs, axis=-1, keepdims=True)
    dists_uncontract = (
        jnp.linalg.norm(
            points_uncontract - origins_uncontract, axis=-1, keepdims=True
        )
        / rays_norm
    )

    # Contract distances and points
    dists_contract = self.contract_fn(dists_uncontract)
    points_contract = self.contract_fn(points_uncontract)
    points_contract_original = points_contract

    # Run network
    net_input = self.get_network_input(points_contract, viewdirs, times)
    net_output = net_input

    for i in range(self.mlp_depth):
      net_output = nn.relu(
          dense_layer(self.mlp_width, name=f"layer_{i}")(net_output)
      )

    net_output = dense_layer(output_channels, name="output_layer")(net_output)

    # Get output
    output = {}
    all_outputs = jnp.split(net_output, output_slices, axis=-1)

    for i, sh in enumerate(output_shapes):
      cur_output = activations[i](train_frac, all_outputs[i])
      output[output_names[i]] = cur_output

    # Add distances
    dist_offset_contract = output["z_vals"] * (1.0 - output["sigma"])
    dists_uncontract = self.inv_contract_fn(
        dists_contract + dist_offset_contract
    )
    points_uncontract = origins_uncontract + viewdirs * dists_uncontract

    # Contract points
    points_contract = self.contract_fn(points_uncontract)

    # Add point offsets
    point_offset_contract = output["point_offset"] * (
        1.0 - output["point_sigma"]
    )
    points_contract = points_contract + point_offset_contract

    # Unnormalize and uncontract
    points_uncontract = self.inv_contract_fn(points_contract)
    points_uncontract = self.unnormalize_points(points_uncontract)

    return dict(
        point_offset=points_uncontract_original - points_uncontract,
        point_offset_contract=points_contract_original - points_contract,
    )


def pluecker(origins, directions, **kwargs):
  directions = normalize(directions)
  moment = jnp.cross(origins, directions, axis=-1)
  return jnp.concatenate([directions, moment], axis=-1)


def dot(a, b, axis=-1, keepdims=False):
  return (a * b).sum(axis=axis, keepdims=keepdims)


def intersect_sphere(
    origins,
    directions,
    radius,
    continuous=False,
):
  o = origins
  d = directions

  dot_o_o = dot(o, o)
  dot_d_d = dot(d, d)
  dot_o_d = dot(o, d)

  a = dot_d_d
  b = 2 * dot_o_d
  c = dot_o_o - radius * radius
  disc = b * b - 4 * a * c

  disc = jnp.where(disc < 0, jnp.zeros_like(disc), disc)

  t1 = (-b + jnp.sqrt(disc + 1e-8)) / (2 * a)
  t2 = (-b - jnp.sqrt(disc + 1e-8)) / (2 * a)

  t1 = jnp.where(disc <= 0, jnp.zeros_like(t1), t1)
  t2 = jnp.where(disc <= 0, jnp.zeros_like(t2), t2)

  return (t1, t2)


def sphere_origins(origins, directions, **kwargs):
  _, t2 = intersect_sphere(origins, directions, radius=kwargs["radius"])
  return origins + directions * t2[Ellipsis, None]
