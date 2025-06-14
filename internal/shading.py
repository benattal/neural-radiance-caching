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
"""NeRF and its MLPs, with helper functions for construction and rendering."""

from collections import namedtuple
import dataclasses
import functools
import operator
import time
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
from internal import coord
from internal import geopoly
from internal import grid_utils
from internal import image
from internal import math
from internal import ref_utils
from internal import render
from internal import utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


gin.config.external_configurable(math.abs, module='math')
gin.config.external_configurable(math.safe_exp, module='math')
gin.config.external_configurable(math.power_3, module='math')
gin.config.external_configurable(math.laplace_cdf, module='math')
gin.config.external_configurable(math.scaled_softplus, module='math')
gin.config.external_configurable(math.power_ladder, module='math')
gin.config.external_configurable(math.inv_power_ladder, module='math')
gin.config.external_configurable(coord.contract, module='coord')
gin.config.external_configurable(coord.contract_constant, module='coord')
gin.config.external_configurable(coord.contract_radius_5, module='coord')
gin.config.external_configurable(
    coord.contract_radius_2, module='coord'
)
gin.config.external_configurable(coord.contract_cube, module='coord')
gin.config.external_configurable(coord.contract_projective, module='coord')


@gin.configurable
class BaseShader(nn.Module):
  """A PosEnc MLP."""

  config: Any = None  # A Config class, must be set upon construction.

  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.

  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.

  bottleneck_width: int = 256  # The width of the bottleneck vector.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.

  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 4  # Max degree of positional encoding for 3D points.

  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  use_posenc_with_grid: bool = False
  
  num_rgb_channels: int = 3  # The number of RGB channels.
  rgb_premultiplier: float = 1.0  # Premultiplier on RGB before activation.
  rgb_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.0  # The shift added to raw colors pre-activation.
  rgb_bias_diffuse: float = (
      -1.0
  )  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.

  isotropize_gaussians: bool = False  # If True, make Gaussians isotropic.
  gaussian_covariance_scale: float = 1.0  # Amount to scale covariances.
  gaussian_covariance_pad: float = 0.0  # Amount to add to covariance diagonals.

  squash_before: bool = False  # Apply squash before computing density gradient.
  warp_fn: Callable[Ellipsis, Any] = None

  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).

  unscented_mip_basis: str = 'mean'  # Which unscented transform basis to use.
  unscented_sqrt_fn: str = 'sqrtm'  # How to sqrt covariance matrices in the UT.
  unscented_scale_mult: float = 0.0  # Unscented scale, 0 == disabled.

  use_density_feature: bool = True
  affine_density_feature: bool = False
  use_grid: bool = False
  grid_representation: str = 'ngp'
  grid_params: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({})
  )

  normals_target: str = 'normals_to_use'

  backfacing_target: str = 'normals_to_use'
  backfacing_noise: float = 0.0
  backfacing_noise_rate: float = float('inf')
  backfacing_near: float = 1e-1

  def run_network(self, x):
    inputs = x

    # Evaluate network to produce the output density.
    for i in range(self.net_depth):
      x = self.layers[i](x)
      x = self.net_activation(x)

      if i % self.skip_layer == 0 and i > 0:
        x = jnp.concatenate([x, inputs], axis=-1)

    return x

  def predict_appearance_feature(
      self,
      sampler_results,
      train_frac = 1.0,
      train = True,
      **kwargs,
  ):
    means, covs = sampler_results['means'], sampler_results['covs']

    # Note that isotropize and the gaussian scaling/padding is done *before*
    # applying warp_fn. For some applications, applying these steps afterwards
    # instead may be desirable.
    if self.isotropize_gaussians:
      # Replace each Gaussian's covariance with an isotropic version with the
      # same determinant.
      covs = coord.isotropize(covs)

    if self.gaussian_covariance_scale != 1:
      covs *= self.gaussian_covariance_scale

    if self.gaussian_covariance_pad > 0:
      covs += jnp.diag(jnp.full(covs.shape[-1], self.gaussian_covariance_pad))

    x = []

    if self.use_density_feature:
      x.append(sampler_results['feature'])

    # Encode input positions.
    if self.grid is not None:
      control_offsets = kwargs['control_offsets']
      control = means[Ellipsis, None, :] + control_offsets
      perp_mag = kwargs['perp_mag']

      # Add point offset
      if 'point_offset' in kwargs:
        control = control + kwargs['point_offset'][Ellipsis, None, :]

      # Warp
      scale = None

      if not self.squash_before and self.warp_fn is not None:
        if perp_mag is not None and self.unscented_scale_mult > 0:
          if self.warp_fn.__wrapped__ == coord.contract:
            # We can accelerate the contraction a lot by special-casing
            # on the contraction and computing the cube root of the
            # determinant of the Jacobian directly.
            s = coord.contract3_isoscale(control)
            scale = self.unscented_scale_mult * (perp_mag * s)[Ellipsis, None]
            control = self.warp_fn(control)  # pylint: disable=not-callable
          else:
            control, perp_mag = coord.track_isotropic(
                self.warp_fn, control, perp_mag
            )
            scale = self.unscented_scale_mult * perp_mag[Ellipsis, None]
        else:
          control = self.warp_fn(control)  # pylint: disable=not-callable

      x.append(
          self.grid(
              control,
              x_scale=scale,
              per_level_fn=math.average_across_multisamples,
              train=train,
              train_frac=train_frac,
          )
      )

      if self.use_posenc_with_grid:
        # Encode using the strategy used in mip-NeRF 360.
        if not self.squash_before and self.warp_fn is not None:
          means, covs = coord.track_linearize(self.warp_fn, means, covs)

        lifted_means, lifted_vars = coord.lift_and_diagonalize(
            means, covs, self.pos_basis_t
        )

        x.append(
            coord.integrated_pos_enc(
                lifted_means,
                lifted_vars,
                self.min_deg_point,
                self.max_deg_point,
            )
        )

    x = jnp.concatenate(x, axis=-1)
    return self.run_network(x)

  def get_predict_appearance_kwargs(self, rng, rays, sampler_results, **kwargs):
    means, covs = sampler_results['means'], sampler_results['covs']
    predict_appearance_kwargs = {}

    if self.grid is not None:
      # Grid/hash structures don't give us an easy way to do closed-form
      # integration with a Gaussian, so instead we sample each Gaussian
      # according to an unscented transform (or something like it) and average
      # the sampled encodings.
      control_points_key, rng = utils.random_split(rng)

      if 'tdist' in sampler_results:
        control, perp_mag = coord.compute_control_points(
            means,
            covs,
            rays,
            sampler_results['tdist'],
            control_points_key,
            self.unscented_mip_basis,
            self.unscented_sqrt_fn,
            self.unscented_scale_mult,
        )
      else:
        control = means[Ellipsis, None, :]
        perp_mag = jnp.zeros_like(control)

      control_offsets = control - means[Ellipsis, None, :]
      predict_appearance_kwargs['control_offsets'] = control_offsets
      predict_appearance_kwargs['perp_mag'] = perp_mag

    return dict(
        **predict_appearance_kwargs,
        **kwargs,
    )

  def get_bottleneck_feature(
      self,
      rng,
      feature,
  ):
    if self.bottleneck_width > 0:
      # Output of the first part of MLP.
      bottleneck = self.bottleneck_layer(feature)

      # Add bottleneck noise.
      if (rng is not None) and (self.bottleneck_noise > 0):
        key, rng = utils.random_split(rng)
        bottleneck += self.bottleneck_noise * random.normal(key, bottleneck.shape)
    else:
      bottleneck = jnp.zeros_like(feature[Ellipsis, 0:0])

    return bottleneck

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      is_secondary = None,
      is_secondary_noise = None,
      shading_only=False,
      **kwargs,
  ):
    # Appearance model
    key, rng = utils.random_split(rng)
    shading_results = self.predict_appearance(
        rng=key,
        rays=rays,
        sampler_results=sampler_results,
        train_frac=train_frac,
        train=train,
        is_secondary=is_secondary,
        **kwargs,
    )

    # Add random value to colors
    if train and (rng is not None) and self.backfacing_noise > 0:
      # Appearance mask
      dotprod = math.dot(
          sampler_results[self.backfacing_target],
          -rays.directions[Ellipsis, None, :],
      )
      app_mask = dotprod > 0.0

      key, rng = utils.random_split(rng)
      rgb_noise = (
          random.normal(key, shading_results['rgb'].shape)
          * self.backfacing_noise
          * jnp.clip(1.0 - train_frac / self.backfacing_noise_rate, 0.0, 1.0)
      )
      rgb = jnp.maximum(
          rgb_noise + jax.lax.stop_gradient(shading_results['rgb']),
          -float('inf'),
      )

      shading_results['rgb'] = jnp.where(
          app_mask,
          shading_results['rgb'],
          rgb,
      )
    
    # sampler_results = jax.tree_util.tree_map(lambda x: x, sampler_results)
    # sampler_results['normals'] = sampler_results['normals'] * rays.lossmult[..., None, :]
    # sampler_results['normals_to_use'] = sampler_results['normals_to_use'] * rays.lossmult[..., None, :]
    # sampler_results['normals_pred'] = sampler_results['normals_pred'] * rays.lossmult[..., None, :]
    # sampler_results['weights'] = sampler_results['weights'] * rays.lossmult

    # Return
    if shading_only:
      return shading_results
    else:
      return dict(
          **shading_results,
          **{k: v for k, v in sampler_results.items() if k not in shading_results.keys()},
      )
