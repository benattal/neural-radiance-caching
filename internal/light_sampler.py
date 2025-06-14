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
from internal import math
from internal import ref_utils
from internal import shading
from internal import utils
from internal.inverse_render import render_utils
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
gin.config.external_configurable(coord.contract, module="coord")
gin.config.external_configurable(coord.contract_constant, module="coord")
gin.config.external_configurable(coord.contract_radius_5, module="coord")
gin.config.external_configurable(coord.contract_radius_2, module="coord")
gin.config.external_configurable(coord.contract_radius_1_2, module='coord')
gin.config.external_configurable(coord.contract_radius_1_4, module='coord')
gin.config.external_configurable(coord.contract_cube, module='coord')
gin.config.external_configurable(coord.contract_cube_5, module='coord')
gin.config.external_configurable(coord.contract_cube_2, module='coord')
gin.config.external_configurable(coord.contract_cube_1_4, module='coord')
gin.config.external_configurable(coord.contract_projective, module='coord')


@gin.configurable
class LightMLP(shading.BaseShader):
  """A PosEnc MLP."""

  config: Any = None

  num_components: int = 64  # Learned BRDF layer width
  vmf_scale: float = 20.0  #  Learned BRDF layer depth
  random_seed: int = 1

  vmf_bias: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({
      'vmf_means': 0.0,
      'vmf_kappas': 1.0,
      'vmf_logits': 1.0,
  })
  vmf_activation: ml_collections.FrozenConfigDict = (
      ml_collections.FrozenConfigDict({
          'vmf_means': lambda x: x,
          'vmf_kappas': lambda x: jnp.minimum(jax.nn.softplus(x), 50.0),
          'vmf_logits': lambda x: jnp.maximum(x, -50.0),
      })
  )

  normals_target: str = 'normals_to_use'

  num_light_features: int = 64  # GLO vector length, disabled if 0.
  use_illumination_feature: bool = False  # GLO vector length, disabled if 0.
  multiple_illumination_outputs: bool = True  # GLO vector length, disabled if 0.

  def setup(self):
    self.dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
    )

    # Light idx
    if self.config.multi_illumination:
      self.light_vecs = nn.Embed(
          self.config.num_illuminations, self.num_light_features, name='light_vecs'
      )

      if self.config.multiple_illumination_outputs:
        self.num_illumination_outputs = self.config.num_illuminations
      else:
        self.num_illumination_outputs = 1
    else:
      self.num_illumination_outputs = 1

    # VMF prediction
    self.layers = [
        self.dense_layer(self.net_width) for i in range(self.net_depth)
    ]

    self.output_layer = self.dense_layer(self.num_components * self.num_illumination_outputs * 5)

    # Grid
    if self.use_grid:
      self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
          self.grid_representation.lower()
      ](name='light_grid', **self.grid_params)
    else:
      self.grid = None

  def get_light_vec(self, rays, feature):
    light_vec = jnp.zeros_like(feature[..., 0:0])

    if self.config.multi_illumination > 0:
      light_idx = rays.light_idx[Ellipsis, 0]
      light_vec = self.light_vecs(light_idx)
      light_vec = light_vec[..., None, :] * jnp.ones_like(feature[..., 0:1])

    return light_vec

  def get_vmfs(self, vmf_params):
    rng = random.PRNGKey(self.random_seed)

    means_key, rng = utils.random_split(rng)
    # kappas_key, rng = utils.random_split(rng)
    # weights_key, rng = utils.random_split(rng)

    means_random = jax.random.normal(
        means_key, shape=vmf_params.shape[:-1] + (3,)
    ) * self.vmf_scale / 2.0

    vmfs = {
        'vmf_means': self.vmf_activation['vmf_means'](
            vmf_params[Ellipsis, 0:3] * self.vmf_scale
            + self.vmf_bias['vmf_means']
            + means_random
        ),
        'vmf_kappas': self.vmf_activation['vmf_kappas'](
            vmf_params[Ellipsis, 3:4] + self.vmf_bias['vmf_kappas']
        ),
        'vmf_logits': self.vmf_activation['vmf_logits'](
            vmf_params[Ellipsis, 4:5] + self.vmf_bias['vmf_logits']
        ),
    }

    return vmfs

  def predict_lighting(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      zero_glo = False,
      **kwargs,
  ):
    outputs = {}

    means, covs = sampler_results['means'], sampler_results['covs']
    viewdirs = rays.viewdirs

    # Appearance feature
    key, rng = utils.random_split(rng)
    predict_appearance_kwargs = self.get_predict_appearance_kwargs(
        key,
        rays,
        sampler_results,
    )

    feature = self.predict_appearance_feature(
        sampler_results,
        train=train,
        **predict_appearance_kwargs,
    )

    if self.config.multi_illumination and self.use_illumination_feature:
      light_vec = self.get_light_vec(rays, feature)
      feature = jnp.concatenate([feature, light_vec], axis=-1)

    # Predict VMFs
    vmf_params = self.output_layer(feature)
    vmf_params = vmf_params.reshape(means.shape[:-1] + (self.num_illumination_outputs * self.num_components * 5,))

    # Light idx
    if self.config.multi_illumination and self.multiple_illumination_outputs:
      light_idx = rays.light_idx[Ellipsis, None, :] * jnp.ones_like(feature[..., 0:1]).astype(rays.light_idx.dtype)
      vmf_params = vmf_params.reshape(vmf_params.shape[:-1] + (self.num_illumination_outputs, -1))
      vmf_params = jnp.take_along_axis(vmf_params, light_idx[..., None], axis=-2)[..., 0, :]

    vmf_params = vmf_params.reshape(vmf_params.shape[:-1] + (self.num_components, 5))
    vmfs = self.get_vmfs(vmf_params)

    # Ouptut
    vmfs['vmf_means'] = vmfs['vmf_means'] - jax.lax.stop_gradient(means[Ellipsis, None, :])
    vmfs['vmf_origins'] = jax.lax.stop_gradient(means[Ellipsis, None, :])
    vmfs['vmf_normals'] = jax.lax.stop_gradient(sampler_results[self.normals_target][Ellipsis, None, :])
    vmfs['weights'] = jax.lax.stop_gradient(sampler_results['weights'][Ellipsis, None, None])

    return vmfs

  @nn.compact
  def __call__(
      self,
      rng,
      rays,
      sampler_results,
      train_frac = 1.0,
      train = True,
      is_secondary = None,
      **kwargs,
  ):
    # Appearance model
    return self.predict_lighting(
        rng=rng,
        rays=rays,
        sampler_results=sampler_results,
        train_frac=train_frac,
        train=train,
        **kwargs,
    )
