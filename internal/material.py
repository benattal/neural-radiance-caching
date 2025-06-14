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
from internal import math
from internal import ref_utils
from internal import shading
from internal import utils
from internal import grid_utils
from internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
import pdb

gin.config.external_configurable(math.abs, module="math")
gin.config.external_configurable(math.sine_plus, module="math")
gin.config.external_configurable(math.safe_exp, module="math")
gin.config.external_configurable(math.power_3, module="math")
gin.config.external_configurable(math.laplace_cdf, module="math")
gin.config.external_configurable(math.scaled_softplus, module="math")
gin.config.external_configurable(math.power_ladder, module="math")
gin.config.external_configurable(math.inv_power_ladder, module="math")
gin.config.external_configurable(coord.contract, module="coord")
gin.config.external_configurable(coord.contract_constant, module="coord")
gin.config.external_configurable(coord.contract_radius_5, module="coord")
gin.config.external_configurable(coord.contract_radius_2, module="coord")
gin.config.external_configurable(coord.contract_radius_1_2, module="coord")
gin.config.external_configurable(coord.contract_radius_1_4, module="coord")
gin.config.external_configurable(coord.contract_cube, module="coord")
gin.config.external_configurable(coord.contract_cube_5, module="coord")
gin.config.external_configurable(coord.contract_cube_2, module="coord")
gin.config.external_configurable(coord.contract_cube_1_4, module="coord")
gin.config.external_configurable(coord.contract_projective, module="coord")


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = jnp.unbind(quaternions, -1)
    r, i, j, k = quaternions
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jnp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def eval_gaussian(quaternion, scale, mean, points):
    R = quaternion_to_matrix(quaternion)
    S = jnp.diag(scale)
    diff = points - mean
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    diff = diff / (dist + 1e-5)
    exponent = ((R @ S @ S.T @ R.T @ diff.T) * diff.T).sum(0)
    intensity = jnp.exp(-exponent)
    return intensity


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    # r, i, j, k = jnp.unbind(quaternions, -1)
    r, i, j, k = quaternions
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = jnp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def eval_gaussian(quaternion, scale, mean, points):
    R = quaternion_to_matrix(quaternion)
    S = jnp.diag(scale)
    diff = points - mean
    dist = jnp.linalg.norm(diff, axis=-1)
    diff = diff / dist[:, None]
    exponent = ((R @ S @ S.T @ R.T @ diff.T) * diff.T).sum(0)
    intensity = jnp.exp(-exponent)
    return intensity


@gin.configurable
class LightSourceMap(nn.Module):
    config: Any = None  # A Config class, must be set upon construction.

    global_light_source: bool = True
    relative_to_camera: bool = True
    use_gaussian: bool = False
    gaussian_scale: float = 1.0

    use_light_source_dir: bool = True
    use_light_source_norm: bool = False
    use_network: bool = True

    optimize_light_position: bool = False
    optimize_transient_shift: bool = False
    optimize_dark_level: bool = False
    optimize_gaussian: bool = False

    deg_points: int = 2
    net_depth: int = 2  # The depth of the second part of MLP.
    net_width: int = 64  # The width of the second part of MLP.
    skip_layer: int = 4  # Add a skip connection to 2nd MLP every N layers.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.

    orthogonal_scale: float = 0.01
    right_scale: float = 0.01
    look_scale: float = 1.0

    light_power_bias: float = 1.0
    light_power_activation: Callable[..., Any] = math.safe_exp
    light_max_angle: float = 0.0

    def setup(self):
        def pos_enc_fn(direction):
            return coord.pos_enc(
                direction, min_deg=0, max_deg=self.deg_points, append_identity=True
            )

        self.pos_enc_fn = pos_enc_fn

        self.dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, "he_uniform")()
        )

        self.layers = [
            self.dense_layer(self.net_width, name="layer_mult_%d" % i)
            for i in range(self.net_depth)
        ]

        self.output_layer = self.dense_layer(1, name="output_layer_mult")

        # Light source pos
        self.light_source_position = jnp.array(self.config.light_source_position)

        def light_init(key, shape):
            return jnp.zeros_like(random.normal(key, shape))

        self.light_source_offset = self.param("light_source_offset", light_init, (3,))

        def transient_shift_init(key, shape):
            return jnp.zeros_like(random.normal(key, shape))

        self.transient_shift_offset = self.param(
            "transient_shift_offset", transient_shift_init, (1,)
        )

        def dark_level_init(key, shape):
            return jnp.zeros_like(random.normal(key, shape))

        self.dark_level_offset = self.param("dark_level_offset", dark_level_init, (1,))

        # Light power
        def light_power_init(key, shape):
            return jnp.ones_like(random.normal(key, shape)) * self.light_power_bias

        self.light_power = self.param("light_power", light_power_init, (1,))

        if self.global_light_source:

            def light_source_dir_init(key, shape):
                return jnp.zeros_like(random.normal(key, shape))

        else:

            def light_source_dir_init(key, shape):
                return jnp.zeros_like(random.normal(key, shape))

        self.light_source_direction = self.param(
            "light_source_direction", light_source_dir_init, (3,)
        )

        # Gaussian
        if self.optimize_gaussian:

            def quaternion_init(key, shape):
                return jnp.array([51.7835, -49.8733, 6.9429, 5.4460])

            self.quaternion = self.param("quaternion", quaternion_init, (4,))

            def scale_init(key, shape):
                return jnp.array([4.5999e00, 2.5764e-05, -4.2560e00])

            self.scale = self.param("scale", scale_init, (4,))
        else:
            self.quaternion = jnp.array([51.7835, -49.8733, 6.9429, 5.4460])
            self.scale = jnp.array([4.5999e00, 2.5764e-05, -4.2560e00])

    def get_dark_level(
        self,
    ):
        if self.optimize_dark_level:
            return jnp.abs(
                self.dark_level_offset[0] * self.config.dark_level_multiplier
            )
        else:
            return 0.0

    def get_transient_shift(
        self,
    ):
        if self.optimize_transient_shift:
            return (
                self.transient_shift_offset[0] * self.config.transient_shift_multiplier
                + self.config.transient_shift
            )
        else:
            return self.config.transient_shift

    def get_light_source_offset(
        self,
    ):
        if self.optimize_light_position:
            return self.light_source_offset[None] * self.config.light_pos_multiplier
        else:
            return jnp.zeros_like(self.light_source_offset[None])

    @nn.compact
    def __call__(
        self,
        points,
        viewdirs,
        lights,
        look,
        up,
        origins,
        **kwargs,
    ):
        # Reshape
        sh = points.shape
        points = jax.lax.stop_gradient(points.reshape(-1, 3))
        viewdirs = jax.lax.stop_gradient(viewdirs.reshape(-1, 3))
        lights = jax.lax.stop_gradient(lights.reshape(-1, 3))
        look = jax.lax.stop_gradient(look.reshape(-1, 3))
        up = jax.lax.stop_gradient(up.reshape(-1, 3))
        origins = jax.lax.stop_gradient(origins.reshape(-1, 3))

        # Lights
        lights = self.get_lights(
            lights,
            look,
            up,
        )

        # Learnable scale in (0, 2)
        if self.config.sl_relight:
            light_radiance_mult = jnp.ones_like(up[..., :1])
        elif self.use_gaussian:
            light_radiance_mult = self.get_input_and_run_gaussian(
                points, origins, lights, look, up
            )
        else:
            if self.relative_to_camera:
                light_radiance_mult = self.get_input_and_run(
                    points - origins, lights, look, up
                )
            else:
                light_radiance_mult = self.get_input_and_run(
                    points - lights, lights, look, up
                )

        # Constant scale in (0, infty)
        light_radiance_mult = light_radiance_mult.reshape(sh[:-1] + (1,))
        light_radiance = light_radiance_mult * self.light_power_activation(
            self.light_power
        )

        # Inverse square falloff
        light_offset = lights - points
        light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
        light_dirs = light_offset / jnp.maximum(light_dists, 1e-5)

        if self.config.use_falloff:
            falloff = 1.0 / jnp.maximum(light_dists.reshape(sh[:-1] + (1,)) ** 2, 1e-5)
            light_radiance = falloff * light_radiance

        # Spotlight
        if self.light_max_angle > 0.0:
            angle_dot = math.dot(-light_dirs, look, keepdims=True)
            angle = jnp.arccos(angle_dot)

            light_radiance = jnp.where(
                ((angle * 180.0 / jnp.pi) > (self.light_max_angle / 2.0))
                | (angle_dot < 0),
                jnp.zeros_like(light_radiance),
                light_radiance,
            )

        return light_radiance, light_radiance_mult

    def get_input_and_run_gaussian(self, points, origins, lights, look, up):
        right = jnp.cross(up, look)

        points = jnp.concatenate(
            [
                -math.dot(points - origins, right),
                -math.dot(points - origins, up),
                math.dot(points - origins, look),
            ],
            axis=-1,
        )

        lights = self.get_lights_opencv(
            lights,
            look,
            up,
            origins,
        )

        light_radiance_mult = (
            eval_gaussian(self.quaternion, self.scale, lights, points)
            * self.gaussian_scale
        )
        return light_radiance_mult

    def get_input_and_run(
        self,
        points,
        lights,
        look,
        up,
    ):
        # Get net input
        if self.use_light_source_dir:
            light_source_look = self.get_light_source_look(
                lights,
                look,
                up,
            )
            light_source_right = jnp.cross(up, light_source_look)

            points = points / (jnp.linalg.norm(points, axis=-1, keepdims=True) + 1e-5)
            dotprod_look = math.dot(
                points,
                light_source_look,
            )
            dotprod_right = math.dot(
                points,
                light_source_right,
            )
            net_input = jnp.concatenate(
                [
                    jnp.abs(dotprod_look),
                    jnp.abs(dotprod_right) * self.right_scale,
                ],
                axis=-1,
            )
        elif self.use_light_source_norm:
            net_input = points / (
                jnp.linalg.norm(points, axis=-1, keepdims=True) + 1e-5
            )
        else:
            net_input = points

        # Run network
        if self.use_network:
            net_input = self.pos_enc_fn(net_input)
            x = self.run_network(net_input)

            light_radiance_mult = nn.sigmoid(self.output_layer(x)) * 2.0
        else:
            light_radiance_mult = jnp.ones_like(net_input[..., :1])

        return light_radiance_mult

    def run_with_viewdirs(
        self,
        rays,
        **kwargs,
    ):
        sh = rays.vcam_look.shape
        viewdirs = rays.viewdirs.reshape(-1, 3)
        lights = jax.lax.stop_gradient(rays.lights.reshape(-1, 3))
        look = rays.vcam_look.reshape(-1, 3)
        up = rays.vcam_up.reshape(-1, 3)
        origins = jax.lax.stop_gradient(rays.vcam_origins.reshape(-1, 3))
        right = jnp.cross(up, look)

        # Lights and light source look
        lights = self.get_lights(
            lights,
            look,
            up,
        )
        light_source_look = self.get_light_source_look(
            lights,
            look,
            up,
        )
        light_source_look = light_source_look / (
            jnp.linalg.norm(light_source_look, axis=-1, keepdims=True) + 1e-5
        )

        # Light source points
        local_points = jnp.concatenate(
            [
                math.dot(viewdirs, right),
                math.dot(viewdirs, up),
                math.dot(viewdirs, look),
            ],
            -1,
        )
        light_source_right = jnp.cross(up, light_source_look)
        light_source_up = jnp.cross(light_source_look, light_source_right)

        if self.config.sl_relight:
            light_radiance_mult = jnp.ones_like(up[..., :1])
            depth = 3.0
            isect_points = local_points * depth

            light_source_points = origins + (
                right * isect_points[..., 0:1]
                + up * isect_points[..., 1:2]
                + look * isect_points[..., 2:3]
            )

            light_radiance_mult = render_utils.get_sl_color(
                kwargs["env_map"],
                kwargs["env_map_w"],
                kwargs["env_map_h"],
                up,
                look,
                light_source_points,
                lights,
                hfov=self.config.sl_hfov,
                vfov=self.config.sl_vfov,
                shift=self.config.sl_shift,
                mult=self.config.sl_mult,
                invert=self.config.sl_invert,
            )
        elif self.use_gaussian:
            depth = 3.0

            # isect_points = local_points * depth / local_points[..., 2:3]
            isect_points = local_points * depth

            light_source_points = origins + (
                right * isect_points[..., 0:1]
                + up * isect_points[..., 1:2]
                + look * isect_points[..., 2:3]
            )

            light_radiance_mult = self.get_input_and_run_gaussian(
                light_source_points, origins, lights, look, up
            ) / (depth**2)
        else:
            light_source_points = (
                lights
                + (
                    light_source_right * local_points[..., 0:1]
                    + light_source_up * local_points[..., 1:2]
                    + light_source_look * local_points[..., 2:3]
                )
                * 1e3
            )

            if self.relative_to_camera:
                light_radiance_mult = self.get_input_and_run(
                    light_source_points - origins, lights, look, up
                )
            else:
                light_radiance_mult = self.get_input_and_run(
                    light_source_points - lights, lights, look, up
                )

        return light_radiance_mult.reshape(sh[:-1] + (1,))[..., None, :]

    def get_lights(
        self,
        lights,
        look,
        up,
    ):
        sh = lights.shape
        lights = lights.reshape(-1, 3)
        look = jax.lax.stop_gradient(look.reshape(-1, 3))
        up = jax.lax.stop_gradient(up.reshape(-1, 3))
        right = jnp.cross(up, look)

        # Add local or global offset
        if self.global_light_source:
            lights = self.get_light_source_offset() + lights
        else:
            light_offset = self.get_light_source_offset()
            light_offset = (
                light_offset[..., 0:1] * right
                + light_offset[..., 1:2] * up
                + light_offset[..., 2:3] * look
            )
            lights = light_offset + lights

        return lights.reshape(sh)

    def get_lights_opencv(
        self,
        lights,
        look,
        up,
        origins,
    ):
        right = jnp.cross(up, look)
        local_lights = jnp.concatenate(
            [
                -math.dot(lights - origins, right),
                -math.dot(lights - origins, up),
                math.dot(lights - origins, look),
            ],
            axis=-1,
        )
        return local_lights

    def get_light_source_look(
        self,
        lights,
        look,
        up,
    ):
        sh = lights.shape
        lights = lights.reshape(-1, 3)
        look = jax.lax.stop_gradient(look.reshape(-1, 3))
        up = jax.lax.stop_gradient(up.reshape(-1, 3))
        right = jnp.cross(up, look)

        # Add local or global offset
        if self.global_light_source:
            light_source_look = (
                self.light_source_direction[None] * self.orthogonal_scale
                - self.light_source_position[None] * self.look_scale
            ) * jnp.ones_like(up)
        else:
            light_source_look = jnp.concatenate(
                [
                    self.light_source_direction[None][..., :2] * self.orthogonal_scale,
                    jnp.ones_like(self.light_source_direction[None][..., :1])
                    * self.look_scale,
                ],
                axis=-1,
            )
            light_source_look = (
                right * light_source_look[..., 0:1]
                + up * light_source_look[..., 1:2]
                + look * light_source_look[..., 2:3]
            )

        return light_source_look.reshape(sh)

    def run_network(self, x):
        inputs = x

        # Evaluate network to produce the output density.
        for i in range(self.net_depth):
            x = self.layers[i](x)
            x = self.net_activation(x)

        if i % self.skip_layer == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

        return x


@gin.configurable
class BaseMaterialMLP(shading.BaseShader):
    config: Any = None

    num_secondary_samples_diff: int = 4
    num_secondary_samples: int = 32

    render_num_secondary_samples_diff: int = 4
    render_num_secondary_samples: int = 32

    random_generator_2d: Any = render_utils.RandomGenerator2D(1, 1, False)
    separate_integration_diffuse_specular: bool = True
    diffuse_sample_fraction: float = 0.5

    diffuse_importance_sampler_configs: Any = (("cosine", 1),)
    diffuse_render_importance_sampler_configs: Any = (("cosine", 1),)
    importance_sampler_configs: Any = (
        ("microfacet", 1),
        ("cosine", 1),
    )
    render_importance_sampler_configs: Any = (
        ("microfacet", 1),
        ("cosine", 1),
    )

    env_importance_samplers: Any = ((render_utils.EnvironmentSampler(), 1.0),)

    active_importance_samplers: Any = ((render_utils.ActiveSampler(), 1.0),)

    use_indirect: bool = True
    use_active: bool = False
    use_env_map: bool = False

    shadow_eps_indirect: bool = False

    material_type: str = "microfacet"
    use_mis: bool = True
    stratified_sampling: bool = False

    use_constant_material: bool = False
    use_constant_fresnel: bool = True
    use_constant_metalness: bool = False
    use_diffuseness: bool = False
    use_mirrorness: bool = False
    use_specular_albedo: bool = False
    reparam_roughness: bool = False

    min_roughness: float = 0.04

    default_F_0: float = 0.04
    max_F_0: float = 1.0

    brdf_bias: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {
            "albedo": -1.0,
            "specular_albedo": -1.0,
            "roughness": 3.0,
            "F_0": 1.0,
            "metalness": 0.0,
            "diffuseness": 0.0,
            "mirrorness": 0.0,
            "specular_multiplier": 0.0,
            "diffuse_multiplier": 0.0,
        }
    )
    brdf_activation: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {
            "albedo": jax.nn.sigmoid,
            "specular_albedo": jax.nn.sigmoid,
            "roughness": jax.nn.softplus,
            "F_0": jax.nn.sigmoid,
            "metalness": jax.nn.sigmoid,
            "diffuseness": jax.nn.sigmoid,
            "mirrorness": jax.nn.sigmoid,
        }
    )
    brdf_stopgrad: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {
            "albedo": 1.0,
            "specular_albedo": 1.0,
            "roughness": 1.0,
            "F_0": 1.0,
            "metalness": 1.0,
            "diffuseness": 1.0,
            "mirrorness": 1.0,
        }
    )

    rgb_emission_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
    rgb_bias_emission: float = -1.0  # The shift added to raw colors pre-activation.

    rgb_residual_albedo_activation: Callable[..., Any] = (
        nn.sigmoid
    )  # The RGB activation.
    rgb_bias_residual_albedo: float = (
        -1.0
    )  # The shift added to raw colors pre-activation.

    use_brdf_correction: bool = True  # Use brdf to weight secondary ray contribution
    anisotropic_brdf_correction: bool = (
        False  # Use brdf to weight secondary ray contribution
    )
    per_point_brdf_correction: bool = (
        False  # Use brdf to weight secondary ray contribution
    )
    global_brdf_correction: bool = (
        False  # Use brdf to weight secondary ray contribution
    )

    use_diffuse_emission: bool = False  # If True, use diffuse emission
    use_residual_albedo: bool = False  # If True, use diffuse emission
    emission_window_frac: float = 0.0  # If True, use diffuse emission
    emission_variate_weight_start: float = 1.0  # If True, use diffuse emission
    emission_variate_weight_end: float = 1.0  # If True, use diffuse emission

    use_irradiance_cache: bool = False  # If True, predict diffuse & specular colors.
    irradiance_cache_weight: float = 1.0  # If True, predict diffuse & specular colors.
    irradiance_cache_stopgrad_weight: float = (
        1.0  # If True, predict diffuse & specular colors.
    )
    irradiance_cache_decay_rate: float = (
        1.0  # If True, predict diffuse & specular colors.
    )

    rgb_irradiance_activation: Any = math.safe_exp
    rgb_bias_irradiance: float = 0.0

    net_width_brdf: int = 64  # Learned BRDF layer width
    net_depth_brdf: int = 2  #  Learned BRDF layer depth
    deg_brdf: int = 2  # Degree of encoding for mlp BRDF output
    deg_brdf_anisotropic: int = 2  # Degree of encoding for mlp BRDF output

    stopgrad_cache_weight: Tuple[float, float] = (1.0, 1.0)
    stopgrad_slf_weight: Tuple[float, float] = (1.0, 1.0)
    stopgrad_env_map_weight: Tuple[float, float] = (1.0, 1.0)

    stopgrad_shading_weight: float = 1.0
    stopgrad_variate_weight: float = 1.0

    use_mesh_points: bool = True
    use_mesh_points_for_prediction: bool = True
    use_mesh_normals: bool = True

    use_corrected_normals: bool = False
    normals_target: str = "normals_to_use"

    stopgrad_samples: bool = False
    stopgrad_rays: bool = False
    stopgrad_rgb: bool = False

    stopgrad_material: bool = True
    stopgrad_light: bool = True

    near_rate: float = 0.1
    near_start_frac: float = 0.1
    near_max: float = 5e-1
    near_min: float = 1e-1

    use_surface_light_field: bool = False  # If True, use diffuse emission

    # Cache sampling strategy
    resample_cache: bool = True
    cache_train_sampling_strategy: Any = None
    cache_render_sampling_strategy: Any = None

    # Multi illumination
    num_light_features: int = 64  # GLO vector length, disabled if 0.
    use_illumination_feature: bool = False  # GLO vector length, disabled if 0.
    multiple_illumination_outputs: bool = True  # GLO vector length, disabled if 0.

    # Active light
    optimize_light: bool = True
    light_power_bias: float = 200.0
    light_power_activation: Callable[..., Any] = math.abs  # The RGB activation.
    light_max_angle: float = 0.0

    stopgrad_occ_weight: float = 0.0  # If True, predict diffuse & specular colors.
    stopgrad_direct_weight: float = 1.0
    stopgrad_indirect_weight: float = 1.0

    rgb_max: float = float("inf")

    def setup(self):
        self._initialize_dense_layers()
        self._initialize_emission_layers()
        self._initialize_integration_strategy()
        self._initialize_integrated_outputs()
        self._initialize_microfacet_properties()
        self._initialize_brdf_layers()
        self._initialize_brdf_encoding()
        self._initialize_brdf_correction_layers()
        self._initialize_multi_illumination()
        self._initialize_grid()
        self._initialize_predicted_normals()
        self._initialize_importance_samplers()
        self._initialize_sampling_parameters()
        self._initialize_light_power()
        self._initialize_learnable_light()

    def _initialize_dense_layers(self):
        weight_init_fn = getattr(jax.nn.initializers, self.weight_init)()
        self.dense_layer = functools.partial(nn.Dense, kernel_init=weight_init_fn)

        self.layers = [self.dense_layer(self.net_width) for _ in range(self.net_depth)]
        self.bottleneck_layer = self.dense_layer(self.bottleneck_width)

    def _initialize_emission_layers(self):
        if self.use_diffuse_emission:
            self.rgb_diffuse_emission_layer = self.dense_layer(self.num_rgb_channels)

        if self.use_residual_albedo:
            self.rgb_residual_albedo_layer = self.dense_layer(self.num_rgb_channels)

    def _initialize_sampling_parameters(self):
        indirect_parameters = {
            (("light_mode", "indirect"), ("light_component", "diffuse"), ("train", False)): {
                "sample_fraction": (
                    self.diffuse_sample_fraction
                    if self.separate_integration_diffuse_specular
                    else 1.0
                ),
                "samplers": self.get_diffuse_importance_samplers(False),
                "material_type": "microfacet_diffuse",
            },
            (("light_mode", "indirect"), ("light_component", "diffuse"), ("train", True)): {
                "sample_fraction": (
                    self.diffuse_sample_fraction
                    if self.separate_integration_diffuse_specular
                    else 1.0
                ),
                "samplers": self.get_diffuse_importance_samplers(True),
                "material_type": "microfacet_diffuse",
            },
            (("light_mode", "indirect"), ("light_component", "specular"), ("train", False)): {
                "sample_fraction": (
                    1.0 - self.diffuse_sample_fraction
                    if self.separate_integration_diffuse_specular
                    else 1.0
                ),
                "samplers": self.get_specular_importance_samplers(False),
                "material_type": "microfacet_specular",
            },
            (("light_mode", "indirect"), ("light_component", "specular"), ("train", True)): {
                "sample_fraction": (
                    1.0 - self.diffuse_sample_fraction
                    if self.separate_integration_diffuse_specular
                    else 1.0
                ),
                "samplers": self.get_specular_importance_samplers(True),
                "material_type": "microfacet_specular",
            },
        }

        direct_parameters = {}

        if self.use_active:
            direct_parameters = {
                (("light_mode", "direct"), ("light_component", "diffuse"), ("train", False)): {
                    "sample_fraction": 1.0,
                    "samplers": self.active_importance_samplers,
                    "material_type": "microfacet_diffuse",
                },
                (("light_mode", "direct"), ("light_component", "diffuse"), ("train", True)): {
                    "sample_fraction": 1.0,
                    "samplers": self.active_importance_samplers,
                    "material_type": "microfacet_diffuse",
                },
                (("light_mode", "direct"), ("light_component", "specular"), ("train", False)): {
                    "sample_fraction": 1.0,
                    "samplers": self.active_importance_samplers,
                    "material_type": "microfacet_specular",
                },
                (("light_mode", "direct"), ("light_component", "specular"), ("train", True)): {
                    "sample_fraction": 1.0,
                    "samplers": self.active_importance_samplers,
                    "material_type": "microfacet_specular",
                },
            }
        elif self.use_env_map:
            direct_parameters = {
                (("light_mode", "direct"), ("light_component", "diffuse"), ("train", False)): {
                    "sample_fraction": (
                        self.diffuse_sample_fraction
                        if self.separate_integration_diffuse_specular
                        else 1.0
                    ),
                    "samplers": self.get_diffuse_importance_samplers(False),
                    "material_type": "microfacet_diffuse",
                },
                (("light_mode", "direct"), ("light_component", "diffuse"), ("train", True)): {
                    "sample_fraction": (
                        self.diffuse_sample_fraction
                        if self.separate_integration_diffuse_specular
                        else 1.0
                    ),
                    "samplers": self.get_diffuse_importance_samplers(True),
                    "material_type": "microfacet_diffuse",
                },
                (("light_mode", "direct"), ("light_component", "specular"), ("train", False)): {
                    "sample_fraction": (
                        1.0 - self.diffuse_sample_fraction
                        if self.separate_integration_diffuse_specular
                        else 1.0
                    ),
                    "samplers": self.get_specular_importance_samplers(False),
                    "material_type": "microfacet_specular",
                },
                (("light_mode", "direct"), ("light_component", "specular"), ("train", True)): {
                    "sample_fraction": (
                        1.0 - self.diffuse_sample_fraction
                        if self.separate_integration_diffuse_specular
                        else 1.0
                    ),
                    "samplers": self.get_specular_importance_samplers(True),
                    "material_type": "microfacet_specular",
                },
            }
        
        indirect_parameters.update(direct_parameters)
        self.sampling_parameters = indirect_parameters

    def _initialize_integrated_outputs(self):
        return {k: 0.0 for k in self.integration_strategy.keys()}

    def _initialize_microfacet_properties(self):
        # Define property configurations for microfacet materials
        self.microfacet_properties = {
            "albedo": {
                "slice": (..., slice(0, self.num_rgb_channels)),
                "activation_key": "albedo",
                "bias_key": "albedo",
                "constant": False,
                "constant_value": None,  # Not used when constant is False
                "scale": None,  # No scaling
                "post_process": None,
            },
            "specular_albedo": {
                "slice": (..., slice(5, 6)),
                "activation_key": "specular_albedo",
                "bias_key": "specular_albedo",
                "constant": False,
                "constant_value": None,
                "scale": None,
                "post_process": None,
            },
            "roughness": {
                "slice": (..., slice(6, 7)),
                "activation_key": "roughness",
                "bias_key": "roughness",
                "constant": False,
                "constant_value": None,
                "scale": None,
                "post_process": self._post_process_roughness,
            },
            "F_0": {
                "slice": (..., slice(9, 10)),
                "activation_key": "F_0",
                "bias_key": "F_0",
                "constant": self.use_constant_fresnel,
                "constant_value": self.default_F_0,
                "scale": self.max_F_0 if not self.use_constant_fresnel else None,
                "post_process": None,
            },
            "metalness": {
                "slice": (..., slice(8, 9)),
                "activation_key": "metalness",
                "bias_key": "metalness",
                "constant": self.use_constant_metalness,
                "constant_value": 0.0,
                "scale": None,
                "post_process": None,
            },
            "diffuseness": {
                "slice": (..., slice(3, 4)),
                "activation_key": "diffuseness",
                "bias_key": "diffuseness",
                "constant": not self.use_diffuseness,
                "constant_value": 0.0,
                "scale": None,
                "post_process": None,
            },
            "mirrorness": {
                "slice": (..., slice(4, 5)),
                "activation_key": "mirrorness",
                "bias_key": "mirrorness",
                "constant": not self.use_mirrorness,
                "constant_value": 0.0,
                "scale": None,
                "post_process": None,
            },
        }

    def _initialize_brdf_layers(self):
        brdf_output_sizes = {"microfacet": 10, "phong": 7, "lambertian": 3}
        output_size = brdf_output_sizes.get(self.material_type)
        if output_size is not None:
            self.pred_brdf_layer = self.dense_layer(output_size)
        else:
            raise ValueError(f"Unsupported material type: {self.material_type}")

    def _initialize_brdf_encoding(self):
        self.brdf_enc_fn = lambda direction: coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_brdf, append_identity=True
        )
        self.brdf_enc_fn_anisotropic = lambda direction: coord.pos_enc(
            direction,
            min_deg=0,
            max_deg=self.deg_brdf_anisotropic,
            append_identity=True,
        )

    def _initialize_brdf_correction_layers(self):
        self.brdf_correction_layers = [
            self.dense_layer(self.net_width_brdf) for _ in range(self.net_depth_brdf)
        ]
        self.output_brdf_correction_layer = self.dense_layer(2)

    def _initialize_multi_illumination(self):
        if self.config.multi_illumination:
            self.light_vecs = nn.Embed(
                num_embeddings=self.config.num_illuminations,
                features=self.num_light_features,
                name="light_vecs",
            )
            self.num_illumination_outputs = (
                self.config.num_illuminations
                if self.config.multiple_illumination_outputs
                else 1
            )
        else:
            self.num_illumination_outputs = 1

    def _initialize_grid(self):
        if self.use_grid:
            grid_class = grid_utils.GRID_REPRESENTATION_BY_NAME.get(
                self.grid_representation.lower()
            )
            if grid_class is None:
                raise ValueError(
                    f"Unsupported grid representation: {self.grid_representation}"
                )
            self.grid = grid_class(name="material_grid", **self.grid_params)
        else:
            self.grid = None

    def _initialize_predicted_normals(self):
        zeros_init = getattr(jax.nn.initializers, "zeros")
        self.pred_normals_layer = nn.Dense(3, kernel_init=zeros_init)

    def _initialize_importance_samplers(self):
        self.importance_samplers = self._create_importance_samplers(
            self.importance_sampler_configs
        )
        self.render_importance_samplers = self._create_importance_samplers(
            self.render_importance_sampler_configs
        )
        self.diffuse_importance_samplers = self._create_importance_samplers(
            self.diffuse_importance_sampler_configs
        )
        self.diffuse_render_importance_samplers = self._create_importance_samplers(
            self.diffuse_render_importance_sampler_configs
        )

    def _create_importance_samplers(self, sampler_configs):
        return [
            (render_utils.IMPORTANCE_SAMPLER_BY_NAME[conf[0]](), conf[1])
            for conf in sampler_configs
        ]

    def _initialize_light_power(self):
        if self.optimize_light:

            def light_init(key, shape):
                return jnp.ones(shape) * self.light_power_bias

            self.light_power = self.param("light_power", light_init, (1,))
        else:
            self.light_power = self.light_power_bias

    def _initialize_learnable_light(self):
        if self.config.learnable_light:
            self.learnable_light = LightSourceMap(
                config=self.config,
                name="LightSource",
            )

    def get_cache_sampling_strategy(self, train):
        if train:
            return self.cache_train_sampling_strategy
        else:
            return self.cache_render_sampling_strategy

    def get_brdf_correction(self, x_input, ref_samples, num_secondary_samples):
        if not self.per_point_brdf_correction:
            # Encode BRDF Inputs
            brdf_input = self._encode_brdf_input(
                x_input, ref_samples, num_secondary_samples
            )

            # Run BRDF Network
            network_output = self._run_brdf_network(brdf_input)

            # Process Network Output
            brdf_correction = self._process_brdf_output(network_output)
        else:
            # When per-point BRDF correction is enabled
            network_output = self.output_brdf_correction_layer(x_input)
            brdf_correction = self._process_brdf_output(
                network_output,
                per_point=True,
                num_secondary_samples=num_secondary_samples,
            )

        return brdf_correction

    def _encode_brdf_input(self, x_input, ref_samples, num_secondary_samples):
        # Direction Dependent Inputs
        brdf_input = jnp.concatenate(
            [
                jnp.broadcast_to(
                    ref_samples["local_viewdirs"][..., 2:3],
                    ref_samples["local_lightdirs"].shape[:-1] + (1,),
                ),
                ref_samples["local_lightdirs"][..., 2:3],
            ],
            axis=-1,
        )
        brdf_input = jnp.concatenate(
            [
                jnp.sort(brdf_input, axis=-1),
                math.dot(
                    ref_samples["local_viewdirs"],
                    ref_samples["local_lightdirs"],
                ),
            ],
            axis=-1,
        )

        # Encode Inputs
        brdf_input = self.brdf_enc_fn(brdf_input)

        # Anisotropic Correction
        if self.anisotropic_brdf_correction:
            brdf_input_anisotropic = jnp.concatenate(
                [
                    ref_samples["global_viewdirs"] + ref_samples["global_lightdirs"],
                    jnp.abs(
                        ref_samples["global_viewdirs"] - ref_samples["global_lightdirs"]
                    ),
                ],
                axis=-1,
            )
            brdf_input_anisotropic = self.brdf_enc_fn_anisotropic(
                brdf_input_anisotropic
            )
            brdf_input = jnp.concatenate([brdf_input, brdf_input_anisotropic], axis=-1)

        # Position Dependent Inputs
        if not self.global_brdf_correction:
            position_inputs = jnp.repeat(
                x_input.reshape(-1, 1, x_input.shape[-1]),
                num_secondary_samples,
                axis=-2,
            )
            brdf_input = jnp.concatenate([brdf_input, position_inputs], axis=-1)

        return brdf_input

    def _run_brdf_network(self, brdf_input):
        x = brdf_input
        for layer in self.brdf_correction_layers[: self.net_depth_brdf]:
            x = layer(x)
            x = self.net_activation(x)
        x = self.output_brdf_correction_layer(x)
        return x

    def _process_brdf_output(self, x, per_point=False, num_secondary_samples=None):
        # Apply sigmoid activation with bias
        specular = nn.sigmoid(x[..., 0:1] + self.brdf_bias["specular_multiplier"])
        diffuse = nn.sigmoid(x[..., 1:2] + self.brdf_bias["diffuse_multiplier"])
        brdf_correction = jnp.concatenate([specular, diffuse], axis=-1)

        if per_point and num_secondary_samples is not None:
            brdf_correction = jnp.repeat(
                brdf_correction.reshape(-1, 1, brdf_correction.shape[-1]),
                num_secondary_samples,
                axis=-2,
            )

        return brdf_correction

    def get_diffuse_importance_samplers(self, train):
        if not self.separate_integration_diffuse_specular:
            return self.get_specular_importance_samplers(train)

        if self.config.compute_relight_metrics or (
            self.config.use_ground_truth_illumination and self.config.multi_illumination
        ):
            return self.env_importance_samplers

        if train:
            return self.diffuse_importance_samplers
        else:
            return self.diffuse_render_importance_samplers

    def get_specular_importance_samplers(self, train):
        if self.config.compute_relight_metrics or (
            self.config.use_ground_truth_illumination and self.config.multi_illumination
        ):
            return self.env_importance_samplers

        if train:
            return self.importance_samplers
        else:
            return self.render_importance_samplers

    def get_num_secondary_samples(self, train):
        if train:
            return self.num_secondary_samples
        else:
            return self.render_num_secondary_samples

    def get_num_secondary_samples_diff(self, train):
        if train:
            return self.num_secondary_samples_diff
        else:
            return self.render_num_secondary_samples_diff

    def get_num_secondary_samples_variate(self, train):
        return self.get_num_secondary_samples(
            train
        ) - self.get_num_secondary_samples_diff(train)

    def get_material(self, brdf_params):
        if self.material_type == "microfacet":
            return self._get_microfacet_material(brdf_params)
        elif self.material_type == "phong":
            return self._get_phong_material(brdf_params)
        elif self.material_type == "lambertian":
            return self._get_lambertian_material(brdf_params)
        else:
            return {}

    def _apply_activation_with_bias(
        self, brdf_params, slice_indices, activation_key, bias_key
    ):
        return self.brdf_activation[activation_key](
            brdf_params[slice_indices] + self.brdf_bias[bias_key]
        )

    def _post_process_roughness(self, roughness):
        if self.reparam_roughness:
            roughness = 1.0 / (roughness + 1.0)

        roughness = roughness * (1.0 - self.min_roughness ** 2) + self.min_roughness ** 2
        return roughness

    def _get_microfacet_material(self, brdf_params):
        material = {}

        for prop, config in self.microfacet_properties.items():
            if config["constant"]:
                # Use a constant value
                material[prop] = jnp.full_like(
                    brdf_params[config["slice"]], config["constant_value"]
                )
            else:
                # Apply activation with bias
                value = self._apply_activation_with_bias(
                    brdf_params,
                    config["slice"],
                    config["activation_key"],
                    config["bias_key"],
                )

                value = utils.stopgrad_with_weight(
                    value, self.brdf_stopgrad[config["activation_key"]]
                )

                # Apply scaling if specified
                if config["scale"] is not None:
                    value = value * config["scale"]

                # Apply post-processing if specified
                if config["post_process"] is not None:
                    value = config["post_process"](value)

                material[prop] = value

        return material

    def _get_phong_material(self, brdf_params):
        return {
            "albedo": jax.nn.sigmoid(brdf_params[..., 0:3]),
            "specular_albedo": jax.nn.sigmoid(brdf_params[..., 3:6]),
            "specular_exponent": math.safe_exp(brdf_params[..., 6:7] - 0.5),
        }

    def _get_lambertian_material(self, brdf_params):
        return {
            "albedo": jax.nn.sigmoid(brdf_params[..., 0:3]),
        }

    def _update_integrated_outputs(self, integrated_outputs):
        for output_key, config in self.integration_strategy.items():
            if "indirect" in output_key and not self.use_indirect:
                continue

            sum_val = 0.0

            for sub_key, reduce_dims in config["sum_over"]:
                if "indirect" in sub_key and not self.use_indirect:
                    continue
                
                sum_val += integrated_outputs[sub_key].sum(axis=reduce_dims)

            sum_val *= config["scale"]
            integrated_outputs[output_key] = sum_val

    def get_outgoing_radiance(
        self,
        rng,
        rays,
        feature,
        sampler_results,
        material,
        num_secondary_samples,
        radiance_cache_fn,
        env_map_fn,
        active_fn=None,
        train_frac=1.0,
        train=True,
        mesh=None,
        light_sampler_results=None,
        last_integrated_outputs=None,
        **kwargs,
    ):
        # Initialize results dictionary with default values
        integrated_outputs = self._initialize_integrated_outputs()

        ref_rays = None
        ref_samples = None
        ref_sampler_results = None

        if self.use_indirect:
            if last_integrated_outputs is not None:
                ref_rays = last_integrated_outputs.get("ref_rays_indirect_specular", None)
                ref_samples = last_integrated_outputs.get("ref_samples_indirect_specular", None)
                ref_sampler_results = None

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="indirect",
                light_component="specular",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=num_secondary_samples,
                radiance_cache_fn=radiance_cache_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=light_sampler_results,
                integrated_outputs=integrated_outputs,
                stopgrad_weight=self.stopgrad_indirect_weight,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )

            if last_integrated_outputs is not None:
                ref_rays = last_integrated_outputs.get("ref_rays_indirect_diffuse", None)
                ref_samples = last_integrated_outputs.get("ref_samples_indirect_diffuse", None)
                ref_sampler_results = None

            elif not self.separate_integration_diffuse_specular:
                ref_rays = integrated_outputs.get("ref_rays_indirect_specular", None)
                ref_samples = integrated_outputs.get("ref_samples_indirect_specular", None)
                ref_sampler_results = integrated_outputs.get("ref_sampler_results_indirect_specular", None)

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="indirect",
                light_component="diffuse",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=num_secondary_samples,
                radiance_cache_fn=radiance_cache_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=light_sampler_results,
                integrated_outputs=integrated_outputs,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                stopgrad_weight=self.stopgrad_indirect_weight,
                **kwargs,
            )

        # Handle direct lighting
        if self.use_active:
            direct_light_sampler_results = self._prepare_light_sampler_results(
                sampler_results,
                light_sampler_results,
                rays,
                direct=True,
                config=self.config,
            )

            if last_integrated_outputs is not None:
                ref_rays = last_integrated_outputs.get("ref_rays_direct_specular", None)
                ref_samples = last_integrated_outputs.get("ref_samples_direct_specular", None)
                ref_sampler_results = last_integrated_outputs.get("ref_sampler_results_direct_specular", None)

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="direct",
                light_component="specular",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=1,
                radiance_cache_fn=active_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=direct_light_sampler_results,
                integrated_outputs=integrated_outputs,
                stopgrad_weight=self.stopgrad_direct_weight,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )

            if last_integrated_outputs is not None:
                ref_rays = last_integrated_outputs.get("ref_rays_direct_diffuse", None)
                ref_samples = last_integrated_outputs.get("ref_samples_direct_diffuse", None)
                ref_sampler_results = last_integrated_outputs.get("ref_sampler_results_direct_diffuse", None)
            else:
                ref_rays = integrated_outputs.get("ref_rays_direct_specular", None)
                ref_samples = integrated_outputs.get("ref_samples_direct_specular", None)
                ref_sampler_results = integrated_outputs.get("ref_sampler_results_direct_specular", None)

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="direct",
                light_component="diffuse",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=1,
                radiance_cache_fn=active_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=direct_light_sampler_results,
                integrated_outputs=integrated_outputs,
                stopgrad_weight=self.stopgrad_direct_weight,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )
        elif self.use_env_map:
            ref_rays = integrated_outputs.get("ref_rays_indirect_specular", None)
            ref_samples = integrated_outputs.get("ref_samples_indirect_specular", None)
            ref_sampler_results = integrated_outputs.get("ref_sampler_results_indirect_specular", None)

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="direct",
                light_component="specular",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=num_secondary_samples,
                radiance_cache_fn=env_map_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=light_sampler_results,
                integrated_outputs=integrated_outputs,
                stopgrad_weight=self.stopgrad_direct_weight,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )

            ref_rays = integrated_outputs.get("ref_rays_indirect_diffuse", None)
            ref_samples = integrated_outputs.get("ref_samples_indirect_diffuse", None)
            ref_sampler_results = integrated_outputs.get("ref_sampler_results_indirect_diffuse", None)

            key, rng = utils.random_split(rng)
            self._process_outgoing_results(
                light_mode="direct",
                light_component="diffuse",
                rng=key,
                rays=rays,
                feature=feature,
                sampler_results=sampler_results,
                material=material,
                num_secondary_samples=num_secondary_samples,
                radiance_cache_fn=env_map_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=light_sampler_results,
                integrated_outputs=integrated_outputs,
                stopgrad_weight=self.stopgrad_direct_weight,
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )

        self._update_integrated_outputs(integrated_outputs)

        return integrated_outputs

    def _prepare_light_sampler_results(
        self, sampler_results, light_sampler_results, rays, direct, config
    ):
        # Retrieve lights based on configuration
        lights = (
            jax.lax.stop_gradient(
                self.learnable_light.get_lights(
                    rays.lights,
                    rays.vcam_look,
                    rays.vcam_up,
                )
            )
            if self.config.learnable_light
            else rays.lights
        )

        """Prepare light sampler results for direct lighting."""
        if direct and self.use_active:
            return {
                "origins": sampler_results["means"][..., None, :],
                "lights": lights[..., None, None, :]
                * jnp.ones_like(sampler_results["means"][..., None, :]),
            }
        else:
            return light_sampler_results

    def _get_sampling_parameters(
        self,
        num_secondary_samples,
        light_mode,
        light_component,
        train=True,
    ):
        query_tuple = (("light_mode", light_mode), ("light_component", light_component), ("train", train))
        sampling_params = self.sampling_parameters[query_tuple]

        return (
            int(np.round(num_secondary_samples * sampling_params["sample_fraction"])),
            sampling_params["samplers"],
            sampling_params["material_type"],
        )

    def _process_outgoing_results(
        self,
        light_mode,
        light_component,
        rng,
        rays,
        feature,
        sampler_results,
        material,
        num_secondary_samples,
        radiance_cache_fn,
        train_frac,
        train,
        mesh,
        light_sampler_results,
        integrated_outputs,
        stopgrad_weight,
        ref_rays=None,
        ref_samples=None,
        ref_sampler_results=None,
        **kwargs,
    ):
        # Get num samples, samplers
        num_samples, samplers, material_type = self._get_sampling_parameters(
            num_secondary_samples,
            light_mode,
            light_component,
            train,
        )

        if num_samples == 0:
            return

        key, rng = utils.random_split(rng)
        light_component_results, ref_rays, ref_samples, ref_sampler_results = (
            self.get_outgoing_radiance_helper(
                key,
                rays,
                feature,
                sampler_results,
                material,
                material_type,
                samplers,
                num_samples,
                radiance_cache_fn,
                train_frac=train_frac,
                train=train,
                mesh=mesh,
                light_sampler_results=light_sampler_results,
                direct=(light_mode == "direct"),
                ref_rays=ref_rays,
                ref_samples=ref_samples,
                ref_sampler_results=ref_sampler_results,
                **kwargs,
            )
        )

        # Store reference variables if in indirect light_mode
        integrated_outputs[f"ref_rays_{light_mode}_{light_component}"] = ref_rays
        integrated_outputs[f"ref_samples_{light_mode}_{light_component}"] = ref_samples
        integrated_outputs[f"ref_sampler_results_{light_mode}_{light_component}"] = (
            ref_sampler_results
        )

        # Apply stop gradient with appropriate weight
        light_component_results = {
            k: utils.stopgrad_with_weight(val, stopgrad_weight)
            for k, val in light_component_results.items()
        }

        # Accumulate results
        for k, val in light_component_results.items():
            put_key = f"{light_mode}_{light_component}_{k}"
            integrated_outputs[put_key] = val

    def get_outgoing_radiance_helper(
        self,
        rng,
        rays,
        feature,
        sampler_results,
        material,
        material_type,
        samplers,
        num_secondary_samples,
        radiance_cache_fn,
        train_frac=1.0,
        train=True,
        mesh=None,
        light_sampler_results=None,
        ref_rays=None,
        ref_samples=None,
        ref_sampler_results=None,
        direct=False,
        **kwargs,
    ):
        sh = sampler_results["points"].shape

        # Prepare material and light based on stop gradient configuration
        material_for_secondary = (
            jax.lax.stop_gradient(material) if self.stopgrad_material else material
        )
        light_for_secondary = (
            jax.lax.stop_gradient(light_sampler_results)
            if self.stopgrad_light
            else light_sampler_results
        )

        # Compute near parameter based on training fraction
        near = self._compute_near(train_frac)

        if ref_samples is None:
            key, rng = utils.random_split(rng)
            ref_rays, ref_samples = render_utils.get_secondary_rays(
                key,
                rays,
                sampler_results["points"],
                rays.viewdirs,
                sampler_results[self.normals_target],
                material_for_secondary,
                refdir_eps=near,
                normal_eps=self.config.secondary_normal_eps,
                random_generator_2d=self.random_generator_2d,
                stratified_sampling=self.stratified_sampling,
                use_mis=self.use_mis,
                samplers=samplers,
                num_secondary_samples=num_secondary_samples,
                light_sampler_results=light_for_secondary,
                offset_origins=mesh is not None,
                far=self.config.secondary_far,
            )

            # Apply mask based on material loss radius
            if self.config.material_loss_radius < float("inf"):
                mask = (
                    jnp.linalg.norm(ref_rays.origins, axis=-1, keepdims=True)
                    < self.config.material_loss_radius
                ).astype(jnp.float32)
                stopgrad_shading_weight = mask * self.stopgrad_shading_weight
            else:
                stopgrad_shading_weight = self.stopgrad_shading_weight

            # Apply stop gradient to various directions
            ref_samples = self._apply_stop_gradient_to_samples(
                ref_samples, stopgrad_shading_weight
            )

            # Zero out weights where local_lightdirs[..., 2:] <= 0
            ref_samples["weight"] = jnp.where(
                ref_samples["local_lightdirs"][..., 2:] > 0.0,
                ref_samples["weight"],
                jnp.zeros_like(ref_samples["weight"]),
            )

            # Apply additional stop gradients if configured
            ref_samples = (
                jax.lax.stop_gradient(ref_samples)
                if self.stopgrad_samples
                else ref_samples
            )
            ref_rays = (
                jax.tree_util.tree_map(jax.lax.stop_gradient, ref_rays)
                if self.stopgrad_rays
                else ref_rays
            )
        else:
            ref_samples = utils.copy_tree(ref_samples)
            ref_rays = utils.copy_tree(ref_rays)

        if ref_sampler_results is None or self.use_env_map:
            # Query radiance cache
            key, rng = utils.random_split(rng)
            rgb, rgb_no_stopgrad, ref_sampler_results = radiance_cache_fn(
                key, ref_rays, ref_samples, ref_sampler_results
            )

            rgb = jnp.nan_to_num(rgb)
            rgb_no_stopgrad = jnp.nan_to_num(rgb_no_stopgrad)

            # Apply stop gradient to RGB if configured
            rgb = (
                jax.lax.stop_gradient(rgb)
                if self.stopgrad_rgb
                else rgb
            )

            # Reshape RGB based on direct flag
            rgb, rgb_no_stopgrad = self._reshape_rgb(
                rgb, rgb_no_stopgrad, direct, num_secondary_samples
            )

            # Reshape ref_samples
            ref_samples = jax.tree_util.tree_map(
                lambda x: x.reshape(rgb.shape[0], -1, x.shape[-1]), ref_samples
            )

            # Apply BRDF correction
            brdf_correction = (
                self.get_brdf_correction(feature, ref_samples, num_secondary_samples)
                if self.use_brdf_correction
                else jnp.ones_like(ref_rays.origins[..., :2])
            )

            # Update ref_samples with radiance and correction
            ref_samples.update(
                {
                    "radiance_in": rgb,
                    "indirect_occ": ref_sampler_results[-1]["acc"][..., None],
                    "radiance_in_no_stopgrad": rgb_no_stopgrad,
                    "brdf_correction": brdf_correction,
                }
            )
        else:
            ref_sampler_results = utils.copy_tree(ref_sampler_results)

        # Integrate the reflect rays
        if self.config.use_transient:
            integrated_outputs = render_utils.transient_integrate_reflect_rays(
                material_type,
                self.use_brdf_correction,
                material,
                ref_samples,
                use_diffuseness=self.use_diffuseness,
                use_mirrorness=self.use_mirrorness,
                use_specular_albedo=self.use_specular_albedo,
                direct=direct,
                max_radiance=self.rgb_max,
            )
        else:
            integrated_outputs = render_utils.integrate_reflect_rays(
                material_type,
                self.use_brdf_correction,
                material,
                ref_samples,
                use_diffuseness=self.use_diffuseness,
                use_mirrorness=self.use_mirrorness,
                use_specular_albedo=self.use_specular_albedo,
                max_radiance=self.rgb_max,
            )

        if direct and self.use_active:
            integrated_outputs["occ"] = ref_sampler_results[-1]["occ"]

        # Reshape integrated outputs to match original sampler_results shape
        if direct or not self.config.use_transient:
            integrated_outputs = jax.tree_util.tree_map(
                lambda x: x.reshape(tuple(sh[:-1]) + (x.shape[-1],)),
                integrated_outputs,
            )
        else:
            integrated_outputs = jax.tree_util.tree_map(
                lambda x: x.reshape(tuple(sh[:-1]) + (-1, x.shape[-1])),
                integrated_outputs,
            )

        return integrated_outputs, ref_rays, ref_samples, ref_sampler_results

    def _compute_near(self, train_frac):
        """Compute the near parameter based on training fraction."""
        if self.near_rate > 0:
            near_weight = jnp.clip(
                (train_frac - self.near_start_frac) / self.near_rate, 0.0, 1.0
            )
            near = near_weight * self.near_min + (1.0 - near_weight) * self.near_max
        else:
            near = self.near_min

        return near

    def _apply_stop_gradient_to_samples(self, ref_samples, weight):
        """Apply stop gradient with weight to various directions in ref_samples."""
        directions = [
            "local_viewdirs",
            "local_lightdirs",
            "global_viewdirs",
            "global_lightdirs",
        ]
        for direction in directions:
            ref_samples[direction] = utils.stopgrad_with_weight(
                ref_samples[direction], weight
            )
        return ref_samples

    def _reshape_rgb(self, rgb, rgb_no_stopgrad, direct, num_secondary_samples):
        """Reshape RGB tensors based on the direct flag."""
        if direct or not self.config.use_transient:
            shape = (rgb.shape[0], num_secondary_samples, self.num_rgb_channels)
        else:
            shape = (rgb.shape[0], num_secondary_samples, -1, self.num_rgb_channels)

        return rgb.reshape(shape), rgb_no_stopgrad.reshape(shape)

    def predict_bottleneck_feature(
        self,
        rng,
        rays,
        sampler_results,
        train=True,
    ):
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

        if self.bottleneck_width > 0:
            key, rng = utils.random_split(rng)
            feature = self.get_bottleneck_feature(key, feature)

        return feature

    def get_light_vec(self, rays, feature):
        light_vec = jnp.zeros_like(feature[..., 0:0])

        if self.config.multi_illumination > 0:
            light_idx = rays.light_idx[..., 0]
            light_vec = self.light_vecs(light_idx)
            light_vec = light_vec[..., None, :] * jnp.ones_like(feature[..., 0:1])

        return light_vec

    def predict_appearance(
        self,
        rng,
        rays,
        sampler_results,
        train_frac=1.0,
        train=True,
        mesh=None,
        radiance_cache=None,
        material_only=False,
        slf_variate=False,
        **kwargs,
    ):
        """
        Predict appearance for the given rays, returning a dictionary of outputs.

        Args:
        rng: JAX PRNG key.
        rays: A namedtuple with ray information (origins, directions, etc.).
        sampler_results: Data structure with sampling results (points, normals, etc.).
        train_frac: Fraction of total training steps completed, used for scheduling.
        train: Whether in training light_mode (as opposed to inference).
        mesh: Optional mesh for offsetting ray origins, if needed.
        radiance_cache: Radiance cache to query for indirect lighting.
        material_only: If True, only compute and return material properties.
        slf_variate: If True, special-case usage of surface light field with multiple variates.
        **kwargs: Additional arguments (e.g. environment maps) used for lighting.

        Returns:
        outputs: Dictionary with predicted material properties, lighting, and final RGB.
        """

        # === 1) Predict or retrieve the material and its feature vector. ===
        key, rng = utils.random_split(rng)
        feature, material = self._predict_material_and_feature(
            key,
            rays,
            sampler_results,
            train_frac,
            train,
            mesh,
            radiance_cache,
            **kwargs,
        )

        # If only material is requested, exit early.
        if material_only:
            return {"material_" + k: v for k, v in material.items()}

        # === 2) Compute emission and residual albedo (if enabled). ===
        key, rng = utils.random_split(rng)
        emission, residual_albedo, outputs = self._compute_emission_and_residual_albedo(
            key, rays, feature, train_frac, train, material
        )

        # === 3) Initialize local closures for radiance/SLF/relighting/active lighting. ===
        # These functions capture variables from the current scope (rays, sampler_results, etc.).
        radiance_cache_fn = self._make_radiance_cache_fn(
            rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
        )
        surface_lf_fn = self._make_surface_lf_fn(
            rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
        )
        env_map_fn = self._make_env_map_fn(
            rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
        )
        active_fn = self._make_active_light_fn(
            rays, sampler_results, mesh, train_frac, train, **kwargs
        )

        # === 4) Compute integrated lighting (direct, indirect, or specialized light_modes). ===
        # Depending on configuration (e.g. compute_relight_metrics, slf_variate, etc.)
        # we may call one or more different integration paths.
        if slf_variate and self.use_surface_light_field:
            # --- SLF variate path ---
            key, rng = utils.random_split(rng)
            integrated_outputs = self._integrate_slf_variate(
                key,
                rays,
                feature,
                sampler_results,
                material,
                radiance_cache_fn,
                surface_lf_fn,
                env_map_fn,
                active_fn,
                train_frac,
                train,
                **kwargs,
            )
        else:
            # --- Standard path (direct + indirect or with SLF if enabled). ---
            key, rng = utils.random_split(rng)
            integrated_outputs = self._integrate_standard(
                key,
                rays,
                feature,
                sampler_results,
                material,
                radiance_cache_fn,
                surface_lf_fn,
                env_map_fn,
                active_fn,
                slf_variate,
                train_frac,
                train,
                mesh,
                **kwargs,
            )

        # === 6) Combine computed emission/residual_albedo with the integrated lighting. ===
        final_rgb = self._apply_emission_and_residual_albedo(
            emission, residual_albedo, material, integrated_outputs
        )

        # === 7) Assemble final outputs and return. ===
        self._finalize_outputs(
            rays,
            outputs,
            integrated_outputs,
            final_rgb,
            material,
            emission,
            residual_albedo,
            sampler_results,
        )

        return outputs

    # ---------------------------------------------------------------------
    # Below are the newly introduced helper methods, which were factored out
    # from the original predict_appearance. They group chunks of logic to
    # simplify the main function.
    # ---------------------------------------------------------------------

    def _predict_material_and_feature(
        self,
        rng,
        rays,
        sampler_results,
        train_frac,
        train,
        mesh,
        radiance_cache,
        **kwargs,
    ):
        """
        Predict the material parameters and the feature vector (bottleneck) used for shading.
        """
        # If we use a constant material, we zero out points in the sampler_results for consistency.
        if self.use_constant_material:
            sr_copy = utils.copy_tree(sampler_results)
            sr_copy["points"] = jnp.zeros_like(sr_copy["points"])
            sr_copy["means"] = jnp.zeros_like(sr_copy["means"])
            sampler_for_bottleneck = sr_copy
        else:
            sampler_for_bottleneck = sampler_results

        # 1) Predict bottleneck feature (which can be used to modulate material or shading).
        key, rng = utils.random_split(rng)
        feature = self.predict_bottleneck_feature(
            key, rays, sampler_for_bottleneck, train=train
        )

        # 2) Predict raw BRDF parameters from the feature.
        brdf_params = self.pred_brdf_layer(feature)
        material = self.get_material(brdf_params)

        # 3) Optionally apply user-supplied albedo ratio.
        if "albedo_ratio" in kwargs and kwargs["albedo_ratio"] is not None:
            ar = kwargs["albedo_ratio"]
            sh = material["albedo"].shape
            clipped_albedo = jnp.clip(
                material["albedo"].reshape(-1, self.num_rgb_channels)
                * ar.reshape(-1, self.num_rgb_channels),
                0.0,
                1.0,
            ).reshape(sh)
            material["albedo"] = clipped_albedo

        # 4) If we are forcing a constant material, override some fields.
        if self.use_constant_material:
            material["metalness"] = jnp.ones_like(material["metalness"])
            material["roughness"] = jnp.ones_like(material["roughness"]) * 0.01

        return feature, material

    def _compute_emission_and_residual_albedo(
        self, rng, rays, feature, train_frac, train, material
    ):
        """
        Compute (optionally) the emission and residual albedo.
        Returns them, plus a dictionary of partial outputs.
        """
        outputs = {}
        emission = jnp.zeros_like(material["albedo"])
        residual_albedo = jnp.zeros_like(material["albedo"])

        # --- Diffuse emission ---
        if self.use_diffuse_emission:
            raw_emission = (
                self.rgb_premultiplier * self.rgb_diffuse_emission_layer(feature)
                + self.rgb_bias_emission
            )
            emission = self.rgb_emission_activation(raw_emission)

            if self.emission_window_frac > 0.0:
                w = jnp.clip(train_frac / self.emission_window_frac, 0.0, 1.0)
            else:
                w = 1.0

            emission_variate_weight = (
                (1.0 - w) * self.emission_variate_weight_start
                + w * self.emission_variate_weight_end
            )

            # Interpolate between the no-stop-gradient and gradient versions of emission.
            emission = emission * emission_variate_weight + jax.lax.stop_gradient(
                emission
            ) * (1.0 - emission_variate_weight)

        # --- Residual albedo ---
        if self.use_residual_albedo:
            raw_resid = (
                self.rgb_premultiplier * self.rgb_residual_albedo_layer(feature)
                + self.rgb_bias_residual_albedo
            )
            residual_albedo = self.rgb_residual_albedo_activation(raw_resid)
            # Overwrite the main "albedo" if desired. In the original code, we do:
            outputs["material_albedo"] = residual_albedo

        # Always record residual_albedo in outputs, even if zero.
        outputs["material_residual_albedo"] = residual_albedo

        return emission, residual_albedo, outputs

    def _make_radiance_cache_fn(
        self, rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
    ):
        def radiance_cache_fn(rng, ref_rays, ref_samples, ref_sampler_results):
            # Prepare shapes
            normals = sampler_results[self.config.shadow_normals_target].reshape(
                ref_rays.origins.shape[:-2] + (-1, 3)
            ) * jnp.ones_like(ref_rays.origins)

            # Conditionally override the "normals" in ref_rays
            if self.shadow_eps_indirect:
                ref_rays = ref_rays.replace(normals=normals)
            else:
                ref_rays = ref_rays.replace(normals=None)

            # Perform the cache call
            key, rng = utils.random_split(rng)
            ref_ray_outputs = radiance_cache.cache(
                key,
                ref_rays,
                train_frac=train_frac,
                train=train,
                compute_extras=False,
                zero_glo=(
                    "glo_vec" not in sampler_results
                    or sampler_results["glo_vec"] is None
                ),
                mesh=mesh,
                stopgrad_proposal=False,
                stopgrad_weights=False,
                is_secondary=True,
                linear_rgb=True,
                resample=self.resample_cache,
                sampling_strategy=self.get_cache_sampling_strategy(train),
                use_env_map=False,
                env_map=kwargs["env_map"],
                env_map_w=kwargs["env_map_w"],
                env_map_h=kwargs["env_map_h"],
                light_power=(
                    self.light_power_activation(self.light_power) if radiance_cache.share_light_power else None
                ),
                radiance_cache=radiance_cache,
                stopgrad_cache_weight=self.stopgrad_cache_weight,
            )

            rgb = ref_ray_outputs["render"]["rgb"]
            rgb_no_stopgrad = ref_ray_outputs["render"]["rgb_no_stopgrad"]

            # Guard against NaNs
            rgb = jnp.maximum(jnp.nan_to_num(rgb), 0.0)
            rgb_no_stopgrad = jnp.maximum(jnp.nan_to_num(rgb_no_stopgrad), 0.0)
            ref_sampler_results = ref_ray_outputs["main"]["sampler"]
            ref_sampler_results[-1]["acc"] = ref_ray_outputs["render"]["acc"].reshape(ref_rays.origins.shape[:-1])
            ref_sampler_results[-1]["acc_no_stopgrad"] = ref_ray_outputs["render"]["acc_no_stopgrad"].reshape(ref_rays.origins.shape[:-1])

            return rgb, rgb_no_stopgrad, ref_sampler_results

        return radiance_cache_fn

    def _make_surface_lf_fn(
        self, rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
    ):
        def surface_lf_fn(rng, ref_rays, ref_samples, ref_sampler_results):
            normals = sampler_results[self.config.shadow_normals_target].reshape(
                ref_rays.origins.shape[:-2] + (-1, 3)
            ) * jnp.ones_like(ref_rays.origins)

            if self.shadow_eps_indirect:
                ref_rays = ref_rays.replace(normals=normals)
            else:
                ref_rays = ref_rays.replace(normals=None)

            key, rng = utils.random_split(rng)
            slf_results = radiance_cache.cache(
                key,
                ref_rays,
                use_slf=True,
                use_env_map=False,
                train=train,
                train_frac=train_frac,
                env_map=kwargs["env_map"],
                env_map_w=kwargs["env_map_w"],
                env_map_h=kwargs["env_map_h"],
                stopgrad_cache_weight=self.stopgrad_slf_weight,
            )
            rgb = slf_results["rgb"].reshape(ref_rays.origins.shape)
            rgb_no_stopgrad = slf_results["rgb_no_stopgrad"].reshape(
                ref_rays.origins.shape
            )

            # Optionally mask out large radii
            if self.config.material_loss_radius < float("inf"):
                mask = (
                    jnp.linalg.norm(ref_rays.origins, axis=-1, keepdims=True)
                    < self.config.material_loss_radius
                ).astype(jnp.float32)
                rgb = utils.stopgrad_with_weight(rgb, mask)
                rgb_no_stopgrad = utils.stopgrad_with_weight(rgb_no_stopgrad, mask)

            rgb = jnp.maximum(rgb, 0.0)
            rgb_no_stopgrad = jnp.maximum(rgb_no_stopgrad, 0.0)

            slf_results["acc"] = slf_results["acc"].reshape(ref_rays.origins.shape[:-1])
            slf_results["acc_no_stopgrad"] = slf_results["acc_no_stopgrad"].reshape(ref_rays.origins.shape[:-1])

            return rgb, rgb_no_stopgrad, [slf_results]

        return surface_lf_fn

    def _make_env_map_fn(
        self, rays, sampler_results, radiance_cache, mesh, train_frac, train, **kwargs
    ):
        def env_map_fn(rng, ref_rays, ref_samples, ref_sampler_results):
            key, rng = utils.random_split(rng)
            env_map_outputs = radiance_cache.cache(
                key,
                ref_rays,
                env_map_only=True,
                use_env_map=True,
                train=train,
                train_frac=train_frac,
                env_map=kwargs["env_map"],
                env_map_w=kwargs["env_map_w"],
                env_map_h=kwargs["env_map_h"],
                stopgrad_cache_weight=self.stopgrad_env_map_weight,
            )
            rgb = env_map_outputs["incoming_rgb"].reshape(ref_rays.origins.shape)
            rgb_no_stopgrad = env_map_outputs["incoming_rgb_no_stopgrad"].reshape(
                ref_rays.origins.shape
            )

            rgb = jnp.maximum(rgb, 0.0) * (1.0 - ref_sampler_results[-1]["acc"].reshape(
                ref_rays.origins.shape[:-1] + (1,))
            )
            rgb_no_stopgrad = jnp.maximum(rgb_no_stopgrad, 0.0) * (1.0 - ref_sampler_results[-1]["acc_no_stopgrad"].reshape(
                ref_rays.origins.shape[:-1] + (1,))
            )

            return rgb, rgb_no_stopgrad, ref_sampler_results

        return env_map_fn

    def _make_active_light_fn(
        self, rays, sampler_results, mesh, train_frac, train, **kwargs
    ):
        def active_fn(rng, ref_rays, ref_samples, ref_sampler_results):
            normals = sampler_results[self.config.shadow_normals_target].reshape(
                ref_rays.origins.shape[:-2] + (-1, 3)
            ) * jnp.ones_like(ref_rays.origins)
            ref_rays = ref_rays.replace(normals=jax.lax.stop_gradient(normals))

            # Possibly learnable lights
            if self.config.learnable_light:
                lights = self.learnable_light.get_lights(
                    ref_rays.lights, ref_rays.vcam_look, ref_rays.vcam_up
                )
                lights = jax.lax.stop_gradient(lights)
            else:
                lights = ref_rays.lights

            light_offset = lights - ref_rays.origins
            light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
            light_dirs = light_offset / jnp.maximum(light_dists, 1e-5)

            # Potentially update ray far if needed
            new_far = jnp.clip(
                light_dists.reshape(ref_rays.far.shape) - self.config.light_near,
                ref_rays.near,
                ref_rays.far,
            )
            ref_rays = ref_rays.replace(far=new_far)

            # Evaluate light power
            if self.config.learnable_light:
                light_radiance, _ = self.learnable_light(
                    ref_rays.origins,
                    ref_rays.viewdirs,
                    ref_rays.lights,
                    ref_rays.vcam_look,
                    ref_rays.vcam_up,
                    ref_rays.vcam_origins,
                    env_map=kwargs["env_map"],
                    env_map_w=kwargs["env_map_w"],
                    env_map_h=kwargs["env_map_h"],
                )
            else:
                light_radiance = jnp.ones_like(
                    light_dists
                ) * self.light_power_activation(self.light_power)

                if self.config.use_falloff:
                    falloff = 1.0 / jnp.maximum(light_dists**2, 1e-5)
                    light_radiance = falloff * light_radiance

                if self.light_max_angle > 0.0:
                    angle_dot = math.dot(
                        -light_dirs, rays.vcam_look[..., None, :], keepdims=True
                    )
                    angle = jnp.arccos(angle_dot)
                    mask = (
                        (angle * 180.0 / jnp.pi) <= (self.light_max_angle / 2.0)
                    ) & (angle_dot > 0)
                    light_radiance = jnp.where(
                        mask, light_radiance, jnp.zeros_like(light_radiance)
                    )

            # Zero light
            if self.config.light_zero:
                light_radiance = jnp.where(
                    light_dists < self.config.light_near,
                    jnp.zeros_like(light_radiance),
                    light_radiance,
                )

            # Apply occlusion
            occ = sampler_results["occ"][..., :1].reshape(
                ref_rays.origins[..., :1].shape
            )
            ref_sampler_results = [
                {
                    "occ": jnp.repeat(occ, self.num_rgb_channels, axis=-1),
                    "acc": jnp.repeat(occ, self.num_rgb_channels, axis=-1)
                },
            ]
            light_radiance = light_radiance * (1.0 - occ)

            if self.config.sl_relight:
                sl_mult = render_utils.get_sl_color(
                    kwargs["env_map"],
                    kwargs["env_map_w"],
                    kwargs["env_map_h"],
                    ref_rays.vcam_up,
                    ref_rays.vcam_look,
                    ref_rays.origins,
                    ref_rays.vcam_origins,
                    hfov=self.config.sl_hfov,
                    vfov=self.config.sl_vfov,
                    shift=self.config.sl_shift,
                    mult=self.config.sl_mult,
                    invert=self.config.sl_invert,
                )
                light_radiance = light_radiance * sl_mult

            # Possibly project onto an environment or structured light (SL)
            rgb = jnp.repeat(light_radiance, self.num_rgb_channels, axis=-1)

            if self.config.material_loss_radius < float("inf"):
                mask = (
                    jnp.linalg.norm(ref_rays.origins, axis=-1, keepdims=True)
                    < self.config.material_loss_radius
                ).astype(jnp.float32)
                rgb = utils.stopgrad_with_weight(rgb, mask)

            rgb = jnp.maximum(rgb, 0.0)

            return rgb, rgb, ref_sampler_results

        return active_fn

    def _integrate_slf_variate(
        self,
        rng,
        rays,
        feature,
        sampler_results,
        material,
        radiance_cache_fn,
        surface_lf_fn,
        env_map_fn,
        active_fn,
        train_frac,
        train,
        **kwargs,
    ):
        """
        SLF variate path: runs two separate integration calls and merges them.
        """
        key, rng = utils.random_split(rng)
        integrated_outputs_cache = self.get_outgoing_radiance(
            rng=key,
            rays=rays,
            feature=feature,
            sampler_results=sampler_results,
            material=material,
            num_secondary_samples=self.get_num_secondary_samples_diff(train),
            radiance_cache_fn=radiance_cache_fn,
            env_map_fn=env_map_fn,
            active_fn=active_fn,
            train_frac=train_frac,
            train=train,
            **kwargs,
        )

        key, rng = utils.random_split(rng)
        integrated_outputs_slf = self.get_outgoing_radiance(
            rng=key,
            rays=rays,
            feature=feature,
            sampler_results=sampler_results,
            material=material,
            num_secondary_samples=self.get_num_secondary_samples_diff(train),
            radiance_cache_fn=surface_lf_fn,
            env_map_fn=env_map_fn,
            active_fn=active_fn,
            train_frac=train_frac,
            train=train,
            last_integrated_outputs=integrated_outputs_cache,
            **kwargs,
        )

        # Merge the two sets of outputs:
        final_outputs = dict(integrated_outputs_cache)  # copy

        for k in [
            "radiance_out",
            "diffuse_radiance_out",
            "specular_radiance_out",
            "direct_radiance_out",
            "indirect_radiance_out",
            "irradiance",
        ]:
            if (
                k not in integrated_outputs_cache
                or k not in integrated_outputs_slf
            ):
                continue

            combined_val = (
                integrated_outputs_cache[k]
                - integrated_outputs_slf[k]
            )
            final_outputs[k] = combined_val

        final_keys = list(final_outputs.keys())

        for f in final_keys:
            final_outputs[f + "_cache"] = integrated_outputs_cache.get(f)
            final_outputs[f + "_slf"] = integrated_outputs_slf.get(f)

        return final_outputs

    def _integrate_standard(
        self,
        rng,
        rays,
        feature,
        sampler_results,
        material,
        radiance_cache_fn,
        surface_lf_fn,
        env_map_fn,
        active_fn,
        slf_variate,
        train_frac,
        train,
        mesh,
        **kwargs,
    ):
        """
        Standard path for direct+indirect lighting, optionally using surface LF if enabled.
        """
        key, rng = utils.random_split(rng)
        integrated_outputs = self.get_outgoing_radiance(
            rng=key,
            rays=rays,
            feature=feature,
            sampler_results=sampler_results,
            material=material,
            num_secondary_samples=self.get_num_secondary_samples(train),
            radiance_cache_fn=(
                surface_lf_fn
                if (self.use_surface_light_field and not slf_variate)
                else radiance_cache_fn
            ),
            env_map_fn=env_map_fn,
            active_fn=active_fn,
            train_frac=train_frac,
            train=train,
            mesh=mesh,
            **kwargs,
        )
        return integrated_outputs

    def _apply_emission_and_residual_albedo(
        self, emission, residual_albedo, material, integrated_outputs
    ):
        """
        Combine the integrated lighting with the diffuse emission or residual albedo.
        """
        # Base lighting
        if self.config.use_transient:
            rgb = integrated_outputs["direct_radiance_out"]
        else:
            rgb = integrated_outputs["radiance_out"]

        # If we have diffuse emission, simply add it to the final color.
        if self.use_diffuse_emission:
            rgb = rgb + emission
        # Otherwise if we have a residual albedo, we add irradiance * that residual.
        elif self.use_residual_albedo:
            rgb = rgb + integrated_outputs["irradiance"] * residual_albedo

        # Finally, clamp the result to the maximum RGB (if specified).
        return rgb

    def _finalize_outputs(
        self,
        rays,
        outputs,
        integrated_outputs,
        final_rgb,
        material,
        emission,
        residual_albedo,
        sampler_results,
    ):
        """
        Gather all relevant outputs (material, lighting, references, etc.) into `outputs`.
        """
        # Record material properties in outputs
        for k in material.keys():
            outputs["material_" + k] = material[k]

        # Record lighting outputs
        outputs["lighting_emission"] = emission
        outputs["lighting_irradiance"] = integrated_outputs["irradiance"].reshape(
            material["albedo"].shape
        )

        # Example: if "occ" not in sampler_results, fill outputs
        if "occ" not in sampler_results:
            if self.use_active:
                outputs["occ"] = integrated_outputs.get("occ", None)
            else:
                outputs["occ"] = jnp.zeros_like(final_rgb)

        # The final main color
        outputs["rgb"] = final_rgb

        # Zero out "invalid" bins for transient indirect
        outputs["direct_diffuse_rgb"] = (
            integrated_outputs.get("direct_diffuse_radiance_out") + emission
            if integrated_outputs.get("direct_diffuse_radiance_out") is not None
            else None
        )
        outputs["direct_specular_rgb"] = integrated_outputs.get(
            "direct_specular_radiance_out"
        )
        outputs["direct_rgb"] = integrated_outputs.get("direct_radiance_out")

        if self.config.use_transient and self.use_indirect:
            (
                transient_indirect_diffuse,
                transient_indirect_specular,
            ) = render_utils.zero_invalid_bins(
                integrated_outputs.get("indirect_diffuse_radiance_out"),
                integrated_outputs.get("indirect_specular_radiance_out"),
                rays,
                sampler_results["means"],
                self.config,
            )

            # Additional lumps of relevant transient or integrated results
            outputs["transient_indirect"] = transient_indirect_diffuse + transient_indirect_specular
            outputs["transient_indirect_diffuse"] = transient_indirect_diffuse 
            outputs["transient_indirect_specular"] = transient_indirect_specular
        elif self.config.use_transient:
            outputs["transient_indirect"] = jnp.repeat(
                jnp.zeros_like(outputs["direct_diffuse_rgb"])[..., None, :],
                self.config.n_bins,
                axis=-2,
            )
            outputs["transient_indirect_diffuse"] = jnp.zeros_like(outputs["transient_indirect"])
            outputs["transient_indirect_specular"] = jnp.zeros_like(outputs["transient_indirect"])

        if self.use_indirect:
            outputs["indirect_diffuse_rgb"] = integrated_outputs.get(
                "indirect_diffuse_radiance_out"
            )
            outputs["indirect_specular_rgb"] = integrated_outputs.get(
                "indirect_specular_radiance_out"
            )
            outputs["indirect_rgb"] = integrated_outputs.get("indirect_radiance_out")
            outputs["indirect_occ"] = integrated_outputs.get("indirect_occ")
        else:
            outputs["indirect_diffuse_rgb"] = jnp.zeros_like(outputs["direct_diffuse_rgb"])
            outputs["indirect_specular_rgb"] = jnp.zeros_like(outputs["direct_specular_rgb"])
            outputs["indirect_rgb"] = jnp.zeros_like(outputs["direct_rgb"])
            outputs["indirect_occ"] = jnp.zeros_like(outputs["direct_rgb"])

        outputs["diffuse_rgb"] = integrated_outputs.get("diffuse_radiance_out")
        outputs["specular_rgb"] = integrated_outputs.get("specular_radiance_out")

        # If the integrator produced references for debug or training re-use, store them:
        for f in integrated_outputs.keys():
            if f.startswith("ref_"):
                outputs[f] = integrated_outputs.get(f)

        # Record distances
        ray_offset = rays.origins[..., None, :] - sampler_results['means']
        ray_dists = jnp.linalg.norm(ray_offset, axis=-1, keepdims=True)
        outputs['ray_dists'] = ray_dists

        if self.use_active:
            if self.config.learnable_light:
                lights = self.learnable_light.get_lights(rays.lights, rays.vcam_look, rays.vcam_up)
                lights = jax.lax.stop_gradient(lights)
            else:
                lights = rays.lights

            light_offset = lights[..., None, :] - sampler_results['means']
            light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
            outputs['light_dists'] = light_dists
        
        # Stopgrads
        mask = (
            jnp.linalg.norm(sampler_results['means'], axis=-1, keepdims=True)
            < self.config.material_loss_radius
        ).astype(jnp.float32)

        for output_key in outputs:
            if isinstance(outputs[output_key], jnp.ndarray) and 'transient' not in output_key:
                outputs[output_key] = utils.stopgrad_with_weight(outputs[output_key], mask)
            elif isinstance(outputs[output_key], jnp.ndarray) and 'transient' in output_key and self.config.use_transient:
                outputs[output_key] = utils.stopgrad_with_weight(outputs[output_key], mask[..., None, :])
    

@gin.configurable
class MaterialMLP(BaseMaterialMLP):
    use_active: bool = False

    def _initialize_integration_strategy(self):
        if self.use_active:
            extra_params = {
                "occ": {
                    "sum_over": [
                        ("direct_diffuse_occ", ()),
                    ],
                    "scale": 1.0,
                }
            }
        else:
            extra_params = {}

        self.integration_strategy = dict(
            **extra_params,
            **{
                "indirect_occ": {
                    "sum_over": [
                        ("indirect_specular_indirect_occ", ()),
                    ],
                    "scale": 0.5,
                },
                "radiance_out": {
                    "sum_over": [
                        ("direct_diffuse_radiance_out", ()),
                        ("direct_specular_radiance_out", ()),
                        ("indirect_diffuse_radiance_out", ()),
                        ("indirect_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "direct_radiance_out": {
                    "sum_over": [
                        ("direct_diffuse_radiance_out", ()),
                        ("direct_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "indirect_radiance_out": {
                    "sum_over": [
                        ("indirect_diffuse_radiance_out", ()),
                        ("indirect_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "diffuse_radiance_out": {
                    "sum_over": [
                        ("direct_diffuse_radiance_out", ()),
                        ("indirect_diffuse_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "specular_radiance_out": {
                    "sum_over": [
                        ("direct_specular_radiance_out", ()),
                        ("indirect_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "direct_diffuse_radiance_out": {
                    "sum_over": [
                        ("direct_diffuse_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "direct_specular_radiance_out": {
                    "sum_over": [
                        ("direct_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "indirect_diffuse_radiance_out": {
                    "sum_over": [
                        ("indirect_diffuse_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "indirect_specular_radiance_out": {
                    "sum_over": [
                        ("indirect_specular_radiance_out", ()),
                    ],
                    "scale": 1.0,
                },
                "irradiance": {
                    "sum_over": [
                        ("direct_diffuse_irradiance", ()),
                        ("indirect_diffuse_irradiance", ()),
                    ],
                    "scale": 0.5,
                },
                "direct_irradiance": {
                    "sum_over": [
                        ("direct_diffuse_irradiance", ()),
                    ],
                    "scale": 1.0,
                },
                "indirect_irradiance": {
                    "sum_over": [
                        ("indirect_diffuse_irradiance", ()),
                    ],
                    "scale": 1.0,
                },
            }
        )


@gin.configurable
class TransientMaterialMLP(BaseMaterialMLP):
    use_active: bool = True

    def _initialize_integration_strategy(self):
        self.integration_strategy = {
            "occ": {
                "sum_over": [
                    ("direct_diffuse_occ", ()),
                ],
                "scale": 1.0,
            },
            "indirect_occ": {
                "sum_over": [
                    ("indirect_specular_indirect_occ", ()),
                ],
                "scale": 0.5,
            },
            "radiance_out": {
                "sum_over": [
                    ("direct_diffuse_radiance_out", ()),
                    ("direct_specular_radiance_out", ()),
                    ("indirect_diffuse_radiance_out", (-2,)),
                    ("indirect_specular_radiance_out", (-2,)),
                ],
                "scale": 1.0,
            },
            "direct_radiance_out": {
                "sum_over": [
                    ("direct_diffuse_radiance_out", ()),
                    ("direct_specular_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "indirect_radiance_out": {
                "sum_over": [
                    ("indirect_diffuse_radiance_out", ()),
                    ("indirect_specular_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "diffuse_radiance_out": {
                "sum_over": [
                    ("direct_diffuse_radiance_out", ()),
                    ("indirect_diffuse_radiance_out", (-2,)),
                ],
                "scale": 1.0,
            },
            "specular_radiance_out": {
                "sum_over": [
                    ("direct_specular_radiance_out", ()),
                    ("indirect_specular_radiance_out", (-2,)),
                ],
                "scale": 1.0,
            },
            "direct_diffuse_radiance_out": {
                "sum_over": [
                    ("direct_diffuse_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "direct_specular_radiance_out": {
                "sum_over": [
                    ("direct_specular_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "indirect_diffuse_radiance_out": {
                "sum_over": [
                    ("indirect_diffuse_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "indirect_specular_radiance_out": {
                "sum_over": [
                    ("indirect_specular_radiance_out", ()),
                ],
                "scale": 1.0,
            },
            "irradiance": {
                "sum_over": [
                    ("direct_diffuse_irradiance", ()),
                    ("indirect_diffuse_irradiance", (-2,)),
                ],
                "scale": 0.5,
            },
            "direct_irradiance": {
                "sum_over": [
                    ("direct_diffuse_irradiance", ()),
                ],
                "scale": 1.0,
            },
            "indirect_irradiance": {
                "sum_over": [
                    ("indirect_diffuse_irradiance", (-2,)),
                ],
                "scale": 1.0,
            },
        }
