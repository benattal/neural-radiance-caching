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

import dataclasses
import functools
import operator
import time
from collections import namedtuple
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

import gin
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from absl import logging
from flax import linen as nn
from jax import random
import pdb

from internal import image, render, utils, coord, math


@gin.configurable
class VolumeIntegrator(nn.Module):
    """A mip-Nerf360 model containing all MLPs."""

    config: Any = None  # A Config class, must be set upon construction.
    bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.

    use_color_net: bool = False  # If True, use IDE to encode directions.

    net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
    net_depth: int = 4  # The depth of the second part of ML.
    net_width: int = 256  # The width of the second part of MLP.
    skip_layer: int = 2  # Add a skip connection to 2nd MLP every N layers.

    deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
    deg_origins: int = 4  # Degree of encoding for viewdirs or refdirs.

    normalize_weights: bool = False  # If True, use PE instead of IPE.

    def setup(self):
        def dir_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True
            )

        self.dir_enc_fn = dir_enc_fn

        def origins_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_origins,
                append_identity=True
            )

        self.origins_enc_fn = origins_enc_fn

        self.dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, "he_uniform")()
        )

        self.layers = [
            self.dense_layer(self.net_width, name=f"layer_{i}")
            for i in range(self.net_depth)
        ]

        self.output_layer = self.dense_layer(
            3, name="output_layer"
        )

    def run_color_network(self, origins, viewdirs):
        # Dir enc
        dir_enc = self.dir_enc_fn(viewdirs)
        origins_enc = self.origins_enc_fn(origins)

        # Run network
        x = jnp.concatenate([dir_enc, origins_enc], axis=-1)
        inputs = x

        # Evaluate network to produce the output density.
        for i in range(self.net_depth):
            x = self.layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return self.output_layer(x)

    @nn.compact
    def __call__(
        self,
        rng,
        rays,
        shader_results,
        train_frac=1.0,
        train=True,
        percentiles=(5, 50, 95),
        linear_rgb=False,
        compute_extras=False,
        compute_distance=True,
        bg_intensity_range=None,
        vignette=None,
        **kwargs
    ):
        if bg_intensity_range is None:
            bg_intensity_range = self.bg_intensity_range

        # Define or sample the background color for each ray.
        random_background = False

        if bg_intensity_range[0] == bg_intensity_range[1]:
            # If the min and max of the range are equal, just take it.
            bg_rgbs = bg_intensity_range[0]
        elif rng is None:
            # If rendering is deterministic, use the midpoint of the range.
            bg_rgbs = ((bg_intensity_range[0] + bg_intensity_range[1]) / 2) * 0.0
        else:
            random_background = True

            # Sample RGB values from the range for each ray.
            key, rng = utils.random_split(rng)

            bg_rgbs = random.normal(
                key,
                shape=shader_results["weights"].shape[:-1] + (3,),
            ) * (bg_intensity_range[1] - bg_intensity_range[0])

        # Render each ray.
        extras_to_render = [
            # Lighting
            "lighting_irradiance",
            # Material
            "material_albedo",
            "material_roughness",
            "material_F_0",
            "material_metalness",
            "material_diffuseness",
            "material_mirrorness",
            # Geometry
            "means",
            "normals",
            "normals_pred",
            "normals_to_use",
            "normals_shading",
            # Other
            "irradiance_cache",
            "irradiance_cache_rgb",
            "incoming_rgb",
            "incoming_s_dist",
            "person_rgb",
            "person_alpha",
            "diffuse_rgb",
            "specular_rgb",
            "occ",
            "indirect_occ",
            "direct_rgb",
            "indirect_rgb",
            "ambient_rgb",
            "irradiance_rgb",
            "light_radiance_rgb",
            "n_dot_l_rgb",
            "albedo_rgb",
            "direct_diffuse_rgb",
            "direct_specular_rgb",
            "indirect_diffuse_rgb",
            "indirect_specular_rgb",
            "ambient_diffuse_rgb",
            "ambient_specular_rgb",
            "light_dists",
            "ray_dists",
            "transient_indirect",
            "transient_indirect_specular",
            "transient_indirect_diffuse",
            "impulse_response",
        ]

        extras_to_always_render = [
            "irradiance_cache",
            "irradiance_cache_rgb",
            "lighting_irradiance",
            "diffuse_rgb",
            "specular_rgb",
            "occ",
            "indirect_occ",
            "direct_rgb",
            "indirect_rgb",
            "ambient_rgb",
            "irradiance_rgb",
            "light_radiance_rgb",
            "n_dot_l_rgb",
            "albedo_rgb",
            "direct_diffuse_rgb",
            "direct_specular_rgb",
            "indirect_diffuse_rgb",
            "indirect_specular_rgb",
            "ambient_diffuse_rgb",
            "ambient_specular_rgb",
            "means",
            "normals",
            "normals_pred",
            "normals_to_use",
            "beta",
            "light_dists",
            "ray_dists",
            "transient_indirect",
            "transient_indirect_specular",
            "transient_indirect_diffuse",
            "impulse_response"
        ]

        if "batch" in kwargs:
            shader_results["rgb"] = jnp.minimum(
                kwargs["batch"].rgb[..., :3].reshape(
                    shader_results["rgb"].shape[:-2] + (-1, 3)
                ),
                shader_results["rgb"]
            )

        rendering = render.volumetric_rendering(
            shader_results["rgb"],
            shader_results["weights"],
            shader_results["weights_no_filter"],
            shader_results["tdist"],
            bg_rgbs,
            compute_extras,
            extras=(
                {k: v for k, v in shader_results.items() if k in extras_to_render}
                if compute_extras else 
                {k: v for k, v in shader_results.items() if k in extras_to_always_render}
            ),
            percentiles=percentiles,
            compute_distance=compute_distance,
        )
        if random_background:
            rendering["bg_noise"] = (1.0 - shader_results["weights"].sum(axis=-1, keepdims=True)) * bg_rgbs
            rendering["rgb"] = rendering["rgb"] - rendering["bg_noise"]

        if self.use_color_net and (not linear_rgb):
            color_correction = self.run_color_network(
                rays.viewdirs, rays.origins,
            )

            color_correction = math.safe_exp(
                color_correction
            )

            rendering["rgb"] = rendering["rgb"] * color_correction
        
        if vignette is not None:
            rendering["rgb"] = rendering["rgb"] * vignette

        # Linear to srgb
        if not linear_rgb and (self.config.linear_to_srgb and rendering["rgb"] is not None):
            rendering["rgb"] = jnp.clip(
                image.linear_to_srgb(rendering["rgb"]),
                0.0,
                float("inf"),
            )
    
        if "beta" in rendering:
            acc = rendering["acc"]
            rendering["beta"] = (
                shader_results["beta"] * jax.lax.stop_gradient(shader_results["weights"][..., None])
            ).sum(axis=-2)
            rendering["beta"] = rendering["beta"] + jax.lax.stop_gradient((1.0 - acc[..., None]))

        return rendering


@gin.configurable
class GeometryVolumeIntegrator(VolumeIntegrator):
    """A mip-Nerf360 model containing all MLPs."""

    config: Any = None  # A Config class, must be set upon construction.
    bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.

    def setup(self):
        pass

    @nn.compact
    def __call__(self, rng, sampler_results, train_frac=1.0, train=True, **kwargs):
        # Geometry buffers
        extras_to_render = [
            "normals_to_use",
            "normals",
            "normals_pred",
            "feature",
            "means",
            "covs",
        ]

        # Reshape covariance
        sampler_results["covs"] = sampler_results["covs"].reshape(sampler_results["covs"].shape[:-2] + (9,))

        # Normalize weights
        if self.normalize_weights:
            sampler_results["weights"] = sampler_results["weights"] / (sampler_results["weights"].sum(axis=-1, keepdims=True) + 1e-8)

        rendering = render.volumetric_rendering(
            sampler_results["means"],
            sampler_results["weights"],
            sampler_results["tdist"],
            0.0,
            True,
            extras={k: v for k, v in sampler_results.items() if k in extras_to_render},
            normalize_weights_for_extras=False,
        )

        del rendering["rgb"]

        # Reshape covariance again
        sampler_results["covs"] = sampler_results["covs"].reshape(sampler_results["covs"].shape[:-1] + (3, 3))
        rendering["covs"] = rendering["covs"].reshape(rendering["covs"].shape[:-1] + (3, 3))

        # Reshape all
        rendering = jax.tree_util.tree_map(lambda x: x[Ellipsis, None, :], rendering)

        return rendering

@gin.configurable
class TransientVolumeIntegrator(VolumeIntegrator):
    """A mip-Nerf360 model containing all MLPs."""

    config: Any = None  # A Config class, must be set upon construction.
    bg_intensity_range: Tuple[float, float] = (1.0, 1.0)  # Background RGB range.

    def setup(self):
        pass

    @nn.compact
    def __call__(
        self,
        rng,
        rays,
        shader_results,
        train_frac=1.0,
        train=True,
        percentiles=(5, 50, 95),
        linear_rgb=False,
        compute_extras=False,
        compute_distance=True,
        bg_intensity_range=None,
        vignette=None,
        is_secondary=False,
        radiance_cache=None,
        material=False,
        **kwargs
    ):
        if bg_intensity_range is None:
            bg_intensity_range = self.bg_intensity_range

        # Define or sample the background color for each ray.
        random_background = False

        if bg_intensity_range[0] == bg_intensity_range[1]:
            # If the min and max of the range are equal, just take it.
            bg_rgbs = bg_intensity_range[0]
        elif rng is None:
            # If rendering is deterministic, use the midpoint of the range.
            bg_rgbs = ((bg_intensity_range[0] + bg_intensity_range[1]) / 2) * 0.0
        else:
            random_background = True

            # Sample RGB values from the range for each ray.
            key, rng = utils.random_split(rng)

            bg_rgbs = random.normal(
                key,
                shape=shader_results["weights"].shape[:-1] + (3,),
            ) * (bg_intensity_range[1] - bg_intensity_range[0])

        # Render each ray.
        extras_to_render = [
            # Lighting
            "lighting_irradiance",
            # Material
            "material_albedo",
            "material_roughness",
            "material_F_0",
            "material_metalness",
            "material_diffuseness",
            "material_mirrorness",
            # Geometry
            "means",
            "normals",
            "normals_pred",
            "normals_to_use",
            "normals_shading",
            # Other
            "irradiance_cache",
            "irradiance_cache_rgb",
            "incoming_rgb",
            "incoming_s_dist",
            "person_rgb",
            "person_alpha",
            "diffuse_rgb",
            "specular_rgb",
            "occ",
            "indirect_occ",
            "direct_rgb",
            "indirect_rgb",
            "ambient_rgb",
            "irradiance_rgb",
            "light_radiance_rgb",
            "n_dot_l_rgb",
            "albedo_rgb",
            "direct_diffuse_rgb",
            "direct_specular_rgb",
            "indirect_diffuse_rgb",
            "indirect_specular_rgb",
            "ambient_diffuse_rgb",
            "ambient_specular_rgb",
            "light_dists",
            "ray_dists",
            "transient_indirect",
            "transient_indirect_specular",
            "transient_indirect_diffuse",
            "impulse_response",
        ]

        extras_to_always_render = [
            "irradiance_cache",
            "irradiance_cache_rgb",
            "lighting_irradiance",
            "diffuse_rgb",
            "specular_rgb",
            "occ",
            "indirect_occ",
            "direct_rgb",
            "indirect_rgb",
            "ambient_rgb",
            "irradiance_rgb",
            "light_radiance_rgb",
            "n_dot_l_rgb",
            "albedo_rgb",
            "direct_diffuse_rgb",
            "direct_specular_rgb",
            "indirect_diffuse_rgb",
            "indirect_specular_rgb",
            "ambient_diffuse_rgb",
            "ambient_specular_rgb",
            "means",
            "normals",
            "normals_pred",
            "normals_to_use",
            "beta",
            "light_dists",
            "ray_dists",
            "transient_indirect",
            "transient_indirect_specular",
            "transient_indirect_diffuse",
            "impulse_response"
        ]

        if self.config.learnable_light:
            transient_shift = radiance_cache.shader.learnable_light.get_transient_shift()
        else:
            transient_shift = self.config.transient_shift
        
        if self.config.learnable_light:
            dark_level = radiance_cache.shader.learnable_light.get_dark_level()
        else:
            dark_level = 0.0
        
        if material:
            transient_shift = jax.lax.stop_gradient(transient_shift)
            dark_level = jax.lax.stop_gradient(dark_level)
        
        rendering = render.volumetric_transient_rendering(
            shader_results["direct_rgb"],
            shader_results["transient_indirect"],
            shader_results["weights"],
            shader_results["weights_no_filter"],
            shader_results["tdist"],
            bg_rgbs,
            compute_extras,
            extras=(
                {k: v for k, v in shader_results.items() if k in extras_to_render}
                if compute_extras else 
                {k: v for k, v in shader_results.items() if k in extras_to_always_render}
            ),
            percentiles=percentiles,
            compute_distance=compute_distance,
            n_bins = self.config.n_bins, 
            shift = transient_shift if not is_secondary else 0.0, 
            dark_level = dark_level if not is_secondary else 0.0, 
            impulse_response = rays.impulse_response if (not is_secondary or not self.config.filter_indirect) else None, 
            tfilter_sigma = self.config.tfilter_sigma if (not is_secondary or not self.config.filter_indirect) else 0.0, 
            exposure_time = self.config.exposure_time,
            filter_indirect = self.config.filter_indirect,
            filter_median = (self.config.filter_median and not is_secondary),
            itof = self.config.use_itof,
            config=self.config,
        )

        if random_background:
            rendering["bg_noise"] = (1.0 - shader_results["weights"].sum(axis=-1, keepdims=True)) * bg_rgbs
            rendering["rgb"] = rendering["rgb"] - rendering["bg_noise"]

        if self.use_color_net and (not linear_rgb):
            color_correction = self.run_color_network(
                rays.viewdirs, rays.origins,
            )

            color_correction = math.safe_exp(
                color_correction
            )

            rendering["rgb"] = rendering["rgb"] * color_correction
        
        if vignette is not None:
            rendering["rgb"] = rendering["rgb"] * vignette[..., None, :]

        # Linear to srgb
        if not linear_rgb and (self.config.linear_to_srgb and rendering["rgb"] is not None):
            rendering["rgb"] = jnp.clip(
                image.linear_to_srgb(rendering["rgb"]),
                0.0,
                float("inf"),
            )
    
        if "beta" in rendering:
            acc = rendering["acc"]
            rendering["beta"] = (
                shader_results["beta"] * jax.lax.stop_gradient(shader_results["weights"][..., None])
            ).sum(axis=-2)
            rendering["beta"] = rendering["beta"] + jax.lax.stop_gradient((1.0 - acc[..., None]))

        return rendering
