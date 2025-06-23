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
import gc
import time
from collections import namedtuple
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Text,
    Tuple,
    Union,
    Dict,
)
import pdb
import gin
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from absl import logging
from flax import linen as nn
from jax import random

from internal import (
    configs,
    integration,
    light_sampler,
    material,
    math,
    sampling,
    shading,
    nerf,
    surface_light_field,
    utils,
    ref_utils,
    coord,
)
from internal.inverse_render import render_utils


@gin.configurable
class Model(nn.Module):
    config: Any = None  # A Config class, must be set upon construction.

    use_material: bool = False

    # Random generator
    random_generator_2d: Any = render_utils.RandomGenerator2D(1, 1, False)

    # Importance samplers
    uniform_importance_samplers: Any = ((render_utils.UniformHemisphereSampler(), 1.0),)
    uniform_sphere_importance_samplers: Any = (
        (render_utils.UniformSphereSampler(), 1.0),
    )
    cosine_importance_samplers: Any = ((render_utils.CosineSampler(), 1.0),)
    light_importance_samplers: Any = ((render_utils.UniformHemisphereSampler(), 1.0),)
    distance_importance_samplers: Any = (
        (render_utils.UniformHemisphereSampler(), 1.0),
    )
    light_field_importance_samplers: Any = (
        (render_utils.UniformHemisphereSampler(), 1),
        (render_utils.MicrofacetSampler(), 1),
    )
    irradiance_importance_samplers: Any = (
        (render_utils.CosineSampler(), 1),
        (render_utils.LightSampler(), 1),
    )
    extra_ray_importance_samplers: Any = (
        (render_utils.UniformHemisphereSampler(), 1),
        (render_utils.IdentitySampler(), 1),
    )
    active_importance_samplers: Any = ((render_utils.ActiveSampler(), 1.0),)

    # Env map
    use_env_map: bool = False
    env_map_near: float = float("inf")
    env_map_far: float = float("inf")

    env_map_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )

    # Surface light field control variate
    use_surface_light_field: bool = False

    surface_lf_mem_distance_near: float = 1e-3
    surface_lf_mem_distance_far: float = 1e6

    surface_lf_mem_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )

    # Resample
    resample: bool = False
    num_resample: int = 1
    resample_render: bool = False
    resample_secondary: bool = False
    resample_argmax: bool = False
    use_raydist_for_secondary_only: bool = False

    logits_mult: float = 1.0
    logits_mult_secondary: float = 1.0
    weights_bias: float = 0.0

    # Geometry stopgrad
    stopgrad_geometry_weight: float = 1.0
    stopgrad_geometry_variate_weight: float = 0.0
    stopgrad_geometry_feature_weight: float = 1.0
    stopgrad_geometry_normals_weight: float = 1.0

    # Variate stopgrad
    stopgrad_weight_variate: float = 1.0
    stopgrad_weight_model: float = 1.0

    # Sampling strategy
    train_sampling_strategy: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 64),
        (1, 1, 64),
        (2, 2, 32),
    )

    render_sampling_strategy: Tuple[Tuple[int, int, int], ...] = (
        (0, 0, 64),
        (1, 1, 64),
        (2, 2, 32),
    )

    # Stopgrads
    stopgrad_cache_weight: Tuple[float, float] = (1.0, 1.0)
    stopgrad_slf_weight: Tuple[float, float] = (1.0, 1.0)
    stopgrad_env_map_weight: Tuple[float, float] = (1.0, 1.0)

    def do_resample(
        self,
        do_resample,
        is_secondary,
        train,
    ):
        return (
            do_resample
            or (train and self.resample)
            or (not train and self.resample_render)
            or (is_secondary and self.resample_secondary)
        )

    def use_volume_variate(
        self,
        is_secondary,
    ):
        return (
            (self.config.volume_variate_secondary and is_secondary)
            or (self.config.volume_variate and not is_secondary)
        )
    
    def get_variate_passes(self, is_secondary):
        return (
            self.config.volume_variate_passes_secondary if is_secondary else self.config.volume_variate_passes
        )

    def get_bg_and_raydist(self, is_secondary):
        if is_secondary:
            bg_intensity_range = (0.0, 0.0)
            use_raydist_fn = True
        else:
            bg_intensity_range = None
            use_raydist_fn = not self.use_raydist_for_secondary_only

        return bg_intensity_range, use_raydist_fn

    def maybe_resample(
        self,
        rng,
        resample,
        sampler_results,
        num_resample,
        inds=None,
        logits_mult=1.0,
    ):
        if not resample:
            sampler_results = utils.copy_tree(sampler_results)
            sampler_results["weights_no_filter"] = sampler_results["weights"]
            return sampler_results, None

        weights = sampler_results["weights"]
        logits = math.safe_log(weights + self.weights_bias) * logits_mult
        probs = jax.nn.softmax(logits, axis=-1)
        acc = weights.sum(axis=-1, keepdims=True)

        if self.resample_argmax:
            samples_shape = logits.shape

            all_inds = (
                jnp.arange(samples_shape[-1])
                .reshape(tuple(1 for _ in samples_shape[:-1]) + (samples_shape[-1],))
                .astype(jnp.int32)
            )
            inds_argmax = jnp.argmax(logits, axis=-1, keepdims=True)
            new_weights = jnp.where(
                all_inds == inds_argmax,
                jnp.zeros_like(weights),
                weights,
            )
            new_acc = new_weights.sum(axis=-1, keepdims=True)
            new_logits = math.safe_log(new_weights + self.weights_bias) * logits_mult
            new_probs = jax.nn.softmax(new_logits, axis=-1)

        if inds is None:
            if self.resample_argmax:
                key, rng = utils.random_split(rng)
                inds = jax.random.categorical(
                    key,
                    logits=new_logits[..., None],
                    axis=-2,
                    shape=(sampler_results["points"].shape[:-2] + (num_resample - 1,)),
                )
                inds = jnp.concatenate([inds_argmax, inds], axis=-1)
            else:
                key, rng = utils.random_split(rng)
                inds = jax.random.categorical(
                    key,
                    logits=logits[..., None],
                    axis=-2,
                    shape=(sampler_results["points"].shape[:-2] + (num_resample,)),
                )

        def take_multiple(path, x):
            if "_no_filter" in path:
                return x

            if isinstance(x, jnp.ndarray):
                if len(x.shape) == (len(sampler_results["points"].shape) - 2):
                    return jnp.take_along_axis(x[..., None], inds, axis=-1)
                elif len(x.shape) < len(sampler_results["points"].shape):
                    return jnp.take_along_axis(x, inds, axis=-1)
                elif len(x.shape) > len(sampler_results["points"].shape):
                    return jnp.take_along_axis(x, inds[..., None, None], axis=-3)
                else:
                    return jnp.take_along_axis(x, inds[..., None], axis=-2)

            return x

        # Filtered sampler results
        filtered_sampler_results = jax.tree_util.tree_map_with_path(
            take_multiple,
            sampler_results,
        )

        filtered_sampler_results["tdist"] = sampler_results["tdist"]
        filtered_sampler_results["sdist"] = sampler_results["sdist"]
        filtered_sampler_results["weights_no_filter"] = sampler_results["weights"]
        filtered_probs = jnp.take_along_axis(probs, inds, axis=-1)

        if self.resample_argmax:
            weights_argmax = filtered_sampler_results["weights"][..., :1]

            filtered_new_probs = jnp.take_along_axis(new_probs, inds[..., 1:], axis=-1)
            weights = filtered_sampler_results["weights"][
                ..., 1:
            ] / jax.lax.stop_gradient((num_resample - 1) * filtered_new_probs + 1e-8)
            filtered_sampler_results["weights"] = jnp.concatenate(
                [weights_argmax, weights], axis=-1
            )
        else:
            weights = filtered_sampler_results["weights"] / jax.lax.stop_gradient(
                num_resample * filtered_probs + 1e-8
            )
            filtered_sampler_results["weights"] = weights

        return filtered_sampler_results, inds

    def get_sampling_strategy(self, train, sampling_strategy):
        if sampling_strategy is not None:
            return sampling_strategy

        if train:
            return self.train_sampling_strategy
        else:
            return self.render_sampling_strategy

    def _get_logits_mult(self, is_secondary):
        if is_secondary:
            return self.logits_mult_secondary
        else:
            return self.logits_mult

    def _handle_secondary(
        self,
        rng,
        rays,
        is_secondary,
        integrator_results,
        train,
        train_frac,
        **render_kwargs,
    ):
        if not is_secondary:
            return integrator_results

        stopgrad_cache_weight = render_kwargs.pop("stopgrad_cache_weight", None) if is_secondary else None

        # Stopgrad cache
        integrator_keys = list(integrator_results.keys())

        for k in integrator_keys:
            if ("rgb" in k) or ("transient" in k) or ("acc" in k):
                integrator_results[f"{k}_no_stopgrad"] = jnp.copy(integrator_results[k])

                if stopgrad_cache_weight != (1.0, 1.0) and stopgrad_cache_weight is not None:
                    integrator_results[k] = utils.stopgrad_with_weight(
                        integrator_results[k],
                        stopgrad_cache_weight[1],
                    )
        
        # Env map
        key, rng = utils.random_split(rng)
        env_map_outputs = self._handle_env_map(
            key,
            rays,
            train,
            train_frac,
            **render_kwargs,
        )

        key, rng = utils.random_split(rng)
        integrator_results = self._composite_env_map(
            key,
            rays,
            integrator_results,
            env_map_outputs,
            train,
            train_frac,
            **render_kwargs,
        )

        return integrator_results

    def _handle_env_map(
        self,
        rng,
        rays,
        train,
        train_frac,
        **render_kwargs,
    ):
        use_env_map = render_kwargs.pop("use_env_map", True)
        env_map_outputs = {}

        # Use env map
        if self.use_env_map and use_env_map:
            # Helpers vars
            env_map = render_kwargs.pop("env_map", None)
            env_map_w = render_kwargs.pop("env_map_w", None)
            env_map_h = render_kwargs.pop("env_map_h", None)
            stopgrad_cache_weight = render_kwargs.pop("stopgrad_cache_weight", None)

            # Stopgrad rays
            env_rays = utils.partial_stopgrad_rays(rays, stopgrad_cache_weight)

            if env_map is not None:
                # Explicit env map provided
                env_map_values = render_utils.get_environment_color(
                    env_rays,
                    env_map,
                    env_map_w,
                    env_map_h,
                )

                env_map_outputs = {
                    "incoming_rgb": env_map_values.reshape(rays.origins.shape[:-1] + (self.config.num_rgb_channels,)),
                }
            else:
                # Use learned env map
                key, rng = utils.random_split(rng)
                env_map_outputs = self.env_map(
                    key,
                    env_rays,
                    {
                        'means': env_rays.origins[..., None, :],
                        'covs': jnp.ones_like(env_rays.origins)[..., None, :],
                    },
                    env_rays.origins[..., None, :],
                    env_rays.viewdirs[..., None, :],
                    roughness=jnp.zeros_like(env_rays.origins[..., None, :1]),
                    shader_bottleneck=None,
                    train=train,
                    train_frac=train_frac,
                )

                env_map_outputs["incoming_rgb_no_stopgrad"] = env_map_outputs["incoming_rgb"]

                if stopgrad_cache_weight != (1.0, 1.0) and stopgrad_cache_weight is not None:
                    env_map_outputs["incoming_rgb"] = utils.stopgrad_with_weight(
                        env_map_outputs["incoming_rgb"],
                        stopgrad_cache_weight[1],
                    )


        return env_map_outputs

    def _composite_env_map(
        self,
        rng,
        rays,
        integrator_results,
        env_map_outputs,
        train,
        train_frac,
        **render_kwargs,
    ):
        use_env_map = render_kwargs.pop("use_env_map", True)

        # Use env map
        if self.use_env_map and use_env_map:
            # Combine env map rgb with integrated rgb
            acc = integrator_results["acc"]

            env_map_rgb = env_map_outputs["incoming_rgb"].reshape(
                integrator_results["rgb"].shape
            )
            env_map_rgb_no_stopgrad = env_map_outputs["incoming_rgb_no_stopgrad"].reshape(
                integrator_results["rgb"].shape
            )
            
            integrator_results["rgb"] = (
                integrator_results["rgb"]
                + env_map_rgb * (1.0 - acc[..., None])
            )

            integrator_results["rgb_no_stopgrad"] = (
                integrator_results["rgb_no_stopgrad"]
                + jax.lax.stop_gradient(env_map_rgb) * (1.0 - acc[..., None])
            )

            integrator_results["env_map_rgb"] = env_map_rgb
            integrator_results["env_map_rgb_no_stopgrad"] = env_map_rgb_no_stopgrad

        return integrator_results
    
    def apply_shader_and_integrator(
        self,
        rng,
        rays,
        cache_rays,
        sampler_results,
        filtered_sampler_results,
        stopgrad_map,
        train,
        train_frac,
        is_secondary,
        bg_intensity_range,
        **render_kwargs,
    ):
        weights_only = render_kwargs.get("weights_only", False)

        filtered_sampler_results = utils.apply_stopgrad_fields(
            filtered_sampler_results, stopgrad_map
        )

        # Shade and integrate
        if weights_only:
            shader_results = self.make_weights_only_shader_results(
                cache_rays, filtered_sampler_results
            )
        else:
            key, rng = utils.random_split(rng)
            shader_results = self.shader(
                rng=key,
                rays=cache_rays,
                sampler_results=filtered_sampler_results,
                filtered_sampler_results=filtered_sampler_results,
                train_frac=train_frac,
                train=train,
                is_secondary=is_secondary,
                **render_kwargs,
            )

        # Integrate
        key, rng = utils.random_split(rng)
        integrator_results = self.integrator(
            rng=key,
            rays=cache_rays,
            shader_results=shader_results,
            train_frac=train_frac,
            train=train,
            bg_intensity_range=bg_intensity_range,
            is_secondary=is_secondary,
            **render_kwargs,
        )

        # Background color
        key, rng = utils.random_split(rng)
        integrator_results = self._handle_secondary(
            key,
            rays,
            is_secondary,
            integrator_results,
            train,
            train_frac,
            **render_kwargs,
        )

        if self.use_volume_variate(is_secondary) and not weights_only:
            # Volume variate
            key, rng = utils.random_split(rng)
            variate_results = self.shader(
                rng=key,
                rays=cache_rays,
                sampler_results=sampler_results[-1],
                filtered_sampler_results=sampler_results[-1],
                train_frac=train_frac,
                train=train,
                is_secondary=is_secondary,
                passes=self.get_variate_passes(is_secondary),
                **render_kwargs,
            )
            variate_results["weights_no_filter"] = variate_results["weights"]

            key, rng = utils.random_split(rng)
            biased_total_integrator_results = self.integrator(
                rng=key,
                rays=cache_rays,
                shader_results=variate_results,
                train_frac=train_frac,
                train=train,
                bg_intensity_range=bg_intensity_range,
                is_secondary=is_secondary,
                **render_kwargs,
            )

            key, rng = utils.random_split(rng)
            biased_total_integrator_results = self._handle_secondary(
                key,
                rays,
                is_secondary,
                biased_total_integrator_results,
                train,
                train_frac,
                **render_kwargs,
            )

            # Filtered volume variate
            key, rng = utils.random_split(rng)
            filtered_variate_results = self.shader(
                rng=key,
                rays=cache_rays,
                sampler_results=filtered_sampler_results,
                filtered_sampler_results=filtered_sampler_results,
                train_frac=train_frac,
                train=train,
                is_secondary=is_secondary,
                passes=self.get_variate_passes(is_secondary),
                **render_kwargs,
            )

            key, rng = utils.random_split(rng)
            biased_integrator_results = self.integrator(
                rng=key,
                rays=cache_rays,
                shader_results=filtered_variate_results,
                train_frac=train_frac,
                train=train,
                bg_intensity_range=bg_intensity_range,
                is_secondary=is_secondary,
                **render_kwargs,
            )

            key, rng = utils.random_split(rng)
            biased_integrator_results = self._handle_secondary(
                key,
                rays,
                is_secondary,
                biased_integrator_results,
                train,
                train_frac,
                **render_kwargs,
            )

            # Combine
            self._handle_volume_variate_pass(
                integrator_results,
                biased_integrator_results,
                biased_total_integrator_results,
                keys=["rgb", "diffuse_rgb", "specular_rgb", "direct_rgb", "indirect_rgb", "transient_indirect"],
                stopgrad_weight_variate=self.stopgrad_weight_variate,
                stopgrad_weight_model=self.stopgrad_weight_model,
            )

            # Shader results
            shader_results = variate_results if not is_secondary else shader_results
        
        return shader_results, integrator_results

    def _handle_volume_variate_pass(
        self,
        unbiased_integrator_results,
        biased_integrator_results,
        biased_total_integrator_results,
        keys,
        stopgrad_weight_variate=1.0,
        stopgrad_weight_model=1.0,
    ):
        for output_key in keys:
            if (
                (output_key not in biased_total_integrator_results or biased_total_integrator_results[output_key] is None)
                or (output_key not in biased_integrator_results or biased_integrator_results[output_key] is None)
                or (output_key not in unbiased_integrator_results or unbiased_integrator_results[output_key] is None)
            ):
                continue

            unbiased_integrator_results[output_key] = (
                utils.stopgrad_with_weight(
                    biased_total_integrator_results[output_key]
                    - biased_integrator_results[output_key].reshape(unbiased_integrator_results[output_key].shape),
                    stopgrad_weight_variate
                )
                + utils.stopgrad_with_weight(unbiased_integrator_results[output_key], stopgrad_weight_model)
            )


@gin.configurable
class BaseNeRFModel(Model):
    sampler_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )
    shader_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({})
    integrator_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    extra_model_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )

    @nn.compact
    def __call__(self, rng, rays, **render_kwargs):
        # Get variables from render_kwargs
        train_frac = render_kwargs.pop("train_frac", 1.0)
        train = render_kwargs.pop("train", True)
        sampling_strategy = render_kwargs.pop("sampling_strategy", None)
        cache_outputs = render_kwargs.pop("cache_outputs", None)
        filtered_sampler_inds = render_kwargs.pop("filtered_sampler_inds", None)

        is_secondary = render_kwargs.pop("is_secondary", False)
        do_resample = self.do_resample(render_kwargs.get("resample", False), is_secondary, train)
        bg_intensity_range, use_raydist_fn = self.get_bg_and_raydist(is_secondary)

        # Rays
        if is_secondary and self.use_env_map:
            rays = rays.replace(
                far=jnp.minimum(rays.far, self.config.env_map_distance)
            )

        stopgrad_cache_weight = render_kwargs.pop("stopgrad_cache_weight", None) if is_secondary else None
        # render_kwargs.pop("stopgrad_cache_weight", None)
        # stopgrad_cache_weight = self.stopgrad_cache_weight if is_secondary else None
        cache_rays = utils.partial_stopgrad_rays(rays, stopgrad_cache_weight)

        # Check for use_slf_variate_direct flag
        use_slf = render_kwargs.pop("use_slf", False)
        
        if use_slf and self.use_surface_light_field:
            # Try to get results from surface light field
            key, rng = utils.random_split(rng)
            slf_results = self.get_slf_results(
                key,
                rays,
                train_frac,
                train,
                **render_kwargs
            )
            
            return slf_results

        env_map_only = render_kwargs.pop("env_map_only", False)

        if env_map_only and self.use_env_map:
            key, rng = utils.random_split(rng)
            env_map_outputs = self._handle_env_map(
                key,
                rays,
                train_frac,
                train,
                **render_kwargs
            )
            
            return env_map_outputs

        # Samples
        if cache_outputs is None:
            key, rng = utils.random_split(rng)
            sampler_results = self.sampler(
                rng=key,
                rays=cache_rays,
                train_frac=train_frac,
                train=train,
                sampling_strategy=self.get_sampling_strategy(train, sampling_strategy),
                use_raydist_fn=use_raydist_fn,
                is_secondary=is_secondary,
                **render_kwargs,
            )
        else:
            sampler_results = utils.copy_tree(cache_outputs["sampler"])

        # Resample
        key, rng = utils.random_split(rng)
        filtered_sampler_results, filtered_sampler_inds = self.maybe_resample(
            rng=key,
            resample=do_resample,
            sampler_results=sampler_results[-1],
            num_resample=self.num_resample,
            logits_mult=self._get_logits_mult(is_secondary),
            inds=filtered_sampler_inds,
        )

        stopgrad_map = {
            "weights": self.stopgrad_geometry_weight,
            "weights_no_filter": self.stopgrad_geometry_weight,
            "feature": self.stopgrad_geometry_feature_weight,
            "normals_pred": self.stopgrad_geometry_normals_weight,
            "normals": self.stopgrad_geometry_normals_weight,
            "normals_to_use": self.stopgrad_geometry_normals_weight,
        }

        stopgrad_map = stopgrad_map if do_resample else {}

        key, rng = utils.random_split(rng)
        shader_results, integrator_results = self.apply_shader_and_integrator(
            key,
            rays,
            cache_rays,
            sampler_results,
            filtered_sampler_results,
            stopgrad_map,
            train,
            train_frac,
            is_secondary,
            bg_intensity_range,
            stopgrad_cache_weight=stopgrad_cache_weight,
            **render_kwargs,
        )

        return {
            "main": {
                "loss_weight": 1.0,
                "sampler": sampler_results,
                "filtered_sampler_inds": filtered_sampler_inds,
                "shader": shader_results,
                "geometry": sampler_results[-1],
                "integrator": integrator_results,
            },
            "render": integrator_results,
        }


@gin.configurable
class NeRFModel(BaseNeRFModel):
    def setup(self):
        self.sampler = sampling.ProposalVolumeSampler(
            config=self.config,
            **self.sampler_params,
            **self.extra_model_params,
            name="Sampler",
        )

        self.shader = nerf.NeRFMLP(
            config=self.config,
            env_map_near=self.env_map_near,
            env_map_far=self.env_map_far,
            **self.shader_params,
            name="Shader",
        )

        self.integrator = integration.VolumeIntegrator(
            config=self.config,
            **self.integrator_params,
            name="Integrator",
        )

        # Env map
        if self.use_env_map:
            env_map_params = {
                k: self.env_map_params[k]
                for k in self.env_map_params.keys()
            }
            env_map_params["distance_near"] = self.env_map_near
            env_map_params["distance_far"] = self.env_map_far

            self.env_map = surface_light_field.SurfaceLightFieldMLP(
                name="EnvMap", config=self.config, **env_map_params
            )

        # Surface light field
        if self.use_surface_light_field:
            surface_lf_mem_params = {
                key: self.surface_lf_mem_params[key]
                for key in self.surface_lf_mem_params.keys()
            }

            if self.use_env_map and self.config.env_map_distance < float("inf"):
                surface_lf_mem_params["distance_near"] = self.surface_lf_mem_distance_near
                surface_lf_mem_params["distance_far"] = self.config.env_map_distance
            else:
                surface_lf_mem_params["distance_near"] = self.surface_lf_mem_distance_near
                surface_lf_mem_params["distance_far"] = self.surface_lf_mem_distance_far

            self.surface_lf_mem = surface_light_field.SurfaceLightFieldMLP(
                name='SurfaceLightFieldMem',
                use_env_alpha=True,
                config=self.config,
                **surface_lf_mem_params
            )

    def make_weights_only_shader_results(
        self,
        rays,
        sampler_results,
    ):
        shader_results = utils.copy_tree(sampler_results)
        shader_results["rgb"] = jnp.repeat(
            jnp.ones_like(shader_results["weights"])[..., None],
            3,
            -1,
        )

        return shader_results

    def get_slf_results(self, rng, rays, train_frac, train, **render_kwargs):
        stopgrad_slf_weight = render_kwargs.pop("stopgrad_slf_weight", None)
        # render_kwargs.pop("stopgrad_slf_weight", None)
        # stopgrad_slf_weight = self.stopgrad_slf_weight
        dist_only = render_kwargs.get("dist_only", False)
        slf_rays = utils.partial_stopgrad_rays(rays, stopgrad_slf_weight)

        # Remove unnecessary
        render_kwargs.pop("origins", None)
        render_kwargs.pop("viewdirs", None)
        
        # Call surface_lf_mem directly
        key, rng = utils.random_split(rng)
        slf_results = self.surface_lf_mem(
            key,
            slf_rays,
            {
                'means': rays.origins[..., None, :],
                'covs': jnp.ones_like(rays.origins[..., None, :]),
            },
            rays.origins[..., None, :],
            rays.viewdirs[..., None, :],
            roughness=jnp.zeros_like(rays.origins[..., None, :1]),
            shader_bottleneck=None,
            train=train,
            train_frac=train_frac,
            **render_kwargs,
        )

        if dist_only:
            return slf_results
        
        # Create a new output structure with SLF results
        slf_integrator_results = {
            "rgb": slf_results["incoming_rgb"],
            "acc": slf_results["incoming_acc"],
        }

        key, rng = utils.random_split(rng)
        slf_integrator_results = self._handle_secondary(
            key,
            rays,
            True,
            slf_integrator_results,
            train,
            train_frac,
            **render_kwargs,
        )

        # Final
        slf_integrator_results = dict(
            **slf_integrator_results,
            **slf_results,
        )

        slf_integrator_results["incoming_rgb"] = slf_integrator_results["rgb_no_stopgrad"]
        slf_integrator_results["incoming_acc"] = slf_integrator_results["acc_no_stopgrad"]

        # Return
        return slf_integrator_results


@gin.configurable
class TransientNeRFModel(BaseNeRFModel):
    sampler_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )
    shader_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({})
    integrator_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    extra_model_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )

    def setup(self):
        self.sampler = sampling.ProposalVolumeSampler(
            config=self.config,
            **self.sampler_params,
            **self.extra_model_params,
            name="Sampler",
        )

        self.shader = nerf.TransientNeRFMLP(
            config=self.config,
            **self.shader_params,
            name="Shader",
        )

        self.integrator = integration.TransientVolumeIntegrator(
            config=self.config,
            **self.integrator_params,
            name="Integrator",
        )

    def make_weights_only_shader_results(
        self,
        rays,
        sampler_results,
    ):
        shader_results = utils.copy_tree(sampler_results)
        light_offset = rays.lights[..., None, :] - sampler_results["means"]
        light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
        shader_results["light_dists"] = light_dists

        ray_offset = rays.origins[..., None, :] - sampler_results["means"]
        ray_dists = jnp.linalg.norm(ray_offset, axis=-1, keepdims=True)
        shader_results["ray_dists"] = ray_dists
        shader_results["transient_indirect"] = jnp.ones(
            shader_results["weights"].shape
            + (self.config.n_bins, self.config.num_rgb_channels)
        )
        shader_results["transient_indirect_specular"] = jnp.ones(
            shader_results["weights"].shape
            + (self.config.n_bins, self.config.num_rgb_channels)
        )
        shader_results["transient_indirect_diffuse"] = jnp.ones(
            shader_results["weights"].shape
            + (self.config.n_bins, self.config.num_rgb_channels)
        )

        shader_results["rgb"] = jnp.repeat(
            jnp.ones_like(shader_results["weights"])[..., None],
            self.config.num_rgb_channels,
            -1,
        )

        shader_results["direct_rgb"] = jnp.repeat(
            jnp.ones_like(shader_results["weights"])[..., None],
            self.config.num_rgb_channels,
            -1,
        )

        return shader_results


@gin.configurable
class VignetteMap(nn.Module):
    config: Any = None  # A Config class, must be set upon construction.

    deg_vignette: int = 2
    net_depth_vignette: int = 2  # The depth of the second part of MLP.
    net_width_vignette: int = 64  # The width of the second part of MLP.
    skip_layer_vignette: int = 4  # Add a skip connection to 2nd MLP every N layers.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.

    def setup(self):
        def vignette_enc_fn(direction):
            return coord.pos_enc(
                direction, min_deg=0, max_deg=self.deg_vignette, append_identity=True
            )

        self.vignette_enc_fn = vignette_enc_fn

        self.dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, "he_uniform")()
        )

        self.vignette_layers = [
            self.dense_layer(self.net_width_vignette, name="layer_%d" % i)
            for i in range(self.net_depth_vignette)
        ]

        self.output_vignette_layer = self.dense_layer(1, name="output_layer")

    @nn.compact
    def __call__(
        self,
        rays,
    ):
        # Run view dependent network
        v_input = math.dot(rays.viewdirs, rays.look, axis=-1, keepdims=True)
        v_input = self.vignette_enc_fn(v_input)
        x = self.run_vignette_network(v_input)

        # Get RGB
        vignette = nn.sigmoid(self.output_vignette_layer(x)) * 2.0

        return vignette

    def run_vignette_network(self, x):
        inputs = x

        # Evaluate network to produce the output density.
        for i in range(self.net_depth_vignette):
            x = self.vignette_layers[i](x)
            x = self.net_activation(x)

        if i % self.skip_layer_vignette == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

        return x


@gin.configurable
class BaseMaterialModel(Model):
    use_material: bool = True
    use_light_sampler: bool = True

    use_resample_depth: bool = False
    depth_key: str = "distance_median"

    loss_weight: float = 1.0
    loss: str = "rawnerf_unbiased"
    linear_to_srgb: bool = False

    cache_loss_weight: float = 1.0
    cache_loss: str = "charb"
    cache_linear_to_srgb: bool = True

    material_loss_weight: float = 1.0
    material_loss: str = "rawnerf_unbiased"
    material_linear_to_srgb: bool = False

    stopgrad_samples: bool = False
    slf_variate: bool = True

    share_material: bool = False
    share_light_power: bool = False

    use_vignette: bool = False

    cache_model_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    light_sampler_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    sampler_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )
    shader_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict({})
    integrator_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    extra_model_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )

    resample: bool = False
    resample_render: bool = False
    resample_secondary: bool = False
    num_resample: int = 1
    logits_mult: float = 1.0
    logits_mult_secondary: float = 1.0
    weights_bias: float = 0.0

    stopgrad_geometry_weight: float = 0.0
    stopgrad_geometry_variate_weight: float = 0.0
    stopgrad_geometry_feature_weight: float = 0.0
    stopgrad_geometry_normals_weight: float = 1.0

    stopgrad_geometry_weight_consistency: float = 0.0
    stopgrad_geometry_feature_weight_consistency: float = 0.0
    stopgrad_geometry_normals_weight_consistency: float = 0.0

    stopgrad_weight_variate: float = 0.0
    stopgrad_weight_model: float = 1.0

    def setup(self):
        self.cache = NeRFModel(
            config=self.config,
            use_surface_light_field=self.use_surface_light_field,
            **self.cache_model_params,
            **self.extra_model_params,
            name="Cache",
        )

        if self.use_light_sampler:
            self.light_sampler = light_sampler.LightMLP(
                config=self.config,
                **self.light_sampler_params,
                name="LightSampler",
            )

        self.shader = material.MaterialMLP(
            config=self.config,
            use_surface_light_field=self.use_surface_light_field,
            **self.shader_params,
            name="MaterialShader",
        )

        self.integrator = integration.VolumeIntegrator(
            config=self.config,
            **self.integrator_params,
            name="MaterialIntegrator",
        )

        if self.use_vignette:
            self.vignette_map = VignetteMap(
                config=self.config,
                name="VignetteMap",
            )

    @nn.compact
    def __call__(self, rng, rays, **render_kwargs):
        train_frac = render_kwargs.pop("train_frac", 1.0)
        train = render_kwargs.pop("train", True)
        passes = render_kwargs.pop("passes", ("cache", "light", "material"))
        compute_extras = render_kwargs.pop("compute_extras", False)
        extra_ray_regularizer = render_kwargs.pop("extra_ray_regularizer", False)
        is_secondary = render_kwargs.pop("is_secondary", False) or (
            "is_secondary" in passes
        )
        cache_outputs = render_kwargs.pop("cache_outputs", None)

        # Possibly handle a bypass pass (geometry-only, SLF-only, etc.).
        key, rng = utils.random_split(rng)
        bypass_outputs, bypass = self._maybe_bypass_pipeline(
            key,
            rays,
            passes,
            train_frac,
            train,
            is_secondary=is_secondary,
            **render_kwargs,
        )

        if bypass_outputs is not None and bypass:
            return bypass_outputs

        # Calculate vignette
        vignette = None

        if self.use_vignette and not is_secondary:
            vignette = self.vignette_map(rays)

        # Obtain the cache outputs
        key, rng = utils.random_split(rng)
        cache_outputs = self._handle_cache_pass(
            rng=key,
            rays=rays,
            train_frac=train_frac,
            train=train,
            is_secondary=is_secondary,
            vignette=vignette,
            cache_outputs=cache_outputs,
            radiance_cache=self,
            **render_kwargs,
        )


        if ("material" in passes) and self.use_material:
            # Resample sampler results for material stage
            filtered_sampler_inds = render_kwargs.pop("filtered_sampler_inds", cache_outputs["filtered_sampler_inds"])

            key, rng = utils.random_split(rng)
            filtered_sampler_results, cache_shader_results = self._get_material_samples(
                key,
                rays,
                cache_outputs["sampler"][-1],
                filtered_sampler_inds,
                train=train,
                train_frac=train_frac,
                is_secondary=is_secondary,
                **render_kwargs,
            )

            # Run learnable light sampler for material stage
            key, rng = utils.random_split(rng)
            light_sampler_results = self._handle_light_sampling_pass(
                rng=key,
                rays=rays,
                filtered_sampler_results=filtered_sampler_results,
                train_frac=train_frac,
                train=train,
                is_secondary=is_secondary,
                **render_kwargs,
            )

            # If we do a material pass, compute that + integrator
            key, rng = utils.random_split(rng)
            final_outputs = self._handle_material_pass(
                rng=key,
                rays=rays,
                train_frac=train_frac,
                train=train,
                is_secondary=is_secondary,
                vignette=vignette,
                cache_outputs=cache_outputs,
                cache_shader_results=cache_shader_results,
                filtered_sampler_results=filtered_sampler_results,
                light_sampler_results=light_sampler_results,
                compute_extras=compute_extras,
                extra_ray_regularizer=extra_ray_regularizer,
                radiance_cache=self,
                **render_kwargs,
            )
        else:
            # Cache only outputs
            final_outputs = self._get_cache_only_outputs(cache_outputs, vignette)
            cache_shader_results = {}
            light_sampler_results = {}

        final_outputs = self._finalize_outputs(
            final_outputs,
            cache_outputs,
            cache_shader_results,
            passes,
            light_sampler_results,
            bypass_outputs,
            rays,
            vignette,
        )

        return final_outputs

    def _maybe_bypass_pipeline(
        self, rng, rays, passes, train_frac, train, **render_kwargs
    ):
        if "material_cache_shader" in passes or "material_shader" in passes:
            key, rng = utils.random_split(rng)
            return self._bypass_material_shaders(
                key, rays, train_frac, train, passes, **render_kwargs
            )

        if "geometry" in passes:
            key, rng = utils.random_split(rng)
            return self._bypass_geometry(key, rays, train_frac, train, **render_kwargs)

        if "surface_light_field" in passes or "surface_light_field_vis" in passes and self.use_surface_light_field:
            key, rng = utils.random_split(rng)
            return self._bypass_surface_light_field(
                key, rays, passes, train_frac, train, **render_kwargs
            )

        return None, False

    def _bypass_material_shaders(
        self, rng, rays, train_frac, train, passes, **render_kwargs
    ):
        sampler_results = render_kwargs.pop("sampler_results")

        key, rng = utils.random_split(rng)
        new_geo_results = self.cache.sampler.mlps[-1](
            rng=key,
            rays=rays,
            gaussians=(sampler_results["means"], sampler_results["covs"]),
            tdist=sampler_results["tdist"],
            train_frac=train_frac,
            train=train,
            **render_kwargs,
        )
        sampler_results["feature"] = new_geo_results["feature"]

        if "material_cache_shader" in passes:
            key, rng = utils.random_split(rng)
            cache_results = self.cache.shader(
                rng=key,
                rays=rays,
                sampler_results=sampler_results,
                filtered_sampler_results=sampler_results,
                train_frac=train_frac,
                train=train,
                radiance_cache=self,
                **render_kwargs,
            )

            key, rng = utils.random_split(rng)
            material_results = self.shader(
                rng=key,
                rays=rays,
                sampler_results=sampler_results,
                train_frac=train_frac,
                train=train,
                **render_kwargs,
            )
            return {"material": material_results, "cache": cache_results}, True

        if "material_shader" in passes:
            key, rng = utils.random_split(rng)
            return (
                self.shader(
                    rng=key,
                    rays=rays,
                    sampler_results=sampler_results,
                    train_frac=train_frac,
                    train=train,
                    **render_kwargs,
                ),
                True,
            )

        return None, False

    def _bypass_geometry(self, rng, rays, train_frac, train, **render_kwargs):
        sampler_results = render_kwargs.pop("sampler_results")
        key, rng = utils.random_split(rng)
        return (
            self.cache.sampler.mlps[-1](
                rng=key,
                rays=rays,
                gaussians=(sampler_results["means"], sampler_results["covs"]),
                tdist=sampler_results["tdist"],
                train_frac=train_frac,
                train=train,
                **render_kwargs,
            ),
            True,
        )

    def _bypass_surface_light_field(
        self, rng, rays, passes, train_frac, train, **render_kwargs
    ):
        if "surface_light_field" in passes or "surface_light_field_vis" in passes and self.use_surface_light_field:
            # Run surface_lf_mem
            key, rng = utils.random_split(rng)
            slf_results = self.cache(
                rng=key,
                rays=rays,
                train_frac=train_frac,
                train=train,
                use_slf=True,
                **render_kwargs
            )
            
            return slf_results, "surface_light_field" in passes

        return None, False

    def _handle_cache_pass(
        self, rng, rays, train_frac, train, is_secondary, vignette, **render_kwargs
    ):
        # Possibly use user-provided cache_outputs
        cache_outputs = render_kwargs.pop("cache_outputs", None)

        key, rng = utils.random_split(rng)
        cache_results = self.cache(
            rng=key,
            rays=rays,
            train_frac=train_frac,
            train=train,
            is_secondary=is_secondary,
            vignette=vignette,
            cache_outputs=cache_outputs,
            **render_kwargs,
        )

        return {
            "loss_weight": self.cache_loss_weight,
            "loss_type": self.cache_loss,
            "linear_to_srgb": self.cache_linear_to_srgb,
            "sampler": cache_results["main"]["sampler"],
            "filtered_sampler_inds": cache_results["main"]["filtered_sampler_inds"],
            "geometry": cache_results["main"]["geometry"],
            "shader": cache_results["main"]["shader"],
            "integrator": cache_results["main"]["integrator"],
        }

    def _get_material_samples(
        self,
        rng,
        rays,
        sampler_results,
        filtered_sampler_inds,
        train=True,
        train_frac=1.0,
        is_secondary=False,
        **render_kwargs,
    ):
        # Copy sampler results
        sampler_results = utils.copy_tree(sampler_results)

        # Filter
        do_resample = self.cache.do_resample(
            render_kwargs.get("resample", False), is_secondary, train
        )

        key, rng = utils.random_split(rng)
        filtered_results, _ = self.maybe_resample(
            rng=key,
            resample=do_resample,
            sampler_results=sampler_results,
            num_resample=self.cache.num_resample,
            inds=filtered_sampler_inds,
        )

        # If the cache already resampled with the same settings, reuse
        if do_resample and (self.cache.num_resample == self.num_resample):
            filtered_results = utils.copy_tree(filtered_results)
        else:
            # Possibly do a second pass resample with a different logits
            key, rng = utils.random_split(rng)
            filtered_results, _ = self.maybe_resample(
                rng=key,
                resample=self.do_resample(render_kwargs.get("resample", False), is_secondary, train),
                sampler_results=filtered_results,
                num_resample=self.num_resample,
                logits_mult=self._get_logits_mult(is_secondary),
            )
            filtered_results["weights_no_filter"] = sampler_results["weights"]

        # Stopgrad
        if self.stopgrad_samples:
            filtered_results = jax.lax.stop_gradient(filtered_results)

        stopgrad_map = {
            "weights": self.stopgrad_geometry_weight,
            "weights_no_filter": self.stopgrad_geometry_weight,
            "feature": self.stopgrad_geometry_feature_weight,
            "normals_pred": self.stopgrad_geometry_normals_weight,
            "normals": self.stopgrad_geometry_normals_weight,
            "normals_to_use": self.stopgrad_geometry_normals_weight,
        }
        stopgrad_map = stopgrad_map if do_resample else {}

        filtered_results_material = utils.apply_stopgrad_fields(
            filtered_results, stopgrad_map
        )

        # Cache shader
        stopgrad_map = {
            "weights": self.stopgrad_geometry_weight_consistency,
            "weights_no_filter": self.stopgrad_geometry_weight_consistency,
            "feature": self.stopgrad_geometry_feature_weight_consistency,
            "normals_pred": self.stopgrad_geometry_normals_weight_consistency,
            "normals": self.stopgrad_geometry_normals_weight_consistency,
            "normals_to_use": self.stopgrad_geometry_normals_weight_consistency,
        }
        filtered_results_cache = utils.apply_stopgrad_fields(
            filtered_results, stopgrad_map
        )

        key, rng = utils.random_split(rng)
        cache_shader_results = self.cache.shader(
            rng=key,
            rays=rays,
            sampler_results=filtered_results_cache,
            filtered_sampler_results=filtered_results_cache,
            train_frac=train_frac,
            train=train,
            is_secondary=is_secondary,
            radiance_cache=self,
            **render_kwargs,
        )

        # Add occlusions
        filtered_results_material["occ"] = jax.lax.stop_gradient(cache_shader_results["occ"])

        # Return
        return filtered_results_material, cache_shader_results

    def _handle_light_sampling_pass(
        self, rng, rays, filtered_sampler_results, train_frac, train, **render_kwargs
    ):
        if self.config.compute_relight_metrics or (
            self.config.use_ground_truth_illumination and self.config.multi_illumination
        ):
            env_map = render_kwargs.get("env_map", None)
            env_map_pmf = render_kwargs.get("env_map_pmf", None)
            env_map_pdf = render_kwargs.get("env_map_pdf", None)
            env_map_dirs = render_kwargs.get("env_map_dirs", None)
            env_map_w = render_kwargs.get("env_map_w", None)
            env_map_h = render_kwargs.get("env_map_h", None)

            return {
                "env_map": env_map,
                "env_map_pmf": env_map_pmf,
                "env_map_pdf": env_map_pdf,
                "env_map_dirs": env_map_dirs,
                "env_map_w": env_map_w,
                "env_map_h": env_map_h,
                "light_idx": jnp.repeat(
                    rays.light_idx[..., None, :],
                    filtered_sampler_results["means"].shape[-2],
                    axis=-2,
                ),
            }
        
        if not self.use_light_sampler:
            return None

        key, rng = utils.random_split(rng)
        return self.light_sampler(
            rng=key,
            rays=rays,
            sampler_results=jax.lax.stop_gradient(filtered_sampler_results),
            train_frac=train_frac,
            train=train,
            **render_kwargs,
        )

    def _handle_material_pass(
        self,
        rng,
        rays,
        train_frac,
        train,
        is_secondary,
        vignette,
        cache_outputs,
        cache_shader_results,
        filtered_sampler_results,
        light_sampler_results,
        compute_extras,
        extra_ray_regularizer,
        **render_kwargs,
    ):
        # Material shading
        key, rng = utils.random_split(rng)
        material_shader_results = self.shader(
            rng=key,
            rays=rays,
            sampler_results=filtered_sampler_results,
            train_frac=train_frac,
            train=train,
            light_sampler_results=light_sampler_results,
            **render_kwargs,
        )

        # Integrate material shading
        key, rng = utils.random_split(rng)
        material_integrator_results = self.integrator(
            rng=key,
            rays=rays,
            shader_results=material_shader_results,
            train_frac=train_frac,
            train=train,
            compute_extras=compute_extras,
            compute_distance=False,
            vignette=vignette,
            material=True,
            **render_kwargs,
        )

        # Control variate for material shading
        if self.slf_variate:
            key, rng = utils.random_split(rng)
            self._handle_slf_variate_pass(
                key,
                rays,
                train_frac,
                train,
                filtered_sampler_results,
                light_sampler_results,
                material_shader_results,
                material_integrator_results,
                **render_kwargs,
            )

        # Additional material outputs
        key, rng = utils.random_split(rng)
        self._get_material_extras(
            key,
            rays,
            train_frac,
            train,
            compute_extras,
            cache_outputs,
            filtered_sampler_results,
            material_integrator_results,
            vignette,
            **render_kwargs,
        )

        # Cache integrator
        key, rng = utils.random_split(rng)
        cache_integrator_results = self.integrator(
            rng=key,
            rays=rays,
            shader_results=cache_shader_results,
            train_frac=train_frac,
            train=train,
            compute_extras=compute_extras,
            compute_distance=False,
            vignette=vignette,
            material=False,
            **render_kwargs,
        )

        # Possibly add extra rays
        if train and extra_ray_regularizer:
            key, rng = utils.random_split(rng)
            self._get_extra_ray_outputs(
                key,
                rays,
                train_frac,
                train,
                filtered_sampler_results,
                light_sampler_results,
                material_shader_results,
                cache_shader_results,
                material_integrator_results,
                cache_integrator_results,
                vignette,
                **render_kwargs,
            )

        stopgrad_map = {
            "weights": self.stopgrad_geometry_weight_consistency,
            "weights_no_filter": self.stopgrad_geometry_weight_consistency,
            "feature": self.stopgrad_geometry_feature_weight_consistency,
            "normals_pred": self.stopgrad_geometry_normals_weight_consistency,
            "normals": self.stopgrad_geometry_normals_weight_consistency,
            "normals_to_use": self.stopgrad_geometry_normals_weight_consistency,
        }

        key, rng = utils.random_split(rng)
        _, cache_consistency_integrator_results = self.cache.apply_shader_and_integrator(
            rng,
            rays,
            rays,
            utils.copy_tree(cache_outputs["sampler"]),
            filtered_sampler_results,
            stopgrad_map,
            train,
            train_frac,
            False,
            None,
            **render_kwargs,
        )

        if self.config.volume_variate_material:
            cache_total_integrator_results = utils.copy_tree(cache_outputs["integrator"])

            self._handle_volume_variate_pass(
                material_integrator_results,
                cache_integrator_results,
                cache_total_integrator_results,
                keys=["rgb", "diffuse_rgb", "specular_rgb", "direct_rgb", "indirect_rgb", "transient_indirect", "transient_indirect_specular", "transient_indirect_diffuse"],
                stopgrad_weight_variate=self.stopgrad_weight_variate,
                stopgrad_weight_model=self.stopgrad_weight_model,
            )
        else:
            cache_total_integrator_results = cache_integrator_results

        # Final outputs
        material_outputs = {
            "loss_weight": self.loss_weight,
            "loss_type": self.loss,
            "linear_to_srgb": self.linear_to_srgb,
            "sampler": None,
            "geometry": None,
            "cache_shader": cache_shader_results,
            "cache_integrator": cache_consistency_integrator_results,
            "shader": material_shader_results,
            "integrator": material_integrator_results,
        }

        final_outputs = {
            "cache_main": cache_outputs,
            "main": material_outputs,
            "render": material_integrator_results,
        }

        return final_outputs

    def _get_light_sampler_extras(self, render_outputs, passes, light_sampler_results):
        if "light_sampler_vis" not in passes:
            return

        # Populate debug keys directly into `render_outputs`
        for k, v in light_sampler_results.items():
            render_outputs[k] = v

    def _get_extra_ray_outputs(
        self,
        rng,
        rays,
        train_frac,
        train,
        filtered_sampler_results,
        light_sampler_results,
        material_shader_results,
        cache_shader_results,
        material_integrator_results,
        cache_integrator_results,
        vignette,
        **render_kwargs,
    ):
        # Generate extra rays
        ray_type = self.config.extra_ray_type

        if ray_type == "train":
            extra_rays = rays
            extra_sr_results = utils.copy_tree(filtered_sampler_results)
            extra_ls_results = utils.copy_tree(light_sampler_results)

        elif ray_type == "incoming":
            key, rng = utils.random_split(rng)
            extra_rays = render_utils.get_outgoing_rays(
                key,
                rays,
                jax.lax.stop_gradient(rays.viewdirs),
                jax.lax.stop_gradient(material_shader_results["normals_to_use"]),
                {},
                random_generator_2d=self.random_generator_2d,
                use_mis=False,
                samplers=self.uniform_importance_samplers,
                num_secondary_samples=1,
            )
            extra_sr_results = filtered_sampler_results
            extra_ls_results = light_sampler_results

        elif ray_type == "outgoing":
            extra_origins = filtered_sampler_results["means"]
            extra_normals = filtered_sampler_results["normals_to_use"]
            key, rng = utils.random_split(rng)
            extra_rays = render_utils.get_secondary_rays(
                key,
                rays,
                jax.lax.stop_gradient(extra_origins),
                jax.lax.stop_gradient(rays.viewdirs),
                jax.lax.stop_gradient(extra_normals),
                {},
                random_generator_2d=self.random_generator_2d,
                use_mis=False,
                samplers=self.uniform_importance_samplers,
                num_secondary_samples=1,
                refdir_eps=self.config.secondary_near,
                normal_eps=self.config.secondary_normal_eps,
                far=self.config.secondary_far,
            )
            extra_sr_results = filtered_sampler_results
            extra_ls_results = light_sampler_results
        else:
            return

        if self.config.extra_ray_light_shuffle:
            new_lights = extra_rays.lights.reshape(-1, 3)
            new_lights = jnp.concatenate(
                [new_lights[1:], new_lights[:1]], axis=0
            ).reshape(extra_rays.lights.shape)

            extra_rays = extra_rays.replace(lights=new_lights)

        # Shader
        key, rng = utils.random_split(rng)
        extra_mat_sh = self.shader(
            rng=key,
            rays=extra_rays,
            sampler_results=extra_sr_results,
            train_frac=train_frac,
            train=train,
            light_sampler_results=extra_ls_results,
            **render_kwargs,
        )

        key, rng = utils.random_split(rng)
        extra_cache_sh = self.cache.shader(
            rng=key,
            rays=extra_rays,
            sampler_results=extra_sr_results,
            filtered_sampler_results=extra_sr_results,
            train_frac=train_frac,
            train=train,
            **render_kwargs,
        )

        # Integrator
        key, rng = utils.random_split(rng)
        extra_mat_int = self.integrator(
            rng=key,
            rays=rays,
            shader_results=extra_mat_sh,
            train_frac=train_frac,
            train=train,
            compute_extras=False,
            compute_distance=False,
            vignette=vignette,
            **render_kwargs,
        )

        key, rng = utils.random_split(rng)
        extra_cache_int = self.integrator(
            rng=key,
            rays=rays,
            shader_results=extra_cache_sh,
            train_frac=train_frac,
            train=train,
            compute_extras=False,
            compute_distance=False,
            vignette=vignette,
            material=False,
            **render_kwargs,
        )

        # Store relevant extra_ fields
        for k in extra_mat_sh:
            if ("rgb" in k) or ("occ" in k) or ("transient" in k):
                material_shader_results[f"extra_{k}"] = extra_mat_sh[k]

        for k in extra_cache_sh:
            if ("rgb" in k) or ("occ" in k) or ("transient" in k):
                cache_shader_results[f"extra_{k}"] = extra_cache_sh[k]
                material_shader_results[f"cache_extra_{k}"] = extra_cache_sh[k]

        for k in extra_mat_int:
            if ("rgb" in k) or ("occ" in k) or ("transient" in k):
                material_integrator_results[f"extra_{k}"] = extra_mat_int[k]

        for k in extra_cache_int:
            if ("rgb" in k) or ("occ" in k) or ("transient" in k):
                cache_integrator_results[f"extra_{k}"] = extra_cache_int[k]
                material_integrator_results[f"cache_extra_{k}"] = extra_cache_int[k]

    def _handle_brdf_pass(
        self,
        rng,
        rays,
        train_frac,
        train,
        compute_extras,
        sampler_results,
        material_integrator_results,
        vignette,
        **render_kwargs,
    ):
        if train or self.use_resample_depth:
            return

        key, rng = utils.random_split(rng)
        mat_only_results = self.shader(
            rng=key,
            rays=rays,
            sampler_results=utils.copy_tree(sampler_results),
            train_frac=train_frac,
            train=train,
            material_only=True,
            **render_kwargs,
        )

        # Stub out fields that won't be used in the integrator
        mat_only_results["rgb"] = jnp.ones_like(mat_only_results["material_albedo"])
        mat_only_results["direct_rgb"] = jnp.ones_like(mat_only_results["material_albedo"])

        if self.config.use_transient:
            mat_only_results["transient_indirect"] = None
            mat_only_results["transient_indirect_specular"] = None
            mat_only_results["transient_indirect_diffuse"] = None

        # Distances
        light_offset = rays.lights[..., None, :] - sampler_results["means"]
        mat_only_results["light_dists"] = jnp.linalg.norm(
            light_offset, axis=-1, keepdims=True
        )

        ray_offset = rays.origins[..., None, :] - sampler_results["means"]
        mat_only_results["ray_dists"] = jnp.linalg.norm(
            ray_offset, axis=-1, keepdims=True
        )
        
        # Weights
        mat_only_results["weights_no_filter"] = sampler_results["weights"]

        # Integrate
        key, rng = utils.random_split(rng)
        mat_only_integ = self.integrator(
            rng=key,
            rays=rays,
            shader_results=mat_only_results,
            train_frac=train_frac,
            train=train,
            compute_extras=True,
            compute_distance=False,
            vignette=vignette,
            material=True,
            **render_kwargs,
        )

        # Merge certain keys
        for k in mat_only_integ.keys():
            if ("rgb" not in k) and ("material" in k) and ("transient" not in k):
                material_integrator_results[k] = mat_only_integ[k]

    def _handle_slf_variate_pass(
        self,
        rng,
        rays,
        train_frac,
        train,
        filtered_sampler_results,
        light_sampler_results,
        material_shader_results,
        material_integrator_results,
        **render_kwargs,
    ):
        if self.config.compute_relight_metrics:
            return

        single_sampler_results = jax.tree_util.tree_map(
            lambda x: x, filtered_sampler_results
        )

        # Possibly re-run or re-use the light sampler
        if self.config.use_ground_truth_illumination and self.config.multi_illumination:
            env_map = render_kwargs.get("env_map", None)
            env_map_pmf = render_kwargs.get("env_map_pmf", None)
            env_map_pdf = render_kwargs.get("env_map_pdf", None)
            env_map_dirs = render_kwargs.get("env_map_dirs", None)
            env_map_w = render_kwargs.get("env_map_w", None)
            env_map_h = render_kwargs.get("env_map_h", None)

            single_light_results = {
                "env_map": env_map,
                "env_map_pmf": env_map_pmf,
                "env_map_pdf": env_map_pdf,
                "env_map_dirs": env_map_dirs,
                "env_map_w": env_map_w,
                "env_map_h": env_map_h,
                "light_idx": jnp.repeat(
                    rays.light_idx[..., None, :],
                    filtered_sampler_results["means"].shape[-2],
                    axis=-2,
                ),
            }
        elif self.use_light_sampler:
            key, rng = utils.random_split(rng)
            single_light_results = self.light_sampler(
                rng=key,
                rays=rays,
                sampler_results=jax.lax.stop_gradient(single_sampler_results),
                train_frac=train_frac,
                train=train,
                **render_kwargs,
            )
        else:
            single_light_results = None

        # Re-run the shader with single-sample geometry
        key, rng = utils.random_split(rng)
        single_material_shader = self.shader(
            rng=key,
            rays=rays,
            sampler_results=jax.lax.stop_gradient(single_sampler_results),
            train_frac=train_frac,
            train=train,
            light_sampler_results=jax.lax.stop_gradient(single_light_results),
            slf_variate=True,
            **render_kwargs,
        )

        # Copy reference fields
        for f in single_material_shader.keys():
            if f.startswith("ref_"):
                material_shader_results[f] = single_material_shader.get(f)

        # Weighted accumulation
        single_sampler_results["weights"] = utils.stopgrad_with_weight(
            single_sampler_results["weights"], self.stopgrad_geometry_variate_weight
        )
        weights_reshaped = single_sampler_results["weights"][..., None]

        for output_key in ["diffuse_rgb", "specular_rgb", "rgb", "lighting_irradiance", "transient_indirect", "transient_indirect_specular", "transient_indirect_diffuse"]:
            if (
                output_key not in material_integrator_results
                or output_key not in single_material_shader
            ):
                continue

            material_integrator_results[output_key] += (
                single_material_shader[output_key] * weights_reshaped
            ).reshape(material_integrator_results[output_key].shape)

    def _get_material_extras(
        self,
        rng,
        rays,
        train_frac,
        train,
        compute_extras,
        cache_outputs,
        filtered_sampler_results,
        material_integrator_results,
        vignette,
        **render_kwargs,
    ):
        if (not train) and (not self.use_resample_depth):
            key, rng = utils.random_split(rng)
            self._handle_brdf_pass(
                key,
                rays,
                train_frac,
                train,
                compute_extras,
                cache_outputs["sampler"][-1],
                material_integrator_results,
                vignette,
                **render_kwargs,
            )

        # Merge geometry distances from cache integrator
        for k in cache_outputs["integrator"].keys():
            if "distance" in k:
                material_integrator_results[k] = cache_outputs["integrator"][k]

    def _get_lossmult(self, final_outputs, rays):
        mat_integrator = final_outputs["render"]

        if self.use_material:
            geometry_results = jax.lax.stop_gradient(
                final_outputs["cache_main"]["integrator"]
            )
            normals = geometry_results["normals_to_use"].reshape(rays.viewdirs.shape)
            points = geometry_results["means"].reshape(rays.viewdirs.shape)

            # Example mask
            lossmult = jnp.ones_like(
                jnp.abs(normals[..., -1:]) < self.config.filter_normals_thresh
            ) & jnp.ones_like(
                jnp.linalg.norm(points, axis=-1, keepdims=True)
                < self.config.material_loss_radius
            )

            mat_integrator["lossmult"] = lossmult
        else:
            shape_rgb = mat_integrator["rgb"].shape
            if len(shape_rgb) == 3:
                mat_integrator["lossmult"] = rays.lossmult[..., None] * jnp.ones_like(
                    mat_integrator["rgb"]
                )
            else:
                mat_integrator["lossmult"] = rays.lossmult * jnp.ones_like(
                    mat_integrator["rgb"]
                )

    def _get_cache_only_outputs(self, cache_outputs, vignette):
        outputs = {
            "cache_main": cache_outputs,
            "main": cache_outputs,
            "render": cache_outputs["integrator"],
        }

        return outputs

    def _finalize_outputs(
        self,
        outputs,
        cache_outputs,
        cache_shader_results,
        passes,
        light_sampler_results,
        bypass_outputs,
        rays,
        vignette,
        **render_kwargs,
    ):
        # Integrator mappings
        integrator_keys = [
            "rgb",
            "normals",
            "normals_pred",
            "incoming_rgb",
            "env_map_rgb",
            "incoming_s_dist",
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
        ]
        for key in integrator_keys:
            if key in cache_outputs["integrator"]:
                outputs["render"][f"cache_{key}"] = cache_outputs["integrator"][key]

        # Shader mappings
        shader_keys = [
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
        ]
        for key in shader_keys:
            if key in cache_shader_results:
                outputs["main"]["shader"][f"cache_{key}"] = cache_shader_results[key]

        # Color
        if "material" in passes:
            outputs["render"]["material_rgb"] = outputs["render"]["rgb"]
        
        # Normals
        outputs["render"]["normals"] = cache_outputs["integrator"]["normals"]
        outputs["render"]["normals_pred"] = cache_outputs["integrator"]["normals_pred"]

        # Lossmult
        outputs["render"].setdefault(
            "lossmult", jnp.ones_like(outputs["render"]["rgb"][..., :1])
        )

        # Vignette
        outputs["render"]["vignette"] = (
            vignette
            if vignette is not None
            else jnp.ones_like(outputs["render"]["rgb"][..., :1])
        )

        # Surface light field
        if "surface_light_field_vis" in passes and bypass_outputs is not None and self.use_surface_light_field:
            self._populate_surface_light_field_vis(outputs, bypass_outputs)

        # Possibly store or visualize light sampler
        outputs["main"]["light_sampler"] = light_sampler_results
        self._get_light_sampler_extras(outputs["render"], passes, light_sampler_results)

        # Compute 'lossmult'
        self._get_lossmult(outputs, rays)

        return outputs

    def _populate_surface_light_field_vis(self, outputs, surface_lf_results):
        # List of keys to process
        keys = ["incoming_rgb", "incoming_acc", "incoming_s_dist"]

        # Get the target shape from outputs["render"]["rgb"]
        render_rgb_shape = outputs["render"]["rgb"].shape

        # Iterate over each key and perform reshape and sum operations
        for key in keys:
            if key in surface_lf_results:
                data = surface_lf_results[key]
                outputs["render"][f"cache_{key}"] = data.reshape(render_rgb_shape[:-1] + (-1,))


@gin.configurable
class MaterialModel(BaseMaterialModel):
    def setup(self):
        ## Cache
        self.cache = NeRFModel(
            config=self.config,
            use_surface_light_field=self.use_surface_light_field,
            **self.cache_model_params,
            **self.extra_model_params,
            name="Cache",
        )

        ## Light sampler
        self.light_sampler = light_sampler.LightMLP(
            config=self.config,
            **self.light_sampler_params,
            name="LightSampler",
        )

        ## Material
        if self.use_light_sampler:
            self.shader = material.MaterialMLP(
                config=self.config,
                use_surface_light_field=self.use_surface_light_field,
                diffuse_importance_sampler_configs=(
                    ("cosine", 1),
                    ("light", 1),
                ),
                diffuse_render_importance_sampler_configs=(
                    ("cosine", 1),
                    ("light", 1),
                ),
                importance_sampler_configs=(
                    ("microfacet", 1),
                ),
                render_importance_sampler_configs=(
                    ("microfacet", 1),
                ),
                **self.shader_params,
                name="MaterialShader",
            )
        else:
            self.shader = material.MaterialMLP(
                config=self.config,
                use_surface_light_field=self.use_surface_light_field,
                diffuse_importance_sampler_configs=(
                    ("cosine", 1),
                ),
                diffuse_render_importance_sampler_configs=(
                    ("cosine", 1),
                ),
                importance_sampler_configs=(
                    ("microfacet", 1),
                ),
                render_importance_sampler_configs=(
                    ("microfacet", 1),
                ),
                **self.shader_params,
                name="MaterialShader",
            )

        self.integrator = integration.VolumeIntegrator(
            config=self.config,
            **self.integrator_params,
            name="MaterialIntegrator",
        )

        if self.use_vignette:
            self.vignette_map = VignetteMap(
                config=self.config,
                name="VignetteMap",
            )


@gin.configurable
class TransientMaterialModel(BaseMaterialModel):
    def setup(self):
        ## Cache
        self.cache = TransientNeRFModel(
            config=self.config,
            use_surface_light_field=self.use_surface_light_field,
            **self.cache_model_params,
            **self.extra_model_params,
            name="Cache",
        )

        ## Light sampler
        self.light_sampler = light_sampler.LightMLP(
            config=self.config,
            **self.light_sampler_params,
            name="LightSampler",
        )

        ## Material
        if self.use_light_sampler:
            self.shader = material.TransientMaterialMLP(
                config=self.config,
                use_surface_light_field=self.use_surface_light_field,
                diffuse_importance_sampler_configs=(
                    ("cosine", 1),
                    ("light", 1),
                ),
                diffuse_render_importance_sampler_configs=(
                    ("cosine", 1),
                    ("light", 1),
                ),
                importance_sampler_configs=(("microfacet", 1),),
                render_importance_sampler_configs=(("microfacet", 1),),
                **self.shader_params,
                name="MaterialShader",
            )
        else:
            self.shader = material.TransientMaterialMLP(
                config=self.config,
                use_surface_light_field=self.use_surface_light_field,
                diffuse_importance_sampler_configs=(("cosine", 1),),
                diffuse_render_importance_sampler_configs=(("cosine", 1),),
                importance_sampler_configs=(("microfacet", 1),),
                render_importance_sampler_configs=(("microfacet", 1),),
                **self.shader_params,
                name="MaterialShader",
            )

        self.integrator = integration.TransientVolumeIntegrator(
            config=self.config,
            **self.integrator_params,
            name="MaterialIntegrator",
        )

        if self.use_vignette:
            self.vignette_map = VignetteMap(
                config=self.config,
                name="VignetteMap",
            )


def construct_model(rng, rays, config, dataset=None):
    # Grab just 10 rays, to minimize memory overhead during construction.
    ray = rays
    extra_model_params = {}

    if dataset is not None and dataset.max_exposure is not None:
        extra_model_params["max_exposure"] = dataset.max_exposure

    if config.use_transient:
        model = TransientMaterialModel(
            config=config,
            extra_model_params=extra_model_params,
        )
    else:
        model = MaterialModel(
            config=config,
            extra_model_params=extra_model_params,
        )

    init_variables = model.init(
        rng,  # The RNG used by flax to initialize random weights.
        rng=random.PRNGKey(0),  # The RNG used by sampling within the model.
        rays=ray,
        train_frac=1.0,
        mesh=dataset.mesh,
        env_map=dataset.env_map,
        env_map_pmf=dataset.env_map_pmf,
        env_map_pdf=dataset.env_map_pdf,
        env_map_dirs=dataset.env_map_dirs,
        env_map_w=dataset.env_map_w,
        env_map_h=dataset.env_map_h,
        compute_extras=True,
        passes=("cache", "material",),
    )

    return model, init_variables


def render_image(
    render_fn,
    rng,
    rays,
    config: configs.Config,
    passes: Tuple[Text, ...],
    verbose: bool = True,
    resample: Any = None,
    num_repeats: int = 1,
    compute_variance: bool = False,
):
    """Render an image with memory optimization.
    
    Args:
        render_fn: Function that renders a chunk of rays.
        rng: Random number generator.
        rays: Rays to render.
        config: Config.
        passes: Passes to render.
        verbose: Whether to print progress.
        resample: Resampling function.
        num_repeats: Number of times to repeat each render for averaging.
        compute_variance: Whether to compute variance across repetitions.

    Returns:
        Tuple of (rendering, rng).
        If compute_variance is True, rendering contains both means and variances.
    """
    height, width = rays.origins.shape[:2]
    num_rays = height * width

    # Reshape rays once at the beginning
    rays = jax.tree_util.tree_map(
        lambda r: (r.reshape((num_rays, -1)) if jnp.size(r) >= num_rays else r), rays
    )

    # Keys for which we'll compute statistics
    stat_keys = [
        "rgb", "integrated_rgb", "lighting_irradiance", "direct_rgb", "indirect_rgb", 
        "material_rgb", "specular_rgb", "diffuse_rgb", "material_albedo", "acc",
    ]
    var_keys = ["rgb", "integrated_rgb"]
    transient_keys = ["transient_direct_viz", "transient_indirect_viz"]

    # Initialize the final output once we know structure (after first render)
    rendering = None
    
    # Process in chunks
    idx0s = range(0, num_rays, config.render_chunk_size)
    start_time = time.time()

    for i_chunk, idx0 in enumerate(idx0s):
        if verbose and i_chunk % max(1, len(idx0s) // 10) == 0:
            print(f"Rendering chunk {i_chunk}/{len(idx0s)-1}")
            utils.log_memory_usage(f"Chunk {i_chunk}/{len(idx0s)-1}, chunk size {config.render_chunk_size}")
        
        # Extract chunk rays
        chunk_size = min(config.render_chunk_size, num_rays - idx0)
        chunk_rays = jax.tree_util.tree_map(
            lambda r: r[idx0 : idx0 + chunk_size], rays
        )
        chunk_rays = chunk_rays.replace(impulse_response=rays.impulse_response)
        
        # Calculate padding needed for sharding
        padding = 0

        if chunk_size % config.render_chunk_size != 0:
            padding = config.render_chunk_size - (chunk_size % config.render_chunk_size)
            chunk_rays = jax.tree_util.tree_map(
                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode="edge"), chunk_rays
            )
            
        # Shard the rays for processing
        chunk_rays = jax.tree_util.tree_map(
            lambda r: utils.shard(r), chunk_rays
        )
        
        # Initialize chunk statistics storage
        chunk_means = {}    # For all keys
        chunk_M2 = {}       # For variance calculation (var_keys only)
        
        # Perform multiple renders
        for i_repeat in range(num_repeats):
            # Render this chunk
            cur_renderings, rng = render_fn(rng, chunk_rays, passes, resample)
            
            # Unshard and remove padding
            cur_renderings = jax.tree_util.tree_map(
                lambda v: np.array(utils.unshard(v[0], padding)), cur_renderings
            )
            
            # Initialize the output arrays if this is the first chunk and first repeat
            if rendering is None:
                rendering = {}

                for k in cur_renderings.keys():
                    sample_shape = cur_renderings[k].shape[1:] if cur_renderings[k].ndim > 1 else ()

                    if "transient" in k in k not in transient_keys:
                        continue

                    full_shape = (height, width) + sample_shape
                    rendering[k] = np.zeros(full_shape, dtype=cur_renderings[k].dtype)
                    
                    # Also initialize variance arrays if needed
                    if compute_variance and k in var_keys:
                        var_key = f"{k}_variance"
                        rendering[var_key] = np.zeros_like(rendering[k])
            
            # Update statistics for each key
            for k in cur_renderings.keys():
                if "transient" in k in k not in transient_keys:
                    continue

                if k not in chunk_means:
                    # First time seeing this key in this chunk
                    chunk_means[k] = cur_renderings[k].copy()
                    
                    # Initialize M2 for variance calculation only for stat_keys
                    if compute_variance and num_repeats > 1 and k in var_keys:
                        chunk_M2[k] = np.zeros_like(cur_renderings[k])

                elif k in stat_keys:
                    # Update mean and M2 for stat_keys using Welford's algorithm
                    delta = cur_renderings[k] - chunk_means[k]
                    chunk_means[k] += delta / (i_repeat + 1)
                    
                    if compute_variance and num_repeats > 1 and k in var_keys:
                        delta2 = cur_renderings[k] - chunk_means[k]
                        chunk_M2[k] += delta * delta2

            # Free memory immediately
            del cur_renderings
        
        # Compute indices for placement vectorized 
        # (idx0 + offset) // width gives y, (idx0 + offset) % width gives x
        indices = np.arange(chunk_size)
        y_indices = (idx0 + indices) // width
        x_indices = (idx0 + indices) % width
        
        # Transfer results to final output arrays - vectorized approach
        for k in chunk_means.keys():
            # Get data for this key
            chunk_data = chunk_means[k]
            rendering[k][y_indices, x_indices] = chunk_data[:chunk_size]
            
            # Calculate and store variance if needed (for stat_keys only)
            if compute_variance and num_repeats > 1 and k in var_keys and k in chunk_M2:
                var_key = f"{k}_variance"
                # Unbiased variance estimate
                variance = (chunk_M2[k] / (num_repeats - 1)) * num_repeats if num_repeats > 1 else np.ones_like(chunk_means[k])
                
                # Store variance with vectorized assignment
                rendering[var_key][y_indices, x_indices] = variance[:chunk_size]
        
        # Clean up chunk data to free memory
        del chunk_means, chunk_rays, y_indices, x_indices, indices

        if compute_variance and num_repeats > 1:
            del chunk_M2

    elapsed_time = (time.time() - start_time) * 1000
    print("Milliseconds per ray", elapsed_time / (height * width))

    return rendering, rng