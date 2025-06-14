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
from internal import image
from internal import surface_light_field
from internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
import pdb

gin.config.external_configurable(math.abs, module="math")
gin.config.external_configurable(math.safe_exp, module="math")
gin.config.external_configurable(math.safe_tanh, module="math")
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


@gin.configurable
class BaseNeRFMLP(shading.BaseShader):
    """A PosEnc MLP."""

    config: Any = None  # A Config class, must be set upon construction.

    use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
    use_occlusions: bool = False  # If True, use refdirs instead of viewdirs.
    cull_backfacing: bool = True  # If True, use refdirs instead of viewdirs.

    use_normals_feature: bool = False  # If True, use refdirs instead of viewdirs.
    use_pred_normals_feature: bool = False  # If True, use refdirs instead of viewdirs.

    # If False and if use_directional_enc is True, use zero roughness in IDE.
    enable_pred_roughness: bool = False

    # Roughness activation function.
    roughness_activation: Callable[Ellipsis, Any] = nn.softplus
    roughness_bias: float = -1.0  # Shift added to raw roughness pre-activation.

    use_specular_tint: bool = False  # If True, predict tint.

    cache_grid_representation: str = "ngp"
    cache_grid_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    net_depth_cache: int = 2  # The depth of the second part of MLP.
    net_width_cache: int = 64  # The width of the second part of MLP.
    skip_layer_cache: int = 4  # Add a skip connection to 2nd MLP every N layers.

    use_learned_vignette_map: bool = False
    use_exposure_at_bottleneck: bool = False
    learned_exposure_scaling: bool = False  # Learned exposure scaling (RawNeRF).

    num_glo_features: int = 0  # GLO vector length, disabled if 0.
    num_glo_embeddings: int = 1000  # Upper bound on max number of train images.

    num_light_features: int = 64  # GLO vector length, disabled if 0.
    use_illumination_feature: bool = False  # GLO vector length, disabled if 0.
    multiple_illumination_outputs: bool = True  # GLO vector length, disabled if 0.

    # GLO vectors can either be 'concatenate'd onto the `bottleneck` or used to
    # construct an 'affine' transformation on the `bottleneck``.
    glo_mode: str = "concatenate"

    # The MLP architecture used to transform the GLO codes before they are used.
    # Setting to () is equivalent to not using an MLP.
    glo_mlp_arch: Tuple[int, Ellipsis] = tuple()
    glo_mlp_act: Callable[Ellipsis, Any] = nn.silu  # The activation for the GLO MLP.
    glo_premultiplier: float = 1.0  # Premultiplier on GLO vectors before process.

    net_depth_integrated_brdf: int = 2  # The depth of the second part of MLP.
    net_width_integrated_brdf: int = 64  # The width of the second part of MLP.
    skip_layer_integrated_brdf: int = (
        4  # Add a skip connection to 2nd MLP every N layers.
    )

    deg_brdf: int = 2  # Degree of encoding for mlp BRDF output
    net_depth_brdf: int = 2  # The depth of the second part of MLP.
    net_width_brdf: int = 64  # The width of the second part of MLP.
    skip_layer_brdf: int = 4  # Add a skip connection to 2nd MLP every N layers.
    brdf_bias: float = (
        -1.09861228867
    )  # Add a skip connection to 2nd MLP every N layers.

    simple_brdf: bool = False  # If True, use a simple BRDF model.
    use_ambient: bool = True  # If True, use a simple BRDF model.
    use_indirect: bool = True  # If True, use a simple BRDF model.
    use_active: bool = False  # If True, use a simple BRDF model.

    run_surface_light_field: bool = True

    use_surface_lf_roughness_ease: bool = False
    surface_lf_roughness_ease: float = 1.0
    surface_lf_roughness_factor: float = 1.0

    use_corrected_normals: bool = False

    stopgrad_normals_weight: float = 1.0
    stopgrad_shading_normals_weight: float = 1.0
    stopgrad_normals_weight_clip: float = 1.0
    stopgrad_normals_rate: float = float("inf")

    surface_lf_distance_near: float = float("inf")
    surface_lf_distance_far: float = float("inf")
    surface_lf_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    microfacet_sampler: Any = render_utils.MicrofacetSampler()
    random_generator_2d: Any = render_utils.RandomGenerator2D(1, 1, False)

    use_feature_filter: bool = False
    use_feature_filter_secondary_only: bool = True
    use_feature_filter_far_field: bool = False
    feature_filter_radius: float = float("inf")
    feature_filter_size: int = 64

    rgb_max: float = float("inf")

    # Env map
    use_env_map: bool = False
    env_map_near: float = float("inf")
    env_map_far: float = float("inf")

    env_map_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {}
    )

    ## Lighting independent
    weight_thold: float = 0.0
    irradiance_activation: Callable[Ellipsis, Any] = nn.softplus  # The RGB activation.
    irradiance_bias: float = -2.0  # The RGB activation.

    ambient_irradiance_activation: Callable[Ellipsis, Any] = (
        nn.softplus
    )  # The RGB activation.
    ambient_irradiance_bias: float = -2.0  # The RGB activation.

    albedo_activation: Callable[Ellipsis, Any] = nn.sigmoid  # The RGB activation.
    albedo_bias: float = -1.0  # The RGB activation.

    ## Lighting dependent
    deg_lights: int = 2  # Degree of encoding for mlp lights output
    net_depth_irradiance: int = 2  # The depth of the second part of MLP.
    net_width_irradiance: int = 64  # The width of the second part of MLP.
    bottleneck_irradiance: int = 64  # The width of the second part of MLP.
    skip_layer_irradiance: int = 4  # Add a skip connection to 2nd MLP every N layers.

    ## Active light
    optimize_light: bool = True
    light_power_bias: float = 200
    light_power_activation: Callable[Ellipsis, Any] = math.abs  # The RGB activation.
    light_max_angle: float = 0.0

    stopgrad_occ_weight: Tuple[float, float] = (0.0, 0.0)
    stopgrad_direct_weight: float = 1.0
    stopgrad_indirect_weight: float = 1.0
    stopgrad_ambient_weight: float = 1.0
    stopgrad_light_radiance_weight: float = 1.0

    indirect_scale: float = 1.0  # Shift added to raw roughness pre-activation.

    def setup(self):
        """
        One-time setup for the MLP, called automatically by Flax's module system.
        Initializes all sub-components of the model.
        """
        self._initialize_pos_basis()
        self._initialize_dense_layers()
        self._initialize_multi_illumination()
        self._initialize_grid()
        self._initialize_incoming()
        self._initialize_output_layers()
        self._initialize_brdf_layers()
        self._initialize_irradiance_layers()
        self._initialize_light_power()

    # -------------------------------------------------------------------------
    # Initialization methods
    # -------------------------------------------------------------------------
    def _initialize_pos_basis(self):
        """Generate and store the positional basis matrix for geometry expansions."""
        self.pos_basis_t = jnp.array(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
        ).T

    def _initialize_dense_layers(self):
        """Initialize the shared dense layers for the main MLP body."""
        weight_init_fn = getattr(jax.nn.initializers, self.weight_init)()
        self.dense_layer = functools.partial(nn.Dense, kernel_init=weight_init_fn)

        self.layers = [self.dense_layer(self.net_width) for _ in range(self.net_depth)]
        self.bottleneck_layer = self.dense_layer(self.bottleneck_width)

    def _initialize_multi_illumination(self):
        """Initialize embedding for multi-illumination if configured."""
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
        """
        Initialize the appearance grid if needed.
        The grid class is pulled from the grid_utils registry by name.
        """
        if self.use_grid:
            grid_class = grid_utils.GRID_REPRESENTATION_BY_NAME.get(
                self.grid_representation.lower()
            )
            if grid_class is None:
                raise ValueError(
                    f"Unsupported grid representation: {self.grid_representation}"
                )
            self.grid = grid_class(name="appearance_grid", **self.grid_params)
        else:
            self.grid = None

    def _initialize_incoming(self):
        """Initialize surface and environment-based light fields if enabled."""
        surface_lf_params = {
            key: self.surface_lf_params[key]
            for key in self.surface_lf_params.keys()
        }

        if self.use_env_map and self.config.env_map_distance < float("inf"):
            surface_lf_params["distance_near"] = self.surface_lf_distance_near
            surface_lf_params["distance_far"] = self.config.env_map_distance
        else:
            surface_lf_params["distance_near"] = self.surface_lf_distance_near
            surface_lf_params["distance_far"] = self.surface_lf_distance_far

        if self.config.use_transient:
            self.surface_lf = surface_light_field.TransientSurfaceLightFieldMLP(
                name="SurfaceLightField",
                use_env_alpha=True,
                config=self.config,
                **surface_lf_params,
            )
        else:
            self.surface_lf = surface_light_field.SurfaceLightFieldMLP(
                name="SurfaceLightField",
                use_env_alpha=True,
                config=self.config,
                **surface_lf_params,
            )

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

    def _initialize_output_layers(self):
        """Initialize top-level output layers for MLP predictions."""
        self.irradiance_layer = self.dense_layer(self.config.num_rgb_channels)

        if self.config.use_transient:
            self.transient_indirect_layer = self.dense_layer(
                self.config.num_rgb_channels * self.config.n_bins
            )
        else:
            self.indirect_layer = self.dense_layer(
                self.config.num_rgb_channels
            )

        self.ambient_irradiance_layer = self.dense_layer(self.config.num_rgb_channels)

        self.albedo_layer = self.dense_layer(self.config.num_rgb_channels)
        self.direct_tint_layer = self.dense_layer(self.config.num_rgb_channels)
        self.tint_layer = self.dense_layer(self.config.num_rgb_channels)
        self.roughness_layer = self.dense_layer(1)

    def _initialize_brdf_layers(self):
        """Initialize integrated and standard BRDF layers."""
        self.integrated_brdf_layers = [
            self.dense_layer(self.net_width_integrated_brdf)
            for _ in range(self.net_depth_integrated_brdf)
        ]
        self.output_integrated_brdf_layer = self.dense_layer(1)

        self.brdf_layers = [
            self.dense_layer(self.net_width_brdf) for _ in range(self.net_depth_brdf)
        ]
        self.output_brdf_layer = self.dense_layer(1)

        def brdf_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_brdf,
                append_identity=True,
            )

        self.brdf_enc_fn = brdf_enc_fn

    def _initialize_irradiance_layers(self):
        """Initialize layers for irradiance computation."""

        def lights_enc_fn(lights):
            return coord.pos_enc(
                lights,
                min_deg=0,
                max_deg=self.deg_lights,
                append_identity=True,
            )

        self.lights_enc_fn = lights_enc_fn
        self.irradiance_layers = [
            self.dense_layer(self.net_width_irradiance)
            for _ in range(self.net_depth_irradiance - 1)
        ] + [self.dense_layer(self.bottleneck_irradiance)]

    def _initialize_light_power(self):
        """
        Initialize learnable light power if enabled, otherwise just store
        a constant bias value.
        """
        if self.optimize_light:

            def light_init(key, shape):
                return jnp.ones_like(random.normal(key, shape)) * self.light_power_bias

            self.light_power = self.param("light_power", light_init, (1,))
        else:
            self.light_power = self.light_power_bias

    def get_bottleneck_feature(
        self,
        rng,
        feature,
        exposure,
    ):
        if self.bottleneck_width > 0:
            # Output of the first part of MLP.
            bottleneck = self.bottleneck_layer(feature)

            # Add bottleneck noise.
            if (rng is not None) and (self.bottleneck_noise > 0):
                key, rng = utils.random_split(rng)
                bottleneck += self.bottleneck_noise * random.normal(
                    key, bottleneck.shape
                )

            # Exposure
            if self.use_exposure_at_bottleneck and exposure is not None:
                bottleneck += jnp.log(exposure)[Ellipsis, None, :]
        else:
            bottleneck = jnp.zeros_like(feature[Ellipsis, 0:0])

        return bottleneck

    def get_light_vec(self, rays, feature):
        light_vec = jnp.zeros_like(feature[..., 0:0])

        if self.config.multi_illumination > 0:
            light_idx = rays.light_idx[Ellipsis, 0]
            light_vec = self.light_vecs(light_idx)
            light_vec = light_vec[..., None, :] * jnp.ones_like(feature[..., 0:1])

        return light_vec

    def run_integrated_brdf_network(self, x):
        inputs = x

        # Evaluate network to produce the output density.
        for i in range(self.net_depth_integrated_brdf):
            x = self.integrated_brdf_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_integrated_brdf == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return x

    def run_brdf_network(self, x):
        inputs = x

        # Evaluate network to produce the output density.
        for i in range(self.net_depth_brdf):
            x = self.brdf_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_brdf == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return x

    def run_irradiance_network(self, x):
        inputs = x
        # Evaluate network to produce the output density.
        for i in range(self.net_depth_irradiance):
            x = self.irradiance_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_irradiance == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return x

    def get_integrated_brdf(
        self,
        normals,
        viewdirs,
        bottleneck,
    ):
        # View dependent network input
        x = []

        # Add bottleneck input
        x.append(bottleneck)

        # Append dot product between normal vectors and view directions.
        dotprod = math.dot(normals, -viewdirs[..., None, :])
        x.append(dotprod)

        # Run view dependent network
        x = self.run_integrated_brdf_network(jnp.concatenate(x, axis=-1))

        # Get RGB
        brdf = nn.sigmoid(self.output_integrated_brdf_layer(x) + jnp.log(3.0))
        return brdf

    def get_brdf_light(
        self,
        normals,
        viewdirs,
        lightdirs,
        bottleneck,
        roughness,
    ):
        brdf_input = []

        halfdirs = math.normalize(-viewdirs[..., None, :] + lightdirs)
        brdf_dot = math.dot(normals, halfdirs)

        # Half vector
        if self.simple_brdf:
            brdf_input.append(
                brdf_dot,
            )

            brdf_input.append(
                brdf_dot,
            )

            brdf_input = jnp.concatenate(brdf_input, axis=-1)
        else:
            # Append normals, viewdirs dotprod
            brdf_input.append(math.dot(normals, -viewdirs[..., None, :]))

            # Append normals, lightdirs dotprod
            brdf_input.append(math.dot(normals, lightdirs))

            # Append viewdirs, lightdirs dotprod
            brdf_input = jnp.concatenate(brdf_input, axis=-1)
            brdf_input = jnp.concatenate(
                [
                    jnp.sort(brdf_input, axis=-1),
                    brdf_dot,
                ],
                axis=-1,
            )

        # Run view dependent network
        brdf_input = self.brdf_enc_fn(brdf_input)
        brdf_input = jnp.concatenate(
            [
                bottleneck,
                brdf_input,
            ],
            axis=-1,
        )
        x = self.run_brdf_network(brdf_input)

        # Get RGB
        brdf = nn.softplus(self.output_brdf_layer(x) + self.brdf_bias)
        return brdf

    def get_irradiance(
        self,
        lights,
        bottleneck,
    ):
        # Run view dependent network
        irradiance_input = self.lights_enc_fn(lights)
        irradiance_input = jnp.concatenate(
            [
                bottleneck,
                irradiance_input,
            ],
            axis=-1,
        )
        x = self.run_irradiance_network(irradiance_input)

        # Get RGB
        return self.irradiance_activation(
            self.irradiance_layer(x) + self.irradiance_bias
        )

    def predict_appearance(
        self,
        rng,
        rays,
        sampler_results,
        train_frac=1.0,
        train=True,
        zero_glo=False,
        is_secondary=False,
        radiance_cache=None,
        light_power=None,
        passes=(),
        **kwargs,
    ):
        """
        Predict appearance (diffuse, specular, ambient, and indirect
        components) given ray samples, view directions, and learned scene parameters.

        Args:
        rng: PRNG key for random operations.
        rays: A Rays object with ray origins, directions, possibly light directions, etc.
        sampler_results: Dictionary containing samples ('means', 'covs', 'normals', etc.).
        train_frac: Scalar for controlling schedule-based features (e.g., thresholding).
        train: Whether we are in training mode (affects e.g. dropout).
        zero_glo: Whether to set GLO features to zero (if used).
        is_secondary: Whether this call is for a secondary/indirect pass.
        radiance_cache: An optional RadianceCache for caching partial results.
        light_power: An optional override for the light power (if sharing across samples).
        **kwargs: Additional arguments, such as environment maps, reflection directions, etc.

        Returns:
        outputs: A dictionary of predicted appearance components, including
            'rgb', 'diffuse_rgb', 'specular_rgb', 'ambient_rgb', 'indirect_rgb',
            'transient_indirect', and various other debug/auxiliary fields.
        """
        outputs = {}
        key, rng = utils.random_split(rng)

        # -------------------------------------------------------------------------
        # 1. Extract / compute primary variables
        # -------------------------------------------------------------------------
        means = sampler_results["means"]
        viewdirs = rays.viewdirs
        exposure = rays.exposure_values

        # -------------------------------------------------------------------------
        # 2. Predict appearance feature (main MLP or grid features)
        # -------------------------------------------------------------------------
        predict_appearance_kwargs = self.get_predict_appearance_kwargs(
            key, rays, sampler_results
        )
        feature = self.predict_appearance_feature(
            sampler_results,
            train=train,
            is_secondary=is_secondary,
            **predict_appearance_kwargs,
        )

        # Optionally include the per-illumination feature if configured.
        if self.config.multi_illumination and self.use_illumination_feature:
            light_vec = self.get_light_vec(rays, feature)
            feature = jnp.concatenate([feature, light_vec], axis=-1)

        # -------------------------------------------------------------------------
        # 3. Compute bottleneck from the feature (optionally add noise, exposure)
        # -------------------------------------------------------------------------
        key, rng = utils.random_split(rng)
        bottleneck = self.get_bottleneck_feature(key, feature, exposure)

        # -------------------------------------------------------------------------
        # 4. Compute normals and roughness
        # -------------------------------------------------------------------------
        raw_roughness = self.roughness_layer(feature)
        roughness = self.roughness_activation(raw_roughness + self.roughness_bias)

        # Possibly stop-gradient on geometry normals
        normals = sampler_results[self.normals_target]
        if self.stopgrad_normals_weight < 1.0:
            normals = utils.stopgrad_with_weight(normals, self.stopgrad_normals_weight)

        # Possibly stop-gradient on shading normals
        shading_normals = sampler_results[self.normals_target]
        if self.stopgrad_shading_normals_weight < 1.0:
            shading_normals = utils.stopgrad_with_weight(
                shading_normals, self.stopgrad_shading_normals_weight
            )
        
        # Different logic for active/passive sensors
        if self.use_active:
            key, rng = utils.random_split(rng)
            outputs = self._predict_appearance_active(
                key,
                rays,
                sampler_results,
                feature,
                bottleneck,
                roughness,
                normals,
                shading_normals,
                train_frac=train_frac,
                train=train,
                zero_glo=zero_glo,
                is_secondary=is_secondary,
                radiance_cache=radiance_cache,
                light_power=light_power,
                passes=passes,
                **kwargs,
            )
        else:
            key, rng = utils.random_split(rng)
            outputs = self._predict_appearance_passive(
                key,
                rays,
                sampler_results,
                feature,
                bottleneck,
                roughness,
                normals,
                shading_normals,
                train_frac=train_frac,
                train=train,
                zero_glo=zero_glo,
                is_secondary=is_secondary,
                radiance_cache=radiance_cache,
                passes=passes,
                **kwargs,
            )
        
        return outputs

    def _predict_appearance_active(
        self,
        rng,
        rays,
        sampler_results,
        feature,
        bottleneck,
        roughness,
        normals,
        shading_normals,
        train_frac=1.0,
        train=True,
        zero_glo=False,
        is_secondary=False,
        radiance_cache=None,
        light_power=None,
        passes=("direct", "occ", "indirect"),
        **kwargs,
    ):
        means = sampler_results["means"]
        viewdirs = rays.viewdirs

        outputs = {}

        # -------------------------------------------------------------------------
        # Prepare lighting directions and distances
        # -------------------------------------------------------------------------
        light_offset = rays.lights[..., None, :] - means
        light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
        light_dirs = light_offset / jnp.maximum(light_dists, 1e-5)

        # -------------------------------------------------------------------------
        # Handle direct light power (learnable or constant). Apply falloff/angle.
        # -------------------------------------------------------------------------
        key, rng = utils.random_split(rng)
        (light_radiance, light_radiance_mult, light_radiance_before_occ) = (
            self._compute_light_radiance(
                key,
                rays,
                sampler_results,
                radiance_cache,
                light_dirs,
                light_power,
                light_dists,
                **kwargs,
            )
        )

        # -------------------------------------------------------------------------
        # Optionally compute occlusions (shadows)
        # -------------------------------------------------------------------------
        n_dot_l = jnp.maximum(0.0, math.dot(shading_normals, light_dirs))

        if len(passes) == 0 or "occ" in passes:
            key, rng = utils.random_split(rng)
            occ = self._compute_occlusions(
                key,
                rays,
                sampler_results,
                shading_normals,
                light_dirs,
                light_dists,
                radiance_cache,
                train_frac,
                train,
                is_secondary,
                **kwargs,
            )
        else:
            occ = jnp.zeros_like(n_dot_l)

        occ = jnp.where(n_dot_l <= 0.0, jnp.ones_like(occ), occ)

        # Apply occlusion to the light radiance
        light_radiance = light_radiance * (1.0 - occ)

        # -------------------------------------------------------------------------
        # Compute direct lighting (diffuse + specular)
        # -------------------------------------------------------------------------
        key, rng = utils.random_split(rng)
        (albedo, direct_diffuse, direct_specular) = self._compute_direct_lighting(
            key,
            rays,
            sampler_results,
            feature,
            shading_normals,
            roughness,
            bottleneck,
            normals,
            n_dot_l,
            light_radiance,
            light_dirs,
            radiance_cache,
            train_frac,
            train,
        )

        direct = direct_diffuse + direct_specular

        # -------------------------------------------------------------------------
        # Compute environment / surface light fields (indirect or ambient)
        # -------------------------------------------------------------------------
        key, rng = utils.random_split(rng)
        incoming_outputs = self.surface_lf(
            key,
            rays,
            sampler_results,
            means,
            self._get_refdirs(viewdirs, normals, kwargs),
            roughness=roughness,
            shader_bottleneck=bottleneck,
            train=train,
            train_frac=train_frac,
        )

        ref_rgb = incoming_outputs["incoming_rgb"]
        ambient_ref_rgb = incoming_outputs["incoming_ambient_rgb"]

        # -------------------------------------------------------------------------
        # Compute indirect lighting
        # -------------------------------------------------------------------------
        (
            indirect_diffuse,
            indirect_specular,
            transient_indirect,
            transient_indirect_diffuse,
            transient_indirect_specular,
        ) = self._compute_indirect_lighting(
            feature,
            means,
            normals,
            shading_normals,
            ref_rgb,
            bottleneck,
            viewdirs,
            roughness,
            rays,
            light_dirs,
            light_radiance_mult,
        )

        # -------------------------------------------------------------------------
        # Compute ambient term
        # -------------------------------------------------------------------------
        ambient_irradiance = self.ambient_irradiance_activation(
            self.ambient_irradiance_layer(feature) + self.ambient_irradiance_bias
        )

        if self.use_ambient:
            ambient_diffuse = ambient_irradiance
            ambient_specular = self._compute_ambient_specular(
                normals, viewdirs, feature, bottleneck, ambient_ref_rgb
                )
        else:
            ambient_diffuse = jnp.zeros_like(ambient_ref_rgb)
            ambient_specular = jnp.zeros_like(ambient_ref_rgb)

        ambient_diffuse = jnp.clip(ambient_diffuse, 0.0, self.rgb_max)
        ambient_specular = jnp.clip(ambient_specular, 0.0, self.rgb_max)

        ambient_diffuse = utils.stopgrad_with_weight(
            ambient_diffuse,
            self.stopgrad_ambient_weight,
        )

        ambient_specular = utils.stopgrad_with_weight(
            ambient_specular,
            self.stopgrad_ambient_weight,
        )

        ambient = ambient_diffuse + ambient_specular

        # -------------------------------------------------------------------------
        # Combine all components (ambient, direct, indirect)
        # -------------------------------------------------------------------------
        indirect_diffuse = utils.stopgrad_with_weight(
            indirect_diffuse,
            self.stopgrad_indirect_weight,
        )

        indirect_specular = utils.stopgrad_with_weight(
            indirect_specular,
            self.stopgrad_indirect_weight,
        )

        indirect = indirect_diffuse + indirect_specular

        diffuse = direct_diffuse + indirect_diffuse + ambient_diffuse
        specular = direct_specular + indirect_specular + ambient_specular

        # Final RGB = direct + ambient
        rgb = direct + ambient + indirect

        if len(passes) > 0 and "indirect" not in passes:
            return {
                "rgb": direct,
                "direct_rgb": direct,
                "indirect_rgb": None,
                "transient_indirect": None,
            }

        # -------------------------------------------------------------------------
        # Populate outputs dict
        # -------------------------------------------------------------------------
        outputs["rgb"] = rgb
        outputs["diffuse_rgb"] = diffuse
        outputs["specular_rgb"] = specular
        outputs["ambient_rgb"] = ambient
        outputs["indirect_rgb"] = indirect + ambient
        outputs["albedo_rgb"] = albedo

        if "occ" not in sampler_results:
            outputs["occ"] = occ * jnp.ones_like(rgb)
        else:
            outputs["occ"] = jnp.zeros_like(rgb)

        outputs["indirect_occ"] = incoming_outputs["incoming_acc"][..., None] * jnp.ones_like(rgb)

        outputs["direct_rgb"] = direct
        outputs["indirect_diffuse_rgb"] = indirect_diffuse + ambient_diffuse
        outputs["direct_diffuse_rgb"] = direct_diffuse
        outputs["direct_specular_rgb"] = direct_specular
        outputs["indirect_specular_rgb"] = indirect_specular + ambient_specular
        outputs["ambient_diffuse_rgb"] = ambient_diffuse
        outputs["ambient_specular_rgb"] = ambient_specular

        if transient_indirect is not None:
            outputs["transient_indirect"] = utils.stopgrad_with_weight(transient_indirect, self.stopgrad_indirect_weight)
            outputs["transient_indirect_diffuse"] = utils.stopgrad_with_weight(transient_indirect_diffuse, self.stopgrad_indirect_weight)
            outputs["transient_indirect_specular"] = utils.stopgrad_with_weight(transient_indirect_specular, self.stopgrad_indirect_weight)
        else:
            outputs["transient_indirect"] = None

        # Useful debug outputs
        outputs["n_dot_l_rgb"] = n_dot_l * jnp.ones_like(rgb)
        outputs["light_radiance_rgb"] = light_radiance_mult * jnp.ones_like(rgb)
        outputs["irradiance_rgb"] = (
            n_dot_l * light_radiance_before_occ / jnp.pi
            if self.use_active
            else jnp.ones_like(rgb)
        )

        # Distances
        ray_offset = rays.origins[..., None, :] - means
        outputs["ray_dists"] = jnp.linalg.norm(ray_offset, axis=-1, keepdims=True)
        outputs["light_dists"] = light_dists

        return outputs

    def _predict_appearance_passive(
        self,
        rng,
        rays,
        sampler_results,
        feature,
        bottleneck,
        roughness,
        normals,
        shading_normals,
        train_frac=1.0,
        train=True,
        zero_glo=False,
        is_secondary=False,
        radiance_cache=None,
        light_power=None,
        passes=("diffuse", "specular",),
        **kwargs,
    ):
        means = sampler_results["means"]
        viewdirs = rays.viewdirs

        outputs = {}

        # Direct diffuse
        ambient_irradiance = self.ambient_irradiance_activation(
            self.ambient_irradiance_layer(feature) + self.ambient_irradiance_bias
        )
        ambient_diffuse = ambient_irradiance
        ambient_diffuse = jnp.clip(ambient_diffuse, 0.0, self.rgb_max)
        ambient_diffuse = utils.stopgrad_with_weight(
            ambient_diffuse,
            self.stopgrad_ambient_weight,
        )

        # BRDF
        tint = nn.sigmoid(self.tint_layer(feature))
        integrated_brdf = self.get_integrated_brdf(
            normals,
            viewdirs,
            bottleneck,
        )

        # Direct specular
        if self.use_env_map:
            key, rng = utils.random_split(rng)
            env_incoming_outputs = self.env_map(
                key,
                rays,
                sampler_results,
                means,
                self._get_refdirs(viewdirs, normals, kwargs),
                roughness=roughness,
                shader_bottleneck=bottleneck,
                train=train,
                train_frac=train_frac,
            )
            env_rgb = env_incoming_outputs["incoming_ambient_rgb"]
            ambient_specular = self._compute_ambient_specular(
                normals, viewdirs, feature, bottleneck, tint, integrated_brdf, env_rgb
            )
        else:
            env_rgb = jnp.zeros_like(ambient_diffuse)
            ambient_specular = jnp.zeros_like(ambient_diffuse)

        ambient_specular = jnp.clip(ambient_specular, 0.0, self.rgb_max)

        # Indirect diffuse
        indirect_irradiance = self.irradiance_activation(
            self.irradiance_layer(feature) + self.irradiance_bias
        )
        indirect_diffuse = indirect_irradiance
        indirect_diffuse = jnp.clip(indirect_diffuse, 0.0, self.rgb_max)
        indirect_diffuse = utils.stopgrad_with_weight(
            indirect_diffuse,
            self.stopgrad_indirect_weight,
        )

        # Indirect specular + direct occlusion
        key, rng = utils.random_split(rng)
        incoming_outputs = self.surface_lf(
            key,
            rays,
            sampler_results,
            means,
            self._get_refdirs(viewdirs, normals, kwargs),
            roughness=roughness,
            shader_bottleneck=bottleneck,
            train=train,
            train_frac=train_frac,
        )
        ref_rgb = incoming_outputs["incoming_ambient_rgb"]
        ref_acc = incoming_outputs["incoming_acc"][..., None]

        ambient_specular = self._compute_ambient_specular(
            normals, viewdirs, feature, bottleneck, tint, integrated_brdf, env_rgb * (1.0 - ref_acc)
        )
        ambient_specular = jnp.clip(ambient_specular, 0.0, self.rgb_max)

        indirect_specular = self._compute_ambient_specular(
            normals, viewdirs, feature, bottleneck, tint, integrated_brdf, ref_rgb * ref_acc
        )
        indirect_specular = jnp.clip(indirect_specular, 0.0, self.rgb_max)

        # Combined ambient
        ambient = ambient_diffuse + ambient_specular
        indirect = indirect_diffuse + indirect_specular

        # -------------------------------------------------------------------------
        # Populate outputs dict
        # -------------------------------------------------------------------------
        rgb = ambient + indirect
        diffuse = ambient_diffuse + indirect_diffuse
        specular = ambient_specular + indirect_specular

        if len(passes) > 0 and "specular" not in passes:
            return {
                "rgb": diffuse,
                "diffuse_rgb": diffuse,
                "specular_rgb": None,
            }

        outputs["rgb"] = rgb
        outputs["diffuse_rgb"] = diffuse
        outputs["specular_rgb"] = specular
        outputs["ambient_rgb"] = ambient
        outputs["indirect_rgb"] = indirect
        outputs["albedo_rgb"] = tint

        outputs["occ"] = jnp.zeros_like(rgb)
        outputs["indirect_occ"] = ref_acc * jnp.ones_like(rgb)

        outputs["direct_rgb"] = ambient
        outputs["indirect_diffuse_rgb"] = indirect_diffuse
        outputs["direct_diffuse_rgb"] = ambient_diffuse
        outputs["direct_specular_rgb"] = ambient_specular
        outputs["indirect_specular_rgb"] = indirect_specular
        outputs["ambient_diffuse_rgb"] = ambient_diffuse
        outputs["ambient_specular_rgb"] = ambient_specular
        outputs["transient_indirect"] = None

        # Useful debug outputs
        outputs["n_dot_l_rgb"] = jnp.zeros_like(rgb)
        outputs["light_radiance_rgb"] = jnp.zeros_like(rgb)
        outputs["irradiance_rgb"] = jnp.zeros_like(rgb)

        # Distances
        ray_offset = rays.origins[..., None, :] - means
        outputs["ray_dists"] = jnp.linalg.norm(ray_offset, axis=-1, keepdims=True)

        return outputs

    # -------------------------------------------------------------------------
    # Below are some example helper methods that can be placed in the same class
    # to break up the monolithic `predict_appearance`.
    # -------------------------------------------------------------------------

    def _compute_light_radiance(
        self,
        rng,
        rays,
        sampler_results,
        radiance_cache,
        light_dirs,
        light_power,
        light_dists,
        **kwargs,
    ):
        """
        Computes light radiance based on either a learned light power or a constant,
        applies distance falloff, cone-angle cutoffs, etc.
        Returns:
        (light_radiance, light_radiance_mult, light_radiance_before_occ)
        """
        light_radiance_before_occ = None
        light_radiance_mult = jnp.ones_like(light_dists)

        if (
            self.config.learnable_light
            and radiance_cache
            and radiance_cache.share_light_power
        ):
            (light_radiance, light_radiance_mult) = (
                radiance_cache.shader.learnable_light(
                    sampler_results["means"],
                    rays.viewdirs[..., None, :]
                    * jnp.ones_like(sampler_results["means"]),
                    rays.lights[..., None, :] * jnp.ones_like(sampler_results["means"]),
                    rays.vcam_look[..., None, :]
                    * jnp.ones_like(sampler_results["means"]),
                    rays.vcam_up[..., None, :]
                    * jnp.ones_like(sampler_results["means"]),
                    rays.vcam_origins[..., None, :]
                    * jnp.ones_like(sampler_results["means"]),
                    env_map=kwargs.get("env_map"),
                    env_map_w=kwargs.get("env_map_w"),
                    env_map_h=kwargs.get("env_map_h"),
                )
            )
        else:
            # Use constant or passed-in light power
            if (light_power is None) or not (
                radiance_cache and radiance_cache.share_light_power
            ):
                light_radiance = jnp.ones_like(
                    light_dists
                ) * self.light_power_activation(self.light_power)
            else:
                light_radiance = jnp.ones_like(light_dists) * light_power

            # Apply distance falloff if configured
            if self.config.use_falloff:
                falloff = 1.0 / jnp.maximum(light_dists**2, 1e-5)
                light_radiance = light_radiance * falloff

            # Apply angle cutoff
            if self.light_max_angle > 0.0:
                angle_dot = math.dot(
                    -light_dirs, rays.vcam_look[..., None, :], keepdims=True
                )
                angle = jnp.arccos(angle_dot)
                angle_mask = (angle * 180.0 / jnp.pi) <= (self.light_max_angle / 2.0)
                angle_mask = angle_mask & (angle_dot > 0.0)
                light_radiance = jnp.where(angle_mask, light_radiance, 0.0)

        if self.config.light_zero:
            light_radiance = jnp.where(
                light_dists < self.config.light_near,
                jnp.zeros_like(light_radiance),
                light_radiance,
            )

        light_radiance_before_occ = light_radiance

        # Possibly apply structured light transforms
        if self.config.sl_relight:
            sl_mult = render_utils.get_sl_color(
                kwargs["env_map"],
                kwargs["env_map_w"],
                kwargs["env_map_h"],
                rays.vcam_up[..., None, :] * jnp.ones_like(sampler_results["means"]),
                rays.vcam_look[..., None, :] * jnp.ones_like(sampler_results["means"]),
                sampler_results["means"],
                rays.vcam_origins[..., None, :]
                * jnp.ones_like(sampler_results["means"]),
                hfov=self.config.sl_hfov,
                vfov=self.config.sl_vfov,
                shift=self.config.sl_shift,
                mult=self.config.sl_mult,
                invert=self.config.sl_invert,
            )
            light_radiance = light_radiance * sl_mult
        
        light_radiance = utils.stopgrad_with_weight(light_radiance, self.stopgrad_light_radiance_weight)
        return light_radiance, light_radiance_mult, light_radiance_before_occ

    def _compute_occlusions(
        self,
        rng,
        rays,
        sampler_results,
        shading_normals,
        light_dirs,
        light_dists,
        radiance_cache,
        train_frac,
        train,
        is_secondary,
        **kwargs,
    ):
        """
        If real occlusions or shadow mapping is enabled, uses secondary ray sampling
        to compute occlusion. Otherwise returns zero occlusion.
        """
        if (
            not self.config.use_occlusions
            or (not is_secondary and self.config.occlusions_secondary_only)
            or (is_secondary and self.config.occlusions_primary_only)
        ):
            # No real occlusion logic
            occ = jnp.zeros_like(sampler_results["means"][..., :1])
            # Repeated across num_rgb_channels
            occ = jnp.repeat(occ, self.num_rgb_channels, axis=-1)
            return occ

        # Real occlusion logic: trace shadow rays
        filtered_sampler_results = kwargs["filtered_sampler_results"]

        # Shadow near rate
        if self.config.shadow_near_rate > 0:
            shadow_near_weight = jnp.clip(
                (train_frac - self.config.shadow_near_start_frac)
                / self.config.shadow_near_rate,
                0.0,
                1.0,
            )
            shadow_near = (
                shadow_near_weight * self.config.shadow_near_min
                + (1.0 - shadow_near_weight) * self.config.shadow_near_max
            )
        else:
            shadow_near = self.config.shadow_near_min

        key, rng = utils.random_split(rng)
        ref_rays, ref_samples = render_utils.get_secondary_rays(
            key,
            rays,
            filtered_sampler_results["means"],
            rays.viewdirs,
            filtered_sampler_results[self.config.shadow_normals_target],
            {"roughness": jnp.ones_like(light_dists)},
            refdir_eps=shadow_near,
            normal_eps=self.config.secondary_normal_eps,
            random_generator_2d=(
                radiance_cache.random_generator_2d if radiance_cache else None
            ),
            stratified_sampling=False,
            use_mis=True,
            samplers=(
                radiance_cache.active_importance_samplers if radiance_cache else None
            ),
            num_secondary_samples=1,
            light_sampler_results={
                "origins": filtered_sampler_results["means"][..., None, :],
                "lights": rays.lights[..., None, None, :]
                * jnp.ones_like(filtered_sampler_results["means"][..., None, :]),
            },
            offset_origins=False,
            far=self.config.secondary_far,
        )

        # Limit the far distance by the actual light distance minus some near offset
        single_light_offset = (
            rays.lights[..., None, :] - filtered_sampler_results["means"]
        )
        single_light_dists = jnp.linalg.norm(
            single_light_offset, axis=-1, keepdims=True
        )
        ref_rays = ref_rays.replace(
            far=jnp.clip(
                single_light_dists.reshape(ref_rays.far.shape) - self.config.light_near,
                ref_rays.near,
                ref_rays.far,
            ),
            normals=jax.lax.stop_gradient(
                filtered_sampler_results[self.config.shadow_normals_target].reshape(
                    ref_rays.viewdirs.shape
                )
            ),
        )

        # Fire these shadow rays into the radiance cache, but only gather weights
        key, rng = utils.random_split(rng)
        ref_ray_outputs = radiance_cache.cache(
            key,
            ref_rays,
            train_frac=train_frac,
            train=train,
            compute_extras=False,
            zero_glo=(
                "glo_vec" not in filtered_sampler_results
                or filtered_sampler_results["glo_vec"] is None
            ),
            stopgrad_proposal=True,
            stopgrad_weights=True,
            is_secondary=True,
            weights_only=True,
            radiance_cache=radiance_cache,
            stopgrad_cache_weight=self.stopgrad_occ_weight,
        )
        # Accumulated transmittance
        acc = ref_ray_outputs["render"]["acc"].reshape(
            single_light_dists.shape[:-1] + (1,)
        )

        # Repeat occlusion for RGB channels
        occ = jax.lax.stop_gradient(jnp.repeat(acc, self.num_rgb_channels, axis=-1))

        # If light is extremely close, do not occlude
        light_offset_baseline = rays.lights[..., None, :] - rays.origins[..., None, :]
        light_dists_baseline = jnp.linalg.norm(light_offset_baseline, axis=-1, keepdims=True)
        occ = jnp.where(light_dists_baseline < 1e-3, jnp.zeros_like(occ), occ)

        # Possibly threshold the occlusion
        if self.config.occ_threshold_rate > 0:
            occ_thresh_weight = jnp.clip(
                (
                    (train_frac - self.config.occ_threshold_start_frac)
                    / self.config.occ_threshold_rate
                ),
                0.0,
                1.0,
            )
            occ_threshold = (
                occ_thresh_weight * self.config.occ_threshold_min
                + (1.0 - occ_thresh_weight) * self.config.occ_threshold_max
            )
        else:
            occ_threshold = self.config.occ_threshold_min

        occ = jnp.where(occ <= occ_threshold, jnp.zeros_like(occ), occ)

        return occ

    def _get_refdirs(self, viewdirs, normals, kwargs):
        """
        Returns reflection directions if 'refdirs' is in kwargs; otherwise
        computes reflection from normals and viewdirs, or just uses viewdirs.
        """
        if "refdirs" in kwargs:
            refdirs = kwargs["refdirs"].reshape(normals.shape[:-2] + (-1, 3))
            del kwargs["refdirs"]  # remove it to avoid accidental re-use
        else:
            refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals)

        if not self.use_reflections:
            # Use view directions instead of reflection vectors
            refdirs = viewdirs[..., None, :] * jnp.ones_like(refdirs)
        return refdirs

    def _compute_direct_from_material(
        self,
        rng,
        rays,
        sampler_results,
        material,
        material_feature,
        normals,
        light_dirs,
        light_radiance,
        train,
        train_frac,
        radiance_cache,
        **kwargs,
    ):
        # Diffuse
        diffuse_config = type('', (), {})()
        diffuse_config.shading = 'microfacet_diffuse'
        diffuse_config.use_brdf_correction = radiance_cache.shader.use_brdf_correction
        diffuse_config.use_diffuseness = radiance_cache.shader.use_diffuseness
        diffuse_config.use_mirrorness = radiance_cache.shader.use_mirrorness
        diffuse_config.use_specular_albedo = radiance_cache.shader.use_specular_albedo

        material_diffuse = render_utils.get_lobe(
            light_dirs[..., None, :],
            -rays.viewdirs[..., None, None, :] * jnp.ones_like(light_dirs[..., None, :]),
            normals[..., None, :],
            material,
            jnp.ones_like(light_dirs[..., None, :2]),
            diffuse_config,
        )

        # Specular
        specular_config = type('', (), {})()
        specular_config.shading = 'microfacet_specular'
        specular_config.use_brdf_correction = radiance_cache.shader.use_brdf_correction
        specular_config.use_diffuseness = radiance_cache.shader.use_diffuseness
        specular_config.use_mirrorness = radiance_cache.shader.use_mirrorness
        specular_config.use_specular_albedo = radiance_cache.shader.use_specular_albedo

        material_specular = render_utils.get_lobe(
            light_dirs[..., None, :],
            -rays.viewdirs[..., None, None, :] * jnp.ones_like(light_dirs[..., None, :]),
            normals[..., None, :],
            material,
            jnp.ones_like(light_dirs[..., None, :2]),
            specular_config,
        )

        # Output
        direct_diffuse = material_diffuse.reshape(material['albedo'].shape) * light_radiance
        direct_specular = material_specular.reshape(material['albedo'].shape) * light_radiance

        direct_diffuse = utils.stopgrad_with_weight(
            direct_diffuse, radiance_cache.shader.stopgrad_direct_weight
        )
        direct_specular = utils.stopgrad_with_weight(
            direct_specular, radiance_cache.shader.stopgrad_direct_weight
        )

        return direct_diffuse, direct_specular

    def _compute_direct_lighting(
        self,
        rng,
        rays,
        sampler_results,
        feature,
        shading_normals,
        roughness,
        bottleneck,
        normals,
        n_dot_l,
        light_radiance,
        light_dirs,
        radiance_cache,
        train_frac,
        train,
    ):
        """
        Computes direct lighting contributions (diffuse and specular), optionally
        from a shared material or from the local network predictions.
        Returns: (albedo, direct_diffuse, direct_specular).
        """
        if not self.use_active:
            # Short-circuit if direct lighting is disabled
            num_ch = self.config.num_rgb_channels
            zero_rgb = jnp.zeros_like(feature[..., :num_ch])
            return zero_rgb, zero_rgb, zero_rgb

        if radiance_cache.share_material:
            # If material is shared from another network
            key, rng = utils.random_split(rng)
            material_feature, material = radiance_cache.shader._predict_material_and_feature(
                rng=key,
                rays=rays,
                sampler_results=sampler_results,
                train_frac=train_frac,
                train=train,
                mesh=None,
                radiance_cache=radiance_cache,
            )
            albedo = material["albedo"]

            # Use the 'active' sampling approach from the material
            key, rng = utils.random_split(rng)
            direct_diffuse, direct_specular = self._compute_direct_from_material(
                key,
                rays,
                sampler_results,
                material,
                material_feature,
                normals,
                light_dirs,
                light_radiance,
                train_frac,
                train,
                radiance_cache,
            )
        else:
            # Standard approach using local predictions
            albedo = self.albedo_activation(
                self.albedo_layer(feature) + self.albedo_bias
            )
            direct_tint = nn.sigmoid(self.direct_tint_layer(feature))

            light_brdf = self.get_brdf_light(
                shading_normals,
                rays.viewdirs,
                light_dirs,
                bottleneck,
                roughness,
            )
            # Mask out invalid or back-facing
            light_brdf = jnp.where(
                n_dot_l == 0.0, jnp.zeros_like(light_brdf), light_brdf
            )

            direct_diffuse = albedo * n_dot_l * light_radiance / jnp.pi
            direct_specular = direct_tint * light_brdf * light_radiance

        # Optionally stop-grad for direct
        direct_diffuse = jnp.clip(direct_diffuse, 0.0, self.rgb_max)
        direct_specular = jnp.clip(direct_specular, 0.0, self.rgb_max)

        direct_diffuse = utils.stopgrad_with_weight(
            direct_diffuse, self.stopgrad_direct_weight
        )
        direct_specular = utils.stopgrad_with_weight(
            direct_specular, self.stopgrad_direct_weight
        )

        return albedo, direct_diffuse, direct_specular

    def _compute_ambient_specular(self, normals, viewdirs, feature, bottleneck, tint, integrated_brdf, ref_rgb):
        """
        Computes the ambient portion of the specular term, if 'use_ambient' is enabled.
        Otherwise returns zeros.
        """
        # We assume ref_rgb is the environment reflection
        ambient_specular = tint * integrated_brdf * ref_rgb
        return ambient_specular


    def get_predict_appearance_kwargs(
        self,
        rng,
        rays,
        sampler_results,
        **kwargs,
    ):
        means, covs = sampler_results["means"], sampler_results["covs"]
        predict_appearance_kwargs = {}

        if self.grid is not None:
            # Grid/hash structures don't give us an easy way to do closed-form
            # integration with a Gaussian, so instead we sample each Gaussian
            # according to an unscented transform (or something like it) and average
            # the sampled encodings.
            control_points_key, rng = utils.random_split(rng)

            if "tdist" in sampler_results:
                control, perp_mag = coord.compute_control_points(
                    means,
                    covs,
                    rays,
                    sampler_results["tdist"],
                    control_points_key,
                    self.unscented_mip_basis,
                    self.unscented_sqrt_fn,
                    self.unscented_scale_mult,
                )
            else:
                control = means[Ellipsis, None, :]
                perp_mag = jnp.zeros_like(control)

            control_offsets = control - means[Ellipsis, None, :]
            predict_appearance_kwargs["control_offsets"] = control_offsets
            predict_appearance_kwargs["perp_mag"] = perp_mag
            predict_appearance_kwargs["viewdirs"] = (
                jnp.ones_like(means[..., None, :]) * rays.viewdirs[..., None, None, :]
            )

        return dict(
            **predict_appearance_kwargs,
            **kwargs,
        )


@gin.configurable
class NeRFMLP(BaseNeRFMLP):
    use_active: bool = False  # If True, use a simple BRDF model.

    def _compute_indirect_lighting(
        self,
        feature,
        means,
        normals,
        shading_normals,
        ref_rgb,
        bottleneck,
        viewdirs,
        roughness,
        rays,
        light_dirs,
        light_radiance_mult,
    ):
        """
        Computes indirect lighting contributions (diffuse + specular) using
        integrated BRDF and/or transient indirect from the network.

        Returns:
        (indirect_diffuse, indirect_specular,
        transient_indirect, transient_indirect_diffuse, transient_indirect_specular)
        """
        if not self.use_indirect:
            num_ch = self.config.num_rgb_channels
            zero_rgb = jnp.zeros_like(feature[..., :num_ch])
            return zero_rgb, zero_rgb, None, None, None

        # Integrated BRDF
        integrated_brdf = self.get_integrated_brdf(normals, viewdirs, bottleneck)
        tint = nn.sigmoid(self.tint_layer(feature))

        # Indirect diffuse
        indirect_diffuse = (
            self.get_indirect(
                lights=rays.lights[..., None, :] * jnp.ones_like(normals),
                bottleneck=feature,
            ) * self.indirect_scale
        )

        indirect_specular = (
            tint * integrated_brdf * ref_rgb * self.indirect_scale
        )

        # Light intensity conditioning (if configured)
        if self.config.light_intensity_conditioning:
            scale_factor = (
                light_radiance_mult * self.config.light_intensity_conditioning_scale
                + self.config.light_intensity_conditioning_bias
            )
            indirect_diffuse *= scale_factor
            indirect_specular *= scale_factor

        return (
            indirect_diffuse,
            indirect_specular,
            None,
            None,
            None,
        )

    def get_indirect(
        self,
        lights,
        bottleneck,
    ):
        # Run view dependent network
        irradiance_input = self.lights_enc_fn(lights)
        irradiance_input = jnp.concatenate(
            [
                bottleneck,
                irradiance_input,
            ],
            axis=-1,
        )

        x = self.run_irradiance_network(irradiance_input)

        # Get RGB
        return self.irradiance_activation(
            self.indirect_layer(x) + self.irradiance_bias
        )


@gin.configurable
class TransientNeRFMLP(BaseNeRFMLP):
    use_active: bool = True  # If True, use a simple BRDF model.

    def _compute_indirect_lighting(
        self,
        feature,
        means,
        normals,
        shading_normals,
        ref_rgb,
        bottleneck,
        viewdirs,
        roughness,
        rays,
        light_dirs,
        light_radiance_mult,
    ):
        """
        Computes indirect lighting contributions (diffuse + specular) using
        integrated BRDF and/or transient indirect from the network.

        Returns:
        (indirect_diffuse, indirect_specular,
        transient_indirect, transient_indirect_diffuse, transient_indirect_specular)
        """
        if not self.use_indirect:
            num_ch = self.config.num_rgb_channels
            zero_rgb = jnp.zeros_like(feature[..., :num_ch])
            zero_rgb_transient = jnp.repeat(zero_rgb[..., None, :], self.config.n_bins, axis=-2)
            return zero_rgb, zero_rgb, zero_rgb_transient, zero_rgb_transient, zero_rgb_transient

        # Integrated BRDF
        integrated_brdf = self.get_integrated_brdf(normals, viewdirs, bottleneck)
        tint = nn.sigmoid(self.tint_layer(feature))

        # Transient indirect specular
        # We expand the tint and integrated_brdf to match shape
        tint_expanded = jnp.repeat(
            tint[..., None, :], self.config.n_bins, axis=-2
        ).reshape(ref_rgb.shape)

        # Transient indirect diffuse
        transient_indirect_diffuse = (
            self.get_indirect(
                lights=rays.lights[..., None, :] * jnp.ones_like(normals),
                bottleneck=feature,
            ) * self.indirect_scale
        )

        transient_indirect_specular = (
            tint_expanded * integrated_brdf * ref_rgb * self.indirect_scale
        )

        # Light intensity conditioning (if configured)
        if self.config.light_intensity_conditioning:
            scale_factor = (
                light_radiance_mult * self.config.light_intensity_conditioning_scale
                + self.config.light_intensity_conditioning_bias
            )
            transient_indirect_diffuse *= scale_factor
            transient_indirect_specular *= scale_factor

        # Zero out "invalid" bins for transient indirect
        shape_trans = transient_indirect_diffuse.shape
        transient_indirect_diffuse = transient_indirect_diffuse.reshape(
            shape_trans[:-1] + (self.config.n_bins, self.config.num_rgb_channels)
        )
        transient_indirect_specular = transient_indirect_specular.reshape(
            shape_trans[:-1] + (self.config.n_bins, self.config.num_rgb_channels)
        )

        (
            transient_indirect_diffuse,
            transient_indirect_specular,
        ) = render_utils.zero_invalid_bins(
            transient_indirect_diffuse,
            transient_indirect_specular,
            rays,
            means,
            self.config,
        )

        transient_indirect_diffuse = jnp.clip(transient_indirect_diffuse, 0.0, self.rgb_max)
        transient_indirect_specular = jnp.clip(transient_indirect_specular, 0.0, self.rgb_max)

        # Sum over bins
        indirect_diffuse = transient_indirect_diffuse.sum(-2)
        indirect_specular = transient_indirect_specular.sum(-2)

        # Combine for final transient
        transient_indirect = transient_indirect_diffuse + transient_indirect_specular

        return (
            indirect_diffuse,
            indirect_specular,
            transient_indirect,
            transient_indirect_diffuse,
            transient_indirect_specular,
        )

    def get_indirect(
        self,
        lights,
        bottleneck,
    ):
        # Run view dependent network
        irradiance_input = self.lights_enc_fn(lights)
        irradiance_input = jnp.concatenate(
            [
                bottleneck,
                irradiance_input,
            ],
            axis=-1,
        )

        x = self.run_irradiance_network(irradiance_input)

        # Get RGB
        return self.irradiance_activation(
            self.transient_indirect_layer(x) + self.irradiance_bias
        )

