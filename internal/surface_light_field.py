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

from collections import namedtuple
import dataclasses
import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Tuple, Union

from absl import logging
from flax import linen as nn
import gin
from internal import coord
from internal import geopoly
from internal import grid_utils
from internal import math
from internal import render
from internal import ref_utils
from internal import shading
from internal import utils
from internal.inverse_render import render_utils
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np


# Register external configurables with gin
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
gin.config.external_configurable(coord.contract_radius_2, module='coord')
gin.config.external_configurable(coord.contract_radius_1_2, module='coord')
gin.config.external_configurable(coord.contract_radius_1_4, module='coord')
gin.config.external_configurable(coord.contract_cube, module='coord')
gin.config.external_configurable(coord.contract_cube_5, module='coord')
gin.config.external_configurable(coord.contract_cube_2, module='coord')
gin.config.external_configurable(coord.contract_cube_1_4, module='coord')
gin.config.external_configurable(coord.contract_projective, module='coord')


@gin.configurable
class BaseSurfaceLightFieldMLP(shading.BaseShader):
    """Base class for Surface Light Field MLP implementation.
    
    This class implements a neural network for rendering surface light fields,
    with support for various features like directional encoding, distance prediction,
    and reflectance modeling.
    """
    config: Any = None  # A Config class, must be set upon construction.

    # Bottleneck configuration
    use_bottleneck: bool = True
    use_shader_bottleneck: bool = False

    # Directional encoding configuration
    use_directional_enc: bool = False
    use_ide: bool = False  # Integrated Directional Encoding

    # View-dependent network configuration
    net_depth_viewdirs: int = 1
    net_width_viewdirs: int = 128
    bottleneck_viewdirs: int = 128
    skip_layer_dir: int = 4
    deg_view: int = 4

    # Far field configuration
    use_far_field_points: bool = False
    use_env_alpha: bool = False

    # Distance prediction configuration
    use_distance_prediction: bool = False
    use_distance_ide: bool = False
    use_sorted_distances: bool = False
    net_depth_distance: int = 1
    net_width_distance: int = 128
    skip_layer_distance: int = 4
    deg_view_distance: int = 2

    # Density prediction configuration
    use_density_prediction: bool = False
    net_depth_density: int = 2
    net_width_density: int = 64
    skip_layer_density: int = 2
    density_activation: Callable[Ellipsis, Any] = math.safe_exp
    density_bias: float = -1.0
    density_noise: float = 0.0

    # Alpha configuration
    alpha_bias: float = 2.0
    alpha_activation: Callable[Ellipsis, Any] = nn.sigmoid

    # Origins configuration
    use_origins: bool = False
    deg_origins: int = 4

    # Lights configuration
    use_lights: bool = True
    deg_lights: int = 2

    # Points configuration
    use_points: bool = False
    use_points_ide: bool = False
    deg_points: int = 4

    # Sphere points configuration
    use_sphere_points: bool = False
    deg_sphere_points: int = 4
    sphere_radius: float = 5.0

    # Distance sampling configuration
    num_distance_samples: int = 1
    num_far_samples: int = 0
    distance_scale: float = 1.0
    distance_bias: float = -2.0
    distance_near: float = 1e-3
    distance_far: float = 1e6
    distance_far_field: float = float("inf")
    use_uniform_distance: bool = False
    use_uniform_loss: bool = False
    use_indirect: bool = False
    use_voxel_grid: bool = False
    voxel_start: float = 0.0
    voxel_end: float = 10.0
    use_uniform_grid: bool = True

    raydist_fn: Union[Tuple[Callable[Ellipsis, Any], Ellipsis], Callable[Ellipsis, Any]] = None

    # Point offset configuration
    use_point_offsets: bool = False
    point_offset_scale: float = 0.25
    point_offset_bias: float = -3.0

    window_points_frac: float = 0.0

    # Reflectance grid configuration
    use_reflectance_grid: bool = False
    reflectance_grid_representation: str = 'ngp'
    reflectance_grid_params: ml_collections.FrozenConfigDict = (
        ml_collections.FrozenConfigDict({})
    )
    ref_warp_fn: Callable[Ellipsis, Any] = None
    use_roughness: bool = False
    roughness_scale: float = 0.001

    per_ref_feature_output: bool = False

    # Light and illumination configuration
    num_light_features: int = 64
    use_illumination_feature: bool = False
    multiple_illumination_outputs: bool = True
    rotate_illumination: bool = False

    # RGB output configuration
    rgb_max: float = float('inf')
    ambient_rgb_max: float = float('inf')
    ambient_rgb_activation: Callable[Ellipsis, Any] = nn.softplus
    ambient_rgb_bias: float = -1.0

    def setup(self):
        """Initialize the model components."""
        # Initialize basis for positional encoding
        self._setup_basis()
        
        # Initialize encoding functions
        self._setup_encoding_functions()
        
        # Initialize illumination components
        self._setup_illumination()
        
        # Initialize grids
        self._setup_grids()
        
        # Initialize network layers
        self._setup_network_layers()

    def _setup_basis(self):
        """Set up positional encoding basis."""
        self.pos_basis_t = jnp.array(
            geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)
        ).T

    def _setup_encoding_functions(self):
        """Set up various encoding functions used by the model."""
        # Directional encoding function
        if self.use_ide:
            self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
        else:
            def dir_enc_fn(direction, _):
                return coord.pos_enc(
                    direction,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True
                )
            self.dir_enc_fn = dir_enc_fn

        # Origins encoding function
        def origins_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_origins,
                append_identity=True
            )
        self.origins_enc_fn = origins_enc_fn

        # Lights encoding function
        def lights_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_lights,
                append_identity=True
            )
        self.lights_enc_fn = lights_enc_fn

        # Sphere points encoding function
        def sphere_points_enc_fn(direction):
            return coord.pos_enc(
                direction,
                min_deg=0,
                max_deg=self.deg_sphere_points,
                append_identity=True
            )
        self.sphere_points_enc_fn = sphere_points_enc_fn

        # Points encoding function
        if self.use_points_ide:
            self.points_enc_fn = ref_utils.generate_ide_fn(self.deg_points)
        elif self.window_points_frac > 0.0:
            def points_enc_fn(direction, train_frac):
                alpha = jnp.clip(train_frac / self.window_points_frac, 0.0, 1.0) * self.deg_points
                return coord.windowed_pos_enc(
                    direction,
                    min_deg=0,
                    max_deg=self.deg_points,
                    alpha=alpha,
                    append_identity=True
                )
            self.points_enc_fn = points_enc_fn
        else:
            def points_enc_fn(direction, _):
                return coord.pos_enc(
                    direction,
                    min_deg=0,
                    max_deg=self.deg_points,
                    append_identity=True
                )
            self.points_enc_fn = points_enc_fn

        # Distance encoding function
        if self.use_distance_prediction:
            if self.use_distance_ide:
                self.dir_enc_fn_distance = ref_utils.generate_ide_fn(self.deg_view_distance)
            else:
                def dir_enc_fn_distance(direction, _):
                    return coord.pos_enc(
                        direction,
                        min_deg=0,
                        max_deg=self.deg_view_distance,
                        append_identity=True,
                    )
                self.dir_enc_fn_distance = dir_enc_fn_distance

    def _setup_illumination(self):
        """Set up illumination-related components."""
        # Handle light features and illumination outputs
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
        
        # Set up illumination rotation matrices if needed
        if self.rotate_illumination and self.config.rotate_illumination:
            light_rotation_matrix = []

            for i in range(self.config.num_illuminations):
                horizontal_angle = jnp.array(self.config.light_rotations[i] / 180 * jnp.pi).astype(jnp.float32)

                rotation_matrix = jnp.array(
                    [
                        [jnp.cos(horizontal_angle), -jnp.sin(horizontal_angle), 0],
                        [jnp.sin(horizontal_angle), jnp.cos(horizontal_angle), 0],
                        [0, 0, 1]
                    ]
                ).astype(jnp.float32)

                light_rotation_matrix.append(rotation_matrix)

            self.light_rotation_matrix = jnp.stack(light_rotation_matrix, axis=0)

    def _setup_grids(self):
        """Initialize grid representations."""
        # Set up appearance grid if needed
        if self.use_grid:
            self.grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
                self.grid_representation.lower()
            ](
                name='distance_grid',
                **self.grid_params
            )
        else:
            self.grid = None

        # Set up reflectance grid if needed
        if self.use_reflectance_grid:
            self.reflectance_grid = grid_utils.GRID_REPRESENTATION_BY_NAME[
                self.reflectance_grid_representation.lower()
            ](
                name='reflectance_grid',
                **self.reflectance_grid_params
            )
        else:
            self.reflectance_grid = None

    def _setup_network_layers(self):
        """Initialize neural network layers."""
        # Helper for creating dense layers
        self.dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)()
        )
        self.zeros_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, 'zeros')
        )

        # Bottleneck layers
        self.layers = [
            self.dense_layer(self.net_width) for i in range(self.net_depth)
        ]
        self.bottleneck_layer = self.dense_layer(self.bottleneck_width)

        # View-dependent layers
        self.view_dependent_layers = [
            self.dense_layer(self.net_width_viewdirs, name=f"layer_{i}")
            for i in range(self.net_depth_viewdirs - 1)
        ] + [self.dense_layer(self.bottleneck_viewdirs, name=f"layer_bottleneck")]

        # Ambient view-dependent layers
        self.ambient_view_dependent_layers = [
            self.dense_layer(self.net_width_viewdirs, name=f"ambient_layer_{i}")
            for i in range(self.net_depth_viewdirs - 1)
        ] + [self.dense_layer(self.bottleneck_viewdirs, name=f"ambient_layer_bottleneck")]

        # Output layers
        output_channels = self.config.num_rgb_channels
        if self.use_indirect:
            output_channels *= self.config.n_bins
        
        self.output_rgba_layer = self.dense_layer(
            output_channels * self.num_illumination_outputs + 1, 
            name="output_rgba_layer"
        )
        
        self.output_ambient_rgb_layer = self.dense_layer(
            self.config.num_rgb_channels * self.num_illumination_outputs, 
            name="output_ambient_rgb_layer"
        )

        # Distance prediction network
        if self.use_distance_prediction:
            self.distance_layers = [
                self.dense_layer(self.net_width_distance, name=f"distance_layer_{i}")
                for i in range(self.net_depth_distance)
            ]
            self.output_distance_layer = self.zeros_layer(
                8 * self.num_distance_samples + 4, name="distance_output_layer"
            )

        # Density prediction network
        if self.use_density_prediction:
            self.density_layers = [
                self.dense_layer(self.net_width_density, name=f"density_layer_{i}")
                for i in range(self.net_depth_density)
            ]
        
        self.output_density_layer = self.dense_layer(1, name="output_density_layer")

    def get_light_vec(self, rays, feature):
        """Get the light feature vector for the given rays.
        
        Args:
            rays: Ray information containing light indices
            feature: Features to match the shape for broadcasting
            
        Returns:
            Light feature vector or empty tensor if not using multi-illumination
        """
        light_vec = jnp.zeros_like(feature[..., 0:0])

        if self.config.multi_illumination > 0:
            light_idx = rays.light_idx[Ellipsis, 0]
            light_vec = self.light_vecs(light_idx)
            light_vec = light_vec[..., None, :] * jnp.ones_like(feature[..., 0:1])

        return light_vec

    def run_distances_network(self, bottleneck, origins, refdirs, roughness):
        """Run the distance prediction network.
        
        Args:
            bottleneck: Bottleneck features
            origins: Ray origins
            refdirs: Ray directions
            roughness: Surface roughness values
            
        Returns:
            Distance network outputs
        """
        # Encode inputs
        dir_enc_distance = self.dir_enc_fn_distance(refdirs, roughness)
        origins_enc = self.origins_enc_fn(self.warp_fn(origins))

        # Prepare network input
        x = jnp.concatenate([bottleneck, origins_enc, dir_enc_distance], axis=-1)
        inputs = x

        # Feed through network layers with skip connections
        for i in range(self.net_depth_distance):
            x = self.distance_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_distance == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return self.output_distance_layer(x)

    def run_density_network(self, ref_grid_feat):
        """Run the density prediction network.
        
        Args:
            ref_grid_feat: Grid features for density prediction
            
        Returns:
            Predicted density values
        """
        x = ref_grid_feat
        inputs = x

        # Feed through network layers with skip connections
        for i in range(self.net_depth_density):
            x = self.density_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_density == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        # Get density output
        raw_density = self.output_density_layer(x)[..., 0]
        density = self.density_activation(raw_density + self.density_bias)

        return density

    def run_surface_lightfield_network(self, x):
        """Run the surface light field network.
        
        Args:
            x: Input features
            
        Returns:
            Surface light field features
        """
        inputs = x

        # Feed through network layers with skip connections
        for i in range(self.net_depth_viewdirs):
            x = self.view_dependent_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_dir == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return x

    def run_ambient_surface_lightfield_network(self, x):
        """Run the ambient surface light field network.
        
        Args:
            x: Input features
            
        Returns:
            Ambient surface light field features
        """
        inputs = x

        # Feed through network layers with skip connections
        for i in range(self.net_depth_viewdirs):
            x = self.ambient_view_dependent_layers[i](x)
            x = self.net_activation(x)

            if i % self.skip_layer_dir == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        return x
    
    def get_raydist_fns(self, rays_near):
        """Get ray distance transformation functions.
        
        Args:
            rays_near: Near plane distances for rays
            
        Returns:
            Tuple of (t_to_s, s_to_t) transformation functions
        """
        if self.raydist_fn is not None:
            fn, fn_inv, raydist_kwargs = self.raydist_fn  # pylint: disable=unpacking-non-sequence
        else:
            fn = lambda x: x
            fn_inv = lambda x: x
            raydist_kwargs = {}

        # Create ray warping functions
        t_to_s, s_to_t = coord.construct_ray_warps(
            functools.partial(fn, **raydist_kwargs),
            jnp.ones_like(rays_near) * self.distance_near,
            jnp.ones_like(rays_near) * self.distance_far,
            fn_inv=functools.partial(fn_inv, **raydist_kwargs),
        )
        
        # Use uniform distance if specified
        if self.use_uniform_distance:
            s_to_t = lambda x: x * (self.distance_far - self.distance_near) + self.distance_near
            t_to_s = lambda x: (x - self.distance_near) / (self.distance_far - self.distance_near)
        elif self.use_uniform_loss:
            t_to_s = lambda x: (x - self.distance_near) / (self.distance_far - self.distance_near)
        
        return t_to_s, s_to_t

    def get_voxel_raydist_fns(self, rays_near):
        """Get voxel grid ray distance transformation functions.
        
        Args:
            rays_near: Near plane distances for rays
            
        Returns:
            Tuple of (t_to_s, s_to_t) transformation functions for voxel grid
        """
        fn, fn_inv, raydist_kwargs = self.raydist_fn  # pylint: disable=unpacking-non-sequence
        
        # Create ray warping functions for voxel grid
        t_to_s, s_to_t = coord.construct_ray_warps(
            functools.partial(fn, **raydist_kwargs),
            jnp.ones_like(rays_near) * self.voxel_start,
            jnp.ones_like(rays_near) * self.voxel_end,
            fn_inv=functools.partial(fn_inv, **raydist_kwargs),
        )
        
        # Use uniform grid if specified
        if self.use_uniform_grid:
            s_to_t = lambda x: x * (self.voxel_end - self.voxel_start) + self.voxel_start
            t_to_s = lambda x: (x - self.voxel_start) / (self.voxel_end - self.voxel_start)
        
        return t_to_s, s_to_t
    
    def convert_to_sdist(self, rays, tdist):
        """Convert ray t-distance to s-distance.
        
        Args:
            rays: Ray information
            tdist: T-distances to convert
            
        Returns:
            Converted s-distances
        """
        t_to_s, _ = self.get_raydist_fns(rays.near)
        return t_to_s(tdist)
    
    def predict_points(
        self,
        rng,
        rays,
        origins,
        refdirs,
        bottleneck,
        roughness,
        near: float = 0.0,
        far: float = float("inf"),
        **kwargs,
    ):
        """Predict 3D sample points along rays.
        
        Args:
            rng: Random number generator key
            rays: Ray information
            origins: Ray origins
            refdirs: Ray directions
            bottleneck: Bottleneck features
            roughness: Surface roughness values
            near: Near plane distance
            far: Far plane distance
            **kwargs: Additional arguments
            
        Returns:
            Tuple containing predicted points and related information
        """
        # Get ray distance transformation functions
        t_to_s, s_to_t = self.get_raydist_fns(rays.near[..., None])
        
        # Get distance network outputs
        dist_net_outputs = self.run_distances_network(
            bottleneck, origins, refdirs, roughness,
        )

        # Extract environment information
        env_rgb = self.rgb_activation(
            self.rgb_premultiplier * dist_net_outputs[..., -4:-1] + self.rgb_bias
        )
        if self.use_env_alpha:
            env_alpha = self.alpha_activation(dist_net_outputs[..., -1:] + self.alpha_bias)
        else:
            env_alpha = jnp.ones_like(dist_net_outputs[..., -1:])

        # Process distance outputs
        dist_net_outputs = dist_net_outputs[..., :-4]
        dist_net_outputs = dist_net_outputs.reshape(
            dist_net_outputs.shape[:-1] + (
                self.num_distance_samples,
                dist_net_outputs.shape[-1] // self.num_distance_samples
            )
        )
        
        # Extract different components from network output
        distance_offsets = dist_net_outputs[..., 0]
        distance_sigma = dist_net_outputs[..., 1]
        point_sigma = dist_net_outputs[..., 2]
        raw_weights = dist_net_outputs[..., 4]
        point_offsets = dist_net_outputs[..., 5:8]

        # Process distance offsets
        distance_offsets = (
            distance_offsets
            * self.distance_scale
            / self.num_distance_samples
            * nn.sigmoid(distance_sigma + self.distance_bias)
        )

        # Handle different distance sampling approaches
        if self.use_voxel_grid:
            # Voxel grid sampling
            _, voxel_s_to_t = self.get_voxel_raydist_fns(rays.near[..., None, None])

            # Reshape distance offsets for voxel grid
            distance_offsets = distance_offsets.reshape(
                distance_offsets.shape[:-1] + (-1,) + (3,)
            )

            # Calculate grid positions
            distance_offsets = (
                2.0 * distance_offsets
                + jnp.linspace(-1.0, 1.0, self.num_distance_samples // 3).reshape(
                    tuple(1 for _ in distance_offsets.shape[:-2]) + (self.num_distance_samples // 3, 1)
                )
            )
            distance_offsets = voxel_s_to_t(jnp.abs(distance_offsets)) * jnp.sign(distance_offsets)

            # Calculate ray-point intersections
            d = jnp.where(
                jnp.abs(refdirs) < 1e-5,
                jnp.ones_like(refdirs) * 1e12,
                refdirs
            )

            distances = (distance_offsets - origins[..., None, :]) / d[..., None, :]
            distances = distances.reshape(distance_offsets.shape[:-2] + (self.num_distance_samples,))

            # Convert to s-space
            s_distances = t_to_s(distances)
        else:
            # Regular distance sampling
            s_distance_offsets = distance_offsets

            # Calculate base s-distances
            if self.num_far_samples > 0:
                # Split into near and far samples
                start_s_distances_1 = jnp.linspace(
                    1e-8, 0.9, (self.num_distance_samples - self.num_far_samples)
                ).reshape(tuple(1 for _ in s_distance_offsets.shape[:-1]) + (-1,))

                start_s_distances_2 = jnp.linspace(
                    0.9, 1.0 - 1e-8, self.num_far_samples
                ).reshape(tuple(1 for _ in s_distance_offsets.shape[:-1]) + (-1,))

                start_s_distances = jnp.concatenate(
                    [start_s_distances_1, start_s_distances_2], axis=-1
                )
            else:
                # Uniform sampling
                start_s_distances = jnp.linspace(
                    1e-8, 1.0 - 1e-8, self.num_distance_samples,
                ).reshape(tuple(1 for _ in s_distance_offsets.shape[:-1]) + (-1,))

            # Apply offsets and fold back into [0,1]
            s_distances = s_distance_offsets + start_s_distances
            s_distances_floor = jnp.floor(s_distances).astype(jnp.int32)
            s_distances_frac = s_distances - s_distances_floor.astype(jnp.float32)
            s_distances = jnp.where(
                (s_distances_floor % 2) == 0,
                s_distances_frac,
                1.0 - s_distances_frac,
            )

            # Convert to t-space
            distances = s_to_t(s_distances)
        
        # Sort distances if needed
        if self.use_sorted_distances:
            idx = jnp.argsort(distances, axis=-1)
            distances = jnp.take_along_axis(distances, idx, axis=-1)
            s_distances = jnp.take_along_axis(s_distances, idx, axis=-1)
            raw_weights = jnp.take_along_axis(raw_weights, idx, axis=-1)
            point_sigma = jnp.take_along_axis(point_sigma, idx, axis=-1)
            point_offsets = jnp.take_along_axis(
                point_offsets,
                jnp.repeat(idx[..., None], 3, axis=-1),
                axis=-1
            )

        # Create validity mask for points
        ref_mask = (distances > self.distance_near).astype(jnp.float32)
        ref_mask = ref_mask * (distances < self.distance_far).astype(jnp.float32)
        ref_mask = ref_mask * (distances > near).astype(jnp.float32)
        ref_mask = ref_mask * (distances < far).astype(jnp.float32)
        
        # Clamp distances to valid range
        distances = jnp.clip(distances, self.distance_near, self.distance_far)

        # Calculate 3D points from ray parameters
        if self.use_point_offsets:
            # Apply offset to points
            point_offsets = (
                nn.tanh(point_offsets) 
                * self.point_offset_scale
                * nn.sigmoid(point_sigma + self.point_offset_bias)[..., None]
            )

            # Calculate base points
            points = origins[..., None, :] + distances[..., None] * refdirs[..., None, :]
            
            # Apply offsets
            points = points + point_offsets
        else:
            # Calculate points without offsets
            points = origins[..., None, :] + distances[..., None] * refdirs[..., None, :]

        # Return all computed values
        return (
            points,
            raw_weights,
            ref_mask,
            s_distances,
            distances,
            env_rgb,
            env_alpha,
        )

    def __call__(
        self,
        rng,
        rays,
        sampler_results,
        origins,
        refdirs,
        roughness: Any = None,
        shader_bottleneck: Any = None,
        train: bool = True,
        train_frac: float = 1.0,
        dist_only: bool = False,
        **kwargs
    ):
        """Forward pass of the surface light field MLP.
        
        Args:
            rng: Random number generator key
            rays: Ray information
            sampler_results: Results from the sampler
            origins: Ray origins
            refdirs: Ray directions
            roughness: Surface roughness values
            shader_bottleneck: Shader bottleneck features
            train: Whether in training mode
            train_frac: Training progress fraction
            dist_only: Whether to only compute distances
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of outputs
        """
        outputs = {}
        
        # Reshape origins to match refdirs
        origins = origins.reshape(refdirs.shape[:-2] + (-1, 3)) * jnp.ones_like(refdirs)

        # Handle cached distance if provided
        if "cache_tdist" in kwargs:
            outputs["cache_sdist"] = self.convert_to_sdist(rays, kwargs["cache_tdist"])

            if dist_only:
                return outputs
        
        # Apply rotation to directions if needed
        if self.rotate_illumination and self.config.rotate_illumination:
            light_idx = rays.light_idx
            sh = light_idx.shape

            # Get rotation matrices for the given light indices
            light_rotation = self.light_rotation_matrix.reshape(
                tuple(1 for _ in sh[:-1]) + (self.config.num_illuminations, 9,)
            )
            light_rotation = jnp.take_along_axis(light_rotation, light_idx[..., None], axis=-2)[..., 0, :]
            light_rotation = light_rotation.reshape(sh[:-1] + (1, 3, 3))

            # Apply rotation to directions
            refdirs = (
                light_rotation[..., :3, 0] * refdirs[..., 0:1]
                + light_rotation[..., :3, 1] * refdirs[..., 1:2]
                + light_rotation[..., :3, 2] * refdirs[..., 2:3]
            )

        # Initialize network input components
        x = []

        # Add ray origins if enabled
        if self.use_origins:
            x.append(self.origins_enc_fn(origins))

        # Get bottleneck features
        if self.use_grid:
            # Get bottleneck from appearance grid
            key, rng = utils.random_split(rng)
            predict_appearance_kwargs = self.get_predict_appearance_kwargs(key, rays, sampler_results)

            bottleneck = self.predict_appearance_feature(
                sampler_results,
                train=train,
                **predict_appearance_kwargs,
            ) * jnp.ones_like(refdirs[..., :1])
        elif self.use_shader_bottleneck:
            # Use provided shader bottleneck
            bottleneck = shader_bottleneck
        else:
            # No bottleneck
            bottleneck = jnp.zeros_like(refdirs)

        # Add bottleneck to input if enabled
        if self.use_bottleneck:
            x.append(bottleneck)

        # Add light vector if enabled
        if self.config.multi_illumination and self.use_illumination_feature:
            light_vec = self.get_light_vec(rays, bottleneck)
            x.append(light_vec)

        # Add shader bottleneck if enabled
        if self.use_shader_bottleneck:
            x.append(shader_bottleneck)

        # Initialize distance-related variables
        s_distances = jnp.zeros_like(bottleneck[..., 0:1])
        distances = jnp.zeros_like(bottleneck[..., 0:1])
        raw_weights = jnp.ones_like(bottleneck[..., 0:1])
        ref_weights = jnp.ones_like(bottleneck[..., 0:1])
        ref_mask = jnp.ones_like(bottleneck[..., 0:1])
        env_rgb = jnp.zeros_like(bottleneck[..., 0:3])
        env_alpha = jnp.zeros_like(bottleneck[..., 0:1])
        acc = jnp.ones_like(bottleneck[..., 0:1])

        # Predict points if distance prediction is enabled
        if self.use_distance_prediction:
            key, rng = utils.random_split(rng)
            (
                points, raw_weights, ref_mask, s_distances, distances, env_rgb, env_alpha
            ) = self.predict_points(
                key, rays, origins, refdirs, bottleneck, roughness, **kwargs,
            )
            
            # Apply reference warping
            points = self.ref_warp_fn(points)

            # Calculate weights
            ref_weights = jax.nn.softmax(raw_weights, axis=-1)
            s_distances = (s_distances * ref_weights).sum(axis=-1, keepdims=True)
            ref_weights = ref_weights * ref_mask * env_alpha
        
        # Use normalized directions as points for far-field modeling
        if self.use_far_field_points:
            points = ref_utils.l2_normalize(refdirs)[..., None, :]

        # Handle reflectance grid if enabled
        if self.use_reflectance_grid:
            # Calculate features from reflectance grid
            ref_roughness = (
                roughness[..., None, :] * distances[..., None] * self.roughness_scale
            ) if self.use_roughness else None

            ref_grid_feat = self.reflectance_grid(
                points,
                x_scale=ref_roughness,
                per_level_fn=lambda x: x,
                train=train,
                train_frac=train_frac,
            )

            # Calculate weights with density prediction if enabled
            if self.use_density_prediction:
                ref_density = self.run_density_network(ref_grid_feat)
                ref_weights, _, _ = render.compute_alpha_weights(
                    (
                        ref_density * self.density_activation(raw_weights + self.density_bias)
                    ),
                    None,
                    refdirs,
                    opaque_background=False,
                    delta=jnp.ones_like(distances) / self.num_distance_samples,
                )

                # Apply mask and renormalize
                ref_weights = ref_weights * ref_mask
                
                # Recalculate weighted average s-distance
                s_distances = (s_distances * ref_weights).sum(axis=-1, keepdims=True)

            # Handle per-feature output mode
            if self.per_ref_feature_output:
                # Run network on per-feature basis
                feature_x = [ref_grid_feat]
                x = self.run_surface_lightfield_network(jnp.concatenate(feature_x, axis=-1))
                
                # Get RGBA output
                raw_rgba = self.output_rgba_layer(x)
                rgb = self.rgb_activation(
                    self.rgb_premultiplier * raw_rgba[..., :3] + self.rgb_bias
                )
                alpha = nn.sigmoid(raw_rgba[..., -1:] - 1.0)

                # Weight and sum RGB
                rgb = (rgb[..., :3] * ref_weights[..., None]).sum(axis=-2)

                # Set outputs
                outputs['incoming_rgb'] = rgb
                outputs['incoming_alpha'] = alpha
                outputs['incoming_env_rgba'] = jnp.concatenate([env_rgb, env_alpha], axis=-1)
                outputs['incoming_weights'] = ref_weights
                outputs['incoming_s_dist'] = s_distances
                outputs['incoming_dist'] = distances

                outputs["incoming_acc"] = outputs['incoming_weights'].sum(axis=-1, keepdims=False)

                return outputs
            else:
                # Weight and sum grid features
                ref_grid_feat = (ref_grid_feat * ref_weights[..., None]).sum(axis=-2)
                
                # Add to input
                x.append(ref_grid_feat)
        else:
            # Average s-distances if not using reflectance grid
            s_distances = s_distances.mean(axis=-1, keepdims=True)

        # Add encoded points if enabled
        if self.use_points:
            # Normalize points
            points = ref_utils.l2_normalize(points)

            # Encode points
            if self.use_points_ide:
                x.append(
                    self.points_enc_fn(points, roughness[..., None, :]).reshape(
                        origins.shape[:-1] + (-1,)
                    )
                )
            else:
                x.append(
                    self.points_enc_fn(points, train_frac).reshape(
                        origins.shape[:-1] + (-1,)
                    )
                )

        # Add sphere intersection points if enabled
        if self.use_sphere_points:
            t = jnp.ones_like(refdirs) * self.sphere_radius
            sphere_points = ref_utils.l2_normalize(origins + t * refdirs)
            x.append(self.sphere_points_enc_fn(sphere_points))

        # Add directional encoding if enabled
        if self.use_directional_enc:
            x.append(self.dir_enc_fn(refdirs, roughness))

        # Process input through networks
        if self.use_lights:
            # Run ambient network
            ambient_x = self.run_ambient_surface_lightfield_network(
                jnp.concatenate(x, axis=-1)
            )

            # Add light encoding
            lights_enc = self.lights_enc_fn(
                self.warp_fn(rays.lights[..., None, :] * jnp.ones_like(origins))
            )
            x.append(lights_enc)

            # Run main network with lights
            x = self.run_surface_lightfield_network(jnp.concatenate(x, axis=-1))
        else:
            # Run main network without lights
            x = self.run_surface_lightfield_network(jnp.concatenate(x, axis=-1))
            
            # Use same features for ambient
            ambient_x = x

        # Get RGB output
        raw_rgba = self.output_rgba_layer(x)
        rgb = self.rgb_activation(
            self.rgb_premultiplier * raw_rgba[..., :-1] + self.rgb_bias
        )
        alpha = jnp.clip(
            self.alpha_activation(raw_rgba[..., -1:] + self.alpha_bias), 
            0.0, 1.0
        )

        # Handle multiple illumination outputs if enabled
        if self.config.multi_illumination and self.multiple_illumination_outputs:
            light_idx = rays.light_idx[Ellipsis, None, :] * jnp.ones_like(bottleneck[..., 0:1]).astype(rays.light_idx.dtype)
            rgb = rgb.reshape(rgb.shape[:-1] + (self.num_illumination_outputs, -1))
            rgb = jnp.take_along_axis(rgb, light_idx[..., None], axis=-2)[..., 0, :]

        # Get ambient RGB
        ambient_rgb = self.ambient_rgb_activation(
            self.output_ambient_rgb_layer(ambient_x) + self.ambient_rgb_bias
        )

        # Set outputs
        outputs['incoming_rgb'] = jnp.clip(rgb, 0.0, self.rgb_max)
        outputs['incoming_ambient_rgb'] = jnp.clip(ambient_rgb, 0.0, self.ambient_rgb_max)
        outputs['incoming_alpha'] = alpha
        outputs['incoming_weights'] = ref_weights
        outputs['incoming_s_dist'] = s_distances
        outputs['incoming_dist'] = distances
        outputs['incoming_env_rgba'] = jnp.concatenate([env_rgb, env_alpha], axis=-1)

        # Acc
        outputs["incoming_acc"] = outputs['incoming_weights'].sum(axis=-1, keepdims=False)

        return outputs


@gin.configurable
class SurfaceLightFieldMLP(BaseSurfaceLightFieldMLP):
    """Implementation of surface light field MLP."""
    pass


@gin.configurable
class TransientSurfaceLightFieldMLP(BaseSurfaceLightFieldMLP):
    """Implementation of transient surface light field MLP."""
    pass