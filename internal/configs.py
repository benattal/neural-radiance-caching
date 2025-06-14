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
"""Utility functions for handling configurations."""

import dataclasses
import enum
import functools
import os
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

import gin
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf

gfile = tf.io.gfile
from absl import flags
from flax.core import FrozenDict

from internal import math

BboxType = Tuple[Tuple[float, float, float], Tuple[float, float, float]]


gin.add_config_file_search_path("configs/")

configurables = {
    "math": [math.power_ladder, math.create_learning_rate_decay],
    "jnp": [
        jnp.reciprocal,
        jnp.log,
        jnp.log1p,
        jnp.exp,
        jnp.sqrt,
        jnp.square,
        jnp.sum,
        jnp.mean,
        jnp.abs,
    ],
    "jax.nn": [jax.nn.relu, jax.nn.softplus, jax.nn.silu, jax.nn.sigmoid],
    "jax.nn.initializers.he_normal": [jax.nn.initializers.he_normal()],
    "jax.nn.initializers.he_uniform": [jax.nn.initializers.he_uniform()],
    "jax.nn.initializers.glorot_normal": [jax.nn.initializers.glorot_normal()],
    "jax.nn.initializers.glorot_uniform": [jax.nn.initializers.glorot_uniform()],
    "optax": [
        optax.adam,
        optax.sgd,
        optax.adamw,
        optax.warmup_exponential_decay_schedule,
        optax.warmup_cosine_decay_schedule,
        optax.linear_schedule,
        optax.constant_schedule,
        optax.polynomial_schedule,
        optax.join_schedules,
        optax.piecewise_constant_schedule,
        optax.piecewise_interpolate_schedule,
    ],
}

for module, configurables in configurables.items():
    for configurable in configurables:
        gin.config.external_configurable(configurable, module=module)


# CallDef is a construct that makes it easier to use callables with arguments
# in Gin configs. A CallDef is simply a tuple containing a callable and keyword
# arguments the callable should be called with.
#
# See: `parse_call_def` and `parse_call_def_partial`.
#
# Example:
#   ```
#   >> def add(a, b):
#   >>   return a + b
#
#   >> call_def = (add, {'a': 1, 'b': 2})
#   >> config_utils.parse_call_def(call_def)
#   3
#   ```
CallDef = Tuple[Callable[Ellipsis, Any], Mapping[str, Any]]


def parse_call_def(call_def):
    """Parses a function call definition.

    Args:
      call_def: A tuple containing (fn, kwargs).

    Returns:
      The result of `fn(**kwargs)`.
    """
    fn, kwargs = call_def
    return fn(**kwargs)


def parse_call_def_partial(call_def):
    """Parses a function call definition partially.

    Parses a CallDef, but instead of evaluating the function immediately,
    return a partial function with the given kwargs.

    Args:
      call_def: A tuple containing (fn, kwargs).

    returns:
      a partial function `fn(**kwargs)`.
    """
    fn, kwargs = call_def
    return functools.partial(fn, **kwargs)


@gin.configurable
def join_schedule_defs(schedule_defs, boundaries):
    """A gin configurable wrapper around `optax.join_schedules`."""
    schedules = [parse_call_def(s) for s in schedule_defs]
    return optax.join_schedules(schedules, boundaries)


@gin.constants_from_enum
class MaskInput(enum.Enum):
    """Specify the format of mask.

    Attributes:
      NONE: No mask used
      PNG: Masks are `.png` images inside a `masks/` subfolder
      PROTO: Masks are `.proto`
        `geo_machine_perception.semantic_index.SemanticIndexDataEntry`
    """

    NONE = enum.auto()
    PNG = enum.auto()
    PROTO = enum.auto()

    def __bool__(self):
        # `if config.use_mask:` is `False` when `NONE`
        return self is not MaskInput.NONE


@gin.constants_from_enum
class ModelType(enum.Enum):
    DEFAULT = enum.auto()
    TRANSIENT = enum.auto()
    MATERIAL = enum.auto()
    TRANSIENT_MATERIAL = enum.auto()


@gin.configurable()
@dataclasses.dataclass
class Config:
    """Configuration flags for everything."""

    #-----------------------------------------------------------------------------
    # General Configuration
    #-----------------------------------------------------------------------------
    debug_mode: bool = False  # If True, compute some expensive debug outputs.
    use_transient: bool = False  # If True, compute some expensive debug outputs.
    model_type: ModelType = ModelType.DEFAULT  # Model type to use
    checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
    partial_checkpoint_dir: str | None = None
    render_dir: Optional[str] = None  # Output rendering directory.
    data_dir: Optional[str] = None  # Input data directory.
    jax_rng_seed: int = 20200823  # The seed that JAX's RNG uses.
    np_rng_seed: int = 20201473  # The seed that Numpy's RNG uses.
    disable_pmap_and_jit: bool = False  # If True disable the training pmap.
    scene_bbox: None | float | BboxType = None
    
    # Dataset configuration
    dataset_loader: str = "llff"  # The type of dataset loader to use.
    load_alphabetical: bool = True  # Load images in COLMAP vs alphabetical ordering (affects heldout test set).
    forward_facing: bool = False  # Set to True for forward-facing LLFF captures.
    llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
    llff_use_all_images_for_training: bool = False  # If true, use all input images for training.
    llff_load_from_poses_bounds: bool = False  # If True, load camera poses of LLFF data from poses_bounds.npy.
    multi_illumination: bool = False  # If True, load and compute normal MAE.
    rotate_illumination: bool = False  # If True, load and compute normal MAE.
    use_ground_truth_illumination: bool = False  # If True, load and compute normal MAE.
    light_rotations: Optional[List[float]] = None
    num_illuminations: int = 1  # If True, load and compute normal MAE.
    factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    test_factor: int = 0  # The downsample factor of images, 0 for no downsampling.
    num_dataset_images: int = -1  # Number of frames in render path.
    
    # Loading specific formats
    load_ngp_format_poses: bool = False  # Use `transforms.json` file for poses.
    arcore_format_pose_file: Optional[str] = None  # Use `metadata.json` for new ARCore poses, `original_metadata.json` for old.
    colmap_subdir: Optional[str] = None  # Where to find COLMAP pose data.
    image_subdir: Optional[str] = None  # Where to find image data.
    load_colmap_points: bool = False
    use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
    use_exrs: bool = False  # If True, use EXR files.
    vocab_tree_path: Optional[str] = None  # Path to vocab tree for COLMAP.
    meshfile: str | None = None
    # Function for transforming loaded poses in non-forward-facing scenes.
    transform_poses_fn: Optional[Callable[Ellipsis, Any]] = None
    use_mesh_face_normals: bool = False
    
    # Multiscale configuration
    multiscale_train_factors: Optional[List[int]] = None  # Integer downsampling factors to use for multiscale training. Note 1 is included by default! Use [2, 4, 8] for mip-NeRF 2021 convention.
    disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
    
    # Mask configuration
    use_masks: MaskInput = MaskInput.NONE  # If not `NONE`, loads image masks from 'masks' directory.
    mask_threshold: float = 0.0
    
    # DTU dataset specific
    dtu_light_cond: int = 3  # Which DTU dataset lighting condition to load.
    
    # Batching and sampling configuration
    batching: str = "all_images"  # Batch composition, [single_image, all_images].
    batch_size: int = 16384  # The number of rays/pixels in each batch.
    patch_size: int = 1  # Resolution of patches sampled for training batches.
    randomized: bool = True  # Use randomized stratified sampling.
    cast_rays_in_train_step: bool = False  # If True, compute rays in train step.
    cast_rays_in_eval_step: bool = False  # If True, compute rays in eval step.
    jitter_rays: int = 0  # If True, compute rays in train step.
    base_batch_size: Optional[int] = 65536
    
    #-----------------------------------------------------------------------------
    # Camera and Scene Configuration
    #-----------------------------------------------------------------------------
    near: float = 2.0  # Near plane distance.
    far: float = 6.0  # Far plane distance.
    secondary_far: float = 1e6  # Far plane distance.
    # Near and far plane distance in meters. If not None, calibration images are
    # used for conversion to scene units.
    near_plane_meters: Optional[float] = None
    far_plane_meters: Optional[float] = None
    render_path: bool = False  # If True, render a path. Used only by LLFF.
    y_up: bool = False  # The padding used for Charbonnier loss.
    flip_secondary: bool = False  # The padding used for Charbonnier loss.
    secondary_rays_no_vis_transform: bool = False  # The padding used for Charbonnier loss.
    fixed_light: bool = False
    fixed_camera: bool = False
    dataset_scale: Optional[float] = 20.0
    
    #-----------------------------------------------------------------------------
    # Training Configuration
    #-----------------------------------------------------------------------------
    max_steps: int = 250000  # The number of optimization steps.
    early_exit_steps: Optional[int] = None  # Early stopping, for debugging.
    train_length_mult: int = 1  # The initial learning rate.
    
    # Learning rate configuration
    lr_init: float = 0.002  # The initial learning rate.
    lr_final: float = 0.00002  # The final learning rate.
    lr_delay_steps: int = 512  # The number of "warmup" learning steps.
    lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
    lr_factor: float = 1.0  # The initial learning rate.
    lr_factor_mult: float = 1.0  # The initial learning rate.
    
    # Adam optimizer configuration
    adam_beta1: float = 0.9  # Adam's beta2 hyperparameter.
    adam_beta2: float = 0.999  # Adam's beta2 hyperparameter.
    adam_eps: float = 1e-6  # Adam's epsilon hyperparameter.
    extra_opt_params: FrozenDict[str, Any] = FrozenDict({})
    
    # Gradient handling
    grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
    grad_max_val: float = 0.0  # Gradient clipping value, disabled if == 0.
    use_grad_accum: bool = False  # Use gradient accumulation
    grad_accum_steps: int = 1  # Gradient clipping value, disabled if == 0.
    use_secondary_grad_accum: bool = False  # Use gradient accumulation
    secondary_grad_accum_steps: int = 1  # Gradient clipping value, disabled if == 0.
    unbiased_resample_cache: bool = False  # Use gradient accumulation
    
    # Parameter regularization
    param_regularizers: FrozenDict[str, Any] = FrozenDict({})
    # An example of total L2 loss (weight decay) on the NeRF MLP and average
    # Geman-McClure loss on the first layer of the proposal MLP:
    #   param_regularizers = {
    #       'NerfMLP_0': (0.00001, @jnp.sum, 2, 1),
    #       'PropMLP_0/Dense_0': (0.01, @jnp.mean, -2, 1),
    #   }
    
    # Checkpoint and visualization
    visualize_every: int = 25000  # How many steps between model visualizations.
    checkpoint_every: int = 25000  # The number of steps between checkpoint saves.
    checkpoint_keep: int = 2  # Keep the last N checkpoints saved to disk.
    print_every: int = 100  # The number of steps between reports to tensorboard.
    train_render_every: int = 5000  # Steps between test set renders when training
    gc_every: int = 1000  # The number of steps between garbage collections.
    
    # Loss scaling
    enable_loss_scaler: bool = False
    loss_scale: float = 1000.0
    
    # Grid optimizer
    enable_grid_c2f: bool = False  # If True, use coarse-to-fine which will be applied to the grid optimizer as an additional scale to the learning rate.
    # The grid size containing the whole -2 to 2 volume, including the contracted area.
    # coarse to fine weights.
    grid_c2f_resolution_schedule_def: CallDef = (
        optax.linear_schedule,
        {
            "init_value": 1024,
            "end_value": 8192,
            "transition_steps": 2500,
            "transition_begin": 0,
        },
    )
    grid_c2f_weight_method: str = "cosine_sequential"
    
    # Gradient scaling
    use_gradient_scaling: bool = False  # If True, use gradient-scaling.github.io
    gradient_scaling_sigma: float = 1.0  # The gradient-scaling scale factor.
    use_gradient_debias: bool = True  # Multiplier on the distortion loss.
    
    #-----------------------------------------------------------------------------
    # Loss Function Configuration
    #-----------------------------------------------------------------------------
    # Data loss
    data_loss_type: str = "charb"  # What kind of loss to use ('mse' or 'charb').
    charb_padding: float = 0.001  # The padding used for Charbonnier loss.
    data_loss_mult: float = 1.0  # Mult for the finest data term in the loss.
    data_loss_gauss_mult: float = 0.0  # Mult for the finest data term in the loss.
    data_coarse_loss_mult: float = 0.0  # Multiplier for the coarser data terms.
    
    # Distortion loss
    distortion_loss_mult: float = 0.01  # Multiplier on the distortion loss.
    distortion_loss_target: str = "sdist"  # The distance that distortion uses.
    # The curve applied to distortion_loss_target before computing distortion of
    # the form (fn, **kwargs), like (@math.power_ladder, {'p':-2, 'premult':10}).
    distortion_loss_curve_fn: Optional[Tuple[Callable[Ellipsis, Any], Dict[str, Any]]] = None
    normalize_distortion_loss: bool = False  # Makes distortion scale invariant.
    
    # Interlevel loss
    # Multiplier(s) for the interlevel loss that supervises the proposal MLP(s).
    # Setting value to 0 indicates no semantic head in proposal MLP(s).
    interlevel_loss_mults: Union[float, Tuple[float, Ellipsis]] = 1.0
    material_interlevel_loss_mults: Union[float, Tuple[float, Ellipsis]] = 1.0
    use_spline_interlevel_loss: bool = False  # Enable a spline-based loss.
    # How much to blur in the spline-based loss.
    interlevel_loss_blurs: Tuple[float, Ellipsis] = (0.01, 0.001)
    
    # Normal and orientation loss
    predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    material_predicted_normal_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    predicted_normal_reverse_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    predicted_normal_weight_loss_mult: float = 0.0  # Mult. on the predicted normal loss.
    # Mult. on the coarser predicted normal loss.
    predicted_normal_coarse_loss_mult: float = 0.0
    predicted_normal_loss_normalize: bool = False
    predicted_normal_loss_stopgrad: bool = True
    predicted_normal_loss_stopgrad_weight: float = 1.0
    
    orientation_loss_mult: float = 0.0  # Multiplier on the orientation loss.
    orientation_coarse_loss_mult: float = 0.0  # Coarser orientation loss weights.
    # What that loss is imposed on, options are 'normals' or 'normals_pred'.
    orientation_loss_target: str = "normals_pred"
    orientation_loss_normalize: bool = False
    orientation_loss_stopgrad: bool = False
    
    # Eikonal loss
    eikonal_loss_mult: float = 0.0  # Multiplier on the eikonal loss.
    eikonal_coarse_loss_mult: float = 0.0  # Multiplier on the coarser eikonal.
    
    # Semantic loss
    semantic_loss_mult: float = 1.0  # Mult for the loss on the semantic MLP.
    semantic_coarse_loss_mult: float = 0.0  # Mult for the coarser semantic terms.
    
    # Extra losses and configurations
    extra_losses: FrozenDict[str, Any] = FrozenDict({})
    extra_loss_params: FrozenDict[str, Any] = FrozenDict({})
    
    # Weight decay and easing
    use_normal_weight_ease: bool = False
    use_normal_weight_ease_backward: bool = False
    normal_weight_ease_frac: float = 0.0
    normal_weight_ease_start: float = 0.0
    normal_weight_ease_min: float = 0.0
    
    use_normal_weight_decay: bool = False  # Enable normal weight decay
    use_normal_weight_decay_backward: bool = False  # Enable normal weight decay
    normal_weight_decay_start: float = 0.0
    normal_weight_decay_frac: float = 0.1
    normal_weight_decay_min: float = 0.01
    
    use_mask_weight_decay: bool = False
    mask_weight_decay_frac: float = 0.0
    mask_weight_decay_start: float = 0.0
    mask_weight_decay_min: float = 0.0
    
    use_mask_weight_ease: bool = False
    mask_weight_ease_frac: float = 0.0
    mask_weight_ease_start: float = 0.0
    mask_weight_ease_min: float = 0.0
    
    use_geometry_weight_decay: bool = False
    geometry_weight_decay_frac: float = 0.0
    geometry_weight_decay_start: float = 0.0
    geometry_weight_decay_min: float = 0.0
    
    use_geometry_weight_ease: bool = False
    geometry_weight_ease_frac: float = 0.0
    geometry_weight_ease_start: float = 0.0
    geometry_weight_ease_min: float = 0.0
    
    use_material_weight_ease: bool = False
    material_weight_ease_frac: float = 0.0
    material_weight_ease_start: float = 0.0
    material_weight_ease_min: float = 0.0
    
    use_extra_ray_weight_ease: bool = False
    extra_ray_weight_ease_frac: float = 0.0
    extra_ray_weight_ease_start: float = 0.0
    extra_ray_weight_ease_min: float = 0.0
    extra_ray_type: str = "train"  # What kind of loss to use ('mse' or 'charb').
    
    use_consistency_weight_ease: bool = False
    consistency_weight_ease_frac: float = 0.0
    consistency_weight_ease_start: float = 0.0
    consistency_weight_ease_min: float = 0.0
    
    use_smoothness_weight_ease: bool = False
    smoothness_weight_ease_frac: float = 0.0
    use_smoothness_weight_decay: bool = False  # Enable smoothness weight decay
    smoothness_weight_decay_start_frac: float = 0.0
    smoothness_weight_decay_rate: float = 0.1
    smoothness_weight_decay_amount: float = 0.01
    
    use_surface_light_field_weight_ease: bool = False
    surface_light_field_importance_sample_weights: bool = False
    surface_light_field_is_secondary: bool = False
    surface_light_field_weight_ease_frac: float = 0.0
    surface_light_field_weight_ease_start: float = 0.0
    surface_light_field_weight_ease_min: float = 0.0
    
    # Loss clipping
    use_loss_clip: bool = False  # Multiplier on the distortion loss.
    loss_thresh: float = 1000000.0  # Multiplier on the distortion loss.
    loss_clip: float = 1000000.0  # Multiplier on the distortion loss.
    loss_clip_max: float = 1000000.0  # Multiplier on the distortion loss.
    loss_clip_min: float = 0.0  # Multiplier on the distortion loss.
    
    # Various specific loss weights
    backward_mask_loss: bool = False  # The padding used for Charbonnier loss.
    opaque_loss_weight: float = 0.0  # The padding used for Charbonnier loss.
    empty_loss_weight: float = 1.0  # The padding used for Charbonnier loss.
    backward_mask_loss_weight: float = 0.0  # The padding used for Charbonnier loss.
    use_color_mask_loss: bool = False  # The padding used for Charbonnier loss.
    color_mask_loss_max: float = 1.0  # The padding used for Charbonnier loss.
    mask_lossmult: bool = False  # The padding used for Charbonnier loss.
    mask_lossmult_weight: float = 0.0  # The padding used for Charbonnier loss.
    
    # Other loss weights
    extra_ray_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    radiometric_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    light_sampling_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    sample_prediction_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    secondary_rgb_loss_mult: float = 1.0  # Multiplier on the distortion loss.
    cache_rgb_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    residual_albedo_loss_mult: float = 1.0  # Multiplier on the distortion loss.
    emission_zero_loss_mult: float = 0.01  # Multiplier on the distortion loss.
    emission_constant_loss_mult: float = 0.01  # Multiplier on the distortion loss.
    beta_loss_weight: float = 0.01  # Multiplier on the distortion loss.
    
    #-----------------------------------------------------------------------------
    # Material and Light Configuration
    #-----------------------------------------------------------------------------
    is_material: bool = False  # Use gradient accumulation
    linear_to_srgb: bool = False  # Convert linear radiance to sRGB
    light_sampling_linear_to_srgb: bool = True  # Multiplier on the distortion loss.
    surface_light_field_linear_to_srgb: bool = True  # Multiplier on the distortion loss.
    surface_light_field_loss_type: str = 'charb'  # Multiplier on the distortion loss.
    
    # Light conditioning
    light_intensity_conditioning: bool = False  # Set to True for forward-facing LLFF captures.
    light_intensity_conditioning_scale: float = 1.0  # Set to True for forward-facing LLFF captures.
    light_intensity_conditioning_bias: float = 0.01  # Set to True for forward-facing LLFF captures.
    light_source_position: Optional[List[float]] = None
    light_transforms: Optional[list] = None
    light_transform_idx: int = 0
    light_name: Optional[str] = None # exposure time per bin
    light_near: Optional[float] = 0
    light_zero: Optional[bool] = True
    light_pos_multiplier: float = 0.1  # Multiplier on the distortion loss.
    light_static_wrt_camera: Optional[bool] = False
    
    compute_relight_metrics: bool = False  # If True, load and compute normal MAE.
    env_map_name: str = 'sunset'  # If True, load and compute normal MAE.
    env_map_distance: float = float("inf")  # If True, load and compute normal MAE.
    
    sl_relight: bool = False  # If True, load and compute normal MAE.
    sl_paths: Optional[List] = None   # If True, load and compute normal MAE.
    sl_invert: bool = False  # If True, load and compute normal MAE.
    sl_hfov: float = 10.0  # If True, load and compute normal MAE.
    sl_vfov: float = 10.0  # If True, load and compute normal MAE.
    sl_mult: float = 5.0  # If True, load and compute normal MAE.
    sl_shift: Tuple[float, Ellipsis] = (0.0, 0.0)  # If True, load and compute normal MAE.
    
    # Surface Light Field
    surface_light_field_stopgrad_weight_forward: float = 0.0  # Multiplier on the distortion loss.
    surface_light_field_stopgrad_weight_backward: float = 1.0  # Multiplier on the distortion loss.
    emission_radius: float = 0.0  # Multiplier on the distortion loss.
    emission_only_radius: float = float("inf")  # Multiplier on the distortion loss.
    far_field_radius: float = float("inf")  # Multiplier on the distortion loss.
    surface_light_field_loss_near: float = 1e-1  # Multiplier on the distortion loss.
    surface_light_field_loss_far: float = float("inf")  # Multiplier on the distortion loss.
    surface_light_field_loss_radius: float = float("inf")  # Multiplier on the distortion loss.
    surface_light_field_loss_roughness: float = 1e-1  # Multiplier on the distortion loss.
    surface_light_field_lossmult_bias: float = 0.1  # Multiplier on the distortion loss.
    surface_light_field_loss_depth_scale: float = 1.0  # Multiplier on the distortion loss.
    surface_light_field_loss_acc: bool = True  # Multiplier on the distortion loss.
    surface_light_field_loss_acc_scale_opaque: float = 1.0  # Multiplier on the distortion loss.
    surface_light_field_loss_acc_scale_empty: float = 1.0  # Multiplier on the distortion loss.
    surface_light_field_loss_bound_scale: float = 0.0  # Multiplier on the distortion loss.
    surface_light_field_loss_bounce: bool = True  # Multiplier on the distortion loss.
    surface_light_field_loss_outward: bool = False  # Multiplier on the distortion loss.
    
    # Material properties
    material_loss_radius: float = float("inf")  # Multiplier on the distortion loss.
    material_acc_threshold: float = 0.01  # Multiplier on the distortion loss.
    material_normals_target: str = 'normals_to_use'  # Numpy render pose file to load.
    material_ray_sampler_interlevel_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    material_ray_sampler_distortion_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    material_ray_sampler_orientation_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    material_ray_sampler_normal_loss_mult: float = 0.0  # Multiplier on the distortion loss.
    material_correlation_weight_albedo: float = 1.0  # Multiplier on the distortion loss.
    material_correlation_weight_other: float = 1.0  # Multiplier on the distortion loss.
    material_smoothness_weight_albedo: float = 1.0  # Multiplier on the distortion loss.
    material_smoothness_weight_other: float = 1.0  # Multiplier on the distortion loss.
    material_smoothness_base: float = 1.0  # Multiplier on the distortion loss.
    material_smoothness_irradiance_multiplier: float = 1.0  # Multiplier on the distortion loss.
    material_smoothness_l1_loss: bool = False  # Numpy render pose file to load.
    material_smoothness_albedo_stopgrad: bool = False  # Numpy render pose file to load.
    material_smoothness_irradiance_weight: bool = False  # Numpy render pose file to load.
    material_smoothness_tensoir_albedo: bool = True  # Numpy render pose file to load.
    material_smoothness_noise: float = 0.01  # Multiplier on the distortion loss.
    
    scale_smoothness_by_irradiance: bool = False  # Numpy render pose file to load.
    irradiance_cache_stopgrad_weight: float = 1.0  # Multiplier on the distortion loss.
    irradiance_cache_stopgrad_weight_backwards: float = 1.0  # Multiplier on the distortion loss.
    irradiance_cache_loss_weight: float = 1.0  # Multiplier on the distortion loss.
    irradiance_cache_color_loss_weight: float = 1.0  # Multiplier on the distortion loss.
    whitening_loss_weight: float = 1.0  # Multiplier on the distortion loss.
    bounded_albedo_loss_weight: float = 0.0  # Multiplier on the distortion loss.
    cache_consistency_loss_type: str = "rawnerf_unbiased"  # What kind of loss to use ('mse' or 'charb').
    cache_consistency_loss_mode: str = "integrator"  # What kind of loss to use ('mse' or 'charb').
    cache_consistency_use_diffuse_specular: bool = False  # What kind of loss to use ('mse' or 'charb').
    cache_consistency_use_total: bool = False  # What kind of loss to use ('mse' or 'charb').
    cache_consistency_fix_shader: bool = False  # What kind of loss to use ('mse' or 'charb').
    cache_consistency_stopgrad_weight_material: float = 0.0
    cache_consistency_stopgrad_weight_cache: float = 1.0
    cache_consistency_loss_weight: float = 1.0
    cache_consistency_direct_weight: float = 1.0
    cache_consistency_indirect_weight: float = 1.0
    cache_integrator_loss_weight: float = 0.0
    cache_consistency_use_integrated: bool = True
    cache_consistency_use_gauss: bool = True
    
    # Sampling configuration
    num_light_samples: int = 8  # Multiplier on the distortion loss.
    num_extra_samples: int = 1  # Multiplier on the distortion loss.
    num_radiometric_samples: int = 1  # Multiplier on the distortion loss.
    num_distance_samples: int = 32  # Multiplier on the distortion loss.
    num_irradiance_samples: int = 8  # Multiplier on the distortion loss.
    num_surface_light_field_samples: int = 1  # Multiplier on the distortion loss.
    
    # RawNeRF specific parameters
    rawnerf_mode: bool = False  # Load raw images and train in raw color space.
    rawnerf_exponent: float = 2
    rawnerf_exponent_material: float = 2
    rawnerf_eps: float = 1e-4
    rawnerf_eps_material: float = 1e-4
    rawnerf_eps_direct_consistency: float = 1e-2
    rawnerf_min: float = 1e-4  # Multiplier on the distortion loss.
    rawnerf_min_material: float = 1e-4  # Multiplier on the distortion loss.
    rawnerf_max: float = 10000.0  # Multiplier on the distortion loss.
    rawnerf_max_material: float = 10000.0  # Multiplier on the distortion loss.
    use_gt_rawnerf: bool = False  # Multiplier on the distortion loss.
    use_combined_rawnerf: bool = True  # Multiplier on the distortion loss.
    use_combined_rawnerf_material: bool = True  # Multiplier on the distortion loss.
    use_norm_rawnerf: bool = False  # Multiplier on the distortion loss.
    
    # Filters and special effects
    round_roughness: bool = False  # If True, load and compute normal MAE.
    filter_retroreflective: bool = False  # Multiplier on the distortion loss.
    filter_retroreflective_thresh: float = 4.0
    filter_normals_thresh: float = 1.01
    filter_direct: bool = False  # If True, load and compute normal MAE.
    filter_indirect: bool = False
    filter_median: bool = False  # If True, load and compute normal MAE.
    filter_median_thresh: float = 35.0  # If True, load and compute normal MAE.
    finetune_multiplier: float = 10.0  # If True, load and compute normal MAE.
    finetune_cache: bool = False  # If True, load and compute normal MAE.
    
    # Volume variating
    volume_variate: bool = False
    volume_variate_material: bool = False
    volume_variate_secondary: bool = False
    volume_variate_passes: Optional[List[str]] = None
    volume_variate_passes_secondary: Optional[List[str]] = None
    
    # Other parameters
    dark_level: float = 0.0  # Multiplier on the distortion loss.
    dark_level_multiplier: float = 0.001  # Multiplier on the distortion loss.
    rgb_max: float = float("inf")  # Rays start at this Z value.
    learnable_light: bool = False  # If True, use refdirs instead of viewdirs.
    no_material_regularizers: bool = False  # Multiplier on the distortion loss.
    
    #-----------------------------------------------------------------------------
    # Geometry Configuration
    #-----------------------------------------------------------------------------
    use_falloff: bool = True  # If True, use refdirs instead of viewdirs.
    use_occlusions: bool = False  # If True, use refdirs instead of viewdirs.
    occlusions_secondary_only: bool = False  # If True, use refdirs instead of viewdirs.
    occlusions_primary_only: bool = False  # If True, use refdirs instead of viewdirs.
    use_surfaces: bool = False  # If True, use refdirs instead of viewdirs.
    stopgrad_with_occlusions: bool = False  # If True, use refdirs instead of viewdirs.
    
    # Occlusion parameters
    occ_threshold_max: float = 1.0  # The RGB activation.
    occ_threshold_min: float = 0.5  # The RGB activation.
    occ_threshold_rate: float = 0.1  # The RGB activation.
    occ_threshold_start_frac: float = 0.1  # The RGB activation.
    
    # Shadow parameters
    shadow_near_max: float = 2e-1  # Far plane distance.
    shadow_near_min: float = 2e-1  # Far plane distance.
    shadow_near_rate: float = 0.1  # Far plane distance.
    shadow_near_start_frac: float = 0.1  # Far plane distance.
    shadow_normal_eps_rate: float = 1e-3  # Far plane distance.
    shadow_normal_eps_dot_min: float = 1e-2  # Far plane distance.
    shadow_sampling_strategy: Any = None
    shadow_normals_target: str = 'normals_to_use'  # Multiplier on the distortion loss.
    secondary_normal_eps: float = 1e-2  # Far plane distance.
    
    # Geometry smoothing
    use_geometry_smoothness: bool = False  # Multiplier on the distortion loss.
    geometry_smoothness_noise: float = 0.01  # Multiplier on the distortion loss.
    geometry_smoothness_weight_normals: float = 1.0  # Multiplier on the distortion loss.
    geometry_smoothness_weight_normals_pred: float = 0.0  # Multiplier on the distortion loss.
    geometry_smoothness_weight_density: float = 0.0  # Multiplier on the distortion loss.
    
    # Geometry cache
    stopgrad_cache_geometry: bool = False  # Multiplier on the distortion loss.
    stopgrad_cache_geometry_weight: float = 0.0  # Multiplier on the distortion loss.
    stopgrad_cache_geometry_feature_weight: float = 0.0  # Multiplier on the distortion loss.
    stopgrad_cache_geometry_normals_weight: float = 0.0  # Multiplier on the distortion loss.
    
    # Weight normalization
    normalize_weight_in_model: bool = False  # Multiplier on the distortion loss.
    normalize_weight_radius: float = 0.5  # Multiplier on the distortion loss.
    normalize_weight_min_thresh: float = 0.01  # Multiplier on the distortion loss.
    normalize_weight_max_thresh: float = 0.01  # Multiplier on the distortion loss.
    normalize_weight_damp_fac: float = 0.0  # Multiplier on the distortion loss.
    normalize_weight_loss: float = 0.0  # Multiplier on the distortion loss.
    
    # Shift invariance
    use_shift_invariance: bool = False  # Multiplier on the distortion loss.
    shift_invariant_start: int = -4  # Multiplier on the distortion loss.
    shift_invariant_end: int = 4  # Multiplier on the distortion loss.
    shift_invariant_step: int = 1  # Multiplier on the distortion loss.
    
    # Difference loss
    use_difference_loss: bool = False  # Multiplier on the distortion loss.
    difference_dist_scale: float = 1.0  # Multiplier on the distortion loss.
    difference_dist_min: float = 1e-3  # Multiplier on the distortion loss.
    difference_pix_dist_scale: float = 10.0  # Multiplier on the distortion loss.
    difference_loss_scale: float = 1.0  # Multiplier on the distortion loss.
    
    # Extra ray configuration
    extra_ray_regularizer: bool = False  # Multiplier on the distortion loss.
    extra_ray_light_shuffle: bool = False  # Multiplier on the distortion loss.
    extra_ray_cams_as_lights: bool = False  # Multiplier on the distortion loss.
    extra_ray_loss_stopgrad_weight_gt: float = 0.0  # Multiplier on the distortion loss.
    extra_ray_loss_stopgrad_weight_pred: float = 1.0  # Multiplier on the distortion loss.
    
    #-----------------------------------------------------------------------------
    # Transient Configuration
    #-----------------------------------------------------------------------------
    transient_start_idx: int = 0  # If True, load and compute normal MAE.
    transient_end_idx: int = 1000  # If True, load and compute normal MAE.
    transient_shift: Optional[float] = 0.0
    transient_shift_multiplier: float = 0.01  # Multiplier on the distortion loss.
    transient_gauss_sigma_scales: Optional[List] = None # exposure time per bin
    transient_gauss_constant_scale: float = 1.0 # exposure time per bin
    
    # Time-of-flight configurations
    n_bins: Optional[int] = 700
    start_bin: Optional[int] = 0
    test_start_bin: Optional[int] = 0
    n_impulse_response_bins: Optional[int] = 101
    impulse_response_start_bin: Optional[int] = 0
    impulse_response: Optional[str] = None
    
    # Exposure time configuration
    exposure_time: Optional[float] = 0.1 # exposure time per bin
    bin_zero_threshold_light: Optional[int] = 1000000 # exposure time per bin
    no_shift_direct: bool = False  # If True, load and compute normal MAE.
    
    # Indirect time filtering
    tfilter_sigma: Optional[float] = 3.0
    
    # Visualization flags for transient render
    calib_checkpoint: Optional[str] = None
    optimize_calib_on_load: bool = True
    
    # Integrating time-of-flight
    use_itof: bool = False
    itof_frequency_phase_shifts: Optional[List] = None # exposure time per bin
    
    #-----------------------------------------------------------------------------
    # Evaluation and Rendering Configuration
    #-----------------------------------------------------------------------------
    # Metric configuration
    compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
    compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
    compute_albedo_metrics: bool = False  # If True, load and compute normal MAE.
    compute_depth_metrics: bool = True
    compute_transient_metrics: bool = False  # If True, load and compute normal MAE.
    evaluate_without_masks: bool = False  # The padding used for Charbonnier loss.
    clip_eval: bool = False  # The padding used for Charbonnier loss.
    correct_eval: bool = False  # The padding used for Charbonnier loss.
    eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
    eval_save_output: bool = True  # If True save predicted images to disk.
    eval_save_ray_data: bool = False  # If True save individual ray traces.
    eval_render_interval: int = 1  # The interval between images saved to disk.
    eval_dataset_limit: int = jnp.iinfo(jnp.int32).max  # Num test images to eval.
    eval_quantize_metrics: bool = True  # If True, run metrics on 8-bit images.
    eval_crop_borders: int = 0  # Ignore c border pixels in eval (x[c:-c, c:-c]).
    
    # Evaluation paths
    eval_train: bool = False  # If True, load and compute normal MAE.
    eval_path: bool = False  # If True, load and compute normal MAE.
    
    # Visualization configuration
    vis_num_rays: int = 16  # The number of rays to visualize.
    # Decimate images for tensorboard (ie, x[::d, ::d]) to conserve memory usage.
    vis_decimate: int = 0
    no_vis: bool = False
    vis_only: bool = False
    vis_render_path: bool = False
    viz_index: Optional[int] = 1000000 # exposure time per bin
    viz_name: Optional[str] = None # exposure time per bin
    
    # Image/video rendering parameters
    render_chunk_size: int = 16384  # Chunk size for whole-image renderings.
    num_showcase_images: int = 5  # The number of test-set images to showcase.
    deterministic_showcase: bool = True  # If True, showcase the same images.
    render_video_fps: int = 60  # Framerate in frames-per-second.
    render_video_crf: int = 18  # Constant rate factor for ffmpeg video quality.
    render_path_frames: int = 120  # Number of frames in render path.
    z_variation: float = 0.0  # How much height variation in render path.
    z_phase: float = 0.0  # Phase offset for height variation in render path.
    rad_mult_min: float = 1.0  # How close to get to the object, relative to 1.
    rad_mult_max: float = 1.0  # How far to get from the object, relative to 1.
    render_rotate_xaxis: float = 0.0  # Rotate camera around x axis.
    render_rotate_yaxis: float = 0.0  # Rotate camera around y axis.
    lock_up: bool = False  # If True, locks the up axis (good for sideways paths).
    render_dist_percentile: float = 0.5  # How much to trim from near/far planes.
    render_dist_curve_fn: Callable[Ellipsis, Any] = jnp.log  # How depth is curved.
    render_path_file: Optional[str] = None  # Numpy render pose file to load.
    render_job_id: int = 0  # Render job id.
    render_num_jobs: int = 1  # Total number of render jobs.
    render_rgb_only: bool = False  # Render spherical 360 panoramas.
    
    # Resolution settings
    width: Optional[int] = None
    height: Optional[int] = None
    test_width: Optional[int] = None
    test_height: Optional[int] = None
    test_ratio: Optional[int] = None
    img_scale: Optional[float] = 1
    var_scale: Optional[float] = 1
    render_resolution: Optional[Tuple[int, int]] = None
    render_focal: Optional[float] = None  # Render focal length.
    render_camtype: Optional[str] = None  # 'perspective', 'fisheye', or 'pano'.
    
    # Special rendering modes
    render_spherical: bool = False  # Render spherical 360 panoramas.
    render_save_async: bool = True  # Save using a separate thread.
    
    # Spline rendering parameters
    # Text file containing names of images to be used as spline keyframes, OR
    # directory containing those images.
    render_spline_keyframes: Optional[str] = None
    # Comma-separated list of possible values for option
    # "render_spline_keyframes". If set, the render pipeline will be executed
    # once per entry, overwriting "render_spline_keyframes" in the process.
    render_spline_keyframes_choices: Optional[str] = None
    render_spline_n_interp: int = 30  # Num. frames to interpolate per keyframe.
    render_spline_degree: int = 5  # Polynomial degree of B-spline interpolation.
    render_spline_lock_up: bool = False  # If True, no up/down tilt in path.
    # B-spline smoothing factor, 0 for exact interpolation of keyframes.
    # Interpolate per-frame exposure value from spline keyframes.
    render_spline_smoothness: float = 0.03
    # Weight associated with rotation dimensions. Higher values means preserving
    # view direction is more important than camera position. Must be >0.
    render_spline_rot_weight: float = 0.1
    render_spline_interpolate_exposure_smoothness: int = 20
    render_spline_interpolate_exposure: bool = False
    render_spline_lookahead_i: Optional[int] = None
    render_spline_fixed_up: bool = False
    render_spline_meters_per_sec: Optional[float] = None
    # If both parameters below are specified, spline keyframes that are far from
    # their neighbors will be ignored.
    render_spline_outlier_keyframe_quantile: Optional[float] = None
    render_spline_outlier_keyframe_multiplier: Optional[float] = None
    # Text file or directory with image pairs for calibrating metric scale.
    render_calibration_keyframes: Optional[str] = None
    render_calibration_distance: float = 3.0  # Default calibration is 3 meters.
    render_spline_const_speed: bool = False  # Retime spline to have const speed.
    render_spline_n_buffer: Optional[int] = None  # Extra keyframes for path.
    
    # Video rendering
    # A tuple of video formats to render. Accepted formats: 'mp4', 'webm', 'gif'.
    render_video_exts: Tuple[str, Ellipsis] = ("mp4",)
    # Whether or not to delete the still images after rendering a video.
    render_delete_images_when_done: bool = True
    # Whether or not videos should be rendered looped (going forwards then the
    # same way backwards)
    render_looped_videos: bool = False
    # Make videos in the main render.py binary
    render_make_videos_in_main_binary: bool = True
    
    # Metric harness configurations
    # During training, disable LPIPS, SSIM, and shift-invariant metrics as they're
    # expensive to evaluate.
    metric_harness_train_config: FrozenDict[str, Any] = FrozenDict(
        {
            "disable_lpips": True,
            "disable_ssim": True,
            "disable_search_invariant": True,
        }
    )
    # During evaluation, let LPIPS and SSIM be turned on by default but still
    # disable shift-invariant metrics as they are expensive and currently unused.
    metric_harness_eval_config: FrozenDict[str, Any] = FrozenDict({"disable_search_invariant": True})
    
    # Parameters for the local color correction used in evaluating the color
    # corrected error metrics. Note that increasing any of these numbers is
    # virtually guaranteed to improve all error metrics, so these parameter should
    # be tuned by visual inspection.
    color_correction_config: FrozenDict[str, Union[int, Tuple[int, int]]] = FrozenDict(
        {
            "num_spatial_bins": [6, 10],
            "num_luma_bins": 9,
            "num_chroma_bins": 3,
        }
    )
    
    #-----------------------------------------------------------------------------
    # Raw Image Processing Configuration
    #-----------------------------------------------------------------------------
    exposure_percentile: float = 97.0  # Image percentile to expose as white.
    # During training, discard N-pixel border around each input image.
    num_border_pixels_to_mask: int = 0
    apply_bayer_mask: bool = False  # During training, apply Bayer mosaic mask.
    autoexpose_renders: bool = False  # During rendering, autoexpose each image.
    # For raw test scenes, use affine raw-space color correction.
    eval_raw_affine_cc: bool = False
    optimize_vignette_on_load: Optional[bool] = False
    
    # Exposure prediction
    exposure_prediction_loss_mult: float = 0.0  # Loss weight for ExposureMLP
    # Weight of the following loss (penalizes for too small or too large
    # exposures):
    # 0.5 * ReLU(min_dataset_exposure - exposure)**2 +
    # 0.5 * ReLU(exposure - max_dataset_exposure)**2
    # Not used when exposure_prediction_loss_mult = 0
    exposure_prediction_bounds_loss_mult: float = 0.1
    
    # World scale for aerial datasets
    world_scale: float = 1.0  # Camera positions are divided by this quantity.
    z_min: Optional[float] = None  # Rays end at this Z value.
    z_max: Optional[float] = None  # Rays start at this Z value.
    
    #-----------------------------------------------------------------------------
    # Semantic Data Configuration
    #-----------------------------------------------------------------------------
    # Flags for performing semantic consensus similar to Semantic-NeRF and
    # Distilled Feature Fields (DFFs). Aside from color and density, the model
    # also predicts an arbitrary size semantic feature for each pixel.
    # If semantic_dir is provided, an additional Semantic MLP will be used to make
    # such prediction. For each input image, we should have a corresponding file
    # under semantic_dir. The file should have the same file name, and dimensions
    # of [H, W, C].
    semantic_dir: Optional[str] = None  # Input directory of semantic data.
    semantic_format: Optional[str] = None  # Semantic format ('npy' or 'image').
    train_exclude_prefixes: Optional[List[str]] = None
    
    # Patch loss-related parameters
    patch_loss_mult: float = 0.0  # Multiplier on patchwise depth loss.
    bilateral_strength: float = 0.0  # Strength of RGB bilateral weights on above.
    # Modulates RGB patch variance weighting of patchwise depth loss.
    patch_variance_weighting: float = 0.0
    
    #-----------------------------------------------------------------------------
    # Miscellaneous
    #-----------------------------------------------------------------------------
    num_rgb_channels: Optional[int] = 3


def define_common_flags():
    # Define the flags used by both train.py and eval.py
    flags.DEFINE_string("mode", None, "Required by GINXM, not used.")
    flags.DEFINE_string("base_folder", None, "Required by GINXM, not used.")
    flags.DEFINE_multi_string("gin_bindings", None, "Gin parameter bindings.")
    flags.DEFINE_multi_string("gin_configs", None, "Gin config files.")
    flags.DEFINE_bool("is_xm_sweep", False, "Whether the run is an xm sweep.")


def load_config(save_config=True):
    """Loads the config, and optionally checkpoints it."""
    gin_bindings = flags.FLAGS.gin_bindings
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True)

    config = Config()

    if save_config and jax.host_id() == 0:
        gfile.makedirs(config.checkpoint_dir)
        with gfile.GFile(config.checkpoint_dir + "/config.gin", "w") as f:
            f.write(gin.config_str())

    return config


def load_config_with_bindings(save_config=True, bindings=[]):
    """Loads the config, and optionally checkpoints it."""
    gin_bindings = flags.FLAGS.gin_bindings
    gin.parse_config_files_and_bindings(flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings + bindings, skip_unknown=True)

    config = Config()

    if save_config and jax.host_id() == 0:
        gfile.makedirs(config.checkpoint_dir)
        with gfile.GFile(config.checkpoint_dir + "/config.gin", "w") as f:
            f.write(gin.config_str())

    return config
