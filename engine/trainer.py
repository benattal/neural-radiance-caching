# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple, Union
import pdb
import chex
import flax
import gin
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from absl import app, logging
from flax.metrics import tensorboard
from flax.training import checkpoints
from jax import random
from rich import box, style
from rich.panel import Panel
from rich.table import Table
import h5py

from internal import camera_utils, configs, datasets, image, models, train_utils, utils, vis, ref_utils, math, image_utils
from internal.inverse_render import render_utils


TIME_PRECISION = 1000


@gin.configurable
@dataclasses.dataclass
class Trainer:
    """Neural rendering model trainer.
    
    This class manages the training, evaluation, and visualization of neural rendering models.
    It handles configuration, model setup, training loops, checkpointing, and evaluation.
    """
    # Core configuration
    stage: str = "cache"
    viewer_only: bool = False
    relight: bool = False
    save_results: bool = True

    # Albedo processing options
    albedo_clip: float = 1.0
    albedo_correct_median: bool = False
    albedo_gamma: bool = True

    # Visualization options
    vis_only: bool = False
    vis_restart: bool = False
    vis_start: int = 0
    vis_end: int = 200
    vis_secondary: bool = False
    vis_extra: bool = False
    vis_surface_light_field: bool = False
    vis_light_sampler: bool = False

    # Training options
    stopgrad: bool = False
    resample: bool = False
    resample_depth: bool = False
    sample_factor: int = 2
    num_resample: int = 1
    resample_render: bool = False
    sample_render_factor: int = 2
    render_repeats: int = 1

    # Stage parameters
    stage_params: ml_collections.FrozenConfigDict = ml_collections.FrozenConfigDict(
        {
            "cache": {
                "render_chunk_size": 4096,
                "use_light_sampler": False,
                "use_material": False,
                "use_surface_light_field": False,
                "optimize_cache": True,
                "optimize_light": False,
                "optimize_surface_light_field": False,
                "reduce_cache_factor": 1.0,
                "reduce_surface_light_field_factor": 1.0,
                "prefixes": None,
                "exclude_prefixes": None,
                "replace_dict": None,
                "extra_losses": {},
            },
        },
    )

    def setup(self):
        """Initialize the training environment and load necessary resources."""
        self._setup_locks()
        self._setup_config_parameters()
        self._setup_binding_configs()
        self._setup_rng()
        self._load_datasets()
        self._setup_cameras_and_lights()
        self._setup_model()
        self._setup_checkpointing()
        self._initialize_visualization()
        self._initialize_metrics()

    def _setup_locks(self):
        """Initialize thread locks for training and rendering."""
        self.train_lock = Lock()
        self.use_transient = gin.query_parameter("Config.use_transient")

        if self.use_transient:
            self.nerf_mlp_name = "TransientNeRFMLP"
            self.nerf_model_name = "TransientNeRFModel"
            self.material_mlp_name = "TransientMaterialMLP"
            self.material_model_name = "TransientMaterialModel"
        else:
            self.nerf_mlp_name = "NeRFMLP"
            self.nerf_model_name = "NeRFModel"
            self.material_mlp_name = "MaterialMLP"
            self.material_model_name = "MaterialModel"

    def _setup_config_parameters(self):
        """Set up configuration parameters from gin config."""
        # Load base configuration
        self.calib_checkpoint = gin.query_parameter("Config.calib_checkpoint")
        self.optimize_calib_on_load = gin.query_parameter("Config.optimize_calib_on_load")
        self.checkpoint_dir = gin.query_parameter("Config.checkpoint_dir")
        
        # Setup gradient accumulation steps
        self.secondary_grad_accum_steps = self.stage_params[self.stage].get(
            "secondary_grad_accum_steps", 
            gin.query_parameter("Config.secondary_grad_accum_steps")
        )
        self.grad_accum_steps = self.stage_params[self.stage].get(
            "grad_accum_steps", 
            gin.query_parameter("Config.grad_accum_steps")
        )
        self.grad_accum_steps *= self.secondary_grad_accum_steps
        
        # Setup model feature flags
        self._setup_feature_flags()
        
        # Setup learning rate parameters
        self._setup_learning_rate_params()
        
        # Setup loss weights and regularization
        self._setup_loss_params()
        
        # Setup sampler parameters
        self._setup_sampling_params()

    def _setup_feature_flags(self):
        """Set up feature flags for the model."""
        # Material and geometry settings
        self.use_material = self.stage_params[self.stage]["use_material"]
        self.use_geometry_smoothness = gin.query_parameter("Config.use_geometry_smoothness")
        self.stopgrad_cache_geometry = gin.query_parameter("Config.stopgrad_cache_geometry") and (self.use_material) and ("from_scratch" not in self.stage)
        self.stopgrad_cache_geometry_weight = gin.query_parameter("Config.stopgrad_cache_geometry_weight")
        self.stopgrad_cache_geometry_feature_weight = gin.query_parameter("Config.stopgrad_cache_geometry_feature_weight")
        self.stopgrad_cache_geometry_normals_weight = gin.query_parameter("Config.stopgrad_cache_geometry_normals_weight")
        
        # Sampler and field settings
        self.use_light_sampler = self.stage_params[self.stage]["use_light_sampler"]
        self.use_surface_light_field = self.stage_params[self.stage]["use_surface_light_field"]
        
        # Optimization settings
        self.optimize_surface_light_field = self.stage_params[self.stage]["optimize_surface_light_field"]
        self.optimize_geometry = self.stage_params[self.stage].get("optimize_geometry", True)
        self.optimize_cache = self.stage_params[self.stage]["optimize_cache"]
        self.optimize_light = self.stage_params[self.stage]["optimize_light"]

        # Occlusions
        self.use_occlusions = gin.query_parameter("Config.use_occlusions")
        self.occlusions_secondary_only = gin.query_parameter("Config.occlusions_secondary_only")
        self.occlusions_primary_only = gin.query_parameter("Config.occlusions_primary_only")
        self.light_near = gin.query_parameter("Config.light_near")

        if self.vis_only:
            self.use_occlusions = True
            self.occlusions_secondary_only = False
            self.occlusions_primary_only = False
            self.light_near = gin.query_parameter("Config.near")
        
        if "finetune" in self.stage:
            self.use_occlusions = True
            self.occlusions_secondary_only = False
            self.occlusions_primary_only = False

    def _setup_learning_rate_params(self):
        """Set up learning rate parameters."""
        # Base parameters
        self.factor = gin.query_parameter("Config.factor")
        self.base_batch_size = gin.query_parameter("Config.base_batch_size")
        self.batch_size = gin.query_parameter("Config.batch_size")
        self.render_chunk_size = gin.query_parameter("Config.render_chunk_size")
        self.train_length_mult = gin.query_parameter("Config.train_length_mult")
        
        # Calculate scaling factors
        self.scale_factor = self.base_batch_size // (
            (self.batch_size * self.grad_accum_steps) // self.secondary_grad_accum_steps
        )
        self.total_batch_size = self.batch_size * self.grad_accum_steps

        # Learning rate parameters
        self.lr_factor = gin.query_parameter("Config.lr_factor") * gin.query_parameter("Config.lr_factor_mult")
        self.max_steps = gin.query_parameter("Config.max_steps")
        self.lr_init = gin.query_parameter("Config.lr_init")
        self.lr_final = gin.query_parameter("Config.lr_final")
        self.lr_delay_steps = gin.query_parameter("Config.lr_delay_steps")
        self.extra_opt_params = gin.query_parameter("Config.extra_opt_params")
        
        # Scale learning rates based on batch size and other factors
        self.max_steps = (self.max_steps * self.scale_factor) // self.train_length_mult
        self.lr_delay_steps = (self.lr_delay_steps * self.scale_factor) // self.train_length_mult
        self.lr_init = (self.lr_init / self.scale_factor) * self.lr_factor
        self.lr_final = (self.lr_final / self.scale_factor) * self.lr_factor

    def _setup_loss_params(self):
        """Set up loss weights and regularization parameters."""
        # Cache consistency and finetuning
        self.cache_consistency_loss_weight = gin.query_parameter("Config.cache_consistency_loss_weight")
        self.cache_consistency_loss_type = gin.query_parameter("Config.cache_consistency_loss_type")
        self.finetune_multiplier = self.stage_params[self.stage].get("finetune_multiplier", gin.query_parameter("Config.finetune_multiplier"))
        self.finetune_cache = False
        
        # Field reduction factors
        self.reduce_cache_factor = self.stage_params[self.stage]["reduce_cache_factor"]
        self.reduce_surface_light_field_factor = self.stage_params[self.stage]["reduce_surface_light_field_factor"]
        self.anneal_slope = self.stage_params[self.stage].get("anneal_slope", gin.query_parameter("ProposalVolumeSampler.anneal_slope"))
        
        # Material loss weights
        self.material_interlevel_loss_mults = gin.query_parameter("Config.material_interlevel_loss_mults")
        self.material_predicted_normal_loss_mult = gin.query_parameter("Config.material_predicted_normal_loss_mult")
        
        # Sampling parameters
        self.secondary_near = gin.query_parameter(f"{self.material_mlp_name}.near_min")
        self.secondary_far = gin.query_parameter("Config.secondary_far")
        self.material_use_active = gin.query_parameter(f"{self.material_mlp_name}.use_active")
        self.cache_use_active = gin.query_parameter(f"{self.nerf_mlp_name}.use_active")
        
        # Prefixes and extra losses
        self.prefixes = self.stage_params[self.stage]["prefixes"]
        self.exclude_prefixes = []
        self.replace_dict = {"": ""}
        self.extra_losses = dict(**(self.stage_params[self.stage]["extra_losses"]))
        self.param_regularizers = self.stage_params[self.stage].get("param_regularizers", gin.query_parameter("Config.param_regularizers"))
        self.no_material_regularizers = gin.query_parameter("Config.no_material_regularizers")

    def _setup_sampling_params(self):
        """Set up sampling parameters."""
        # Get secondary sample counts from config
        self.num_secondary_samples = self.stage_params[self.stage].get(
            "num_secondary_samples", 
            gin.query_parameter(f"{self.material_mlp_name}.num_secondary_samples")
        )
        self.num_secondary_samples_render = self.stage_params[self.stage].get(
            "num_secondary_samples_render", 
            self.num_secondary_samples
        )
        self.num_secondary_samples_diff = self.stage_params[self.stage].get(
            "num_secondary_samples_diff", 
            gin.query_parameter(f"{self.material_mlp_name}.num_secondary_samples_diff")
        )
        self.num_secondary_samples_diff_render = self.stage_params[self.stage].get(
            "num_secondary_samples_diff_render", 
            self.num_secondary_samples_diff
        )
        
        # Surface light field samples
        self.num_surface_light_field_samples = self.stage_params[self.stage].get("num_surface_light_field_samples", None)
        self.slf_variate = (
            self.stage_params[self.stage].get("slf_variate", gin.query_parameter(f"{self.material_model_name}.slf_variate"))
            and self.use_surface_light_field
        )
        self.surface_light_field_loss_far = self.stage_params[self.stage].get("surface_light_field_loss_far", None)
        self.surface_light_field_loss_radius = self.stage_params[self.stage].get("surface_light_field_loss_radius", None)
        
        # Apply sampling factors
        self.num_secondary_samples_render *= self.sample_render_factor
        self.num_secondary_samples *= self.sample_factor

        self.num_secondary_samples_diff_render *= self.sample_render_factor
        self.num_secondary_samples_diff *= self.sample_factor
        
    def _process_extra_losses(self):
        """Process and configure extra losses based on model configuration."""
        # Handle SLF variate
        if self.slf_variate:
            self.extra_losses["material_surface_light_field"] = {
                "main": {"mult": 1.0, "start_frac": 0.0}
            }
            if "surface_light_field" in self.extra_losses:
                del self.extra_losses["surface_light_field"]

        # Add geometry smoothness losses if needed
        if self.use_geometry_smoothness:
            if not self.use_material:
                self.extra_losses["geometry_smoothness"] = {
                    "main": {"mult": 1.0, "start_frac": 0.0},
                }
            elif self.use_material and "from_scratch" in self.stage:
                self.extra_losses["geometry_smoothness"] = {
                    "cache_main": {"mult": 1.0, "start_frac": 0.0},
                }

        # Add consistency losses if using material model
        if self.use_material:
            if self.use_transient:
                self.extra_losses["direct_indirect_consistency"] = {
                    "main": {"mult": self.cache_consistency_loss_weight, "start_frac": 0.0},
                }
            else:
                self.extra_losses["direct_indirect_consistency"] = {
                    "main": {"mult": self.cache_consistency_loss_weight, "start_frac": 0.0},
                }

    def _process_opt_params(self):
        """Process optimization parameters for different model components."""
        # Mapping function to update training parameters
        def replace_training_params(path, x, *rest):
            path = tuple(k.key for k in path)
            if "lr_init" in path:
                return (x / self.scale_factor) * self.lr_factor
            elif "lr_init_material" in path:
                return (x / self.scale_factor) * self.lr_factor
            elif "lr_final" in path:
                return (x / self.scale_factor) * self.lr_factor
            elif "lr_final_material" in path:
                return (x / self.scale_factor) * self.lr_factor
            elif "lr_delay_steps" in path:
                return (x * self.scale_factor) // self.train_length_mult
            elif "lr_delay_steps_material" in path:
                return (x * self.scale_factor) // self.train_length_mult
            else:
                return x

        # Apply transformations to optimization parameters
        self.extra_opt_params = jax.tree_util.tree_map_with_path(
            replace_training_params,
            self.extra_opt_params
        )

        # Update material optimization parameters if needed
        if self.use_material and ("from_scratch" not in self.stage):
            for key in self.extra_opt_params.keys():
                self.extra_opt_params[key]["lr_delay_steps"] = self.extra_opt_params[key]["lr_delay_steps_material"]
                self.extra_opt_params[key]["lr_init"] = self.extra_opt_params[key]["lr_init_material"]
                self.extra_opt_params[key]["lr_final"] = self.extra_opt_params[key]["lr_final_material"]

        # Disable vignette optimization if needed
        if (self.calib_checkpoint is not None and self.calib_checkpoint != "") and not self.optimize_calib_on_load:
            self.extra_opt_params["VignetteMap"] = {
                "lr_delay_steps": 0,
                "lr_final": 0.0,
                "lr_init": 0.0,
            }
        
        # Disable geometry optimization if needed
        if not self.optimize_geometry:
            self._disable_optimization_for_keys(["Sampler", "MLP_1", "MLP_2", "density_grid"])
        
        # Configure finetuning parameters
        if "finetune" in self.stage:
            self._configure_finetune()
        
        # Disable cache optimization if needed
        if not self.optimize_cache:
            self._disable_optimization_for_keys(["Cache", "SurfaceLightField", "PersonLightField"])

        # Disable light optimization if needed
        if not self.optimize_light:
            self._disable_optimization_for_keys(["LightSampler"])

        # Disable surface light field optimization if needed
        if not self.optimize_surface_light_field:
            self._disable_optimization_for_keys(["SurfaceLightFieldMem"])

    def _disable_optimization_for_keys(self, keys):
        """Disable optimization for the given parameter keys."""
        for key in keys:
            self.extra_opt_params[key] = {
                "lr_delay_steps": 0,
                "lr_final": 0.0,
                "lr_init": 0.0,
            }

    def _configure_finetune(self):
        """Configure parameters for finetuning."""
        self.param_regularizers = None
        self.finetune_cache = True
        self.cache_consistency_loss_weight *= self.finetune_multiplier

        # Disable optimization for various components during finetuning
        keys_to_disable = [
            "Sampler", "MLP_1", "MLP_2", "density_grid", 
            "MaterialShader", "VignetteMap", "LightSource"
        ]
        self._disable_optimization_for_keys(keys_to_disable)

    def _setup_binding_configs(self):
        """Set up configuration bindings for gin."""
        # Process extra losses and optimization parameters
        self._process_extra_losses()
        self._process_opt_params()
        
        # Generate base bindings
        self.bindings = [
            f"Config.max_steps = {self.max_steps}",
            f"Config.batch_size = {self.batch_size}",
            f"Config.grad_accum_steps = {self.grad_accum_steps}",
            f"Config.lr_init = {self.lr_init}",
            f"Config.lr_final = {self.lr_final}",
            f"Config.lr_delay_steps = {self.lr_delay_steps}",
            f"Config.extra_opt_params = {self.extra_opt_params}",
            f"Config.extra_losses = {self.extra_losses}",
            f"Config.finetune_cache = {self.finetune_cache}",
            f"Config.cache_consistency_loss_type = '{self.cache_consistency_loss_type}'",
            f"Config.use_occlusions = {self.use_occlusions}",
            f"Config.occlusions_secondary_only = {self.occlusions_secondary_only}",
            f"Config.occlusions_primary_only = {self.occlusions_primary_only}",
            f"Config.light_near = {self.light_near}",
            f"{self.material_model_name}.use_material = {self.use_material}",
            f"{self.material_model_name}.use_light_sampler = {self.use_light_sampler}",
            f"{self.material_model_name}.use_surface_light_field = {self.use_surface_light_field}",
            f"ProposalVolumeSampler.anneal_slope = {self.anneal_slope}",
        ]
        
        # Add visualization-specific bindings
        if self.vis_only:
            self._add_visualization_bindings()
            
        # Add material-specific bindings
        if self.use_material and "from_scratch" not in self.stage:
            self._add_material_bindings()
            
        # Add stopgrad bindings if needed
        if self.stopgrad_cache_geometry:
            self._add_stopgrad_bindings()
            
        # Add stopgrad bindings for rays and samples if needed
        if self.stopgrad:
            self._add_rays_stopgrad_bindings()
            
        # Add resampling bindings if needed
        if self.resample_render:
            self.bindings.append(f"{self.material_model_name}.resample_render = {self.resample_render}")
            
        if self.resample:
            self._add_resample_bindings()
            
        # Add render chunk size binding if specified
        if self.render_chunk_size is not None:
            self.bindings.append(f"Config.render_chunk_size = {self.render_chunk_size}")
            
        # Add surface light field loss bindings if needed
        if self.surface_light_field_loss_far is not None:
            self._add_surface_light_field_loss_bindings()
            
        # Add SLF variation binding if needed
        if self.slf_variate is not None:
            self.bindings.append(f"{self.material_model_name}.slf_variate = {self.slf_variate}")
            
        # Add secondary samples bindings if needed
        if self.num_secondary_samples is not None:
            self._add_secondary_samples_bindings()
            
        # Add surface light field samples binding if needed
        if self.num_surface_light_field_samples is not None:
            self.bindings.append(f"Config.num_surface_light_field_samples = {self.num_surface_light_field_samples}")
            
        # Add parameter regularizers binding if needed
        if self.param_regularizers is not None:
            self.bindings.append(f"Config.param_regularizers = {self.param_regularizers}")
            
        # Disable normals offset if not optimizing cache
        if not self.optimize_cache:
            self.bindings.append(f"{self.material_mlp_name}.enable_normals_offset = False")
            
        # Load configuration with bindings
        gin.clear_config()
        self.config = configs.load_config_with_bindings(bindings=self.bindings)
        logging.info("Configuration loaded:\n%s", gin.config_str())

    def _add_visualization_bindings(self):
        """Add visualization-specific bindings."""
        metric_harness_config = {
            "disable_lpips": False,
            "disable_ssim": False,
        }
        self.bindings.append(f"Config.metric_harness_train_config = {metric_harness_config}")
        
        # Set test factor based on visualization settings
        if self.vis_only or not self.use_material:
            self.bindings.append(f"Config.test_factor = {self.factor}")
        else:
            self.bindings.append(f"Config.test_factor = {self.factor * 2}")

    def _add_material_bindings(self):
        """Add material-specific bindings."""
        material_bindings = [
            f"Config.occ_threshold_start_frac = 0.0",
            f"Config.occ_threshold_rate = 0.0",
            f"Config.shadow_near_start_frac = 0.0",
            f"Config.shadow_near_rate = 0.0",
            f"{self.material_mlp_name}.near_start_frac = 0.0",
            f"{self.material_mlp_name}.near_rate = 0.0",
            f"Config.use_normal_weight_ease = False",
            f"Config.use_normal_weight_ease_backward = False",
            f"Config.use_material_weight_ease = False",
            f"Config.use_consistency_weight_ease = False",
            f"Config.use_surface_light_field_weight_ease = False",
            f"Config.interlevel_loss_mults = {self.material_interlevel_loss_mults}",
            f"Config.predicted_normal_loss_mult = {gin.query_parameter('Config.predicted_normal_loss_mult') * self.material_predicted_normal_loss_mult}",
            f"Config.predicted_normal_reverse_loss_mult = {gin.query_parameter('Config.predicted_normal_reverse_loss_mult') * self.material_predicted_normal_loss_mult}",
        ]
        self.bindings.extend(material_bindings)

    def _add_stopgrad_bindings(self):
        """Add stopgrad-specific bindings."""
        stopgrad_bindings = [
            f"{self.nerf_model_name}.stopgrad_geometry_weight = {self.stopgrad_cache_geometry_weight}",
            f"{self.nerf_model_name}.stopgrad_geometry_feature_weight = {self.stopgrad_cache_geometry_feature_weight}",
            f"{self.nerf_model_name}.stopgrad_geometry_normals_weight = {self.stopgrad_cache_geometry_normals_weight}",
        ]
        self.bindings.extend(stopgrad_bindings)

    def _add_rays_stopgrad_bindings(self):
        """Add ray stopgrad bindings."""
        stopgrad_ray_bindings = [
            f"{self.material_mlp_name}.stopgrad_rays = True",
            f"{self.material_mlp_name}.stopgrad_samples = True",
            f"Config.cache_consistency_stopgrad_weight_cache = 0.0",
        ]
        self.bindings.extend(stopgrad_ray_bindings)

    def _add_resample_bindings(self):
        """Add resampling bindings."""
        resample_bindings = [
            f"{self.material_model_name}.resample = {self.resample}",
            f"{self.material_model_name}.num_resample = {self.num_resample}",
            f"{self.material_model_name}.use_resample_depth = {self.resample_depth}",
        ]
        self.bindings.extend(resample_bindings)

    def _add_surface_light_field_loss_bindings(self):
        """Add surface light field loss bindings."""
        slf_loss_bindings = [
            f"Config.surface_light_field_loss_far = {self.surface_light_field_loss_far}",
            f"Config.surface_light_field_loss_radius = {self.surface_light_field_loss_radius}",
        ]
        self.bindings.extend(slf_loss_bindings)

    def _add_secondary_samples_bindings(self):
        """Add secondary samples bindings."""
        secondary_samples_bindings = [
            f"{self.material_mlp_name}.num_secondary_samples = {self.num_secondary_samples}",
            f"{self.material_mlp_name}.render_num_secondary_samples = {self.num_secondary_samples_render}",
        ]
        self.bindings.extend(secondary_samples_bindings)
        
        if self.num_secondary_samples_diff is not None:
            diff_bindings = [
                f"{self.material_mlp_name}.num_secondary_samples_diff = {self.num_secondary_samples_diff}",
                f"{self.material_mlp_name}.render_num_secondary_samples_diff = {self.num_secondary_samples_diff_render}",
            ]
            self.bindings.extend(diff_bindings)

    def _setup_rng(self):
        """Set up random number generators."""
        self.rng = random.PRNGKey(self.config.jax_rng_seed)
        self.render_rng = random.PRNGKey(self.config.jax_rng_seed)
        np.random.seed(self.config.np_rng_seed + jax.host_id())
        
        # Handle JIT and multi-GPU settings
        if self.config.disable_pmap_and_jit:
            chex.fake_pmap_and_jit().start()
            
        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

    def _load_datasets(self):
        """Load training and test datasets."""
        # Adjust light source position if y-up
        if self.config.y_up:
            self.config.light_source_position = [
                self.config.light_source_position[1],
                self.config.light_source_position[0],
                self.config.light_source_position[2],
            ]
        
        # Load datasets
        self.dataset = datasets.load_dataset("train", self.config.data_dir, self.config)
        self.test_dataset = datasets.load_dataset("test", self.config.data_dir, self.config)
        self.test_raybatcher = datasets.RayBatcher(self.test_dataset)
        
        # Set up postprocessing function
        if self.config.clip_eval:
            self.postprocess_fn = lambda x: np.clip(
                image.linear_to_srgb(x * self.test_dataset.exposure), 0.0, 1.0
            )
        else:
            self.postprocess_fn = self._create_postprocess_fn()

    def _create_postprocess_fn(self):
        """Create a postprocessing function for rendering output."""
        def p_fn(x):
            if len(x.shape) == 4:
                x = x.sum(-2) 
                x = np.clip(x / self.config.img_scale, 0, 1)

            if x.shape[-1] == 1:
                shape_size = len(x.shape)
                x = np.tile(x, (shape_size-1)*(1, ) + (3,))

            return image.linear_to_srgb(x * self.test_dataset.exposure)
        
        return p_fn

    def _setup_cameras_and_lights(self):
        """Set up cameras and lights for training and testing."""
        # Convert numpy arrays to JAX arrays
        np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
        
        # Set up training cameras
        self.cameras = self.dataset.get_train_cameras(self.config)
        self.cameras = jax.tree_util.tree_map(np_to_jax, self.cameras)
        self.cameras_replicated = flax.jax_utils.replicate(self.cameras)
        
        # Set up virtual cameras for training
        if "finetune" in self.stage:
            self.virtual_cameras = self.dataset.get_train_virtual_cameras(self.config)
            self.virtual_cameras = jax.tree_util.tree_map(np_to_jax, self.virtual_cameras)
            self.virtual_cameras_replicated = flax.jax_utils.replicate(self.virtual_cameras)
        else:
            self.virtual_cameras = self.cameras
            self.virtual_cameras_replicated = self.cameras_replicated
        
        # Set up lights for training
        self.lights = self.dataset.lights
        self.lights = jax.tree_util.tree_map(np_to_jax, self.lights)
        self.lights_replicated = flax.jax_utils.replicate(self.lights)
        
        # Set up test cameras and lights
        self.test_cameras = self.test_dataset.cameras
        self.test_cameras = jax.tree_util.tree_map(np_to_jax, self.test_cameras)
        self.test_cameras_replicated = flax.jax_utils.replicate(self.test_cameras)
        
        self.test_lights = self.test_dataset.lights
        self.test_lights = jax.tree_util.tree_map(np_to_jax, self.test_lights)
        self.test_lights_replicated = flax.jax_utils.replicate(self.test_lights)

    def _setup_model(self):
        """Set up the model and training state."""
        # Create a random key for initialization
        key, self.rng = random.split(self.rng)
        
        # Set up the model, state, and functions
        (
            self.model, 
            self.state, 
            self.render_eval_pfn, 
            self.train_pstep, 
            self.lr_fn
        ) = train_utils.setup_model(
            self.config, key, dataset=self.dataset
        )
        
        # Define shape function for parameter summaries
        def shape_fn(x):
            return x.shape if isinstance(x, jnp.ndarray) else train_utils.tree_len(x)
        
        # Summarize model parameters
        self.param_summary = train_utils.summarize_tree(shape_fn, self.state.params["params"])
        num_chars = max([len(x) for x in self.param_summary])
        logging.info("Optimization parameter sizes/counts:")
        
        for k, v in self.param_summary.items():
            logging.info("%s %s", k.ljust(num_chars), str(v))
        
        # Set up metric harness for evaluation
        self.metric_harness = image.MetricHarness(**self.config.metric_harness_train_config)
        
        # Update exclude prefixes for partial checkpointing
        if "finetune" not in self.stage and self.use_material and self.config.partial_checkpoint_dir is not None and not self.vis_only:
            self.exclude_prefixes += ["params/MaterialShader"]
        
        if "finetune" in self.stage and self.config.sl_relight and self.config.partial_checkpoint_dir is not None and not self.vis_only:
            self.exclude_prefixes += ["params/LightSampler"]

    def _setup_checkpointing(self):
        """Set up checkpointing and restore from checkpoint if needed."""
        # Create save directory
        self.save_dir = os.path.join(self.config.checkpoint_dir, "save")
        
        # Restore from partial checkpoint if specified
        if (self.config.partial_checkpoint_dir is not None):
            self.state = train_utils.restore_partial_checkpoint(
                self.config,
                self.state,
                prefixes=self.prefixes,
                exclude_prefixes=self.exclude_prefixes,
                replace_dict=self.replace_dict,
            )
        else:
            # Otherwise restore from regular checkpoint
            self.state = checkpoints.restore_checkpoint(self.config.checkpoint_dir, self.state)
        
        # Restore calibration if no checkpoint exists
        if self.config.calib_checkpoint is not None and self.config.calib_checkpoint != "":
            self.state = train_utils.restore_partial_checkpoint(
                self.config,
                self.state,
                prefixes=self.prefixes,
                exclude_prefixes=self.exclude_prefixes,
                replace_dict={
                    "params/VignetteMap": "params/VignetteMap",
                },
                checkpoint_dir=self.config.calib_checkpoint,
            )
        
        # Create checkpoint directory if it doesn't exist
        if (not utils.isdir(self.config.checkpoint_dir)):
            utils.makedirs(self.config.checkpoint_dir)

    def _initialize_visualization(self):
        """Initialize visualization parameters."""
        # Set up dimensions for visualization
        self.H = self.test_dataset.height // self.config.vis_decimate
        self.W = self.test_dataset.width // self.config.vis_decimate
        
        # Set up selection coordinates
        self.select_x_start = int(np.round(self.W * 0.3)) // self.config.vis_decimate
        self.select_y_start = int(np.round(self.H * 0.6)) // self.config.vis_decimate
        self.select_x, self.select_y = self.select_x_start, self.select_y_start
        
        # Set up light dimensions
        self.light_H = min(256, self.H)
        self.light_W = min(512, self.W * 2)
        theta, phi, self.light_xyz, dtheta_dphi = utils.get_sphere_directions(
            self.light_H, self.light_W, flip=self.config.flip_secondary
        )
        
        # Set up visualization passes
        self.vis_passes = ("cache", "light", "material")
        self.vis_passes_secondary = ("cache", "light", "is_secondary")
        
        if self.use_light_sampler and self.vis_light_sampler:
            self.vis_passes = self.vis_passes + ("light_sampler_vis",)
        
        if self.vis_surface_light_field:
            self.vis_passes_secondary = self.vis_passes_secondary + ("surface_light_field_vis",)
        
        # Set up evaluation function for secondary rays
        self.render_eval_pfn_secondary = self.render_eval_pfn
        
        # Set visualization start step
        self.vis_start_step = self.vis_start 
        
        if self.vis_only and self.vis_restart:
            if utils.isdir(os.path.join(self.save_dir, "color_cache")):
                num_files = len(os.listdir(os.path.join(self.save_dir, "color_cache")))
                self.vis_start_step = max(num_files - 2, 0)

    def _initialize_metrics(self):
        """Initialize metrics tracking for evaluation."""
        self.avg_step = 0
        self.avg_metrics = {
            "albedo_psnr": 0.0,
            "psnr": 0.0,
            "mae": 0.0,
            "transient_iou": 0.0, 
            "l1_median": 0.0, 
            "l1_mean": 0.0, 
            "lpips": 0.0, 
            "ssim": 0.0
        }
        self.metric_list = {
            "albedo_psnr": [],
            "psnr": [],
            "mae": [],
            "transient_iou": [], 
            "l1_median": [], 
            "l1_mean": [], 
            "lpips": [], 
            "ssim": []
        }
        
        self.all_albedo_gt = []
        self.all_albedo_pred = []
        self.albedo_ratio = None

    def render_primary_rays(self, test_rays, train_frac):
        """Render primary rays for evaluation.
        
        Args:
            test_rays: Ray batch to render
            train_frac: Training fraction for evaluation
            
        Returns:
            rendering: Dictionary of rendering outputs
        """
        def render_fn(rng, rays, passes, resample):
            return self.render_eval_pfn(
                self.state.params,
                rng,
                train_frac,
                self.test_cameras_replicated,
                self.test_lights_replicated,
                rays,
                passes,
                resample,
            )

        rendering, self.render_rngs = models.render_image(
            render_fn,
            rng=self.render_rngs,
            rays=test_rays,
            config=self.config,
            passes=self.vis_passes,
            resample=None,
            num_repeats=self.render_repeats,
            compute_variance=True,
        )

        rendering = jax.tree_util.tree_map(np.array, rendering)
        return rendering

    def render_secondary_rays(self, test_rays, distance_median, normals_to_use, select_x, select_y, train_frac):
        """Render secondary rays for evaluation.
        
        Args:
            test_rays: Primary ray batch
            distance_median: Median distance for primary rays
            normals_to_use: Surface normals at hit points
            select_x: X-coordinate for secondary ray origin
            select_y: Y-coordinate for secondary ray origin
            train_frac: Training fraction for evaluation
            
        Returns:
            rendering_secondary: Dictionary of rendering outputs for secondary rays
        """
        # Calculate ray origin position from primary ray hit point
        positions = (
            test_rays.origins.reshape(self.H, self.W, 3)[select_y, select_x]
            + test_rays.directions.reshape(self.H, self.W, 3)[select_y, select_x]
            * distance_median.reshape(self.H, self.W)[select_y, select_x, Ellipsis, None]
        )

        # Get normal at hit point and offset position slightly along normal
        normals = normals_to_use.reshape(self.H, self.W, 3)[select_y, select_x]
        positions = positions + 4e-1 * normals

        # Create rotation matrix based on surface normal
        if self.config.secondary_rays_no_vis_transform:
            rot_mat = np.eye(3)
        else:
            rot_mat = render_utils.get_rotation_matrix(-normals[None], y_up=self.config.y_up)[0]

        # Use identity rotation for camera-to-world transform
        rot_mat = np.eye(3)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = rot_mat
        cam_to_world[Ellipsis, :3, -1] = positions

        # Cast spherical rays from the hit point
        secondary_rays = camera_utils.cast_spherical_rays(
            cam_to_world,
            self.light_H,
            self.light_W,
            self.secondary_near,
            self.secondary_far,
            light_idx=int(np.array(test_rays.light_idx.reshape(-1)[0]).astype(np.int32)),
            xnp=jnp,
        )

        # Update ray directions and viewdirs with light xyz directions
        secondary_rays = self._update_secondary_rays(secondary_rays, test_rays)

        # Render the secondary rays
        def render_fn(rng, rays, passes, resample):
            return self.render_eval_pfn_secondary(
                self.state.params,
                rng,
                train_frac,
                self.test_cameras_replicated,
                self.test_lights_replicated,
                rays,
                passes,
                resample,
            )

        rendering_secondary, self.render_rngs = models.render_image(
            render_fn,
            rng=self.render_rngs,
            rays=secondary_rays,
            config=self.config,
            passes=self.vis_passes_secondary,
            resample=None,
        )

        rendering_secondary = jax.tree_util.tree_map(np.array, rendering_secondary)
        return rendering_secondary

    def _update_secondary_rays(self, secondary_rays, test_rays):
        """Update secondary ray properties with test ray information.
        
        Args:
            secondary_rays: Secondary rays to update
            test_rays: Primary rays with source information
            
        Returns:
            Updated secondary rays
        """
        # Update directions and viewdirs
        secondary_rays = dataclasses.replace(
            secondary_rays,
            directions=self.light_xyz.reshape(secondary_rays.directions.shape),
        )
        secondary_rays = dataclasses.replace(
            secondary_rays,
            viewdirs=self.light_xyz.reshape(secondary_rays.viewdirs.shape),
        )
        
        # Copy light information from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            lights=(
                test_rays.lights.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.lights.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.lights),
        )
        
        # Copy imageplane information from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            imageplane=(
                test_rays.imageplane.reshape(-1, 2)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.imageplane.shape[:-1]) + (2,)
                )
            ) * jnp.ones_like(secondary_rays.imageplane),
        )
        
        # Copy look vector from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            look=(
                test_rays.look.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.look.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.look),
        )
        
        # Copy up vector from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            up=(
                test_rays.up.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.up.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.up),
        )
        
        # Copy camera origins from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            cam_origins=(
                test_rays.cam_origins.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.cam_origins.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.cam_origins),
        )
        
        # Copy virtual camera look vector from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            vcam_look=(
                test_rays.look.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.vcam_look.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.look),
        )
        
        # Copy up vector from test rays (duplicated intentionally from original code)
        secondary_rays = dataclasses.replace(
            secondary_rays,
            up=(
                test_rays.up.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.vcam_up.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.up),
        )
        
        # Copy virtual camera origins from test rays
        secondary_rays = dataclasses.replace(
            secondary_rays,
            vcam_origins=(
                test_rays.cam_origins.reshape(-1, 3)[0:1].reshape(
                    tuple(1 for _ in secondary_rays.vcam_origins.shape[:-1]) + (3,)
                )
            ) * jnp.ones_like(secondary_rays.cam_origins),
        )
        
        return secondary_rays

    def render_vmf(self, rendering, select_x, select_y):
        """Render von Mises-Fisher distribution visualization.
        
        Args:
            rendering: Dictionary of rendering outputs
            select_x: X-coordinate for VMF center
            select_y: Y-coordinate for VMF center
            
        Returns:
            vmf_image: Rendered VMF distribution as an image
        """
        # Extract VMF parameters
        true_means = rendering['vmf_means'][select_y, select_x]
        true_normals = rendering['vmf_normals'][select_y, select_x]
        
        # Normalize means
        true_means = true_means / np.maximum(np.linalg.norm(true_means, axis=-1, keepdims=True), 1e-5)
        
        # Get kappas and weights
        true_kappas = rendering['vmf_kappas'][select_y, select_x, Ellipsis, 0]
        true_weights = math.safe_exp(rendering['vmf_logits'][select_y, select_x, Ellipsis, 0])
        
        # Reshape normals and create mask
        true_normals = true_normals.reshape(true_means.shape[:-2] + (1, 3))
        mask = np.ones_like(true_weights)
        
        # Normalize weights
        true_weights = true_weights / true_weights.sum(-1, keepdims=True)
        
        # Evaluate VMF distribution
        vmf_image = np.sum(
            true_weights
            * mask
            * np.array(render_utils.eval_vmf(
                self.light_xyz[Ellipsis, None, :], true_means, true_kappas
            )),
            axis=-1,
        ).reshape(self.light_H, self.light_W, 1)
        
        # Repeat values across RGB channels and convert to sRGB
        vmf_image = jnp.repeat(vmf_image, 3, axis=-1)
        vmf_image = image.linear_to_srgb(vmf_image)
        
        return np.array(vmf_image)

    def log_stats(self, step):
        """Log training statistics.
        
        Args:
            step: Current training step
        """
        # Get current learning rate
        learning_rate = self.lr_fn(step)
        
        # Calculate timing information
        elapsed_time = time.time() - self.train_start_time
        steps_per_sec = self.config.print_every / elapsed_time
        rays_per_sec = self.config.batch_size * steps_per_sec * (self.grad_accum_steps // self.secondary_grad_accum_steps)
        
        # Track total training time
        TIME_PRECISION = 1000.0  # Constant from original code (not defined above)
        self.total_time += int(round(TIME_PRECISION * elapsed_time))
        self.total_steps += self.config.print_every
        approx_total_time = int(round(step * self.total_time / self.total_steps))
        
        # Process statistics from buffer
        fs = [flax.traverse_util.flatten_dict(s, sep="/") for s in self.stats_buffer]
        stats_stacked = {k: jnp.stack([f[k] for f in fs]) for k in fs[0].keys()}
        
        # Split statistics into individual components
        stats_split = self._split_statistics(stats_stacked)
        
        # Calculate average and max statistics
        kv = [(k, v) for k, v in stats_split.items() if not k.startswith("ray_")]
        avg_stats = {k: jnp.mean(v) for k, v in kv}
        max_stats = {k: jnp.max(v) for k, v in kv}
        
        # Format and log statistics
        precision = int(np.ceil(np.log10(self.config.max_steps))) + 1
        avg_loss = avg_stats["loss"]
        avg_psnr = avg_stats["psnr"]
        
        # Extract loss components
        str_losses = [
            (k[7:], (f"{v:0.5f}" if v >= 1e-4 and v < 10 else f"{v:0.1e}"))
            for k, v in avg_stats.items()
            if k.startswith("losses/")
        ]
        
        # Log message
        msg = (
            f"%{precision}d/%d: loss=%0.5f, psnr=%6.3f, lr=%0.2e | "
            + ", ".join([f"{k}={s}" for k, s in str_losses])
            + ", %0.0f r/s"
        )
        
        logging.info(
            msg,
            step,
            self.config.max_steps,
            avg_loss,
            avg_psnr,
            learning_rate,
            rays_per_sec,
        )
        
        self.reset_stats = True

    def _split_statistics(self, stats_stacked):
        """Split statistics into individual components.
        
        Args:
            stats_stacked: Dictionary of stacked statistics
            
        Returns:
            stats_split: Dictionary of split statistics
        """
        stats_split = {}
        
        for k, v in stats_stacked.items():
            if v.ndim not in [1, 2] and v.shape[0] != len(self.stats_buffer):
                raise ValueError("statistics must be of size [n], or [n, k].")
            
            if v.ndim == 1:
                stats_split[k] = v
            elif v.ndim == 2:
                # Keep ray statistics as vectors of percentiles
                if k.startswith("ray_"):
                    stats_split[k] = v
                else:
                    for i, vi in enumerate(tuple(v.T)):
                        stats_split[f"{k}/{i}"] = vi
        
        return stats_split

    def log_test_set_evaluation(self, step, train_frac, compute_albedo_ratio=False):
        """Evaluate and log model performance on test set.
        
        Args:
            step: Current training step
            train_frac: Training fraction for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        eval_start_time = time.time()
        
        # Get test rays and apply decimation if needed
        test_case = self._get_test_case(step)
        test_rays = self._cast_test_rays(test_case)
        
        # Render primary rays
        rendering = self.render_primary_rays(test_rays, train_frac)
        
        # Get masks
        masks = self._get_masks(test_case)
        
        # Process and log evaluation results on host 0
        if jax.host_id() == 0:
            self._log_evaluation_results(step, eval_start_time, test_case, rendering, masks, train_frac, compute_albedo_ratio)
            
            # Increment step counter and collect garbage
            self.avg_step += 1
            gc.collect()

    def _get_test_case(self, step):
        """Get test case for evaluation.
        
        Args:
            step: Current step
            
        Returns:
            test_case: Test case with rays and ground truth
        """
        # Apply decimation if needed
        if self.config.vis_decimate > 1:
            d = self.config.vis_decimate
            decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
        else:
            decimate_fn = lambda x: x
        
        # Select camera index
        if self.vis_only:
            cam_idx = step - self.init_step
        else:
            cam_idx = np.random.randint(0, self.test_dataset._n_examples)
        
        # Generate ray batch and apply decimation
        test_case = self.test_dataset.generate_ray_batch(cam_idx)
        test_case = jax.tree_util.tree_map(decimate_fn, test_case)
        
        return test_case

    def _cast_test_rays(self, test_case):
        """Cast rays for test case.
        
        Args:
            test_case: Test case with rays
            
        Returns:
            test_rays: Cast rays for rendering
        """
        return camera_utils.cast_ray_batch(
            cameras=self.test_dataset.cameras,
            lights=self.test_dataset.lights,
            pixels=test_case.rays,
            camtype=self.test_dataset.camtype,
            xnp=jnp,
            impulse_response=self.test_dataset.impulse_response,
            virtual_cameras=self.test_dataset.virtual_cameras,
        )

    def _get_masks(self, test_case):
        """Get masks for test case.
        
        Args:
            test_case: Test case with masks
            
        Returns:
            masks: Masks for evaluation
        """
        if test_case.masks is not None:
            return test_case.masks
        else:
            return jnp.ones_like(test_case.rgb[..., :1])

    def _log_evaluation_results(self, step, eval_start_time, test_case, rendering, masks, train_frac, compute_albedo_ratio):
        """Log evaluation results.
        
        Args:
            step: Current step
            eval_start_time: Start time of evaluation
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            train_frac: Training fraction
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Log timing information
        eval_time = time.time() - eval_start_time
        num_rays = np.prod(test_case.rays.near.shape[:-1])
        rays_per_sec = num_rays / eval_time
        logging.info("Eval %d: %0.3fs., %0.0f rays/sec", step, eval_time, rays_per_sec)
        
        # Compute metrics
        metric_start_time = time.time()
        metrics = self._compute_metrics(test_case, rendering, masks)
        
        # Save visualization outputs
        if not compute_albedo_ratio:
            self._save_visualization_outputs(step, test_case, rendering, masks)
        
        # Compute and log metrics
        self._compute_and_log_metrics(step, test_case, rendering, masks, metrics, compute_albedo_ratio)
        
        # Visualize results
        self._visualize_results(step, test_case, rendering, masks, compute_albedo_ratio)
        
        # Visualize surface light field if requested
        if self.use_light_sampler and self.vis_light_sampler:
            self._visualize_vmf(step, rendering, compute_albedo_ratio)
        
        # Visualize secondary rays if requested
        if self.vis_secondary:
            self._visualize_secondary_rays(step, test_case, rendering, train_frac, compute_albedo_ratio)
        else:
            del rendering

    def _compute_metrics(self, test_case, rendering, masks):
        """Compute metrics for evaluation.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        if self.config.evaluate_without_masks and self.config.compute_relight_metrics:
            return self.metric_harness(
                self.postprocess_fn(rendering["cache_rgb"]),
                self.postprocess_fn(test_case.rgb)
            )
        else:
            # Apply color correction if needed
            rgb = rendering["rgb"].reshape(-1, self.config.num_rgb_channels)
            rgb_gt = test_case.rgb.reshape(-1, self.config.num_rgb_channels)
            
            if self.config.correct_eval:
                mask = np.repeat((masks > 0.0).reshape(-1, 1), self.config.num_rgb_channels, axis=-1)
                masked_rgb = rgb[mask].reshape(-1, self.config.num_rgb_channels)
                masked_rgb_gt = rgb_gt[mask].reshape(-1, self.config.num_rgb_channels)
                
                masked_rgb_ratio = np.median(masked_rgb_gt / np.clip(masked_rgb, 1e-6, 1.0), axis=0, keepdims=True)
                rgb[mask] = (masked_rgb * masked_rgb_ratio).reshape(-1)
                rendering["rgb"] = rgb.reshape(rendering["rgb"].shape)
            
            return self.metric_harness(
                self.postprocess_fn(rendering["cache_rgb"]) * masks,
                self.postprocess_fn(test_case.rgb) * masks
            )

    def _save_visualization_outputs(self, step, test_case, rendering, masks):
        """Save visualization outputs to disk.
        
        Args:
            step: Current step
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
        """
        # Create output directories if they don't exist
        output_dirs = [
            "color_residual", "color_gt", "color", "masks", 
            "normals_to_use", "cache_albedo_rgb"
        ]
        
        for dir_name in output_dirs:
            dir_path = os.path.join(self.save_dir, dir_name)
            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
        
        # Save residual
        residual = self.postprocess_fn(rendering["cache_rgb"]) - self.postprocess_fn(test_case.rgb)
        utils.save_img_u8(
            np.abs(residual), 
            os.path.join(self.save_dir, "color_residual", f"{self.avg_step+self.vis_start_step:04d}.png")
        )
        
        # Save color ground truth
        color_gt = np.array(self.postprocess_fn(test_case.rgb))

        if self.vis_only and self.vis_extra:
            color_gt[self.select_y-5:self.select_y+5, self.select_x-5:self.select_x+5, :] = np.array([1.0, 0.0, 0.0])

        utils.save_img_u8(
            color_gt,
            os.path.join(self.save_dir, "color_gt", f"{self.avg_step+self.vis_start_step:04d}.png")
        )
        
        # Save NPY files
        np.save(
            os.path.join(self.save_dir, "color_gt", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            self.postprocess_fn(test_case.rgb)
        )
        np.save(
            os.path.join(self.save_dir, "color", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            self.postprocess_fn(rendering["cache_rgb"])
        )
        np.save(
            os.path.join(self.save_dir, "masks", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            masks.reshape(self.H, self.W, 1)
        )
        np.save(
            os.path.join(self.save_dir, "normals_to_use", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            self.postprocess_fn(rendering["normals_to_use"])
        )
        np.save(
            os.path.join(self.save_dir, "cache_albedo_rgb", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            self.postprocess_fn(rendering["cache_albedo_rgb"])
        )

    def _compute_and_log_metrics(self, step, test_case, rendering, masks, metrics, compute_albedo_ratio):
        """Compute and log various metrics.
        
        Args:
            step: Current step
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            metrics: Dictionary of computed metrics
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Compute and log PSNR
        self._compute_and_log_psnr(test_case, rendering, masks, compute_albedo_ratio)
        
        # Log LPIPS and SSIM if available
        self._log_perceptual_metrics(metrics, compute_albedo_ratio)
        
        # Compute and log albedo metrics if requested
        if self.config.compute_albedo_metrics and self.vis_only and not self.config.fixed_light and not self.config.eval_train and not self.config.eval_path:
            self._compute_and_log_albedo_metrics(test_case, rendering, masks, compute_albedo_ratio)
        
        # Compute and log transient metrics if requested
        if self.config.compute_transient_metrics and self.vis_only and not compute_albedo_ratio and not self.config.eval_path:
            self._compute_and_log_transient_metrics(test_case, rendering)
        
        # Handle path evaluation if requested
        if self.config.eval_path:
            self._save_path_evaluation(step, rendering)
        
        # Save depth information
        self._save_depth_information(step, rendering)
        
        # Compute and log depth metrics if requested
        if self.config.compute_depth_metrics and self.vis_only and not self.config.fixed_light and not self.config.eval_train and not self.config.eval_path:
            self._compute_and_log_depth_metrics(test_case, rendering, masks, compute_albedo_ratio)
        
        # Compute and log normal metrics if requested
        if self.config.compute_normal_metrics and self.vis_only and not self.config.fixed_light and not self.config.eval_train and not self.config.eval_path:
            self._compute_and_log_normal_metrics(test_case, rendering, masks, compute_albedo_ratio)

    def _compute_and_log_psnr(self, test_case, rendering, masks, compute_albedo_ratio):
        """Compute and log PSNR metric.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Calculate MSE between rendered and ground truth images
        if self.config.evaluate_without_masks:
            mse = (
                (
                    (self.postprocess_fn(test_case.rgb[..., :3])).reshape(-1, self.config.num_rgb_channels)
                    - (self.postprocess_fn(rendering["cache_rgb"])).reshape(-1, self.config.num_rgb_channels)
                ) ** 2
            ).mean()
        else:
            mse = (
                (
                    (self.postprocess_fn(test_case.rgb[..., :3]) * masks).reshape(-1, self.config.num_rgb_channels)
                    - (self.postprocess_fn(rendering["cache_rgb"]) * masks).reshape(-1, self.config.num_rgb_channels)
                ) ** 2
            ).mean()
        
        # Convert MSE to PSNR
        psnr = utils.mse_to_psnr(mse)
        
        # Update running average for visualization only
        if self.vis_only and not compute_albedo_ratio and not self.config.fixed_light:
            self.avg_metrics["psnr"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["psnr"]
                + (1.0 / (self.avg_step + 1)) * psnr
            )
            print(f"PSNR {self.avg_metrics['psnr']}")
            self.metric_list['psnr'].append(psnr)
        else:
            print(f"PSNR {psnr}")

    def _log_perceptual_metrics(self, metrics, compute_albedo_ratio):
        """Log perceptual metrics like LPIPS and SSIM.
        
        Args:
            metrics: Dictionary of computed metrics
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Log LPIPS if available
        if "lpips" in metrics:
            lpips = metrics["lpips"]
            if self.vis_only and not compute_albedo_ratio:
                self.avg_metrics["lpips"] = (
                    (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["lpips"]
                    + (1.0 / (self.avg_step + 1)) * lpips
                )
                print(f"LPIPS {self.avg_metrics['lpips']}")
                self.metric_list['lpips'].append(lpips)
            else:
                print(f"LPIPS {lpips}")
        
        # Log SSIM if available
        if "ssim" in metrics:
            ssim = metrics["ssim"]
            if self.vis_only and not compute_albedo_ratio:
                self.avg_metrics["ssim"] = (
                    (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["ssim"]
                    + (1.0 / (self.avg_step + 1)) * ssim
                )
                print(f"SSIM {self.avg_metrics['ssim']}")
                self.metric_list['ssim'].append(ssim)
            else:
                print(f"SSIM {ssim}")

    def _compute_and_log_albedo_metrics(self, test_case, rendering, masks, compute_albedo_ratio):
        """Compute and log albedo metrics.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Get albedo from material or cache
        if self.use_material:
            albedo = np.array(rendering["material_albedo"]).reshape(-1, self.config.num_rgb_channels)
        else:
            albedo = np.array(rendering["cache_albedo_rgb"]).reshape(-1, self.config.num_rgb_channels)
        
        # Get accumulation and mask
        acc = np.array(rendering["acc"]).reshape(-1)
        mask = np.repeat((masks > 0.0).reshape(-1, 1), self.config.num_rgb_channels, axis=-1)
        albedo_mask = (mask & (acc[..., None] > 0.5))
        
        # Process ground truth albedo
        albedo_gt = np.array(test_case.albedos.reshape(-1, self.config.num_rgb_channels))
        albedo_gt[~mask] = 1
        
        # Process predicted albedo
        albedo = (albedo + (1.0 - acc[..., None])).reshape(-1, self.config.num_rgb_channels)
        albedo[~mask] = 1.0
        
        # Store albedo for ratio computation if needed
        if compute_albedo_ratio:
            self.all_albedo_gt.append(albedo_gt[albedo_mask].reshape(-1, self.config.num_rgb_channels))
            self.all_albedo_pred.append(albedo[albedo_mask].reshape(-1, self.config.num_rgb_channels))
        
        # Calculate albedo ratio for visualization
        albedo_ratio_im = np.clip(
            (albedo_gt / albedo).reshape(self.H, self.W, self.config.num_rgb_channels), 
            0.0, 
            1.0
        )
        
        # Apply albedo correction if ratio is available
        if self.albedo_ratio is not None:
            albedo[albedo_mask] = np.clip(
                albedo[albedo_mask].reshape(-1, self.config.num_rgb_channels) * self.albedo_ratio, 
                0.0, 
                self.albedo_clip
            ).reshape(-1)
        else:
            # Calculate albedo ratio from current batch
            masked_albedo_gt = albedo_gt[albedo_mask].reshape(-1, self.config.num_rgb_channels)
            masked_albedo = albedo[albedo_mask].reshape(-1, self.config.num_rgb_channels)
            masked_albedo_ratio = np.median(
                masked_albedo_gt / np.clip(masked_albedo, 1e-6, 1.0), 
                axis=0, 
                keepdims=True
            )
            albedo[albedo_mask] = np.clip(
                masked_albedo * masked_albedo_ratio, 
                0.0, 
                self.albedo_clip
            ).reshape(-1)
        
        # Apply gamma correction
        albedo_gt = albedo_gt ** (1.0 / 2.2)
        albedo = albedo ** (1.0 / 2.2)
        
        # Calculate albedo metrics
        albedo_mse = (((albedo*masks.reshape(-1, 1) - albedo_gt*masks.reshape(-1, 1)) ** 2)).mean()
        albedo_psnr = utils.mse_to_psnr(albedo_mse)
        
        # Save albedo visualization if not computing ratio
        if not compute_albedo_ratio:
            self._save_albedo_visualization(test_case, albedo, albedo_gt, albedo_ratio_im)
        
        # Update running averages for visualization only
        if self.vis_only and not compute_albedo_ratio:
            self.avg_metrics["albedo_psnr"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["albedo_psnr"]
                + (1.0 / (self.avg_step + 1)) * albedo_psnr
            )
            print(f"Albedo PSNR {self.avg_metrics['albedo_psnr']}")
            self.metric_list['albedo_psnr'].append(albedo_psnr)
        else:
            print(f"Albedo PSNR {albedo_psnr}")

    def _save_albedo_visualization(self, test_case, albedo, albedo_gt, albedo_ratio_im):
        """Save albedo visualization outputs.
        
        Args:
            test_case: Test case with ground truth
            albedo: Predicted albedo
            albedo_gt: Ground truth albedo
            albedo_ratio_im: Albedo ratio image
        """
        # Create output directories
        dirs = ["albedo_gt", "albedo_ratio_im", "albedo_pred"]
        for dir_name in dirs:
            dir_path = os.path.join(self.save_dir, dir_name)
            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
        
        # Save ground truth albedo
        np.save(
            os.path.join(self.save_dir, "albedo_gt", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            albedo_gt.reshape(self.H, self.W, self.config.num_rgb_channels)
        )
        utils.save_img_u8(
            albedo_gt.reshape(self.H, self.W, self.config.num_rgb_channels), 
            os.path.join(self.save_dir, "albedo_gt", f"{self.avg_step+self.vis_start_step:04d}.png")
        )
        
        # Save albedo ratio image
        utils.save_img_u8(
            albedo_ratio_im.reshape(self.H, self.W, self.config.num_rgb_channels), 
            os.path.join(self.save_dir, "albedo_ratio_im", f"{self.avg_step+self.vis_start_step:04d}.png")
        )
        
        # Save predicted albedo
        np.save(
            os.path.join(self.save_dir, "albedo_pred", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            albedo.reshape(self.H, self.W, self.config.num_rgb_channels)
        )
        utils.save_img_u8(
            albedo.reshape(self.H, self.W, self.config.num_rgb_channels), 
            os.path.join(self.save_dir, "albedo_pred", f"{self.avg_step+self.vis_start_step:04d}.png")
        )

    def _compute_and_log_transient_metrics(self, test_case, rendering):
        """Compute and log transient metrics.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
        """
        # Calculate IoU
        intersection = jnp.minimum(rendering["cache_rgb"], test_case.rgb[..., :3])
        union = jnp.maximum(rendering["cache_rgb"], test_case.rgb[..., :3])
        iou = jnp.sum(intersection) / jnp.sum(union)
        
        # Update running average
        if self.vis_only:
            self.avg_metrics["transient_iou"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["transient_iou"]
                + (1.0 / (self.avg_step + 1)) * iou
            )
            print(f"Transient IOU {self.avg_metrics['transient_iou']}")
            self.metric_list['transient_iou'].append(iou.item())
        else:
            print(f"Transient IOU {iou}")
        
        # Save transient data
        self._save_transient_data(rendering)

    def _save_transient_data(self, rendering, is_secondary=False):
        """Save transient data to HDF5 files.
        
        Args:
            rendering: Rendering outputs
        """
        suffix = "/secondary" if is_secondary else ""

        # Create output directories
        if is_secondary:
            dirs = [f"transients{suffix}"]
        else:
            dirs = [f"transients{suffix}", f"material_transients{suffix}"]

        for dir_name in dirs:
            dir_path = os.path.join(self.save_dir, dir_name)

            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
        
        # Save cache RGB transients
        file = h5py.File(
            os.path.join(self.save_dir, f"transients{suffix}", f"{self.avg_step+self.vis_start_step:04d}.h5"), 
            'w'
        )
        dataset = file.create_dataset(
            "data", rendering["cache_rgb"].shape, dtype='f', data=np.nan_to_num(rendering["cache_rgb"])
        )
        file.close()
        
        # Save material transients
        if not is_secondary:
            file = h5py.File(
                os.path.join(self.save_dir, f"material_transients{suffix}", f"{self.avg_step+self.vis_start_step:04d}.h5"), 
                'w'
            )
            dataset = file.create_dataset(
                "data", rendering["rgb"].shape, dtype='f', data=np.nan_to_num(rendering["rgb"])
            )
            file.close()

    def _save_path_evaluation(self, step, rendering):
        """Save path evaluation outputs.
        
        Args:
            step: Current step
            rendering: Rendering outputs
        """
        # Calculate time slice
        total_steps = self.test_dataset.camtoworlds.shape[0]
        frac = (step - self.init_step) / float(total_steps)
        
        t = self.config.transient_start_idx + frac * (self.config.transient_end_idx - self.config.transient_start_idx)
        t_floor = int(np.floor(t))
        t_ceil = int(np.ceil(t))
        w = t - t_floor
        
        # Create output directory
        dir_path = os.path.join(self.save_dir, "cache_time_slice")
        if not utils.isdir(dir_path):
            utils.makedirs(dir_path)
        
        # Calculate time slice
        cache_rgb = rendering["cache_rgb"]
        cache_time_slice = w * cache_rgb[..., t_ceil, :] + (1.0 - w) * cache_rgb[..., t_floor, :]
        
        # Save time slice
        np.save(
            os.path.join(dir_path, f"{self.avg_step+self.vis_start_step:04d}.npy"),
            cache_time_slice.reshape(self.H, self.W, self.config.num_rgb_channels)
        )
        utils.save_img_u8(
            cache_time_slice.reshape(self.H, self.W, self.config.num_rgb_channels).squeeze(),
            os.path.join(dir_path, f"{self.avg_step+self.vis_start_step:04d}.png")
        )

    def _save_depth_information(self, step, rendering):
        """Save depth information.
        
        Args:
            step: Current step
            rendering: Rendering outputs
        """
        # Get depth maps
        depth_mean = rendering["distance_mean"]
        depth_median = rendering["distance_median"]
        
        # Create output directories
        dirs = ["depth_mean", "depth_median"]
        for dir_name in dirs:
            dir_path = os.path.join(self.save_dir, dir_name)
            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
        
        # Save depth maps
        np.save(
            os.path.join(self.save_dir, "depth_mean", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            depth_mean.reshape(self.H, self.W),
        )
        np.save(
            os.path.join(self.save_dir, "depth_median", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            depth_median.reshape(self.H, self.W),
        )

    def _compute_and_log_depth_metrics(self, test_case, rendering, masks, compute_albedo_ratio):
        """Compute and log depth metrics.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Get depth maps
        depth_mean = rendering["distance_mean"]
        depth_median = rendering["distance_median"]
        
        # Calculate L1 errors
        l1_median = jnp.abs(depth_median - test_case.depth)
        l1_mean = jnp.abs(depth_mean - test_case.depth)
        
        # Apply masking if needed
        if self.config.evaluate_without_masks:
            l1_median = jnp.mean(l1_median)
            l1_mean = jnp.mean(l1_mean)
        else:
            l1_median = (l1_median*masks.squeeze()).sum()/masks.sum()
            l1_mean = (l1_mean*masks.squeeze()).sum()/masks.sum()
        
        # Update running averages
        if self.vis_only and not compute_albedo_ratio:
            self.avg_metrics["l1_median"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["l1_median"]
                + (1.0 / (self.avg_step + 1)) * l1_median
            )
            print(f"L1 median {self.avg_metrics['l1_median']}")
            
            self.avg_metrics["l1_mean"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["l1_mean"]
                + (1.0 / (self.avg_step + 1)) * l1_mean
            )
            print(f"L1 mean {self.avg_metrics['l1_mean']}")
            
            self.metric_list['l1_mean'].append(l1_mean.item())
            self.metric_list['l1_median'].append(l1_median.item())
        else:
            print(f"L1 median {l1_median}")
            print(f"L1 mean {l1_mean}")

    def _compute_and_log_normal_metrics(self, test_case, rendering, masks, compute_albedo_ratio):
        """Compute and log normal metrics.
        
        Args:
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Get accumulation values
        acc = np.array(rendering["acc"]).reshape(-1)
        
        # Process ground truth normals
        normals_gt = np.array(test_case.normals.reshape(-1, 3))
        normals_gt = (normals_gt + (1.0 - masks.reshape(-1, 1))).reshape(-1, 3)
        
        # Normalize ground truth normals
        normals_gt = np.where(
            np.linalg.norm(normals_gt, axis=-1, keepdims=True) < 1e-5,
            np.zeros_like(normals_gt),
            normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True),
        )
        
        # Get predicted normals based on configuration
        if self.config.material_normals_target == "normals":
            normals = np.array(rendering["normals"]).reshape(-1, 3)
        else:
            normals = np.array(rendering["normals_to_use"]).reshape(-1, 3)
        
        # Process and normalize predicted normals
        normals = (normals + (1.0 - acc[..., None])).reshape(-1, 3)
        normals = np.where(
            np.linalg.norm(normals, axis=-1, keepdims=True) < 1e-5,
            np.zeros_like(normals),
            normals / np.linalg.norm(normals, axis=-1, keepdims=True),
        )
        
        # Save normal visualizations if not computing ratio
        if not compute_albedo_ratio:
            self._save_normal_visualizations(test_case, normals, normals_gt, masks)
        
        # Calculate mean angular error
        if self.config.evaluate_without_masks:
            mae = np.mean(
                np.arccos(
                    np.clip(np.sum(normals_gt * normals, axis=-1), -1, 1)
                ) * 180 / np.pi
            )
        else:
            mae = np.mean(
                np.arccos(
                    np.clip(np.sum(normals_gt * normals, axis=-1), -1, 1)
                ) * 180 / np.pi
                * masks.reshape(-1)
            )
        
        # Update running average
        if self.vis_only and not compute_albedo_ratio:
            self.avg_metrics["mae"] = (
                (self.avg_step / (self.avg_step + 1)) * self.avg_metrics["mae"]
                + (1.0 / (self.avg_step + 1)) * mae
            )
            print(f"MAE {self.avg_metrics['mae']}")
            self.metric_list['mae'].append(mae)
        else:
            print(f"MAE {mae}")

    def _save_normal_visualizations(self, test_case, normals, normals_gt, masks):
        """Save normal visualization outputs.
        
        Args:
            test_case: Test case with ground truth
            normals: Predicted normals
            normals_gt: Ground truth normals
            masks: Masks for evaluation
        """
        # Create output directories
        dirs = ["normals_to_use", "normals_gt"]
        for dir_name in dirs:
            dir_path = os.path.join(self.save_dir, dir_name)
            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
        
        # Save predicted normals
        np.save(
            os.path.join(self.save_dir, "normals_to_use", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            normals.reshape(self.H, self.W, 3),
        )
        
        # Save ground truth normals
        mask = masks.reshape(test_case.normals.shape[0], test_case.normals.shape[1], 1) > 0
        mask = np.repeat(mask, 3, axis=-1)
        normals_gt_save = test_case.normals.copy()
        normals_gt_save[~mask] = 1
        
        utils.save_img_u8(
            normals_gt_save / 2.0 + 0.5, 
            os.path.join(self.save_dir, "normals_gt", f"{self.avg_step+self.vis_start_step:04d}.png")
        )
        np.save(
            os.path.join(self.save_dir, "normals_gt", f"{self.avg_step+self.vis_start_step:04d}.npy"),
            normals_gt.reshape(test_case.normals.shape),
        )

    def _visualize_results(self, step, test_case, rendering, masks, compute_albedo_ratio):
        """Visualize rendering results.
        
        Args:
            step: Current step
            test_case: Test case with ground truth
            rendering: Rendering outputs
            masks: Masks for evaluation
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Add ground truth to rendering for visualization
        rendering["color_gt"] = test_case.rgb
        
        if self.vis_only and test_case.depth is not None:
            rendering["depth_gt"] = test_case.depth
        
        # Generate visualization suite
        if self.use_transient:
            vis_suite = vis.visualize_transient_suite(
                rendering,
                self.config, 
                vis_material=self.use_material,
                vis_secondary=False,
            )
        else:
            vis_suite = vis.visualize_suite(
                rendering,
                self.config, 
                vis_material=self.use_material,
                vis_secondary=self.vis_secondary,
            )
        
        # Save visualization outputs
        if self.save_results and not compute_albedo_ratio:
            for k, v in vis_suite.items():
                if isinstance(v, list):
                    pass
                else:
                    # Create directory if it doesn't exist
                    dir_path = os.path.join(self.save_dir, k)
                    if not utils.isdir(dir_path):
                        utils.makedirs(dir_path)
                    
                    # Apply masking to depth visualizations
                    if "depth" in k:
                        if v.shape[-1] == 1:
                            v[~(np.repeat(masks > 0.0, 1, axis=-1))] = 1
                        else:
                            v[~(np.repeat(masks > 0.0, 3, axis=-1))] = 1
                    
                    # Sum over temporal dimension if present
                    if len(v.shape) == 4: 
                        v = v.sum(-2)
                    
                    # Save visualization
                    utils.save_img_u8(
                        v.squeeze(), 
                        os.path.join(self.save_dir, k, f"{self.avg_step+self.vis_start_step:04d}.png")
                    )

    def _visualize_vmf(self, step, rendering, compute_albedo_ratio):
        """Visualize von Mises-Fisher distribution.
        
        Args:
            step: Current step
            rendering: Rendering outputs
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Render VMF visualization
        vmf_image = self.render_vmf(
            rendering,
            self.select_x,
            self.select_y,
        )
        
        # Save VMF visualization
        if self.save_results and not compute_albedo_ratio:
            dir_path = os.path.join(self.save_dir, "vmfs")
            if not utils.isdir(dir_path):
                utils.makedirs(dir_path)
            
            utils.save_img_u8(
                vmf_image,
                os.path.join(self.save_dir, f"vmfs/{self.avg_step+self.vis_start_step:04d}.png")
            )

    def _visualize_secondary_rays(self, step, test_case, rendering, train_frac, compute_albedo_ratio):
        """Visualize secondary rays.
        
        Args:
            step: Current step
            test_case: Test case with ground truth
            rendering: Rendering outputs
            train_frac: Training fraction
            compute_albedo_ratio: Whether to compute albedo ratio
        """
        # Extract depth and normals from rendering
        distance_median = rendering['distance_median'].copy()
        normals_to_use = rendering['normals_to_use'].copy()
        del rendering
        
        # Render secondary rays
        test_rays = self._cast_test_rays(test_case)
        rendering_secondary = self.render_secondary_rays(
            test_rays,
            distance_median,
            normals_to_use,
            self.select_x,
            self.select_y,
            train_frac,
        )
        
        # Generate visualization suite for secondary rays
        if self.use_transient:
            vis_suite_secondary = vis.visualize_transient_suite(
                rendering_secondary,
                self.config,
                vis_secondary=True,
            )
        else:
            vis_suite_secondary = vis.visualize_suite(
                rendering_secondary,
                self.config,
                vis_secondary=True,
            )
        
        # Save visualization outputs
        if self.save_results and not compute_albedo_ratio:
            for k, v in vis_suite_secondary.items():
                if isinstance(v, list):
                    pass
                else:
                    dir_path = os.path.join(self.save_dir, k, "secondary")
                    if not utils.isdir(dir_path):
                        utils.makedirs(dir_path)
                    
                    if len(v.shape) == 4:
                        v = v.sum(-2)
                    
                    utils.save_img_u8(
                        v.squeeze(),
                        os.path.join(self.save_dir, k, f"secondary/{self.avg_step+self.vis_start_step:04d}.png")
                    )
        
        if self.vis_only:
            self._save_transient_data(rendering_secondary, is_secondary=True)
        
        del rendering_secondary

    def save_checkpoint(self, step):
        """Save a checkpoint of the current training state.
        
        Args:
            step: Current training step
        """
        checkpoints.save_checkpoint_multiprocess(
            self.config.checkpoint_dir,
            jax.device_get(flax.jax_utils.unreplicate(self.state)),
            int(step),
            keep=self.config.checkpoint_keep,
            overwrite=True,
        )

    def train(self):
        """Main training loop."""
        # Replicate model state across devices
        self.state = flax.jax_utils.replicate(self.state)
        
        # Initialize logging
        self.total_time = 0
        self.total_steps = 0
        self.train_start_time = time.time()
        self.reset_stats = True
        
        # Determine number of training steps
        if self.config.early_exit_steps is not None:
            num_steps = self.config.early_exit_steps
        else:
            num_steps = self.max_steps
        
        # Set up data loading
        raybatcher = datasets.RayBatcher(self.dataset)
        p_raybatcher = flax.jax_utils.prefetch_to_device(raybatcher, 3)
        
        # Set up RNG for training
        self.rng = self.rng + jax.host_id()  # Make random seed separate across hosts
        self.rngs = random.split(self.rng, jax.local_device_count())  # For pmapping RNG keys
        self.render_rngs = random.split(self.render_rng, jax.local_device_count())  # For pmapping RNG keys
        
        # Initialize training state
        self.init_step = flax.jax_utils.unreplicate(self.state.step) // self.grad_accum_steps + 1
        batch = None
        
        # Handle visualization-only mode
        if self.vis_only:
            self._run_visualization_only()
            return
            
        # Main training loop
        for step in range(self.init_step, num_steps + 1):
            # Reset stats for logging
            if self.reset_stats and (jax.host_id() == 0):
                self.stats_buffer = []
                self.train_start_time = time.time()
                self.reset_stats = False
            
            # Gradient accumulation loop
            for s in range(self.grad_accum_steps):
                with jax.profiler.StepTraceAnnotation("train", step_num=step * self.grad_accum_steps + s):
                    # Get next batch based on accumulation settings
                    if self.secondary_grad_accum_steps > 1:
                        if batch is None or s % self.secondary_grad_accum_steps == 0:
                            batch = next(p_raybatcher)
                    else:
                        batch = next(p_raybatcher)
                    
                    # Update training fraction
                    cur_step = flax.jax_utils.unreplicate(self.state.step) // self.grad_accum_steps
                    train_frac = jnp.clip(
                        cur_step / (self.max_steps - 1),
                        0,
                        1
                    )
                    
                    # Perform training step
                    self.state, stats, self.rngs = self.train_pstep(
                        self.rngs,
                        self.state,
                        batch,
                        self.cameras,
                        self.virtual_cameras,
                        self.lights,
                        train_frac,
                    )
            
            # Garbage collection
            if step % self.config.gc_every == 0:
                gc.collect()
            
            # Checkpointing
            if step == 1 or step % self.config.checkpoint_every == 0:
                self.save_checkpoint(step)
            
            # Log statistics
            if jax.host_id() == 0:
                stats = flax.jax_utils.unreplicate(stats)
                stats = jax.tree_util.tree_map(np.array, stats)
                self.stats_buffer.append(stats)
                
                if step == self.init_step or step % self.config.print_every == 0:
                    self.log_stats(step)
            
            # Log test-set evaluation
            if (
                self.config.train_render_every > 0
                and step % self.config.train_render_every == 0
                and not self.config.no_vis
            ):
                if not (utils.isdir(self.save_dir)):
                    utils.makedirs(self.save_dir)
                
                self.log_test_set_evaluation(step, train_frac)

    def _run_visualization_only(self):
        """Run visualization-only mode."""
        # Create save directory if needed
        if not (utils.isdir(self.save_dir)):
            utils.makedirs(self.save_dir)
        
        # Reset step counter
        self.avg_step = 0
        
        # Compute albedo ratio if needed
        if "albedo" in self.save_dir:
            self._compute_albedo_ratio()
        
        # Evaluate and visualize test set
        self.avg_step = 0
        
        # Iterate through camera views
        for step in range(self.init_step + self.vis_start_step, self.init_step + min(self.test_dataset.camtoworlds.shape[0], self.vis_end)):
            self.log_test_set_evaluation(step, 1.0)
        
        # Calculate mean metrics
        for k, v in self.metric_list.items():
            if len(v) > 0:
                self.metric_list[k].append(sum(v)/len(v))
            else:
                self.metric_list[k].append(0.0)
        
        # Save results
        with open(os.path.join(self.save_dir, "results.txt"), "w") as file:
            for key, values in self.metric_list.items():
                file.write(f"{key}: {values}\n")
        
        exit()

    def _compute_albedo_ratio(self):
        """Compute albedo ratio for visualization-only mode."""
        for step in range(self.init_step, self.init_step + self.test_dataset.camtoworlds.shape[0], 10):
            self.log_test_set_evaluation(step, 1.0, compute_albedo_ratio=True)
        
        if self.albedo_correct_median:
            # Use median for albedo correction
            all_albedo_gt = np.concatenate(self.all_albedo_gt, axis=0)
            all_albedo_pred = np.concatenate(self.all_albedo_pred, axis=0)
            self.albedo_ratio = np.median(all_albedo_gt / np.clip(all_albedo_pred, 1e-6, 1.0), axis=0, keepdims=True)
        else:
            # Use least squares for albedo correction
            all_albedo_pred = np.concatenate(self.all_albedo_pred + self.all_albedo_pred + self.all_albedo_pred, axis=0)
            all_albedo_gt = np.transpose(np.concatenate(self.all_albedo_gt, axis=0), (1, 0)).reshape(-1)
            
            # Zero out components to isolate color channels
            all_albedo_pred_temp = np.copy(all_albedo_pred)
            all_albedo_pred_temp[0*all_albedo_pred_temp.shape[0]//3:1*all_albedo_pred_temp.shape[0]//3, 0] = 0
            all_albedo_pred_temp[1*all_albedo_pred_temp.shape[0]//3:2*all_albedo_pred_temp.shape[0]//3, 1] = 0
            all_albedo_pred_temp[2*all_albedo_pred_temp.shape[0]//3:3*all_albedo_pred_temp.shape[0]//3, 2] = 0
            all_albedo_pred = all_albedo_pred - all_albedo_pred_temp
            
            # Apply gamma correction if needed
            if self.albedo_gamma:
                all_albedo_pred = all_albedo_pred ** (1.0 / 2.2)
                all_albedo_gt = all_albedo_gt ** (1.0 / 2.2)
            
            # Solve least squares problem
            self.albedo_ratio = np.linalg.lstsq(all_albedo_pred, all_albedo_gt)[0].reshape(-1, 3)
            
            # Undo gamma correction if needed
            if self.albedo_gamma:
                self.albedo_ratio = self.albedo_ratio ** (2.2)
        
        # Save albedo ratio
        np.save(
            os.path.join(self.save_dir, "albedo_ratio.npy"),
            self.albedo_ratio
        )

