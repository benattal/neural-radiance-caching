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
"""Training step and model creation functions."""

import collections
import copy
import dataclasses
import functools
from typing import Any, Callable, Dict, MutableMapping, Optional, Tuple, Text

import chex
import flax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from flax.core.scope import FrozenVariableDict
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import random
import pdb
from internal import (
    camera_utils,
    configs,
    coord,
    datasets,
    grid_utils,
    image,
    loss_utils,
    math,
    models,
    ref_utils,
    utils,
)
from internal.inverse_render import render_utils
from third_party.robust_loss_jax import general


# -----------------------------------------------------------------------------
# JAX Tree Utility Functions
# -----------------------------------------------------------------------------

def tree_sum(tree):
    """Sum all values in a pytree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm_sq(tree):
    """Calculate the squared norm of a pytree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(y**2), tree, initializer=0)


def tree_norm(tree):
    """Calculate the norm of a pytree."""
    return jnp.sqrt(tree_norm_sq(tree))


def tree_abs_max(tree):
    """Find the maximum absolute value in a pytree."""
    return jax.tree_util.tree_reduce(lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), tree, initializer=0)


def tree_len(tree):
    """Count the total number of elements in a pytree."""
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.prod(jnp.array(y.shape)), tree, initializer=0)


def summarize_tree(fn, tree, ancestry=(), max_depth=3):
    """Flatten 'tree' while 'fn'-ing values and formatting keys like/this.
    
    Args:
        fn: A function to apply to each value in the tree.
        tree: The pytree to summarize.
        ancestry: The current key path, used for recursion.
        max_depth: Maximum depth to traverse in the tree.
        
    Returns:
        A dictionary mapping flattened keys to their processed values.
    """
    stats = {}
    for k, v in tree.items():
        name = ancestry + (k,)
        stats["/".join(name)] = fn(v)
        if hasattr(v, "items") and len(ancestry) < (max_depth - 1):
            stats.update(summarize_tree(fn, v, ancestry=name, max_depth=max_depth))
    return stats


# -----------------------------------------------------------------------------
# Unbiased Loss Functions
# -----------------------------------------------------------------------------

def compute_unbiased_loss(rendering, gt, gt_nocorr, config):
    """Compute unbiased loss between rendering and ground truth.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        
    Returns:
        The computed loss.
    """
    rgb = rendering["rgb"]
    rgb_nocorr = rendering["rgb_nocorr"]

    diff = rgb - gt
    diff_nocorr = rgb_nocorr - gt_nocorr
    loss = 2 * diff * jax.lax.stop_gradient(diff_nocorr)

    return loss


def compute_unbiased_loss_itof(rendering, gt, gt_nocorr, config):
    """Compute unbiased loss between rendering and ground truth using iToF.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        
    Returns:
        The computed loss.
    """
    rgb = rendering["rgb"]
    rgb_nocorr = rendering["rgb_nocorr"]

    diff = render_utils.dtof_to_itof(rgb - gt, config.itof_frequency_phase_shifts, config.exposure_time)
    diff_nocorr = render_utils.dtof_to_itof(rgb_nocorr - gt_nocorr, config.itof_frequency_phase_shifts, config.exposure_time)
    loss = 2 * diff * jax.lax.stop_gradient(diff_nocorr)

    return loss


def compute_unbiased_loss_transient_gauss(rendering, gt, gt_nocorr, config):
    """Compute unbiased loss between rendering and ground truth using Gaussian blur for transients.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        
    Returns:
        The computed loss.
    """
    rgb = rendering["rgb"]
    rgb_nocorr = rendering["rgb_nocorr"]

    diff = render_utils.dtof_to_gauss(rgb - gt, config.transient_gauss_sigma_scales, config.transient_gauss_constant_scale)
    diff_nocorr = render_utils.dtof_to_gauss(rgb_nocorr - gt_nocorr, config.transient_gauss_sigma_scales, config.transient_gauss_constant_scale)
    loss = 2 * diff * jax.lax.stop_gradient(diff_nocorr)

    return loss


# -----------------------------------------------------------------------------
# RawNeRF Loss Functions
# -----------------------------------------------------------------------------

def compute_unbiased_loss_rawnerf(rendering, gt, gt_nocorr, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute unbiased RawNeRF loss.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip), exponent) + eps)
    data_loss = compute_unbiased_loss(rendering, gt, gt_nocorr, config)
    return data_loss * scaling_grad


def compute_unbiased_loss_rawnerf_transient(rendering, gt, gt_nocorr, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute unbiased RawNeRF loss for transient data.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = compute_unbiased_loss(rendering, gt, gt_nocorr, config)
    return data_loss * scaling_grad


def compute_unbiased_loss_rawnerf_transient_itof(rendering, gt, gt_nocorr, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute unbiased RawNeRF loss for transient data using iToF.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = compute_unbiased_loss_itof(rendering, gt, gt_nocorr, config)
    return data_loss * scaling_grad


def compute_unbiased_loss_rawnerf_transient_gauss(rendering, gt, gt_nocorr, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute unbiased RawNeRF loss for transient data using Gaussian blur.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = compute_unbiased_loss_transient_gauss(rendering, gt, gt_nocorr, config)
    return data_loss * scaling_grad


def compute_loss_rawnerf(rendering, gt, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute standard RawNeRF loss.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip), exponent) + eps)
    data_loss = ((rendering["rgb"] - gt) ** 2)
    return data_loss * scaling_grad


def compute_loss_rawnerf_transient(rendering, gt, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute standard RawNeRF loss for transient data.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = ((rendering["rgb"] - gt) ** 2)
    return data_loss * scaling_grad


def compute_loss_rawnerf_transient_itof(rendering, gt, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute standard RawNeRF loss for transient data using iToF.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = render_utils.dtof_to_itof(rendering["rgb"] - gt, config.itof_frequency_phase_shifts, config.exposure_time) ** 2
    return data_loss * scaling_grad


def compute_loss_rawnerf_transient_gauss(rendering, gt, config, clip_val=10000.0, exponent=1.0, eps=1e-3):
    """Compute standard RawNeRF loss for transient data using Gaussian blur.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        exponent: Exponent for scaling gradient.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The computed loss.
    """
    rgb_clip = _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val)

    scaling_grad = 1.0 / (jnp.power(jax.lax.stop_gradient(rgb_clip.sum(-2)[..., None, :]), exponent) + eps)
    data_loss = render_utils.dtof_to_gauss(rendering["rgb"] - gt, config.transient_gauss_sigma_scales, config.transient_gauss_constant_scale) ** 2
    return data_loss * scaling_grad


def compute_loss_charb(rendering, gt, config):
    """Compute Charbonnier loss.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        
    Returns:
        The computed loss.
    """
    data_loss = jnp.sqrt(
        (rendering["rgb"] - gt) ** 2
        + config.charb_padding**2
    )

    return data_loss


def _get_rgb_clip_for_rawnerf(rendering, gt, config, clip_val):
    """Helper function to get clipped RGB values for RawNeRF loss.
    
    Args:
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        config: Configuration object.
        clip_val: Value to clip RGB values.
        
    Returns:
        Clipped RGB values.
    """
    if config.use_gt_rawnerf:
        rgb_clip = jnp.clip(gt, 0.0, clip_val)
    else:
        if "cache_rgb" in rendering:
            rgb_clip = jnp.clip(rendering["cache_rgb"], 0.0, clip_val)
        else: 
            rgb_clip = jnp.clip(rendering["rgb"], 0.0, clip_val)

        if config.use_combined_rawnerf:
            rgb_clip = jnp.clip(jnp.maximum(rgb_clip, gt), 0.0, clip_val)

    if config.use_norm_rawnerf:
        rgb_clip = jnp.linalg.norm(rgb_clip, axis=-1, keepdims=True)
        
    return rgb_clip


# -----------------------------------------------------------------------------
# Data Loss Functions
# -----------------------------------------------------------------------------

def compute_data_loss(batch, rendering, rays, config, main=False, train_frac=1.0):
    """Computes data loss terms for RGB, normal, and depth outputs.
    
    Args:
        batch: Batch of data.
        rendering: Dictionary containing rendered outputs.
        rays: Ray data.
        config: Configuration object.
        main: Whether this is the main render.
        train_frac: Training fraction.
        
    Returns:
        Tuple of (loss, stats).
    """
    data_losses = []
    stats = collections.defaultdict(lambda: [])

    # lossmult can be used to apply a weight to each ray in the batch.
    # For example: masking out rays, applying the Bayer mosaic mask, upweighting
    # rays from lower resolution images and so on.
    lossmult = rays.lossmult
    lossmult = jnp.broadcast_to(lossmult, batch.rgb[..., :3].shape)

    extra_ray_regularizer_outgoing = (
        config.extra_ray_regularizer
        and config.extra_ray_type == "outgoing"
    )

    if extra_ray_regularizer_outgoing and main:
        lossmult = jnp.concatenate(
            [
                lossmult[:lossmult.shape[0] // 2],
                jnp.zeros_like(lossmult[:lossmult.shape[0] // 2]),
            ],
            axis=0,
        )

    if config.convert_srgb:
        rendering = utils.copy_tree(rendering)
        rendering["rgb"] = image.linear_to_srgb(rendering["rgb"])
        batch = batch.replace(rgb=image.linear_to_srgb(batch.rgb[..., :3]))
    
    if batch.masks is not None:
        masks = batch.masks
    else:
        masks = jnp.ones_like(lossmult)

    if config.mask_lossmult or ("unbiased" in config.data_loss_type):
        lossmult = lossmult * masks

        if ("unbiased" not in config.data_loss_type):
            lossmult += lossmult * (1.0 - masks) * config.mask_lossmult_weight

    # Color loss
    if rendering["rgb"] is None:
        mse = -1.0
        sub_data_loss = 0
    else:
        gt = batch.rgb[..., :3]
        rendering = utils.copy_tree(rendering)

        # Clip loss
        if main and config.use_loss_clip and ("unbiased" not in config.data_loss_type):
            rendering["rgb"] = jnp.clip(rendering["rgb"], config.loss_clip_min, config.loss_clip)

            if "rgb_nocorr" in rendering:
                rendering["rgb_nocorr"] = jnp.clip(rendering["rgb_nocorr"], config.loss_clip_min, config.loss_clip)

            if "gt_nocorr" in rendering:
                rendering["gt_nocorr"] = jnp.clip(rendering["gt_nocorr"], config.loss_clip_min, config.loss_clip)

            gt = jnp.clip(gt, config.loss_clip_min, config.loss_clip)

        # Zero loss
        lossmult = jnp.where(
            gt > config.loss_thresh,
            jnp.zeros_like(lossmult),
            lossmult
        )

        # Compute loss
        if config.clip_eval:
            resid_sq = (jnp.clip(rendering["rgb"], 0.0, 1.0) - jnp.clip(gt, 0.0, 1.0)) ** 2
        else:
            resid_sq = (rendering["rgb"] - gt) ** 2

        if extra_ray_regularizer_outgoing and main:
            mse = (masks * lossmult * resid_sq)[:lossmult.shape[0] // 2].mean()
        else:
            mse = (masks * lossmult * resid_sq).mean()

        data_loss = 0

        if "rgb_nocorr" not in rendering:
            rendering["rgb_nocorr"] = rendering["rgb"]

        if "gt_nocorr" in rendering:
            gt_nocorr = rendering["gt_nocorr"]
        else:
            gt_nocorr = gt
        
        if config.is_material:
            rawnerf_exponent = config.rawnerf_exponent_material
            rawnerf_eps = config.rawnerf_eps_material
        else:
            rawnerf_exponent = config.rawnerf_exponent
            rawnerf_eps = config.rawnerf_eps

        data_loss = _select_data_loss_function(config, rendering, gt, gt_nocorr, 
                                             rawnerf_eps, rawnerf_exponent)

        if "bg_noise" in rendering:
            data_loss += (rendering["bg_noise"] ** 2) * masks

        sub_data_loss = (lossmult * data_loss).mean()

    data_losses.append(sub_data_loss)
    stats["mses"].append(mse * config.data_loss_mult)

    # Additional stats
    _compute_additional_stats(config, rendering, batch, stats)

    data_losses = jnp.array(data_losses)
    loss = jnp.sum(data_losses)

    stats = {k: jnp.array(stats[k]) for k in stats}
    return loss, stats


def compute_transient_data_loss(batch, rendering, rays, config, main=False, train_frac=1.0):
    """Computes data loss terms for transient RGB outputs.
    
    Args:
        batch: Batch of data.
        rendering: Dictionary containing rendered outputs.
        rays: Ray data.
        config: Configuration object.
        main: Whether this is the main render.
        train_frac: Training fraction.
        
    Returns:
        Tuple of (loss, stats).
    """
    data_losses = []
    stats = collections.defaultdict(lambda: [])
    lossmult = rays.lossmult

    if config.convert_srgb:
        rendering = utils.copy_tree(rendering)
        rendering["rgb"] = image.linear_to_srgb(rendering["rgb"])
        batch = batch.replace(rgb=image.linear_to_srgb(batch.rgb[..., :3]))
    
    if batch.masks is not None:
        masks = batch.masks
    else:
        masks = jnp.ones_like(lossmult)

    if config.mask_lossmult or ("unbiased" in config.data_loss_type):
        lossmult = lossmult * masks

        if ("unbiased" not in config.data_loss_type):
            lossmult += lossmult * (1.0 - masks) * config.mask_lossmult_weight

    # Color loss
    if rendering["rgb"] is None:
        mse = -1.0
        sub_data_loss = 0
    else:
        gt = batch.rgb[..., :3]
        rendering = utils.copy_tree(rendering)

        # Clip loss
        if main and config.use_loss_clip and ("unbiased" not in config.data_loss_type):
            rendering["rgb"] = jnp.clip(rendering["rgb"], config.loss_clip_min, config.loss_clip)

            if "rgb_nocorr" in rendering:
                rendering["rgb_nocorr"] = jnp.clip(rendering["rgb_nocorr"], config.loss_clip_min, config.loss_clip)

            if "gt_nocorr" in rendering:
                rendering["gt_nocorr"] = jnp.clip(rendering["gt_nocorr"], config.loss_clip_min, config.loss_clip)

            gt = jnp.clip(gt, config.loss_clip_min, config.loss_clip)

        # Zero loss
        lossmult = jnp.where(
            (gt > config.loss_thresh).sum(-2),
            jnp.zeros_like(lossmult),
            lossmult
        )

        # Compute loss
        if config.clip_eval:
            resid_sq = (jnp.clip(rendering["rgb"], 0.0, 1.0) - jnp.clip(gt, 0.0, 1.0)) ** 2
        elif config.use_itof:
            resid_sq = (render_utils.dtof_to_itof(rendering["rgb"] - gt, config.itof_frequency_phase_shifts, config.exposure_time) ** 2)
            resid_sq = resid_sq / resid_sq.shape[-2]
        else:
            resid_sq = ((rendering["rgb"] - gt) ** 2)

        mse = (masks[..., None, :] * lossmult[..., None, :] * resid_sq).sum(axis=-2).mean()
        data_loss = 0

        if "rgb_nocorr" not in rendering:
            rendering["rgb_nocorr"] = rendering["rgb"]

        if "gt_nocorr" in rendering:
            gt_nocorr = rendering["gt_nocorr"]
        else:
            gt_nocorr = gt

        if config.is_material:
            rawnerf_exponent = config.rawnerf_exponent_material
            rawnerf_eps = config.rawnerf_eps_material
        else:
            rawnerf_exponent = config.rawnerf_exponent
            rawnerf_eps = config.rawnerf_eps

        data_loss = _select_transient_data_loss_function(config, rendering, gt, gt_nocorr, 
                                                      rawnerf_eps, rawnerf_exponent)

        if config.use_itof:
            data_loss = data_loss / data_loss.shape[-2]

        if "bg_noise" in rendering:
            data_loss += (rendering["bg_noise"] ** 2) * masks

        sub_data_loss = (lossmult[..., None, :] * data_loss).sum(axis=-2).mean()

    data_losses.append(sub_data_loss)
    stats["mses"].append(mse * config.data_loss_mult)

    # Additional stats
    _compute_additional_stats(config, rendering, batch, stats)

    data_losses = jnp.array(data_losses)
    loss = jnp.sum(data_losses)

    stats = {k: jnp.array(stats[k]) for k in stats}
    return loss, stats


def _select_data_loss_function(config, rendering, gt, gt_nocorr, rawnerf_eps, rawnerf_exponent):
    """Select the appropriate data loss function based on configuration.
    
    Args:
        config: Configuration object.
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        rawnerf_eps: Epsilon value for RawNeRF loss.
        rawnerf_exponent: Exponent value for RawNeRF loss.
        
    Returns:
        The computed loss.
    """
    if config.data_loss_type == "mse":
        # Mean-squared error (L2) loss.
        return (rendering["rgb"] - gt) ** 2
    elif config.data_loss_type == "mse_unbiased":
        return compute_unbiased_loss(rendering, gt, gt_nocorr, config)
    elif config.data_loss_type == "rawnerf":
        return compute_loss_rawnerf(rendering, gt, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_unbiased":
        return compute_unbiased_loss_rawnerf(rendering, gt, gt_nocorr, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_transient":
        return compute_loss_rawnerf(rendering, gt, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_transient_unbiased":
        return compute_unbiased_loss_rawnerf(rendering, gt, gt_nocorr, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_charb":
        data_loss = compute_loss_rawnerf(rendering, gt, config, exponent=2.0, eps=rawnerf_eps) ** 2
        return jnp.sqrt(data_loss + config.charb_padding**2)
    elif config.data_loss_type == "charb":
        # Charbonnier loss.
        return jnp.sqrt((rendering["rgb"] - gt) ** 2 + config.charb_padding**2)
    elif config.data_loss_type == "charb_clip":
        # Charbonnier loss with clipping.
        rgb_render_clip = jnp.minimum(1.0, rendering["rgb"])
        rgb_gt_clip = jnp.minimum(1.0, gt)
        resid_sq_clip = (rgb_render_clip - rgb_gt_clip) ** 2
        return jnp.sqrt(resid_sq_clip + config.charb_padding**2)
    else:
        raise ValueError(f"Unknown data loss type: {config.data_loss_type}")


def _select_transient_data_loss_function(config, rendering, gt, gt_nocorr, rawnerf_eps, rawnerf_exponent):
    """Select the appropriate transient data loss function based on configuration.
    
    Args:
        config: Configuration object.
        rendering: Dictionary containing rendered outputs.
        gt: Ground truth RGB values.
        gt_nocorr: Ground truth RGB values without corrections.
        rawnerf_eps: Epsilon value for RawNeRF loss.
        rawnerf_exponent: Exponent value for RawNeRF loss.
        
    Returns:
        The computed loss.
    """
    if config.data_loss_type == "mse":
        # Mean-squared error (L2) loss.
        return (rendering["rgb"] - gt) ** 2
    elif config.data_loss_type == "mse_itof":
        # Mean-squared error (L2) loss with iToF.
        return render_utils.dtof_to_itof(rendering["rgb"] - gt, config.itof_frequency_phase_shifts, config.exposure_time) ** 2
    elif config.data_loss_type == "mse_fwp":
        # Mean-squared error (L2) loss with frequency-weighted power.
        return ((rendering["rgb"] + 1e-5)**(1.0/2.0) - (gt + 1e-5)**(1.0/2.0)) ** 2
    elif config.data_loss_type == "mse_unbiased":
        return compute_unbiased_loss(rendering, gt, gt_nocorr, config)
    elif config.data_loss_type == "mse_itof_unbiased":
        return compute_unbiased_loss_itof(rendering, gt, gt_nocorr, config)
    elif config.data_loss_type == "rawnerf":
        return compute_loss_rawnerf(rendering, gt, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_unbiased":
        return compute_unbiased_loss_rawnerf(rendering, gt, gt_nocorr, config, eps=rawnerf_eps, exponent=rawnerf_exponent)
    elif config.data_loss_type == "rawnerf_transient":
        data_loss = compute_loss_rawnerf_transient(rendering, gt, config, exponent=rawnerf_exponent, eps=rawnerf_eps)

        data_loss_gauss = compute_loss_rawnerf_transient_gauss(
            rendering, gt, config, exponent=rawnerf_exponent, eps=rawnerf_eps
        ) * config.data_loss_gauss_mult
        data_loss_gauss = data_loss_gauss / data_loss.shape[-2]
        return data_loss + data_loss_gauss.sum(axis=-2, keepdims=True)
    elif config.data_loss_type == "rawnerf_transient_unbiased":
        data_loss = compute_unbiased_loss_rawnerf_transient(rendering, gt, gt_nocorr, config, exponent=rawnerf_exponent, eps=rawnerf_eps)

        data_loss_gauss = compute_unbiased_loss_rawnerf_transient_gauss(
            rendering, gt, gt_nocorr, config, exponent=rawnerf_exponent, eps=rawnerf_eps
        ) * config.data_loss_gauss_mult
        data_loss_gauss = data_loss_gauss / data_loss.shape[-2]
        return data_loss + data_loss_gauss.sum(axis=-2, keepdims=True)
    elif config.data_loss_type == "rawnerf_transient_itof":
        return compute_loss_rawnerf_transient_itof(rendering, gt, config, exponent=rawnerf_exponent, eps=rawnerf_eps)
    elif config.data_loss_type == "rawnerf_transient_itof_unbiased":
        return compute_unbiased_loss_rawnerf_transient_itof(rendering, gt, gt_nocorr, config, exponent=rawnerf_exponent, eps=rawnerf_eps)
    elif config.data_loss_type == "rawnerf_charb":
        data_loss = compute_loss_rawnerf(rendering, gt, config, exponent=2.0, eps=1e-4) ** 2
        return jnp.sqrt(data_loss + config.charb_padding**2)
    elif config.data_loss_type == "charb":
        # Charbonnier loss.
        return jnp.sqrt((rendering["rgb"] - gt) ** 2 + config.charb_padding**2)
    elif config.data_loss_type == "charb_clip":
        # Charbonnier loss with clipping.
        rgb_render_clip = jnp.minimum(1.0, rendering["rgb"])
        rgb_gt_clip = jnp.minimum(1.0, gt)
        resid_sq_clip = (rgb_render_clip - rgb_gt_clip) ** 2
        return jnp.sqrt(resid_sq_clip + config.charb_padding**2)
    else:
        raise ValueError(f"Unknown data loss type: {config.data_loss_type}")


def _compute_additional_stats(config, rendering, batch, stats):
    """Compute additional statistics for evaluation.
    
    Args:
        config: Configuration object.
        rendering: Dictionary containing rendered outputs.
        batch: Batch of data.
        stats: Dictionary to store statistics.
    """
    if config.compute_disp_metrics:
        # Using mean to compute disparity, but other distance statistics can
        # be used instead.
        disp = 1 / (1 + rendering["distance_mean"])
        stats["disparity_mses"].append(((disp - batch.disps) ** 2).mean())

    if config.compute_normal_metrics and (not hasattr(config, 'vis_only') or config.vis_only):
        if "normals" in rendering:
            weights = rendering["acc"] * batch.alphas
            normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
            normalized_normals = ref_utils.l2_normalize(rendering["normals"])
            normal_mae = ref_utils.compute_weighted_mae(weights, normalized_normals, normalized_normals_gt)
        else:
            # If normals are not computed, set MAE to -1.
            normal_mae = -1.0

        stats["normal_maes"].append(normal_mae)


# -----------------------------------------------------------------------------
# Other Loss Functions
# -----------------------------------------------------------------------------

def compute_mask_loss(batch, rendering, rays, config, train_frac=1.0, empty_loss_weight=None):
    """Computes mask loss terms.
    
    Args:
        batch: Batch of data.
        rendering: Dictionary containing rendered outputs.
        rays: Ray data.
        config: Configuration object.
        train_frac: Training fraction.
        empty_loss_weight: Optional weight for empty regions.
        
    Returns:
        The computed mask loss.
    """
    lossmult = rays.lossmult

    if batch.masks is not None:
        masks = batch.masks
    else:
        masks = jnp.ones_like(lossmult)

    if (
        "acc" in rendering
        and rendering["acc"] is not None
    ):
        # Apply weight decay and ease-in if configured
        mask_loss_weight_decay = _compute_mask_weight_decay(config, train_frac)
        mask_loss_weight_ease = _compute_mask_weight_ease(config, train_frac)

        # Compute Charbonnier loss
        data_loss = jnp.sqrt(
            (rendering["acc"][..., None] - masks) ** 2
            + config.charb_padding ** 2
        ) * mask_loss_weight_decay * mask_loss_weight_ease

        # Apply appropriate weights for opaque vs. empty regions
        if empty_loss_weight is not None:
            data_loss = jnp.where(
                masks > 0.5,
                data_loss * 0.0,
                data_loss * empty_loss_weight,
            )
        else:
            data_loss = jnp.where(
                masks > 0.5,
                data_loss * config.opaque_loss_weight,
                data_loss * config.empty_loss_weight,
            )
    else:
        data_loss = jnp.zeros_like(masks)

    return jnp.mean(lossmult * data_loss)


def compute_weight_ease_in(
    train_frac: float,
    use_weight_schedule: bool,
    start_frac: float,
    transition_frac: float,
    min_value: float = 0.0
) -> float:
    """Compute a weight that eases in from a minimum value to 1.0 over training.
    
    Args:
        train_frac: Current training fraction (0.0 to 1.0).
        use_weight_schedule: Whether to use weight scheduling.
        start_frac: Fraction of training at which to start the transition.
        transition_frac: Length of the transition as a fraction of training.
        min_value: Minimum weight value at the start.
        
    Returns:
        Weight factor between min_value and 1.0.
    """
    if not use_weight_schedule:
        return 1.0
        
    if transition_frac > 0:
        # Linear interpolation from min_value to 1.0
        w = jnp.clip((train_frac - start_frac) / transition_frac, 0.0, 1.0)
        return min_value * (1.0 - w) + w
    else:
        # Step function
        return jnp.float32(train_frac >= start_frac)


def compute_weight_decay(
    train_frac: float,
    use_weight_schedule: bool,
    start_frac: float,
    transition_frac: float,
    min_value: float = 0.0
) -> float:
    """Compute a weight that decays from 1.0 to a minimum value over training.
    
    Args:
        train_frac: Current training fraction (0.0 to 1.0).
        use_weight_schedule: Whether to use weight scheduling.
        start_frac: Fraction of training at which to start the decay.
        transition_frac: Length of the transition as a fraction of training.
        min_value: Minimum weight value at the end.
        
    Returns:
        Weight factor between min_value and 1.0.
    """
    if not use_weight_schedule:
        return 1.0
        
    # Linear interpolation from 1.0 to min_value
    w = jnp.clip((train_frac - start_frac) / transition_frac, 0.0, 1.0)
    return min_value * w + (1.0 - w)


def _compute_mask_weight_decay(config, train_frac):
    """Compute mask weight decay based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight decay factor.
    """
    return compute_weight_decay(
        train_frac=train_frac,
        use_weight_schedule=config.use_mask_weight_decay,
        start_frac=config.mask_weight_decay_start,
        transition_frac=config.mask_weight_decay_frac,
        min_value=config.mask_weight_decay_min
    )


def _compute_mask_weight_ease(config, train_frac):
    """Compute mask weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    return compute_weight_ease_in(
        train_frac=train_frac,
        use_weight_schedule=config.use_mask_weight_ease,
        start_frac=config.mask_weight_ease_start,
        transition_frac=config.mask_weight_ease_frac,
        min_value=config.mask_weight_ease_min
    )


def compute_semantic_loss(batch, renderings, rays, config, stats):
    """Computes semantic loss terms for semantic outputs.
    
    Args:
        batch: Batch of data.
        renderings: Dictionary containing rendered outputs.
        rays: Ray data.
        config: Configuration object.
        stats: Dictionary to store statistics.
        
    Returns:
        The computed semantic loss.
    """
    loss, aux = loss_utils.semantic_loss(
        batch,
        renderings,
        rays,
        coarse_mult=config.semantic_coarse_loss_mult,
        mult=config.semantic_loss_mult,
    )
    stats.update(aux)
    return loss


def interlevel_loss(ray_history, loss_mults, loss_blurs, config):
    """Computes the interlevel loss defined in mip-NeRF 360.
    
    Args:
        ray_history: History of ray samples.
        loss_mults: Loss multipliers.
        loss_blurs: Loss blur factors.
        config: Configuration object.
        
    Returns:
        The computed interlevel loss.
    """
    if config.use_spline_interlevel_loss:
        return loss_utils.spline_interlevel_loss(
            ray_history,
            mults=loss_mults,
            blurs=loss_blurs,
        )
    else:
        return loss_utils.interlevel_loss(ray_history, mults=loss_mults)


def distortion_loss(ray_history, distortion_loss_mult, config):
    """Computes the distortion loss.
    
    Args:
        ray_history: History of ray samples.
        distortion_loss_mult: Loss multiplier.
        config: Configuration object.
        
    Returns:
        The computed distortion loss.
    """
    if config.distortion_loss_curve_fn is None:
        curve_fn = lambda x: x
    else:
        curve_fn, kwargs = config.distortion_loss_curve_fn
        curve_fn = functools.partial(curve_fn, **kwargs)
    return loss_utils.distortion_loss(
        ray_history,
        target=config.distortion_loss_target,
        mult=distortion_loss_mult,
        curve_fn=curve_fn,
        normalize=config.normalize_distortion_loss,
    )


def patch_loss(batch, renderings, config):
    """Computes a smoothing regularizer over output depth patches.
    
    Args:
        batch: Batch of data.
        renderings: Dictionary containing rendered outputs.
        config: Configuration object.
        
    Returns:
        The computed patch loss.
    """
    return loss_utils.patch_loss(
        batch,
        renderings,
        charb_padding=config.charb_padding,
        bilateral_strength=config.bilateral_strength,
        patch_variance_weighting=config.patch_variance_weighting,
        mult=config.patch_loss_mult,
    )


def orientation_loss(rays, ray_results, config):
    """Computes the orientation loss regularizer defined in ref-NeRF.
    
    Args:
        rays: Ray data.
        ray_results: Ray rendering results.
        config: Configuration object.
        
    Returns:
        The computed orientation loss.
    """
    return loss_utils.orientation_loss(
        rays,
        ray_results,
        target=config.orientation_loss_target,
        coarse_mult=config.orientation_coarse_loss_mult,
        mult=config.orientation_loss_mult,
        normalize=config.orientation_loss_normalize,
        stopgrad=config.orientation_loss_stopgrad,
    )


def predicted_normal_loss(ray_results, beta, config):
    """Computes the predicted normal supervision loss defined in ref-NeRF.
    
    Args:
        ray_results: Ray rendering results.
        beta: Beta parameter.
        config: Configuration object.
        
    Returns:
        The computed predicted normal loss.
    """
    return loss_utils.predicted_normal_loss(
        ray_results,
        beta,
        coarse_mult=config.predicted_normal_coarse_loss_mult,
        mult=config.predicted_normal_loss_mult,
        gt="normals_pred",
        pred="normals",
        normalize=config.predicted_normal_loss_normalize,
        stopgrad=config.predicted_normal_loss_stopgrad,
        stopgrad_weight=config.predicted_normal_loss_stopgrad_weight,
    )


def predicted_normal_reverse_loss(ray_results, beta, config):
    """Computes the reverse predicted normal supervision loss.
    
    Args:
        ray_results: Ray rendering results.
        beta: Beta parameter.
        config: Configuration object.
        
    Returns:
        The computed reverse predicted normal loss.
    """
    return loss_utils.predicted_normal_loss(
        ray_results,
        beta,
        coarse_mult=config.predicted_normal_coarse_loss_mult,
        mult=config.predicted_normal_reverse_loss_mult,
        gt="normals",
        pred="normals_pred",
        normalize=config.predicted_normal_loss_normalize,
        stopgrad=True,
    )


def predicted_normal_weight_loss(ray_results, beta, config):
    """Computes the weighted predicted normal supervision loss.
    
    Args:
        ray_results: Ray rendering results.
        beta: Beta parameter.
        config: Configuration object.
        
    Returns:
        The computed weighted predicted normal loss.
    """
    return loss_utils.predicted_normal_loss(
        ray_results,
        beta,
        coarse_mult=config.predicted_normal_coarse_loss_mult,
        mult=config.predicted_normal_weight_loss_mult,
        gt="normals",
        pred="normals_pred",
        normalize=config.predicted_normal_loss_normalize,
        stopgrad=False,
    )


def exposure_prediction_bounds_loss(predicted_exposure: jnp.ndarray, dataset: datasets.Dataset) -> jnp.ndarray:
    """Compute loss for staying in bounds of the dataset exposures.
    
    Args:
        predicted_exposure: Predicted exposure values.
        dataset: Dataset containing reference exposures.
        
    Returns:
        The computed exposure bounds loss.
    """
    min_exp = np.min(np.log(dataset.exposures))
    max_exp = np.max(np.log(dataset.exposures))

    exp_pred = jnp.log(predicted_exposure)

    min_loss = jnp.maximum(0, min_exp - exp_pred) ** 2
    max_loss = jnp.maximum(0, exp_pred - max_exp) ** 2
    loss_bounds = jnp.mean(min_loss + max_loss)

    return loss_bounds


def exposure_prediction_loss(
    rays: utils.Rays,
    renderings: list[dict[str, jnp.ndarray]],
    config: configs.Config,
    dataset: datasets.Dataset,
) -> jnp.ndarray:
    """Compute loss for exposure prediction for each ray.
    
    Args:
        rays: Ray data.
        renderings: List of dictionaries containing rendered outputs.
        config: Configuration object.
        dataset: Dataset containing reference exposures.
        
    Returns:
        The computed exposure prediction loss.
    """
    predicted_exposure = renderings[-1]["exposure_prediction"]
    target_exposure = rays.exposure_values
    exposure_residuals = (predicted_exposure - target_exposure) ** 2
    loss = config.exposure_prediction_loss_mult * jnp.mean(exposure_residuals)
    if config.exposure_prediction_bounds_loss_mult > 0:
        loss += config.exposure_prediction_bounds_loss_mult * exposure_prediction_bounds_loss(
            predicted_exposure, dataset
        )
    return loss


def param_regularizer_loss(variables, config, train_frac):
    """Computes regularizer loss(es) over optimized parameters.
    
    Args:
        variables: Model variables.
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Dictionary of computed regularization losses.
    """
    # Do some bookkeeping to ensure that every regularizer is valid.
    reg_used = {k: False for k in config.param_regularizers}
    params_flat = flax.traverse_util.flatten_dict(variables["params"])
    losses = {k: 0.0 for k in config.param_regularizers.keys()}

    # Calculate material loss weight ease-in if configured
    material_loss_weight_ease = _compute_material_weight_ease(config, train_frac)

    for name_tuple, param in params_flat.items():
        name = "/".join(name_tuple)
        for prefix in config.param_regularizers:
            if 'material' in prefix:
                extra_mult = material_loss_weight_ease
            else:
                extra_mult = 1.0

            if name.startswith(prefix) or prefix in name_tuple:
                reg_used[prefix] = True
                mult, acc_fn, alpha, scale = config.param_regularizers[prefix]
                if (alpha == 2) and (scale == 1):
                    # Special-casing this common setting gives a small speedup and much
                    # faster compilation times.
                    losses[prefix] += mult * 0.5 * acc_fn(param**2) * extra_mult
                else:
                    losses[prefix] += mult * acc_fn(general.lossfun(param, alpha, scale)) * extra_mult

                if not config.disable_pmap_and_jit:
                    print("Regularizing " + f"{mult}*{acc_fn.__name__}(lossfun{(name, alpha, scale)})")

    # If some regularizer was not used, the gin config is probably wrong.
    for reg, used in reg_used.items():
        if not used:
            print(f"Regularizer {reg} not used.")

    return losses


def _compute_material_weight_ease(config, train_frac):
    """Compute material weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    if config.use_material_weight_ease:
        if config.material_weight_ease_frac > 0:
            w = jnp.clip((train_frac - config.material_weight_ease_start) / config.material_weight_ease_frac, 0.0, 1.0)
            return config.material_weight_ease_min * (1.0 - w) + w
        else:
            return ((train_frac - config.material_weight_ease_start) >= 0.0).astype(jnp.float32)
    else:
        return 1.0


def eikonal_equation(n, eps=jnp.finfo(jnp.float32).tiny):
    """Compute the satisfactory of the normals n with the eikonal equations.
    
    Args:
        n: Normal vectors.
        eps: Small epsilon for numerical stability.
        
    Returns:
        The eikonal loss.
    """
    norm = jnp.sqrt(jnp.maximum(jnp.sum(n**2, axis=-1), eps))
    return jnp.mean((norm - 1.0) ** 2.0)


def eikonal_loss(ray_history, config):
    """Computes the eikonal normal regularization loss defined in VolSDF.
    
    Args:
        ray_history: History of ray samples.
        config: Configuration object.
        
    Returns:
        The computed eikonal loss.
    """
    total_loss = 0.0
    for i, ray_results in enumerate(ray_history):
        n = ray_results["normals"]
        if n is None:
            raise ValueError("Gradient normals cannot be None if eikonal loss is on.")
        loss = eikonal_equation(n)
        if i < len(ray_history) - 1:
            total_loss += config.eikonal_coarse_loss_mult * loss
        else:
            total_loss += config.eikonal_loss_mult * loss
    return total_loss


def clip_gradients(grad, config):
    """Clips gradients of each MLP individually based on norm and max value.
    
    Args:
        grad: Gradients to clip.
        config: Configuration object.
        
    Returns:
        Clipped gradients.
    """
    # Clip the gradients of each MLP individually.
    grad_clipped = flax.core.unfreeze(grad)
    for k, g in grad["params"].items():
        # Clip by value.
        if config.grad_max_val > 0:
            g = jax.tree_util.tree_map(lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), g)

        # Then clip by norm.
        if config.grad_max_norm > 0:
            mult = jnp.minimum(1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + tree_norm(g)))
            g = jax.tree_util.tree_map(lambda z: mult * z, g)  # pylint:disable=cell-var-from-loop

        grad_clipped["params"][k] = g
    grad = type(grad)(grad_clipped)
    return grad


# -----------------------------------------------------------------------------
# Material and Geometry Loss Functions
# -----------------------------------------------------------------------------

def extra_ray_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute loss for extra rays.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed extra ray loss.
    """
    # Get extra rays
    rng, cur_key = random.split(cur_key)
    extra_rays = render_utils.get_outgoing_rays(
        rng,
        rays,
        jax.lax.stop_gradient(rays.viewdirs),
        jax.lax.stop_gradient(
            model_results["shader"][config.material_normals_target]
        ),
        {},
        random_generator_2d=model.random_generator_2d,
        use_mis=False,
        samplers=model.uniform_importance_samplers,
        num_secondary_samples=1,
    )

    # Render extra rays
    rng, cur_key = random.split(cur_key)
    extra_results = model.apply(
        variables,
        rng,
        extra_rays,
        train_frac=kwargs["train_frac"],
        compute_extras=False,
        mesh=kwargs["mesh"],
        env_map=kwargs["env_map"],
        env_map_pmf=kwargs["env_map_pmf"],
        env_map_pdf=kwargs["env_map_pdf"],
        env_map_dirs=kwargs["env_map_dirs"],
        env_map_w=kwargs["env_map_w"],
        env_map_h=kwargs["env_map_h"],
    )

    rng, cur_key = random.split(cur_key)
    extra_results_nocorr = model.apply(
        variables,
        rng,
        extra_rays,
        train_frac=kwargs["train_frac"],
        compute_extras=False,
        mesh=kwargs["mesh"],
        env_map=kwargs["env_map"],
        env_map_pmf=kwargs["env_map_pmf"],
        env_map_pdf=kwargs["env_map_pdf"],
        env_map_dirs=kwargs["env_map_dirs"],
        env_map_w=kwargs["env_map_w"],
        env_map_h=kwargs["env_map_h"],
        cache_outputs=extra_results["cache_main"],
    )

    # Create extra batch
    rgb_gt = utils.stopgrad_with_weight(
        extra_results["integrator"]["cache_rgb"],
        config.extra_ray_loss_stopgrad_weight_gt,
    )
    rgb_gt_nocorr = utils.stopgrad_with_weight(
        extra_results_nocorr["integrator"]["cache_rgb"],
        config.extra_ray_loss_stopgrad_weight_gt,
    )
    rgb = utils.stopgrad_with_weight(
        extra_results["integrator"]["material_rgb"].reshape(rgb_gt.shape),
        config.extra_ray_loss_stopgrad_weight_pred,
    )
    rgb_nocorr = utils.stopgrad_with_weight(
        extra_results_nocorr["integrator"]["material_rgb"].reshape(rgb_gt.shape),
        config.extra_ray_loss_stopgrad_weight_pred,
    )

    pred_outputs = {
        "rgb": rgb,
        "rgb_nocorr": rgb_nocorr,
        "cache_rgb": rgb_gt,
    }

    if "rawnerf" in config.data_loss_type:
        data_loss = compute_unbiased_loss_rawnerf(
            pred_outputs,
            rgb_gt,
            rgb_gt_nocorr,
            config,
        ).mean()
    else:
        data_loss = compute_unbiased_loss(
            pred_outputs,
            rgb_gt,
            rgb_gt_nocorr,
            config,
        ).mean()

    return data_loss


def maximum_radiance_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute maximum radiance loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed maximum radiance loss.
    """
    diff = jnp.maximum(
        model_results["shader"]["rgb"] - batch.rgb[..., None, :],
        0.0
    )

    return jnp.square(diff).mean(axis=-1).mean()


def normalize_weight_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute weight normalization loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed weight normalization loss.
    """
    if "geometry" not in model_results or config.normalize_weight_loss == 0.0:
        return 0.0

    geometry_results = model_results["geometry"]

    diff = jnp.abs(
        geometry_results["weights_original"]
        - jax.lax.stop_gradient(geometry_results["weights_new"])
    )

    return (diff).mean(axis=-1).mean() * config.normalize_weight_loss


def emission_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute emission loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed emission loss.
    """
    if "lighting_emission" not in model_results["shader"]:
        return 0.0

    # Outputs
    emission = model_results["shader"]["lighting_emission"]
    cache_rgb = model_results["integrator"]["cache_rgb"]
    lossmult = rays.lossmult.reshape(emission.shape[:-2] + (-1, 1))

    zero_loss = (
        math.safe_sqrt(emission + 1e-5)
        / math.safe_sqrt(cache_rgb.reshape(emission.shape[:-2] + (-1, 3)) + 1e-3)
    ) * config.emission_zero_loss_mult * lossmult

    difference_loss = (
        jnp.square(emission - jax.lax.stop_gradient(emission))
    ) * config.emission_constant_loss_mult * lossmult

    # Loss
    if model_results["geometry"] is not None:
        weights = jax.lax.stop_gradient(
            model_results["geometry"]["weights"]
        )[..., None]
    else:
        weights = jnp.ones_like(zero_loss)

    return (
        + (zero_loss * weights).sum(axis=-2).mean()
        + (difference_loss * weights).sum(axis=-2).mean()
    )


def residual_albedo_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute residual albedo loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed residual albedo loss.
    """
    if "lighting_emission" not in model_results["shader"]:
        return 0.0

    # Outputs
    emission = model_results["shader"]["lighting_emission"]
    irradiance = model_results["shader"]["lighting_irradiance"]
    irradiance_nocorr = model_results["shader"]["lighting_irradiance_nocorr"]
    residual_albedo = model_results["shader"]["material_residual_albedo"]

    # Compute unbiased difference
    material_results = {
        "rgb": residual_albedo * jax.lax.stop_gradient(irradiance),
        "rgb_nocorr": residual_albedo * jax.lax.stop_gradient(irradiance_nocorr),
        "cache_rgb": jax.lax.stop_gradient(emission),
    }

    lossmult = rays.lossmult.reshape(emission.shape[:-2] + (-1, 1))

    if "rawnerf" in config.data_loss_type:
        diff = compute_unbiased_loss_rawnerf(
            material_results,
            jax.lax.stop_gradient(emission),
            jax.lax.stop_gradient(emission),
            config
        ) * lossmult
    else:
        diff = compute_unbiased_loss(
            material_results,
            jax.lax.stop_gradient(emission),
            jax.lax.stop_gradient(emission),
            config
        ) * lossmult

    # Loss
    if model_results["geometry"] is not None:
        weights = jax.lax.stop_gradient(
            model_results["geometry"]["weights"]
        )[..., None]
    else:
        weights = jnp.ones_like(diff)

    return (
        (diff * weights).sum(axis=-2).mean()
    )


def direct_indirect_consistency_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute cache consistency loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed cache consistency loss.
    """
    # Outputs
    loss = 0.0

    prefixes = list(
        zip(
            ["diffuse_rgb", "specular_rgb", "direct_rgb", "indirect_rgb"],
            ["diffuse_rgb", "specular_rgb", "direct_rgb", "indirect_rgb"],
        )
    )

    if config.extra_ray_regularizer:
        prefixes += list(
            zip(
                ["extra_diffuse_rgb", "extra_specular_rgb", "extra_direct_rgb", "extra_indirect_rgb"],
                ["extra_diffuse_rgb", "extra_specular_rgb", "extra_direct_rgb", "extra_indirect_rgb"]
            )
        )
        
    weights = model_results["shader"]["weights"]

    for (cache_prefix, material_prefix) in prefixes:
        rgb = model_results[config.cache_consistency_loss_mode][f"{material_prefix}"]
        rgb_nocorr = model_results[config.cache_consistency_loss_mode][f"{material_prefix}_nocorr"]
        weights = jnp.ones_like(rgb[..., :1])

        rgb_cache = model_results[f"{config.cache_consistency_loss_mode}"][f"cache_{cache_prefix}"].reshape(
            rgb.shape
        )
        rgb_cache_nocorr = model_results[f"cache_{config.cache_consistency_loss_mode}"][f"{cache_prefix}_nocorr"].reshape(
            rgb.shape
        )

        rgb = utils.stopgrad_with_weight(
            rgb,
            config.cache_consistency_stopgrad_weight_material,
        )
        rgb_nocorr = utils.stopgrad_with_weight(
            rgb_nocorr,
            config.cache_consistency_stopgrad_weight_material,
        )
        rgb_cache = utils.stopgrad_with_weight(
            rgb_cache,
            config.cache_consistency_stopgrad_weight_cache,
        )
        rgb_cache_nocorr = utils.stopgrad_with_weight(
            rgb_cache_nocorr,
            config.cache_consistency_stopgrad_weight_cache,
        )

        # Compute unbiased loss
        cur_results = {
            "rgb": rgb,
            "rgb_nocorr": rgb_nocorr,
            "cache_rgb": rgb_cache,
        }

        if not config.cache_consistency_use_integrated:
            cur_results["gt_nocorr"] = jax.lax.stop_gradient(jnp.nan_to_num(rgb_cache_nocorr))

        cur_batch = batch.replace(
            rgb=rgb_cache,
            masks=batch.masks.reshape(rgb.shape[:-1] + (1,)),
        )
        cur_rays = rays.replace(
            lossmult=(rays.lossmult.reshape(rgb.shape[:-1] + (1,)) * weights.reshape(rgb.shape[:-1] + (1,)))
        )

        cur_config = copy.deepcopy(config)
        cur_config.data_loss_type = config.cache_consistency_loss_type if "occ" not in cache_prefix else "mse_unbiased"
        cur_config.is_material = True

        cur_loss = compute_data_loss(
            cur_batch,
            cur_results,
            cur_rays,
            cur_config
        )[0]

        if "indirect" in cache_prefix:
            cur_loss = cur_loss * config.cache_consistency_indirect_weight
        elif "direct" in cache_prefix:
            cur_loss = cur_loss * config.cache_consistency_direct_weight
        
        loss += cur_loss

    return loss


def transient_direct_indirect_consistency_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute transient direct-indirect consistency loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed transient direct-indirect consistency loss.
    """
    # Outputs
    loss = 0.0

    if "shader" in config.cache_consistency_loss_mode:
        prefixes = list(
            zip(
                ["direct_rgb", "transient_indirect"],
                ["direct_rgb", "transient_indirect"],
            )
        )
    else:
        prefixes = list(
            zip(
                ["transient_direct_no_filter", "transient_indirect_no_filter"],
                ["transient_direct_no_filter", "transient_indirect_no_filter"],
            )
        )

    if config.extra_ray_regularizer:
        if "shader" in config.cache_consistency_loss_mode:
            prefixes += list(zip(
                ["extra_direct_rgb", "extra_transient_indirect"],
                ["extra_direct_rgb", "extra_transient_indirect"]
            ))
        else:
            prefixes += list(zip(
                ["extra_transient_direct_no_filter", "extra_transient_indirect_no_filter"],
                ["extra_transient_direct_no_filter", "extra_transient_indirect_no_filter"]
            ))
        
    weights = jax.lax.stop_gradient(model_results["shader"]["weights"])
    base_rgb = model_results[f"cache_{config.cache_consistency_loss_mode}"]["direct_rgb"]

    if "shader" in config.cache_consistency_loss_mode:
        base_shape = base_rgb.shape[:-2] + (-1, 3)
        base_denom_rgb = model_results["integrator"]["cache_rgb"][..., None, :, :] * jnp.ones_like(model_results["cache_shader"]["transient_indirect"])
        base_denom_rgb = jnp.maximum(base_denom_rgb, batch.rgb[..., None, :, :])
    elif "integrator" in config.cache_consistency_loss_mode:
        base_shape = base_rgb.shape
        base_denom_rgb = model_results["integrator"]["cache_rgb"]
        base_denom_rgb = jnp.maximum(base_denom_rgb, batch.rgb)
    else:
        return 0.0

    for (cache_prefix, material_prefix) in prefixes:
        rgb = model_results[config.cache_consistency_loss_mode][f"{material_prefix}"]
        rgb_nocorr = model_results[config.cache_consistency_loss_mode][f"{material_prefix}_nocorr"]

        rgb_cache = model_results[f"cache_{config.cache_consistency_loss_mode}"][f"{cache_prefix}"].reshape(
            rgb.shape
        )
        rgb_cache_nocorr = model_results[f"cache_{config.cache_consistency_loss_mode}"][f"{cache_prefix}_nocorr"].reshape(
            rgb.shape
        )
        weights = weights.reshape(base_rgb[..., :1].shape)

        rgb = utils.stopgrad_with_weight(
            rgb,
            config.cache_consistency_stopgrad_weight_material,
        )
        rgb_nocorr = utils.stopgrad_with_weight(
            rgb_nocorr,
            config.cache_consistency_stopgrad_weight_material,
        )
        rgb_cache = utils.stopgrad_with_weight(
            rgb_cache,
            config.cache_consistency_stopgrad_weight_cache,
        )
        rgb_cache_nocorr = utils.stopgrad_with_weight(
            rgb_cache_nocorr,
            config.cache_consistency_stopgrad_weight_cache,
        )

        # Compute unbiased loss
        if "transient" in cache_prefix and config.cache_consistency_use_total:
            denom_rgb = base_denom_rgb.reshape(rgb.shape)
        elif "direct_rgb" in cache_prefix and config.cache_consistency_use_total:
            denom_rgb = base_denom_rgb.reshape(rgb.shape[:-1] + (-1, rgb.shape[-1])).sum(axis=-2)
        elif "direct_rgb" in cache_prefix and config.cache_consistency_use_total:
            denom_rgb = base_denom_rgb.reshape(rgb.shape[:-1] + (-1, rgb.shape[-1])).sum(axis=-2)
        else:
            denom_rgb = rgb_cache

        cur_results = {
            "rgb": jnp.nan_to_num(rgb),
            "rgb_nocorr": jax.lax.stop_gradient(jnp.nan_to_num(rgb_nocorr)),
            "cache_rgb": jax.lax.stop_gradient(jnp.nan_to_num(denom_rgb)),
        }

        if not config.cache_consistency_use_integrated:
            cur_results["gt_nocorr"] = jax.lax.stop_gradient(jnp.nan_to_num(rgb_cache_nocorr))

        cur_batch = batch.replace(
            rgb=jnp.nan_to_num(rgb_cache),
            masks=batch.masks.reshape(base_shape[:-1] + (1,)),
        )

        if "shader" in config.cache_consistency_loss_mode:
            cur_rays = rays.replace(
                lossmult=(rays.lossmult.reshape(base_shape[:-1] + (1,)) * weights.reshape(base_shape[:-1] + (1,)))
            )
        else:
            cur_rays = rays.replace(
                lossmult=(rays.lossmult.reshape(base_shape[:-1] + (1,)))
            )

        cur_config = copy.deepcopy(config)
        cur_loss = 0.0

        if "transient" in cache_prefix:
            cur_config.data_loss_type = config.cache_consistency_loss_type
            cur_config.is_material = True

            if not config.cache_consistency_use_gauss:
                cur_config.data_loss_gauss_mult = 0.0

            cur_loss = compute_transient_data_loss(
                cur_batch,
                cur_results,
                cur_rays,
                cur_config
            )[0]
        else:
            cur_config.data_loss_type = config.cache_consistency_loss_type
            cur_config.is_material = True
            cur_loss = compute_data_loss(
                cur_batch,
                cur_results,
                cur_rays,
                cur_config
            )[0]
        
        if "indirect" in cache_prefix:
            cur_loss = cur_loss * config.cache_consistency_indirect_weight
        elif "direct" in cache_prefix:
            cur_loss = cur_loss * config.cache_consistency_direct_weight
        
        loss += cur_loss

    return loss


def transient_light_sampling_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute transient light sampling loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed transient light sampling loss.
    """
    if "light_sampler" not in model_results or model_results["light_sampler"] is None:
        return 0.0

    data_loss = 0.0
    multiplier = 1.0
    light_sampler_results = model_results["light_sampler"]

    for suffix in ["_indirect_diffuse", "_indirect_specular"]:
        extra_rays = model_results["shader"].get(f"ref_rays{suffix}", None)

        if extra_rays is None:
            multiplier = 2.0
            continue

        extra_ray_samples = model_results["shader"][f"ref_samples{suffix}"]

        extra_rays = jax.tree_util.tree_map(jax.lax.stop_gradient, extra_rays)
        extra_ray_samples = jax.tree_util.tree_map(jax.lax.stop_gradient, extra_ray_samples)
        function_vals = extra_ray_samples["radiance_in"].sum(-2)

        function_vals = jax.lax.stop_gradient(
            jnp.linalg.norm(
                function_vals, axis=-1
            )
        )

        # Viewdirs
        extra_viewdirs = jax.lax.stop_gradient(extra_rays.viewdirs.reshape(function_vals.shape + (3,)))

        # VMF parameters
        vmf_means = light_sampler_results["vmf_means"]
        vmf_kappas = light_sampler_results["vmf_kappas"]
        vmf_logits = light_sampler_results["vmf_logits"]
        vmf_normals = light_sampler_results["vmf_normals"]

        vmf_means = vmf_means.reshape(-1, vmf_means.shape[-2], 3)
        vmf_kappas = vmf_kappas.reshape(-1, vmf_kappas.shape[-2], 1)
        vmf_logits = vmf_logits.reshape(-1, vmf_logits.shape[-2], 1)
        vmf_normals = vmf_normals.reshape(-1, 3)

        # Loss
        lossmult = rays.lossmult.reshape(-1, 1, 1)
        lossmult = (
            lossmult * jnp.ones_like(function_vals.reshape(lossmult.shape[0], -1, 1))
        )
        lossmult = (lossmult / lossmult.shape[-2]).reshape(function_vals.shape)
        data_loss += render_utils.vmf_loss_fn(
            (vmf_means, vmf_kappas, vmf_logits),
            vmf_normals,
            extra_viewdirs,
            extra_ray_samples,
            function_vals,
            function_vals,
            lossmult,
            linear_to_srgb=config.light_sampling_linear_to_srgb,
        ) / 2.0
    
    return data_loss * multiplier


def light_sampling_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute light sampling loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed light sampling loss.
    """
    if "light_sampler" not in model_results or model_results["light_sampler"] is None:
        return 0.0

    data_loss = 0.0
    multiplier = 1.0
    light_sampler_results = model_results["light_sampler"]

    for suffix in ["_indirect_diffuse", "_indirect_specular"]:
        extra_rays = model_results["shader"][f"ref_rays{suffix}"]

        if extra_rays is None:
            multiplier = 2.0
            continue

        extra_ray_samples = model_results["shader"][f"ref_samples{suffix}"]

        extra_rays = jax.tree_util.tree_map(jax.lax.stop_gradient, extra_rays)
        extra_ray_samples = jax.tree_util.tree_map(jax.lax.stop_gradient, extra_ray_samples)

        function_vals = jax.lax.stop_gradient(
            jnp.linalg.norm(
                extra_ray_samples["radiance_in"], axis=-1
            )
        )

        # Viewdirs
        extra_viewdirs = jax.lax.stop_gradient(extra_rays.viewdirs.reshape(function_vals.shape + (3,)))

        # VMF parameters
        vmf_means = light_sampler_results["vmf_means"]
        vmf_kappas = light_sampler_results["vmf_kappas"]
        vmf_logits = light_sampler_results["vmf_logits"]
        vmf_normals = light_sampler_results["vmf_normals"]

        vmf_means = vmf_means.reshape(-1, vmf_means.shape[-2], 3)
        vmf_kappas = vmf_kappas.reshape(-1, vmf_kappas.shape[-2], 1)
        vmf_logits = vmf_logits.reshape(-1, vmf_logits.shape[-2], 1)
        vmf_normals = vmf_normals.reshape(-1, 3)

        # Loss
        lossmult = rays.lossmult.reshape(-1, 1, 1)
        lossmult = (
            lossmult * jnp.ones_like(function_vals.reshape(lossmult.shape[0], -1, 1))
        )
        lossmult = (lossmult / lossmult.shape[-2]).reshape(function_vals.shape)

        data_loss += render_utils.vmf_loss_fn(
            (vmf_means, vmf_kappas, vmf_logits),
            vmf_normals,
            extra_viewdirs,
            extra_ray_samples,
            function_vals,
            function_vals,
            lossmult,
            linear_to_srgb=config.light_sampling_linear_to_srgb,
        ) / 2.0
    
    return data_loss * multiplier


def material_surface_light_field_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute material surface light field loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed material surface light field loss.
    """
    data_loss = 0.0
    multiplier = 1.0
    shader_results = model_results["shader"]

    for suffix in ["_indirect_diffuse", "_indirect_specular"]:
        extra_rays = shader_results[f"ref_rays{suffix}_cache"]

        if extra_rays is None:
            multiplier = 2.0
            continue

        ref_samples = shader_results[f"ref_samples{suffix}_cache"]
        ref_samples_slf = shader_results[f"ref_samples{suffix}_slf"]

        ref_sampler_results = shader_results[f"ref_sampler_results{suffix}_cache"][-1]
        ref_sampler_results_slf = shader_results[f"ref_sampler_results{suffix}_slf"][-1]

        # Create batch 
        sh = ref_samples["radiance_in_no_stopgrad"].shape
        cache_rgb = utils.stopgrad_with_weight(
            ref_samples["radiance_in_no_stopgrad"],
            config.surface_light_field_stopgrad_weight_forward
        )
        pred_rgb = utils.stopgrad_with_weight(
            ref_samples_slf["radiance_in_no_stopgrad"].reshape(sh),
            config.surface_light_field_stopgrad_weight_backward
        )
        pred_outputs = {
            "rgb": pred_rgb,
            "cache_rgb": cache_rgb,
        }

        # Geometry outputs
        pred_dist = ref_sampler_results_slf["incoming_dist"].reshape(
            sh[:-1] + (-1,)
        )
        pred_weights = ref_sampler_results_slf["incoming_weights"].reshape(
            sh[:-1] + (-1,)
        )
        pred_env_acc = jnp.where(
            pred_dist < config.env_map_distance,
            pred_weights,
            jnp.zeros_like(pred_weights)
        ).sum(axis=-1).reshape(sh[:-1] + (1,))

        cache_tdist = ref_sampler_results["tdist"][..., :-1].reshape(
            extra_rays.viewdirs.shape[:-1] + (-1,)
        )
        cache_weights = ref_sampler_results["weights"].reshape(
            sh[:-1] + (-1,)
        )
        env_acc = jnp.where(
            cache_tdist < config.env_map_distance,
            cache_weights,
            jnp.zeros_like(cache_weights)
        ).sum(axis=-1).reshape(sh[:-1] + (1,))
        acc = cache_weights.sum(axis=-1).reshape(sh[:-1] + (1,))

        if config.surface_light_field_loss_far == float("inf"):
            acc = jnp.ones_like(acc)

        if config.surface_light_field_loss_radius < float("inf"):
            lossmult = (
                jnp.linalg.norm(
                    extra_rays.origins, axis=-1, keepdims=True,
                ) < config.surface_light_field_loss_radius
            ).reshape(sh[:-1] + (1,)).astype(jnp.float32)
        else:
            lossmult = jnp.ones_like(cache_rgb[..., :1])
        
        # Weight
        if config.surface_light_field_is_secondary:
            lossmult = jnp.where(
                ref_samples['local_lightdirs'][..., -1].reshape(lossmult.shape) > 0.0,
                lossmult,
                jnp.zeros_like(lossmult)
            )

        if config.surface_light_field_importance_sample_weights:
            denominator = jnp.maximum(
                ref_samples['pdf'][Ellipsis, 0],
                1e-1
            )
            weight = jnp.clip(
                ref_samples['weight'][Ellipsis, 0],
                0.0,
                10.0,
            )
            lossmult = lossmult * (weight / denominator).reshape(lossmult.shape)

        # Stop gradient
        extra_rays = jax.tree_util.tree_map(jax.lax.stop_gradient, extra_rays)

        extra_batch = batch.replace(
            rgb=cache_rgb,
            masks=jnp.ones_like(cache_rgb[..., :1]),
        )
        extra_rays = extra_rays.replace(
            lossmult=jax.lax.stop_gradient(lossmult),
        )

        # Compute data loss
        config_copy = copy.deepcopy(config)
        config_copy.data_loss_type = config.surface_light_field_loss_type
        config_copy.convert_srgb = config.surface_light_field_linear_to_srgb
        config_copy.loss_clip = float("inf")
        config_copy.loss_thresh = float("inf")
        cur_data_loss, _ = compute_data_loss(
            extra_batch,
            pred_outputs,
            extra_rays,
            config_copy
        )

        # Acc loss
        acc_loss = (
            jnp.square(
                jax.lax.stop_gradient(env_acc) - pred_env_acc
            )
        ) * jax.lax.stop_gradient(lossmult)

        acc_loss = jnp.where(
            env_acc > 0.5,
            acc_loss * config.surface_light_field_loss_acc_scale_opaque,
            acc_loss * config.surface_light_field_loss_acc_scale_empty,
        )

        cur_data_loss += acc_loss.mean()

        # Depth loss
        origins = rays.origins.reshape(-1, 1, 3)
        origins = (
            origins * jnp.ones_like(
                extra_rays.origins.reshape(origins.shape[0], -1, 3)
            )
        ).reshape(extra_rays.origins.shape)
        viewdirs = rays.viewdirs.reshape(-1, 1, 3)
        viewdirs = (
            viewdirs * jnp.ones_like(
                extra_rays.viewdirs.reshape(viewdirs.shape[0], -1, 3)
            )
        ).reshape(extra_rays.viewdirs.shape)

        rng, cur_key = random.split(cur_key)
        results_pred = model.apply(
            variables,
            rng,
            extra_rays,
            train_frac=kwargs["train_frac"],
            compute_extras=False,
            mesh=kwargs["mesh"],
            passes=("surface_light_field",),
            cache_tdist=cache_tdist,
            dist_only=True,
        )

        cache_sdist = results_pred["cache_sdist"].reshape(
            sh[:-1] + (-1,)
        )
        pred_sdist = ref_sampler_results_slf["incoming_s_dist"].reshape(
            sh[:-1] + (1,)
        )

        cur_data_loss += (
            jnp.abs(
                jax.lax.stop_gradient(cache_sdist) - pred_sdist
            )
            * jax.lax.stop_gradient(cache_weights)
            * jax.lax.stop_gradient(lossmult)
        ).sum(axis=-1).mean() * (
            config.surface_light_field_loss_depth_scale
        )

        data_loss += cur_data_loss / 2.0

    return data_loss * multiplier


def material_ray_sampler_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute material ray sampler loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed material ray sampler loss.
    """
    shader_results = model_results["shader"]

    if "ref_sampler_results_indirect_diffuse" not in shader_results:
        return 0.0

    ref_sampler_results = shader_results["ref_sampler_results_indirect_diffuse"]
    ref_rays = shader_results["ref_rays_indirect_diffuse"]

    # Lossmult
    lossmult = rays.lossmult.reshape(-1, 1, 1)
    lossmult = (
        lossmult * jnp.ones_like(ref_rays.viewdirs[..., :1].reshape(lossmult.shape[0], -1, 1))
    ).reshape(ref_rays.viewdirs[..., :1].shape)

    for idx in range(len(ref_sampler_results)):
        ref_sampler_results[idx]["weights"] = ref_sampler_results[idx]["weights"] * lossmult

    # Compute loss
    ref_sample_loss = 0.0

    ref_sample_loss += sum(
        interlevel_loss(
            ref_sampler_results, config.interlevel_loss_mults, config.interlevel_loss_blurs, config
        )
    ) * config.material_ray_sampler_interlevel_loss_mult

    ref_sample_loss += (
        distortion_loss(
            ref_sampler_results, config.distortion_loss_mult, config
        ) * config.material_ray_sampler_normal_loss_mult
    ) * config.material_ray_sampler_distortion_loss_mult

    if config.orientation_loss_mult > 0:
        ref_sample_loss += (
            orientation_loss(ref_rays, ref_sampler_results[-1], config)
        ) * config.material_ray_sampler_orientation_loss_mult

    if config.predicted_normal_loss_mult > 0:
        beta = jnp.ones_like(ref_sampler_results[-1]["normals"][..., :1])
        ref_sample_loss += predicted_normal_loss(
            ref_sampler_results[-1],
            beta,
            config,
        ) * config.material_ray_sampler_normal_loss_mult

    if config.predicted_normal_reverse_loss_mult > 0:
        beta = jnp.ones_like(ref_sampler_results[-1]["normals"][..., :1])
        ref_sample_loss += predicted_normal_reverse_loss(
            ref_sampler_results[-1],
            beta,
            config,
        ) * config.material_ray_sampler_normal_loss_mult

    return jnp.nan_to_num(ref_sample_loss)


def material_correlation_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute material correlation loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed material correlation loss.
    """
    material_weights = {
        'material_albedo': config.material_correlation_weight_albedo,
        'material_roughness': config.material_correlation_weight_other,
        'material_F_0': config.material_correlation_weight_other,
        'material_metalness': config.material_correlation_weight_other,
        'material_diffuseness': config.material_correlation_weight_other,
        'material_mirrorness': config.material_correlation_weight_other,
    }

    rng, cur_key = random.split(cur_key)
    shader_results = utils.filter_jax_dict(
        model_results["shader"]
    )
    shader_results, inds = model.maybe_resample(
        rng,
        True,
        shader_results,
        1,
    )

    # Lossmult
    lossmult = rays.lossmult.reshape(-1, 1, 1)
    lossmult = (
        lossmult * jnp.ones_like(shader_results["lighting_irradiance"][..., :1].reshape(lossmult.shape[0], -1, 1))
    ).reshape(-1, 1) * jax.lax.stop_gradient(shader_results["weights"].reshape(-1, 1) * shader_results["weights"].shape[-1])

    # Irradiance
    irradiance = (
        shader_results["lighting_irradiance"]
    ).reshape(-1, 3)
    irradiance_nocorr = (
        shader_results["lighting_irradiance_nocorr"]
    ).reshape(-1, 3)
    irradiance_cache = (
        shader_results["irradiance_cache"]
    ).reshape(-1, 3)

    # Normalize irradiance
    irradiance_norm = irradiance.reshape(-1, 3) * lossmult
    irradiance_norm = (
        irradiance_norm
        - (
            irradiance_norm.sum(axis=0, keepdims=True)
            / (lossmult.sum(axis=0, keepdims=True) + 1e-3)
        )
    ) * lossmult
    irradiance_norm = jax.lax.stop_gradient(
        irradiance_norm / (jnp.abs(irradiance_norm).sum(axis=0, keepdims=True) + irradiance_norm.shape[0])
    ) * irradiance_norm.shape[0]

    # Loss
    loss = 0.0

    for key in material_weights.keys():
        if key not in shader_results:
            continue

        cur_material_output = shader_results[key].reshape(
            irradiance_norm.shape[0], -1
        ) * lossmult
        cur_material_output = (
            cur_material_output
            - (
                cur_material_output.sum(axis=0, keepdims=True)
                / (lossmult.sum(axis=0, keepdims=True) + 1e-3)
            )
        ) * lossmult
        cur_material_output = (
            cur_material_output / (jnp.abs(cur_material_output).sum(axis=0, keepdims=True) + irradiance_norm.shape[0])
        ) * irradiance_norm.shape[0]

        # Penalize any correlation
        loss += (
            jnp.abs((cur_material_output * irradiance_norm).mean(axis=0)).sum()
        ) * material_weights[key]

    # Irradiance cache loss
    results = {
        "rgb": utils.stopgrad_with_weight(
            irradiance,
            config.irradiance_cache_stopgrad_weight
        ),
        "rgb_nocorr": irradiance_nocorr,
        "cache_rgb": irradiance_cache,
    }

    if "rawnerf" in config.data_loss_type:
        diff = compute_unbiased_loss_rawnerf(
            results,
            utils.stopgrad_with_weight(
                irradiance_cache,
                config.irradiance_cache_stopgrad_weight_backwards
            ),
            irradiance_cache,
            config
        ) * lossmult
    else:
        diff = compute_unbiased_loss(
            results,
            utils.stopgrad_with_weight(
                irradiance_cache,
                config.irradiance_cache_stopgrad_weight_backwards
            ),
            irradiance_cache,
            config
        ) * lossmult

    loss += diff.mean() * config.irradiance_cache_loss_weight

    # Whitening
    loss += (
        compute_unbiased_loss(
            {
                "rgb": irradiance,
                "rgb_nocorr": irradiance_nocorr,
            },
            jax.lax.stop_gradient(irradiance.mean(axis=-1, keepdims=True)),
            jax.lax.stop_gradient(irradiance_nocorr.mean(axis=-1, keepdims=True)),
            config
        ) * lossmult
    ).mean() * config.whitening_loss_weight

    # Return
    return loss


def material_smoothness_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute material smoothness loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed material smoothness loss.
    """
    rng, cur_key = random.split(cur_key)
    shader_results = utils.filter_jax_dict(
        model_results["shader"]
    )
    shader_results, inds = model.maybe_resample(
        rng,
        True,
        shader_results,
        1,
    )

    rng, cur_key = random.split(cur_key)
    cache_shader_results = utils.filter_jax_dict(
        model_results["cache_shader"]
    )
    cache_shader_results, _ = model.maybe_resample(
        rng,
        True,
        cache_shader_results,
        1,
        inds=inds,
    )

    material_weights = {
        'material_albedo': config.material_smoothness_weight_albedo,
        'material_roughness': config.material_smoothness_weight_other,
        'material_F_0': config.material_smoothness_weight_other,
        'material_metalness': config.material_smoothness_weight_other,
        'material_diffuseness': config.material_smoothness_weight_other,
        'material_mirrorness': config.material_smoothness_weight_other,
    }

    # Render perturbed
    shader_results = utils.copy_tree(shader_results)
    origins = shader_results["means"]

    rng, cur_key = random.split(cur_key)

    noise = jax.random.normal(rng, shape=origins.shape)
    # noise_mag = jnp.linalg.norm(noise, axis=-1, keepdims=True)
    # noise = (noise / (noise_mag + 1e-5)) * jnp.clip(noise_mag, 0.0, 1.0)

    shader_results["means"] = origins + noise * config.material_smoothness_noise

    rng, cur_key = random.split(cur_key)
    perturbed_shader_results = model.apply(
        variables,
        rng,
        rays,
        train_frac=kwargs["train_frac"],
        compute_extras=False,
        mesh=kwargs["mesh"],
        passes=("material_cache_shader",),
        sampler_results=jax.lax.stop_gradient(shader_results),
        material_only=True,
        shading_only=True,
        env_map=kwargs["env_map"],
        env_map_pmf=kwargs["env_map_pmf"],
        env_map_pdf=kwargs["env_map_pdf"],
        env_map_dirs=kwargs["env_map_dirs"],
        env_map_w=kwargs["env_map_w"],
        env_map_h=kwargs["env_map_h"],
    )
    perturbed_shader_results = jax.tree_util.tree_map(
        jnp.nan_to_num,
        perturbed_shader_results
    )

    perturbed_cache_shader_results = perturbed_shader_results["cache"]
    perturbed_shader_results = perturbed_shader_results["material"]

    # Lossmult
    lossmult = rays.lossmult.reshape(-1, 1, 1)
    lossmult = (
        lossmult * jnp.ones_like(shader_results["means"][..., :1].reshape(lossmult.shape[0], -1, 1))
    ).reshape(shader_results["means"][..., :1].shape) * (
        jax.lax.stop_gradient(shader_results["weights"][..., None] * shader_results["weights"].shape[-1])
    )

    # Irradiance
    if "irradiance_cache" in shader_results:
        irradiance_cache = jax.lax.stop_gradient(
            shader_results["irradiance_cache"]
        )
        perturbed_irradiance_cache = jax.lax.stop_gradient(
            perturbed_shader_results["irradiance_cache"]
        )
    else:
        irradiance_cache = jnp.ones_like(shader_results["means"][..., :config.num_rgb_channels])
        perturbed_irradiance_cache = jnp.ones_like(shader_results["means"][..., :config.num_rgb_channels])
    
    # Color
    cache_rgb = jax.lax.stop_gradient(
        jnp.abs(cache_shader_results["rgb"]).reshape(irradiance_cache.shape)
    ) / (jnp.maximum(irradiance_cache, 0.0) + 1e-5)
    perturbed_cache_rgb = jax.lax.stop_gradient(
        jnp.abs(perturbed_cache_shader_results["rgb"].reshape(cache_rgb.shape))
    ) / (jnp.maximum(perturbed_irradiance_cache, 0.0) + 1e-5)

    irradiance_weight = 2.0 * jax.nn.sigmoid(
        # -jnp.linalg.norm(
        -jnp.sum(
            jnp.abs(
                cache_rgb - perturbed_cache_rgb
            ) / (jnp.maximum(cache_rgb, perturbed_cache_rgb) + 1e-5),
            # ),
            axis=-1,
            keepdims=True,
        ) * config.material_smoothness_irradiance_multiplier 
    )

    # Compute loss
    loss = 0.0

    for key in material_weights.keys():
        if key not in shader_results:
            continue

        # if "roughness" in key:
        #     cur_diff = (
        #         1.0 / (shader_results[key])
        #         - 1.0 / (perturbed_shader_results[key].reshape(shader_results[key].shape)))

        cur_diff = (
            shader_results[key]
            - perturbed_shader_results[key].reshape(
                shader_results[key].shape
            )
        )

        if ('albedo' in key) and config.material_smoothness_tensoir_albedo:
            cur_denom = jnp.maximum(
                shader_results[key],
                perturbed_shader_results[key].reshape(shader_results[key].shape),
            )

            if config.material_smoothness_albedo_stopgrad:
                cur_denom = jax.lax.stop_gradient(cur_denom)

            cur_diff = cur_diff / jnp.maximum(
                1e-6,
                cur_denom,
            )

        if config.material_smoothness_irradiance_weight:
            cur_weight = irradiance_weight + config.material_smoothness_base
        else:
            cur_weight = jnp.ones_like(irradiance_weight)

        if config.material_smoothness_l1_loss:
            loss += (
                jnp.abs(
                    cur_diff
                ) * (
                    cur_weight
                ) * lossmult.reshape(
                    shader_results[key].shape[:-1] + (-1,)
                ) * material_weights[key]
            ).mean()
        else:
            loss += (
                jnp.square(
                    cur_diff
                ) * (
                    cur_weight
                ) * lossmult.reshape(
                    shader_results[key].shape[:-1] + (-1,)
                ) * material_weights[key]
            ).mean()
    
    return loss


def geometry_smoothness_loss(
    model,
    variables,
    cur_key,
    rays,
    config,
    batch,
    model_results,
    **kwargs,
):
    """Compute geometry smoothness loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        batch: Batch of data.
        model_results: Model rendering results.
        **kwargs: Additional keyword arguments.
        
    Returns:
        The computed geometry smoothness loss.
    """
    rng, cur_key = random.split(cur_key)
    geometry_results = model_results["geometry"]

    geometry_weights = {
        'normals': config.geometry_smoothness_weight_normals,
        'normals_pred': config.geometry_smoothness_weight_normals_pred,
        'density': config.geometry_smoothness_weight_density,
    }

    # Render perturbed
    geometry_results = utils.copy_tree(geometry_results)
    origins = geometry_results["means"]

    rng, cur_key = random.split(cur_key)

    noise = jax.random.normal(rng, shape=origins.shape)
    # noise_mag = jnp.linalg.norm(noise, axis=-1, keepdims=True)
    # noise = (noise / (noise_mag + 1e-5)) * jnp.clip(noise_mag, 0.0, 1.0)

    geometry_results["means"] = origins + noise * config.geometry_smoothness_noise

    rng, cur_key = random.split(cur_key)
    perturbed_geometry_results = model.apply(
        variables,
        rng,
        rays,
        train_frac=kwargs["train_frac"],
        compute_extras=False,
        mesh=kwargs["mesh"],
        passes=("geometry",),
        sampler_results=jax.lax.stop_gradient(geometry_results),
        material_only=True,
        shading_only=True,
    )
    perturbed_geometry_results = jax.tree_util.tree_map(
        jnp.nan_to_num,
        perturbed_geometry_results
    )

    # Compute loss
    geometry_outputs = {
        key: geometry_results[key] for key in geometry_weights.keys()
    }
    geometry_outputs_perturbed = {
        key: perturbed_geometry_results[key].reshape(
            geometry_outputs[key].shape
        ) for key in geometry_weights.keys()
    }

    loss = 0.0

    lossmult = rays.lossmult.reshape(-1, 1, 1)
    lossmult = (
        lossmult * jnp.ones_like(geometry_results["means"][..., :1].reshape(lossmult.shape[0], -1, 1))
    ).reshape(geometry_results["means"][..., :1].shape) * (
        jax.lax.stop_gradient(geometry_results["weights"][..., None] * geometry_results["weights"].shape[-1])
    )

    for key in geometry_weights.keys():
        if key == "density":
            loss += (
                jnp.abs(
                    (
                        geometry_outputs[key]
                        - geometry_outputs_perturbed[key]
                    )
                ) * geometry_weights[key]
                * (
                    lossmult.reshape(geometry_outputs[key].shape)
                )
            ).mean()
        else:
            loss += (
                jnp.abs(
                    (
                        geometry_outputs[key]
                        - geometry_outputs_perturbed[key]
                    )
                ) * geometry_weights[key]
                * (
                    lossmult.reshape(geometry_outputs[key].shape[:-1] + (1,))
                )
            ).mean()
    
    return loss


# -----------------------------------------------------------------------------
# Model Training and Setup Functions
# -----------------------------------------------------------------------------

def create_train_step(
    model: models.Model,
    config: configs.Config,
    dataset: Optional[datasets.Dataset] = None,
):
    """Creates the pmap'ed Nerf training function.

    Args:
        model: The linen model.
        config: The configuration.
        dataset: Training dataset.

    Returns:
        pmap'ed training function.
    """
    if dataset is None:
        camtype = camera_utils.ProjectionType.PERSPECTIVE
    else:
        camtype = dataset.camtype

    def train_step(
        rng,
        state,
        batch,
        cameras,
        virtual_cameras,
        lights,
        train_frac,
    ):
        """One optimization step.

        Args:
            rng: jnp.ndarray, random number generator.
            state: TrainState, state of the model/optimizer.
            batch: dict, a mini-batch of data for training.
            cameras: module containing camera poses.
            virtual_cameras: module containing virtual camera poses.
            lights: module containing light poses.
            train_frac: float, the fraction of training that is complete.

        Returns:
            A tuple (new_state, stats, rng) with
              new_state: TrainState, new training state.
              stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
              rng: jnp.ndarray, updated random number generator.
        """
        def per_output_loss_fn(
            model,
            variables,
            config: Any,
            cur_key: Any,
            rays: utils.Rays,
            cur_batch: Any,
            model_results: Any,
            output_key: Any,
            **kwargs,
        ):
            """Compute loss for a specific output key in the model results.
            
            Args:
                model: Model instance.
                variables: Model variables.
                config: Configuration object.
                cur_key: Random key.
                rays: Ray data.
                cur_batch: Current batch of data.
                model_results: Model rendering results.
                output_key: Output key to compute losses for.
                **kwargs: Additional keyword arguments.
                
            Returns:
                Tuple of (losses, stats).
            """
            losses = {}
            
            ## Sampler losses
            if model_results["sampler"] is not None:
                losses = _compute_sampler_losses(model_results, config, losses)

            # Normal weight ease in
            normal_loss_weight_ease = _compute_normal_weight_ease(config, train_frac)

            if model_results["geometry"] is not None:
                # Normal weight decay
                normal_loss_weight_decay = _compute_normal_weight_decay(config, train_frac)

                losses = _compute_geometry_losses(model_results, rays, config, normal_loss_weight_ease, 
                                               normal_loss_weight_decay, losses)

            ## Integrator losses
            stats = {}
            if model_results["integrator"] is not None:
                data_loss, stats = _compute_integrator_losses(cur_batch, model_results, 
                                                         rays, config, train_frac)
                
                if config.finetune_cache:
                    losses["data"] = jnp.zeros_like(data_loss)
                else:
                    losses["data"] = data_loss * config.data_loss_mult

                # Mask loss
                if not config.is_material:
                    losses["mask"] = compute_mask_loss(
                        cur_batch,
                        model_results["integrator"],
                        rays,
                        config,
                        train_frac=train_frac,
                    ) * (not config.finetune_cache)

                    if config.backward_mask_loss:
                        extra_rays, extra_results = _compute_backward_mask_loss(
                            model, variables, cur_key, rays, config, train_frac, kwargs
                        )
                        
                        extra_batch = cur_batch.replace(
                            masks=jnp.zeros_like(extra_results["main"]["integrator"]["acc"]),
                        )

                        losses["mask_backwards"] = compute_mask_loss(
                            extra_batch,
                            extra_results["main"]["integrator"],
                            extra_rays,
                            config,
                            train_frac=train_frac,
                            empty_loss_weight=config.backward_mask_loss_weight,
                        ) * (not config.finetune_cache)

            ## Extra losses
            losses = _compute_extra_losses(model, variables, cur_key, rays, config, 
                                      cur_batch, model_results, output_key, 
                                      kwargs, losses, train_frac)

            return losses, stats

        def loss_fn(variables: Any, key: Any, loss_scaler: jmp.LossScale):
            """Compute loss for a batch of data.
            
            Args:
                variables: Model variables.
                key: Random key.
                loss_scaler: Loss scaler for mixed precision training.
                
            Returns:
                Tuple of (scaled_loss, (stats, mutable_camera_params))
            """
            cur_key = key
            losses = {}
            rays = batch.rays
            jax_cameras = None
            transformed_jax_cameras = None
            mutable_camera_params = None

            if config.cast_rays_in_train_step:
                transformed_cameras = cameras[:3]
                transformed_virtual_cameras = virtual_cameras[:3]
                rng, cur_key = random.split(cur_key)
                rays = camera_utils.cast_ray_batch(
                    (*transformed_cameras, *cameras[3:]), lights, rays, camtype, rng=rng, jitter=config.jitter_rays, xnp=jnp, impulse_response=batch.impulse_response, virtual_cameras=virtual_cameras,
                )

            # Indicates whether we need to compute output normal or depth maps in 2D
            # or the semantic maps.
            compute_extras = (
                config.compute_disp_metrics
                or config.compute_normal_metrics
                or config.patch_loss_mult > 0.0
                or config.semantic_dir
            )

            # Material loss weight
            material_loss_weight_ease = _compute_material_weight_ease(config, train_frac)

            model_results = _run_model_forward(model, variables, cur_key, rays, train_frac, 
                                            compute_extras, dataset, config)

            # Per output losses
            stats = {}

            for k in model_results.keys():
                if k == "render":
                    continue

                cur_config = copy.deepcopy(config)

                # Batch and rays
                if k == "cache_main":
                    cur_rays = rays
                    cur_batch = batch
                    extra_loss_mult = 1.0
                else:
                    cur_rays = rays
                    cur_batch = batch
                    extra_loss_mult = material_loss_weight_ease

                    # Filter based on normal direction
                    if model.use_material:
                        cur_rays = _filter_rays_by_normal(cur_rays, model_results, config)

                if model.use_material:
                    if k == "main":
                        cur_config.is_material = True
                    elif k == "cache_main":
                        cur_config.is_material = False
                else:
                    cur_config.is_material = False

                # Update loss type
                if "loss_type" in model_results[k]:
                    cur_config.data_loss_type = model_results[k]["loss_type"]

                if "convert_srgb" in model_results[k]:
                    cur_config.convert_srgb = model_results[k]["convert_srgb"]
                else:
                    cur_config.convert_srgb = False

                # Get loss
                rng, cur_key = random.split(cur_key)
                per_output_losses, per_output_stats = per_output_loss_fn(
                    model,
                    variables,
                    cur_config,
                    rng,
                    cur_rays,
                    cur_batch,
                    model_results[k],
                    output_key=k,
                    train_frac=train_frac,
                    compute_extras=compute_extras,
                    mesh=dataset.mesh,
                    env_map=dataset.env_map,
                    env_map_pmf=dataset.env_map_pmf,
                    env_map_pdf=dataset.env_map_pdf,
                    env_map_dirs=dataset.env_map_dirs,
                    env_map_w=dataset.env_map_w,
                    env_map_h=dataset.env_map_h,
                )

                # Add losses
                exclude_list = ["interlevel", "distortion", "orientation", "predicted_normals", 
                              "predicted_normals_reverse", "material_smoothness", 
                              "geometry_smoothness", "mask"]

                for loss_k in per_output_losses.keys():
                    if k == "main":
                        output_k = loss_k
                    else:
                        output_k = f"{k}_{loss_k}"

                    if isinstance(per_output_losses[loss_k], list) or isinstance(per_output_losses[loss_k], tuple):
                        if loss_k in exclude_list:
                            losses[output_k] = [l for l in per_output_losses[loss_k]]
                        else:
                            losses[output_k] = [l * model_results[k]["loss_weight"] * extra_loss_mult for l in per_output_losses[loss_k]]
                    else:
                        if loss_k in exclude_list:
                            losses[output_k] = per_output_losses[loss_k]
                        else:
                            losses[output_k] = per_output_losses[loss_k] * model_results[k]["loss_weight"] * extra_loss_mult

                if k == "main":
                    stats = per_output_stats

            # Regularizers
            if config.param_regularizers:
                losses["regularizer"] = param_regularizer_loss(variables, config, train_frac)

            losses_flat = {}

            for k, v in losses.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    for i, vi in enumerate(v):
                        losses_flat[k + "_" + str(i)] = vi
                elif isinstance(v, dict):
                    for ki, vi in v.items():
                        losses_flat[k + "/" + ki] = vi
                else:
                    losses_flat[k] = v

            stats["loss"] = jnp.sum(jnp.array(list(losses_flat.values())))
            stats["losses"] = losses_flat

            if config.debug_mode:
                stats["weight_l2s"] = summarize_tree(tree_norm_sq, variables["params"])

                # Log some summary statistics of t/s distances along rays and the size
                # of each t/s ray interval.
                def percentile_fn(x):
                    return jnp.percentile(x.flatten(), jnp.linspace(0, 100, 101))

                for ri, rh in enumerate(model_results["sampler"]):
                    s = rh["sdist"]
                    t = rh["tdist"]
                    ds = s[..., 1:] - s[..., :-1]
                    dt = t[..., 1:] - t[..., :-1]
                    stats[f"ray_normalized_distance{ri}"] = percentile_fn(s)
                    stats[f"ray_normalized_distance{ri}_log_delta"] = math.safe_log(percentile_fn(ds))
                    stats[f"ray_metric_distance{ri}_log"] = math.safe_log(percentile_fn(t))
                    stats[f"ray_metric_distance{ri}_log_delta"] = math.safe_log(percentile_fn(dt))

            final_loss = stats["loss"]
            final_loss = loss_scaler.scale(final_loss)

            return final_loss, (stats, mutable_camera_params)

        loss_scaler = jmp.NoOpLossScale()
        if config.enable_loss_scaler:
            loss_scaler = jmp.StaticLossScale(loss_scale=config.loss_scale)

        loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        key, rng = random.split(rng)
        (_, (stats, mutable_camera_params)), grad = loss_grad_fn(state.params, key, loss_scaler)

        # Perform preconditioning before pmean.
        pmean = lambda x: jax.lax.pmean(x, axis_name="batch")
        grad = pmean(grad)
        stats = pmean(stats)
        mutable_camera_params = pmean(mutable_camera_params)

        grad = loss_scaler.unscale(grad)

        if config.debug_mode:
            stats["grad_norms"] = summarize_tree(tree_norm, grad["params"])
            stats["grad_maxes"] = summarize_tree(tree_abs_max, grad["params"])

            for name, g in flax.traverse_util.flatten_dict(grad, sep="/").items():
                # pylint: disable=cell-var-from-loop
                jax.lax.cond(
                    jnp.any(~jnp.isfinite(g)),
                    lambda: jax.debug.print(f"Warning: {name} has non-finite grads"),
                    lambda: None,
                )
                jax.lax.cond(
                    jnp.all(g == 0),
                    lambda: jax.debug.print(f"Warning: {name} has all-zero grads"),
                    lambda: None,
                )
                # pylint: enable=cell-var-from-loop

        grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
        grad = clip_gradients(grad, config)

        new_state = state.apply_gradients(grads=grad)

        if config.debug_mode:
            opt_delta = jax.tree_util.tree_map(lambda x, y: x - y, new_state, state).params["params"]
            stats["opt_update_norms"] = summarize_tree(tree_norm, opt_delta)
            stats["opt_update_maxes"] = summarize_tree(tree_abs_max, opt_delta)

        stats["psnrs"] = jnp.nan_to_num(image.mse_to_psnr(stats["mses"]), nan=-1.0)
        stats["psnr"] = stats["psnrs"][-1]
        return new_state, stats, rng

    train_pstep = jax.pmap(
        train_step,
        axis_name="batch",
        in_axes=(0, 0, 0, None, None, None, None),
        donate_argnums=(0, 1),
    )
    return train_pstep


def _compute_sampler_losses(model_results, config, losses):
    """Compute sampler losses.
    
    Args:
        model_results: Model rendering results.
        config: Configuration object.
        losses: Dictionary to store losses.
        
    Returns:
        Updated losses dictionary.
    """
    if (
        isinstance(config.interlevel_loss_mults, tuple)
        or (config.interlevel_loss_mults > 0 and model_results["sampler"] is not None)
        and model_results["sampler"] is not None
    ):
        losses["interlevel"] = interlevel_loss(
            model_results["sampler"], config.interlevel_loss_mults, config.interlevel_loss_blurs, config
        )

        for i in range(len(losses["interlevel"])):
            losses["interlevel"][i] *= (not config.finetune_cache)

    if isinstance(config.distortion_loss_mult, tuple) or (
        config.distortion_loss_mult > 0 and model_results["sampler"] is not None
    ):
        losses["distortion"] = distortion_loss(
            model_results["sampler"], config.distortion_loss_mult, config
        ) * (not config.finetune_cache)

    if config.eikonal_coarse_loss_mult > 0 or config.eikonal_loss_mult > 0:
        losses["eikonal"] = eikonal_loss(model_results["sampler"], config)
        
    return losses


def _compute_normal_weight_ease(config, train_frac):
    """Compute normal weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    return compute_weight_ease_in(
        train_frac=train_frac,
        use_weight_schedule=config.use_normal_weight_ease,
        start_frac=config.normal_weight_ease_start,
        transition_frac=config.normal_weight_ease_frac,
        min_value=config.normal_weight_ease_min
    )


def _compute_normal_weight_decay(config, train_frac):
    """Compute normal weight decay based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight decay factor.
    """
    return compute_weight_decay(
        train_frac=train_frac,
        use_weight_schedule=config.use_normal_weight_decay,
        start_frac=config.normal_weight_decay_start,
        transition_frac=config.normal_weight_decay_frac,
        min_value=config.normal_weight_decay_min
    )


def _compute_geometry_losses(model_results, rays, config, normal_loss_weight_ease, normal_loss_weight_decay, losses):
    """Compute geometry losses.
    
    Args:
        model_results: Model rendering results.
        config: Configuration object.
        normal_loss_weight_ease: Weight ease-in factor.
        normal_loss_weight_decay: Weight decay factor.
        losses: Dictionary to store losses.
        
    Returns:
        Updated losses dictionary.
    """
    if config.orientation_loss_mult > 0:
        losses["orientation"] = (
            orientation_loss(rays, model_results["geometry"], config) * normal_loss_weight_decay
        ) * (not config.finetune_cache)

    if config.predicted_normal_loss_mult > 0:
        beta = jnp.ones_like(model_results["sampler"][-1]["normals"][..., :1])

        losses["predicted_normals"] = (
            predicted_normal_loss(
                model_results["geometry"],
                beta,
                config,
            )
            * normal_loss_weight_ease
            * normal_loss_weight_decay
        ) * (not config.finetune_cache)

    if config.predicted_normal_reverse_loss_mult > 0:
        beta = jnp.ones_like(model_results["sampler"][-1]["normals"][..., :1])

        losses["predicted_normals_reverse"] = (
            predicted_normal_reverse_loss(
                model_results["geometry"],
                beta,
                config,
            )
            * (normal_loss_weight_ease if config.use_normal_weight_ease_backward else 1.0)
            * (normal_loss_weight_decay if config.use_normal_weight_decay_backward else 1.0)
        ) * (not config.finetune_cache)

    if config.predicted_normal_weight_loss_mult > 0:
        beta = jnp.ones_like(model_results["sampler"][-1]["normals"][..., :1])

        losses["predicted_normals_weight"] = (
            predicted_normal_weight_loss(
                model_results["geometry"],
                beta,
                config,
            )
        )
        
    return losses


def _compute_integrator_losses(cur_batch, model_results, rays, config, train_frac):
    """Compute integrator losses.
    
    Args:
        cur_batch: Current batch of data.
        model_results: Model rendering results.
        rays: Ray data.
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Tuple of (data_loss, stats).
    """
    if config.use_transient:
        data_loss, stats = compute_transient_data_loss(
            cur_batch,
            model_results["integrator"],
            rays,
            config,
            main=True,
            train_frac=train_frac,
        )
    else:
        data_loss, stats = compute_data_loss(
            cur_batch,
            model_results["integrator"],
            rays,
            config,
            main=True,
            train_frac=train_frac,
        )
        
    return data_loss, stats


def _compute_backward_mask_loss(model, variables, cur_key, rays, config, train_frac, kwargs):
    """Compute backward mask loss.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        train_frac: Training fraction.
        kwargs: Additional keyword arguments.
        
    Returns:
        Tuple of (extra_rays, extra_results).
    """
    rng, cur_key = random.split(cur_key)
    extra_rays, _ = render_utils.get_secondary_rays(
        rng,
        rays,
        jax.lax.stop_gradient(rays.origins[..., None, :] + rays.look[..., None, :] * config.shadow_near_max),
        jax.lax.stop_gradient(rays.viewdirs),
        jax.lax.stop_gradient(-rays.look[..., None, :]),
        {},
        refdir_eps=config.shadow_near_max,
        normal_eps=config.secondary_normal_eps,
        random_generator_2d=model.random_generator_2d,
        use_mis=False,
        samplers=model.uniform_importance_samplers,
        num_secondary_samples=1,
        light_sampler_results=None,
        far=config.secondary_far,
    )

    extra_rays = extra_rays.replace(
        near=jnp.ones_like(extra_rays.near) * config.shadow_near_max,
    )

    rng, cur_key = random.split(cur_key)
    extra_results = model.apply(
        variables,
        rng,
        extra_rays,
        train_frac=train_frac,
        compute_extras=False,
        passes=("cache",),
        is_secondary=False,
        linear_rgb=True,
        resample=False,
        run_env=False,
        run_person_lf=False,
        weights_only=True,
    )
    
    return extra_rays, extra_results


def _run_model_forward(model, variables, cur_key, rays, train_frac, compute_extras, dataset, config):
    """Run the model forward pass.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        train_frac: Training fraction.
        compute_extras: Whether to compute extra outputs.
        dataset: Dataset containing mesh and environment map.
        config: Configuration object.
        
    Returns:
        Model results.
    """
    if model.use_material:
        rng, cur_key = random.split(cur_key)
        model_results = model.apply(
            variables,
            rng,
            rays,
            train_frac=train_frac,
            compute_extras=compute_extras,
            mesh=dataset.mesh,
            env_map=dataset.env_map,
            env_map_pmf=dataset.env_map_pmf,
            env_map_pdf=dataset.env_map_pdf,
            env_map_dirs=dataset.env_map_dirs,
            env_map_w=dataset.env_map_w,
            env_map_h=dataset.env_map_h,
            extra_ray_regularizer=config.extra_ray_regularizer,
        )

        if config.use_gradient_debias:
            rng, cur_key = random.split(cur_key)
            model_results_nocorr = model.apply(
                variables,
                rng,
                rays,
                train_frac=train_frac,
                compute_extras=compute_extras,
                mesh=dataset.mesh,
                env_map=dataset.env_map,
                env_map_pmf=dataset.env_map_pmf,
                env_map_pdf=dataset.env_map_pdf,
                env_map_dirs=dataset.env_map_dirs,
                env_map_w=dataset.env_map_w,
                env_map_h=dataset.env_map_h,
                cache_outputs=model_results["cache_main"],
                extra_ray_regularizer=config.extra_ray_regularizer,
            )

            if config.cache_consistency_fix_shader:
                rng, cur_key = random.split(cur_key)
                model_results_nocorr_shader = model.apply(
                    variables,
                    rng,
                    rays,
                    train_frac=train_frac,
                    compute_extras=compute_extras,
                    mesh=dataset.mesh,
                    env_map=dataset.env_map,
                    env_map_pmf=dataset.env_map_pmf,
                    env_map_pdf=dataset.env_map_pdf,
                    env_map_dirs=dataset.env_map_dirs,
                    env_map_w=dataset.env_map_w,
                    env_map_h=dataset.env_map_h,
                    cache_outputs=model_results["cache_main"],
                    filtered_sampler_inds=model_results["cache_main"]["filtered_sampler_inds"],
                    extra_ray_regularizer=config.extra_ray_regularizer,
                )
            else:
                model_results_nocorr_shader = model_results_nocorr

            for field_key in ["integrator", "cache_integrator"]:
                for final_key in model_results_nocorr["main"][field_key].keys():
                    model_results["main"][field_key][f"{final_key}_nocorr"] = model_results_nocorr["main"][field_key][final_key]

            for field_key in ["shader", "cache_shader"]:
                for final_key in model_results_nocorr_shader["main"][field_key].keys():
                    model_results["main"][field_key][f"{final_key}_nocorr"] = model_results_nocorr_shader["main"][field_key][final_key]
    else:
        rng, cur_key = random.split(cur_key)
        model_results = model.apply(
            variables,
            rng,
            rays,
            train_frac=train_frac,
            compute_extras=compute_extras,
            passes=("cache"),
            mesh=dataset.mesh,
            env_map=dataset.env_map,
            env_map_pmf=dataset.env_map_pmf,
            env_map_pdf=dataset.env_map_pdf,
            env_map_dirs=dataset.env_map_dirs,
            env_map_w=dataset.env_map_w,
            env_map_h=dataset.env_map_h,
        )

        if config.use_gradient_debias and (config.unbiased_resample_cache or config.volume_variate):
            rng, cur_key = random.split(cur_key)
            model_results_nocorr = model.apply(
                variables,
                rng,
                rays,
                train_frac=train_frac,
                compute_extras=compute_extras,
                passes=("cache"),
                mesh=dataset.mesh,
                env_map=dataset.env_map,
                env_map_pmf=dataset.env_map_pmf,
                env_map_pdf=dataset.env_map_pdf,
                env_map_dirs=dataset.env_map_dirs,
                env_map_w=dataset.env_map_w,
                env_map_h=dataset.env_map_h,
                cache_outputs=model_results["main"],
                extra_ray_regularizer=config.extra_ray_regularizer,
            )

            rng, cur_key = random.split(cur_key)
            model_results_nocorr_shader = model.apply(
                variables,
                rng,
                rays,
                train_frac=train_frac,
                compute_extras=compute_extras,
                passes=("cache"),
                mesh=dataset.mesh,
                env_map=dataset.env_map,
                env_map_pmf=dataset.env_map_pmf,
                env_map_pdf=dataset.env_map_pdf,
                env_map_dirs=dataset.env_map_dirs,
                env_map_w=dataset.env_map_w,
                env_map_h=dataset.env_map_h,
                cache_outputs=model_results["main"],
                extra_ray_regularizer=config.extra_ray_regularizer,
            )

            for field_key in ["integrator", "shader"]:
                for final_key in model_results_nocorr["main"][field_key].keys():
                    model_results["main"][field_key][f"{final_key}_nocorr"] = model_results_nocorr["main"][field_key][final_key]
                    
    return model_results


def _filter_rays_by_normal(rays, model_results, config):
    """Filter rays based on normal criteria.
    
    Args:
        rays: Ray data.
        model_results: Model rendering results.
        config: Configuration object.
        
    Returns:
        Filtered rays.
    """
    geometry_results = jax.lax.stop_gradient(
        model_results["cache_main"]["integrator"]
    )

    normals = geometry_results[config.material_normals_target].reshape(
        rays.viewdirs.shape
    )
    points = geometry_results["means"].reshape(
        rays.viewdirs.shape
    )

    filtered_lossmult = jnp.ones_like(
        jnp.abs(normals[..., -1:]) < config.filter_normals_thresh
    ) * jnp.ones_like(
        jnp.linalg.norm(
            points, axis=-1, keepdims=True,
        ) < config.material_loss_radius
    ) * rays.lossmult.reshape(rays.viewdirs.shape[:-1] + (1,))

    # Additional filtering for retroreflections if needed
    if config.filter_retroreflective:
        refdirs = ref_utils.reflect(-rays.viewdirs, normals)
        distances = jnp.linalg.norm(points - rays.origins, axis=-1, keepdims=True)
        dotprod = (-rays.viewdirs * refdirs).sum(-1, keepdims=True)
        person_length = (jnp.sqrt(jnp.maximum(1.0 - jnp.square(dotprod), 1e-8)) / jnp.maximum(dotprod, 1e-8)) * distances

        filtered_lossmult = (
            (
                (person_length > config.filter_retroreflective_thresh)
                | (dotprod < 0)
            )
        ) * filtered_lossmult

    return rays.replace(
        lossmult=filtered_lossmult
    )


def _compute_extra_losses(model, variables, cur_key, rays, config, 
                      cur_batch, model_results, output_key, 
                      kwargs, losses, train_frac):
    """Compute extra losses.
    
    Args:
        model: Model instance.
        variables: Model variables.
        cur_key: Random key.
        rays: Ray data.
        config: Configuration object.
        cur_batch: Current batch of data.
        model_results: Model rendering results.
        output_key: Output key to compute losses for.
        kwargs: Additional keyword arguments.
        losses: Dictionary to store losses.
        train_frac: Training fraction.
        
    Returns:
        Updated losses dictionary.
    """
    extra_loss_functions = {
        "emission": emission_loss,
        "residual_albedo": residual_albedo_loss,
        "direct_indirect_consistency": direct_indirect_consistency_loss if not config.use_transient else transient_direct_indirect_consistency_loss,
        "light_sampling": light_sampling_loss if not config.use_transient else transient_light_sampling_loss,
        "material_surface_light_field": material_surface_light_field_loss,
        "material_smoothness": material_smoothness_loss,
        "geometry_smoothness": geometry_smoothness_loss,
        "material_correlation": material_correlation_loss,
        "material_ray_sampler": material_ray_sampler_loss,
        "maximum_radiance": maximum_radiance_loss,
        "normalize_weight": normalize_weight_loss,
    }
    
    # Process each extra loss if configured
    for loss_name, loss_fn in extra_loss_functions.items():
        if loss_name in config.extra_losses and output_key in config.extra_losses[loss_name]:
            # Special case for consistency losses with ease-in
            if loss_name in ["cache_consistency", "direct_indirect_consistency"]:
                consistency_loss_weight_ease = _compute_consistency_weight_ease(config, train_frac)
                loss_mult = config.extra_losses[loss_name][output_key]["mult"] * consistency_loss_weight_ease
            elif loss_name == "surface_light_field" or loss_name == "material_surface_light_field":
                surface_light_field_loss_weight_ease = _compute_surface_light_field_weight_ease(config, train_frac)
                loss_mult = config.extra_losses[loss_name][output_key]["mult"] * surface_light_field_loss_weight_ease
            else:
                loss_mult = config.extra_losses[loss_name][output_key]["mult"]
            
            # Call the loss function
            rng, cur_key = random.split(cur_key)
            cur_loss = loss_fn(
                model,
                variables,
                rng,
                rays,
                config,
                cur_batch,
                model_results,
                **kwargs,
            )
            losses[loss_name] = loss_mult * cur_loss
    
    # Handle extra ray loss separately
    if config.extra_ray_loss_mult > 0.0 and config.is_material:
        extra_ray_loss_weight_ease = _compute_extra_ray_weight_ease(config, train_frac)
        loss_mult = config.extra_ray_loss_mult * extra_ray_loss_weight_ease

        rng, cur_key = random.split(cur_key)
        cur_loss = extra_ray_loss(
            model,
            variables,
            rng,
            rays,
            config,
            cur_batch,
            model_results,
            **kwargs,
        )
        losses["extra_ray"] = loss_mult * cur_loss
        
    return losses


def _compute_consistency_weight_ease(config, train_frac):
    """Compute consistency weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    if config.use_consistency_weight_ease:
        if config.consistency_weight_ease_frac > 0:
            w = jnp.clip((train_frac - config.consistency_weight_ease_start) / config.consistency_weight_ease_frac, 0.0, 1.0)
            return config.consistency_weight_ease_min * (1.0 - w) + w
        else:
            return ((train_frac - config.consistency_weight_ease_start) >= 0.0).astype(jnp.float32)
    else:
        return 1.0


def _compute_surface_light_field_weight_ease(config, train_frac):
    """Compute surface light field weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    if config.use_surface_light_field_weight_ease:
        if config.surface_light_field_weight_ease_frac > 0:
            w = jnp.clip((train_frac - config.surface_light_field_weight_ease_start) / config.surface_light_field_weight_ease_frac, 0.0, 1.0)
            return config.surface_light_field_weight_ease_min * (1.0 - w) + w
        else:
            return ((train_frac - config.surface_light_field_weight_ease_start) >= 0.0).astype(jnp.float32)
    else:
        return 1.0


def _compute_extra_ray_weight_ease(config, train_frac):
    """Compute extra ray weight ease-in based on training fraction.
    
    Args:
        config: Configuration object.
        train_frac: Training fraction.
        
    Returns:
        Weight ease-in factor.
    """
    if config.use_extra_ray_weight_ease:
        if config.extra_ray_weight_ease_frac > 0:
            w = jnp.clip((train_frac - config.extra_ray_weight_ease_start) / config.extra_ray_weight_ease_frac, 0.0, 1.0)
            return config.extra_ray_weight_ease_min * (1.0 - w) + w
        else:
            return ((train_frac - config.extra_ray_weight_ease_start) >= 0.0).astype(jnp.float32)
    else:
        return 1.0


def create_render_fn(
    model: models.Model,
    dataset: Optional[datasets.Dataset] = None,
    mapping_fn: Any = jax.pmap,
):
    """Creates pmap'ed or vmap'ed function for full image rendering.
    
    Args:
        model: The linen model.
        dataset: Training dataset.
        mapping_fn: Function to map over data (pmap or vmap).
        
    Returns:
        Mapping function for rendering.
    """
    camtype = None

    if dataset is not None:
        camtype = dataset.camtype

    def render_eval_fn(
        variables,
        rng,
        train_frac: float,
        cameras: Optional[Tuple[jnp.ndarray, ...]],
        lights: Optional[Tuple[jnp.ndarray, ...]],
        rays: utils.Rays | utils.Pixels,
        passes: Tuple[Text, ...],
        resample: Any = None,
    ):
        """Render function for evaluation.
        
        Args:
            variables: Model variables.
            rng: Random number generator.
            train_frac: Training fraction.
            cameras: Camera parameters.
            lights: Light parameters.
            rays: Ray data or pixel coordinates.
            passes: Render passes to compute.
            resample: Resampling specification.
            
        Returns:
            Tuple of (results, updated rng).
        """
        if isinstance(rays, utils.Pixels):
            assert cameras is not None and camtype is not None, (
                "When passing Pixels into render_eval_fn, cameras and camtype needs"
                f" to be not None. Got cameras={cameras} camtype={camtype}."
            )
            rays = camera_utils.cast_ray_batch(cameras, lights, rays, camtype, xnp=jnp)

        key, rng = utils.random_split(rng)
        results = jax.lax.all_gather(
            model.apply(
                variables,
                key,
                rays,
                compute_extras=True,
                train_frac=train_frac,
                train=False,
                passes=passes,
                resample=resample,
                mesh=dataset.mesh,
                env_map=dataset.env_map,
                env_map_pmf=dataset.env_map_pmf,
                env_map_pdf=dataset.env_map_pdf,
                env_map_dirs=dataset.env_map_dirs,
                env_map_w=dataset.env_map_w,
                env_map_h=dataset.env_map_h,
                albedo_ratio=dataset.albedo_ratio,
            )["render"],
            axis_name="batch",
        )

        key, rng = utils.random_split(rng)
        return results, key

    # call the mapping_fn over only the data input.
    render_eval_mfn = mapping_fn(
        render_eval_fn,
        # Shard variables and rays. Copy train_frac and rng.
        #
        # variables should be replicated manually by calling
        # flax.jax_utils.replicate
        in_axes=(0, 0, None, 0, 0, 0, None, None),
        static_broadcasted_argnums=(6,),
        axis_name="batch",
    )
    return render_eval_mfn


def create_optimizer(
    config: configs.Config,
    variables: FrozenVariableDict,
    model: models.Model | None = None,
) -> Tuple[TrainState, Callable[[int], float]]:
    """Creates optax optimizer for model training.
    
    Args:
        config: Configuration object.
        variables: Model variables.
        model: Optional model instance.
        
    Returns:
        Tuple of (train_state, learning_rate_function).
    """
    adam_kwargs = {
        "b1": config.adam_beta1,
        "b2": config.adam_beta2,
        "eps": config.adam_eps,
    }
    lr_kwargs = {
        "max_steps": config.max_steps,
        "lr_delay_steps": config.lr_delay_steps,
        "lr_delay_mult": config.lr_delay_mult,
    }

    def get_lr_fn(lr_init, lr_final, **lr_kwargs):
        return functools.partial(
            math.learning_rate_decay,
            lr_init=lr_init,
            lr_final=lr_final,
            **lr_kwargs,
        )

    lr_fn_main = get_lr_fn(config.lr_init, config.lr_final, **lr_kwargs)
    tx_model = optax.adam(learning_rate=lr_fn_main, **adam_kwargs)
    all_false = jax.tree_util.tree_map(lambda _: False, variables)

    def construct_optimizer(opt_params, prefix, tx_model):
        """Construct optimizer with custom parameters for a specific prefix.
        
        Args:
            opt_params: Optimizer parameters.
            prefix: Parameter prefix to apply optimizer to.
            tx_model: Base optimizer.
            
        Returns:
            Combined optimizer.
        """
        # Get learning rate kwargs
        lr_kwargs = {
            "max_steps": (opt_params["max_steps"] if "max_steps" in opt_params else config.max_steps),
            "lr_delay_steps": (
                opt_params["lr_delay_steps"] if "lr_delay_steps" in opt_params else config.lr_delay_steps
            ),
            "lr_delay_mult": (opt_params["lr_delay_mult"] if "lr_delay_mult" in opt_params else config.lr_delay_mult),
        }
        
        # Get Adam kwargs
        adam_kwargs = {
            "b1": (opt_params["adam_b1"] if "adam_b1" in opt_params else config.adam_beta1),
            "b2": (opt_params["adam_b2"] if "adam_b2" in opt_params else config.adam_beta2),
            "eps": (opt_params["adam_eps"] if "adam_eps" in opt_params else config.adam_eps),
        }

        # Keep opt for other params the same
        model_traversal = flax.traverse_util.ModelParamTraversal(
            lambda path, _: prefix not in path.split("/")
        )
        model_mask = model_traversal.update(lambda _: True, all_false)
        tx = optax.masked(tx_model, model_mask)

        # Opt for current params
        extra_lr_fn = get_lr_fn(
            opt_params["lr_init"] if "lr_init" in opt_params else config.lr_init,
            opt_params["lr_final"] if "lr_final" in opt_params else config.lr_final,
            **lr_kwargs,
        )
        extra_traversal = flax.traverse_util.ModelParamTraversal(
            lambda path, _: prefix in path.split("/")
        )
        extra_mask = extra_traversal.update(lambda _: True, all_false)
        extra_tx = optax.adam(learning_rate=extra_lr_fn, **adam_kwargs)

        # Return
        return optax.chain(
            tx,
            optax.masked(extra_tx, extra_mask),
        )

    # Apply extra optimizers if configured
    if config.extra_opt_params is not None:
        for prefix, params in config.extra_opt_params.items():
            tx_model = construct_optimizer(params, prefix, tx_model)

    tx = tx_model

    # Apply gradient accumulation if needed
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, config.grad_accum_steps, use_grad_mean=True)

    return TrainState.create(apply_fn=None, params=variables, tx=tx), lr_fn_main


def setup_model(
    config: configs.Config,
    rng: jnp.array,
    dataset: Optional[datasets.Dataset] = None,
) -> Tuple[
    models.Model,
    TrainState,
    Callable[
        [
            FrozenVariableDict,
            float,
            Optional[Tuple[jnp.ndarray, ...]],
            jnp.array,
            utils.Rays | utils.Pixels,
        ],
        MutableMapping[str, Any],
    ],
    Callable[
        [jnp.array, TrainState, utils.Batch, Optional[Tuple[Any, ...]], float],
        Tuple[TrainState, Dict[str, Any], jnp.array],
    ],
    Callable[[int], float],
]:
    """Creates NeRF model, optimizer, and pmap-ed train/render functions.
    
    Args:
        config: Configuration object.
        rng: Random number generator.
        dataset: Optional dataset.
        
    Returns:
        Tuple of (model, state, render_eval_pfn, train_pstep, lr_fn).
    """
    # Create dummy rays for model initialization
    dummy_rays = utils.dummy_rays(include_exposure_idx=config.rawnerf_mode, include_exposure_values=True)
    
    # Construct model and variables
    model, variables = models.construct_model(rng, dummy_rays, config, dataset=dataset)

    # Create optimizer and learning rate function
    state, lr_fn = create_optimizer(config, variables, model=model)
    
    # Create render and train functions
    render_eval_pfn = create_render_fn(model, dataset=dataset)
    train_pstep = create_train_step(model, config, dataset=dataset)

    return model, state, render_eval_pfn, train_pstep, lr_fn


# -----------------------------------------------------------------------------
# Checkpoint Utilities
# -----------------------------------------------------------------------------

def replace_param_subset(
    state,
    cache_state,
    put_prefix="CacheModel/",
    take_prefix="CacheModel/",
    exclude_prefixes=None,
):
    """Replace a subset of parameters in the state.
    
    Args:
        state: Current state.
        cache_state: State to take parameters from.
        put_prefix: Prefix to replace in current state.
        take_prefix: Prefix to take from cache state.
        exclude_prefixes: Prefixes to exclude from replacement.
        
    Returns:
        Updated state.
    """
    flat_params = flax.traverse_util.flatten_dict(state.params, sep="/")
    flat_cache_params = flax.traverse_util.flatten_dict(cache_state["params"], sep="/")

    for put_key in flat_params:
        if (
            not put_key.startswith(put_prefix)
            or (
                exclude_prefixes is not None
                and any(put_key.startswith(prefix) for prefix in exclude_prefixes)
            )
        ):
            print("Excluding:", put_key)
            continue

        cur_key = put_key[len(put_prefix):]
        take_key = take_prefix + cur_key

        if take_key in flat_cache_params:
            print(put_prefix, take_prefix, put_key, take_key)
            flat_params[put_key] = copy.deepcopy(flat_cache_params[take_key])

    params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
    return state.replace(params=flax.core.unfreeze(params))


def restore_partial_checkpoint(
    config,
    state,
    prefixes: Any = None,
    exclude_prefixes: Any = None,
    replace_dict: Any = None,
    checkpoint_dir: Any = None,
):
    """Restore parameters from a partial checkpoint.
    
    Args:
        config: Configuration object.
        state: Current state.
        prefixes: Prefixes to include from checkpoint.
        exclude_prefixes: Prefixes to exclude from checkpoint.
        replace_dict: Dictionary mapping destination prefixes to source prefixes.
        checkpoint_dir: Directory containing the checkpoint.
        
    Returns:
        Updated state.
    """
    if checkpoint_dir is None:
        checkpoint_dir = config.partial_checkpoint_dir

    if checkpoint_dir is not None:
        # Filter params
        partial_params = flax.traverse_util.flatten_dict(flax.core.unfreeze(state.params), sep="/")
        partial_params = {
            k: v for k, v in partial_params.items()
            if (prefixes is None or any(k.startswith(prefix) for prefix in prefixes))
            and (exclude_prefixes is None or not any(k.startswith(prefix) for prefix in exclude_prefixes))
        }
        partial_params = flax.traverse_util.unflatten_dict(partial_params, sep="/")

        # Create partial state
        partial_state = {"params": partial_params}
        partial_state = checkpoints.restore_checkpoint(checkpoint_dir, partial_state)

        # Replace params
        if replace_dict is not None:
            for put_prefix, take_prefix in replace_dict.items():
                state = replace_param_subset(
                    state,
                    partial_state,
                    put_prefix=put_prefix,
                    take_prefix=take_prefix,
                    exclude_prefixes=exclude_prefixes,
                )
        else:
            state = state.replace(
                params=flax.core.unfreeze(partial_state["params"])
            )

    return state


def simple_restore_partial_checkpoint(
    config,
    state,
    prefixes: Any = None,
    exclude_prefixes: Any = None,
    replace_dict: Any = None,
):
    """Simple version of partial checkpoint restoration.
    
    Args:
        config: Configuration object.
        state: Current state.
        prefixes: Prefixes to include from checkpoint.
        exclude_prefixes: Prefixes to exclude from checkpoint.
        replace_dict: Dictionary mapping destination prefixes to source prefixes.
        
    Returns:
        Updated state.
    """
    if config.partial_checkpoint_dir is not None:
        partial_state = copy.deepcopy(state)
        partial_state = checkpoints.restore_checkpoint(config.partial_checkpoint_dir, partial_state)
        state = state.replace(
            params=partial_state.params,
        )

    return state