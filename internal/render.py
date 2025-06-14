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
"""Helper functions for shooting and rendering rays."""

from internal import stepfun
import jax
import jax.numpy as jnp
import math
import pdb


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[Ellipsis, None, :] * t_mean[Ellipsis, None]

    d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[Ellipsis, None] * d_outer_diag[Ellipsis, None, :]
        xy_cov_diag = r_var[Ellipsis, None] * null_outer_diag[Ellipsis, None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[Ellipsis, :, None] * d[Ellipsis, None, :]
        eye = jnp.eye(d.shape[-1])
        null_outer = eye - d[Ellipsis, :, None] * (d / d_mag_sq)[Ellipsis, None, :]
        t_cov = t_var[Ellipsis, None, None] * d_outer[Ellipsis, None, :, :]
        xy_cov = r_var[Ellipsis, None, None] * null_outer[Ellipsis, None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def gaussianize_frustum(t0, t1):
    """Convert intervals along a conical frustum into means and variances."""
    # A more stable version of Equation 7 from https://arxiv.org/abs/2103.13415.
    s = t0 + t1
    d = t1 - t0
    eps = jnp.finfo(jnp.float32).eps ** 2
    ratio = d**2 / jnp.maximum(eps, 3 * s**2 + d**2)
    t_mean = s * (1 / 2 + ratio)
    t_var = (1 / 12) * d**2 - (1 / 15) * ratio**2 * (12 * s**2 - d**2)
    r_var = (1 / 16) * s**2 + d**2 * (5 / 48 - (1 / 15) * ratio)
    return t_mean, t_var, r_var


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag):
    """Approximate a 3D conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
      d: jnp.float32 3-vector, the axis of the cone
      t0: float, the starting distance of the frustum.
      t1: float, the ending distance of the frustum.
      base_radius: float, the scale of the radius as a function of distance.
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean, t_var, r_var = gaussianize_frustum(t0, t1)
    r_var *= base_radius**2
    mean, cov = lift_gaussian(d, t_mean, t_var, r_var, diag)
    return mean, cov


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: jnp.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
      tdist: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
      diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
      a tuple of arrays of means and covariances.
    """

    t0 = tdist[Ellipsis, :-1]
    t1 = tdist[Ellipsis, 1:]
    if ray_shape == "cone":
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == "cylinder":
        gaussian_fn = cylinder_to_gaussian
    else:
        raise ValueError("ray_shape must be 'cone' or 'cylinder'")
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[Ellipsis, None, :]
    return means, covs


def compute_alpha_weights(
    density,
    tdist,
    dirs,
    opaque_background=False,
    delta=None,
):
    """Helper function for computing alpha compositing weights."""
    if delta is None:
        t_delta = tdist[Ellipsis, 1:] - tdist[Ellipsis, :-1]
        delta = t_delta * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)

    density_delta = density * jnp.abs(delta)

    if opaque_background:
        # Equivalent to making the final t-interval infinitely wide.
        density_delta = jnp.concatenate(
            [
                density_delta[Ellipsis, :-1],
                jnp.full_like(density_delta[Ellipsis, -1:], jnp.inf),
            ],
            axis=-1,
        )

    alpha = 1 - jnp.exp(-density_delta)
    trans = jnp.exp(
        -jnp.concatenate(
            [
                jnp.zeros_like(density_delta[Ellipsis, :1]),
                jnp.cumsum(density_delta[Ellipsis, :-1], axis=-1),
            ],
            axis=-1,
        )
    )
    weights = alpha * trans
    return weights, alpha, trans


def volumetric_rendering(
    rgbs,
    weights,
    weights_no_filter,
    tdist,
    bg_rgbs,
    compute_extras,
    extras=None,
    normalize_weights_for_extras=False,
    percentiles=(5, 50, 95),
    compute_distance=True,
):
    """Volumetric Rendering Function.

    Args:
      rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
      weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
      tdist: jnp.ndarray(float32), [batch_size, num_samples].
      bg_rgbs: jnp.ndarray(float32), the color(s) to use for the background.
      compute_extras: bool, if True, compute extra quantities besides color.
      extras: dict, a set of values along rays to render by alpha compositing.
      percentiles: depth will be returned for these percentiles.

    Returns:
      rendering: a dict containing an rgb image of size [batch_size, 3], and other
        visualizations if compute_extras=True.
    """
    eps = jnp.finfo(jnp.float32).eps
    rendering = {}

    acc = weights_no_filter.sum(axis=-1)
    acc_no_filter = weights_no_filter.sum(axis=-1)
    bg_w = jnp.maximum(0, 1 - acc[Ellipsis, None])  # The weight of the background.

    if rgbs is not None:
        rgb = (weights[Ellipsis, None] * rgbs).sum(axis=-2) + bg_w * bg_rgbs
    else:
        rgb = None

    rendering["rgb"] = rgb
    rendering["acc"] = acc

    weights_norm = weights / jnp.maximum(eps, acc[Ellipsis, None])
    weights_norm_no_filter = weights_no_filter / jnp.maximum(
        eps, acc_no_filter[Ellipsis, None]
    )

    if extras is not None:
        for k, v in extras.items():
            if v is not None:
                if normalize_weights_for_extras:
                    rendering[k] = (weights_norm[Ellipsis, None] * v).sum(axis=-2)
                else:
                    rendering[k] = (weights[Ellipsis, None] * v).sum(axis=-2)

    if compute_distance:
        expectation = lambda x: (weights_no_filter * x).sum(axis=-1) / jnp.maximum(
            eps, acc_no_filter
        )
        t_mids = 0.5 * (tdist[Ellipsis, :-1] + tdist[Ellipsis, 1:])
        # For numerical stability this expectation is computing using log-distance.
        rendering["distance_mean"] = jnp.clip(
            jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
            tdist[Ellipsis, 0],
            tdist[Ellipsis, -1],
        )

        distance_percentiles = stepfun.weighted_percentile(
            tdist, weights_norm_no_filter, percentiles
        )

        for i, p in enumerate(percentiles):
            s = "median" if p == 50 else "percentile_" + str(p)
            rendering["distance_" + s] = distance_percentiles[Ellipsis, i]

    return rendering


def volumetric_transient_rendering(
    direct_rgbs,
    transient_indirect,
    weights,
    weights_no_filter,
    tdist,
    bg_rgbs,
    compute_extras,
    extras=None,
    normalize_weights_for_extras=False,
    percentiles=(5, 50, 95),
    compute_distance=True,
    n_bins=700,
    shift=0.0,
    dark_level=0.0,
    impulse_response=None,
    tfilter_sigma=0.0,
    exposure_time=0.01,
    filter_indirect=False,
    filter_median=False,
    itof=False,
    config=None,
):
    eps = jnp.finfo(jnp.float32).eps
    acc = weights_no_filter.sum(axis=-1)
    acc_no_filter = weights_no_filter.sum(axis=-1)
    rendering = {}

    # Depths / extras
    weights_norm = weights / jnp.maximum(eps, acc[Ellipsis, None])
    weights_norm_no_filter = weights_no_filter / jnp.maximum(
        eps, acc_no_filter[Ellipsis, None]
    )

    if extras is not None:
        for k, v in extras.items():
            if v is not None:
                if len(v.shape) == len(weights.shape) + 2:
                    if normalize_weights_for_extras:
                        rendering[k] = (weights_norm[Ellipsis, None, None] * v).sum(
                            axis=-3
                        )
                    else:
                        rendering[k] = (weights[Ellipsis, None, None] * v).sum(axis=-3)

                else:
                    if normalize_weights_for_extras:
                        rendering[k] = (weights_norm[Ellipsis, None] * v).sum(axis=-2)
                    else:
                        rendering[k] = (weights[Ellipsis, None] * v).sum(axis=-2)

    # Distance
    expectation = lambda x: (weights_no_filter * x).sum(axis=-1) / jnp.maximum(
        eps, acc_no_filter
    )
    t_mids = 0.5 * (tdist[Ellipsis, :-1] + tdist[Ellipsis, 1:])

    rendering["distance_mean"] = jnp.clip(
        jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
        tdist[Ellipsis, 0],
        tdist[Ellipsis, -1],
    )

    distance_percentiles = stepfun.weighted_percentile(
        tdist, weights_norm_no_filter, percentiles
    )

    for i, p in enumerate(percentiles):
        s = "median" if p == 50 else "percentile_" + str(p)
        rendering["distance_" + s] = distance_percentiles[Ellipsis, i]

    # Distances
    n_rays = direct_rgbs.shape[0]
    num_rgb_channels = direct_rgbs.shape[-1]

    dists_direct = extras["light_dists"].squeeze() + extras["ray_dists"].squeeze()
    dists_indirect = extras["ray_dists"].squeeze().reshape(-1)

    # Weights
    if len(weights.shape) == 3:
        n_rays = weights.shape[0] * weights.shape[1]
        weights_sq = weights.reshape(
            weights.shape[0] * weights.shape[1], weights.shape[2]
        )
    else:
        weights_sq = weights

    # Filter weights
    if filter_median and transient_indirect is not None:
        distance_median = rendering["distance_median"]
        effective_depth = (
            dists_indirect + config.filter_median_thresh * config.exposure_time
        )

        weights_sq = jnp.where(
            (
                effective_depth.reshape(n_rays, n_samples, 1)
                < distance_median.reshape(n_rays, 1, 1)
            ).reshape(weights_sq.shape),
            jnp.zeros_like(weights_sq),
            weights_sq,
        )
        weights_sq = weights_sq / (weights_sq.sum(axis=-1, keepdims=True) + 1e-5)

    # Offsets
    if config.no_shift_direct and config.vis_only:
        direct_offset = dists_indirect.reshape(n_rays, n_samples)
        indirect_offset = dists_indirect.reshape(-1)
    else:
        direct_offset = 0
        indirect_offset = 0

    # Direct
    n_samples = weights_sq.shape[-1]
    direct_rgbs = direct_rgbs.reshape(
        n_rays, n_samples, num_rgb_channels
    )  # n_rays, n_samples, 3
    dists_direct = dists_direct.reshape(n_rays, n_samples)  # n_rays, n_samples
    dists_direct = (dists_direct) / exposure_time
    transient_direct = shift_direct(
        dists_direct + shift / exposure_time - direct_offset / exposure_time,
        direct_rgbs,
        weights_sq,
        n_bins,
        num_rgb_channels,
        impulse_response,
    )

    # Indirect
    if transient_indirect is not None:
        transient_indirect = transient_indirect.reshape(
            n_rays * n_samples, n_bins, num_rgb_channels
        )
        transient_indirect = shift_map_coordinates(
            transient_indirect,
            dists_indirect + shift - indirect_offset,
            exposure_time,
            n_bins,
            num_rgb_channels,
        )
        transient_indirect = transient_indirect.reshape(
            n_rays, n_samples, n_bins, num_rgb_channels
        )
        transient_indirect = (transient_indirect * weights_sq[..., None, None]).sum(1)
        rendering["transient_indirect_no_integration"] = extras["transient_indirect"]
    else:
        transient_indirect = jnp.zeros((n_rays, n_bins, num_rgb_channels))

    # Apply impulse response
    transient_indirect_no_filter = transient_indirect
    transient_direct_no_filter = transient_direct

    if impulse_response is not None or tfilter_sigma != 0.0:
        if impulse_response is not None:
            filter = impulse_response
        else:
            filter = jnp.arange(round(-4 * tfilter_sigma), round(4 * tfilter_sigma) + 1)
            filter = jnp.exp(-(filter**2) / (2 * tfilter_sigma**2)) - math.exp(-8)
            filter = filter / filter.sum()

        transient_direct = jax.scipy.signal.convolve(
            transient_direct, filter[None, :, None], mode="same"
        )

        if filter_indirect:
            transient_indirect = jax.scipy.signal.convolve(
                transient_indirect, filter[None, :, None], mode="same"
            )

    # Reshape
    integrated_shape = weights.shape[:-1]
    transient_direct = transient_direct.reshape(
        integrated_shape + transient_direct.shape[-2:]
    )
    transient_indirect = transient_indirect.reshape(
        integrated_shape + transient_indirect.shape[-2:]
    )

    # Vis
    rendering["transient_direct_viz"] = transient_direct + dark_level
    rendering["transient_indirect_viz"] = transient_indirect

    # Other outputs
    rendering["dists"] = dists_direct
    rendering["weights"] = weights_sq
    rendering["direct_rgb_viz"] = direct_rgbs.sum(-2)
    rendering["rgb"] = transient_direct + transient_indirect + dark_level
    rendering["acc"] = acc

    rendering["direct_rgb"] = transient_direct.sum(-2)
    rendering["indirect_rgb"] = transient_indirect.sum(-2)
    rendering["integrated_rgb"] = rendering["rgb"].sum(-2)

    rendering["transient_indirect"] = transient_indirect
    rendering["transient_direct"] = transient_direct

    rendering["transient_indirect_no_filter"] = transient_indirect_no_filter
    rendering["transient_direct_no_filter"] = transient_direct_no_filter

    return rendering


def shift_direct(dists, direct_rgbs, weights, n_bins, num_rgb_channels, impulse_response):
    n_rays = direct_rgbs.shape[0]
    rgb = jnp.zeros((n_rays, n_bins, num_rgb_channels))

    dists_low = jnp.maximum(jnp.floor(dists), 0)
    dists_high = jnp.ceil(dists)
    weights_high = dists - dists_low
    weights_low = 1.0 - weights_high

    indices_low = (
        jnp.repeat(jnp.arange(direct_rgbs.shape[0]) * n_bins, direct_rgbs.shape[1])
        + dists_low.reshape(-1).astype(jnp.int32)
    ).astype(jnp.int32)
    indices_high = (
        jnp.repeat(jnp.arange(direct_rgbs.shape[0]) * n_bins, direct_rgbs.shape[1])
        + dists_high.reshape(-1).astype(jnp.int32)
    ).astype(jnp.int32)

    rgb = (
        (rgb.reshape(-1, num_rgb_channels))
        .at[indices_low]
        .add(
            (weights[Ellipsis, None] * direct_rgbs).reshape(-1, num_rgb_channels)
            * weights_low.reshape(-1, 1)
        )
        .reshape(-1, n_bins, num_rgb_channels)
    )

    rgb = (
        (rgb.reshape(-1, num_rgb_channels))
        .at[indices_high]
        .add(
            (weights[Ellipsis, None] * direct_rgbs).reshape(-1, num_rgb_channels)
            * weights_high.reshape(-1, 1)
        )
        .reshape(-1, n_bins, num_rgb_channels)
    )

    return rgb


def shift_map_coordinates(
    transient_indirect, bins_move, exposure_time, n_bins, num_rgb_channels
):
    bins_move = bins_move / exposure_time
    x_dim = transient_indirect.shape[0]
    x = jnp.arange(x_dim)
    y = jnp.arange(n_bins)
    z = jnp.arange(num_rgb_channels)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    Y = Y - bins_move[:, None, None]
    indices = jnp.stack([X, Y, Z])
    indirect = jax.scipy.ndimage.map_coordinates(
        transient_indirect, indices, 1, mode="constant"
    )
    return indirect
