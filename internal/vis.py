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
"""Helper functions for visualizing things."""

import jax.numpy as jnp
import numpy as np
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt 
from internal import stepfun
from internal import image
import pdb

def colorize(
    x,
    percentile=99.0,
    pos_color=(1, 0, 0),
    neg_color=(0, 0, 1),
    bg_color=(1, 1, 1),
    xnp=np,
):
    """Map +x to pos_color, -x to neg_color, and x==0 to bg_color."""
    max_val = xnp.percentile(xnp.abs(x), percentile)
    x_norm = x / max_val
    x_pos = xnp.clip(x_norm, 0, 1)
    x_neg = xnp.clip(-x_norm, 0, 1)
    x_zero = xnp.maximum(0, 1 - xnp.abs(x_norm))
    vis = (
        x_pos[Ellipsis, None] * xnp.array(pos_color)[None, None]
        + x_neg[Ellipsis, None] * xnp.array(neg_color)[None, None]
        + x_zero[Ellipsis, None] * xnp.array(bg_color)[None, None]
    )
    return vis


def weighted_percentile(x, w, ps, assume_sorted=False, xnp=np):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = xnp.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = xnp.cumsum(w)
    return xnp.interp(xnp.array(ps) * (acc_w[-1] / 100), acc_w, x)


def sinebow(h, xnp=np):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: xnp.sin(xnp.pi * x) ** 2
    return xnp.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8, xnp=np):
    return vis + (1.0 - acc[..., None])

    # """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    # bg_mask = xnp.logical_xor(
    #     (xnp.arange(acc.shape[-2]) % (2 * width) // width)[Ellipsis, :, None],
    #     (xnp.arange(acc.shape[-1]) % (2 * width) // width)[Ellipsis, None, :],
    # )
    # bg = xnp.where(bg_mask, light, dark)
    # return vis * acc[Ellipsis, None] + (bg * (1 - acc))[Ellipsis, None]


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
    xnp=np,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = xnp.finfo(xnp.float32).eps
    lo = lo or (lo_auto - eps)
    hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = xnp.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = xnp.clip((value - xnp.minimum(lo, hi)) / xnp.abs(hi - lo), 0, 1)
    value = xnp.nan_to_num(value)

    if colormap:
        colorized = colormap(value)[Ellipsis, :3]
    else:
        if value.shape[-1] != 3:
            raise ValueError(f"value must have 3 channels but has {value.shape[-1]}")
        colorized = value

    # return matte(colorized, weight) if matte_background else colorized
    return colorized

def visualize_rays(
    dist,
    dist_range,
    weights,
    rgbs,
    accumulate=False,
    renormalize=False,
    resolution=2048,
    bg_color=0.8,
    xnp=np,
):
    """Visualize a bundle of rays."""
    dist_vis = xnp.linspace(*dist_range, resolution + 1)
    vis_rgb, vis_alpha = [], []
    for ds, ws, rs in zip(dist, weights, rgbs):
        vis_rs, vis_ws = [], []
        for d, w, r in zip(ds, ws, rs):
            if accumulate:
                # Produce the accumulated color and weight at each point along the ray.
                w_csum = xnp.cumsum(w, axis=0)
                rw_csum = xnp.cumsum((r * w[:, None]), axis=0)
                eps = xnp.finfo(xnp.float32).eps
                r, w = (rw_csum + eps) / (w_csum[:, None] + 2 * eps), w_csum
            vis_rs.append(stepfun.resample(dist_vis, d, r.T, use_avg=True).T)
            vis_ws.append(stepfun.resample(dist_vis, d, w.T, use_avg=True).T)
        vis_rgb.append(xnp.stack(vis_rs))
        vis_alpha.append(xnp.stack(vis_ws))
    vis_rgb = xnp.stack(vis_rgb, axis=1)
    vis_alpha = xnp.stack(vis_alpha, axis=1)

    if renormalize:
        # Scale the alphas so that the largest value is 1, for visualization.
        vis_alpha /= xnp.maximum(xnp.finfo(xnp.float32).eps, xnp.max(vis_alpha))

    if resolution > vis_rgb.shape[0]:
        rep = resolution // (vis_rgb.shape[0] * vis_rgb.shape[1] + 1)
        stride = rep * vis_rgb.shape[1]

        vis_rgb = xnp.tile(vis_rgb, (1, 1, rep, 1)).reshape((-1,) + vis_rgb.shape[2:])
        vis_alpha = xnp.tile(vis_alpha, (1, 1, rep)).reshape((-1,) + vis_alpha.shape[2:])

        # Add a strip of background pixels after each set of levels of rays.
        vis_rgb = vis_rgb.reshape((-1, stride) + vis_rgb.shape[1:])
        vis_alpha = vis_alpha.reshape((-1, stride) + vis_alpha.shape[1:])
        vis_rgb = xnp.concatenate([vis_rgb, xnp.zeros_like(vis_rgb[:, :1])], axis=1).reshape((-1,) + vis_rgb.shape[2:])
        vis_alpha = xnp.concatenate([vis_alpha, xnp.zeros_like(vis_alpha[:, :1])], axis=1).reshape(
            (-1,) + vis_alpha.shape[2:]
        )

    # Matte the RGB image over the background.
    vis = vis_rgb * vis_alpha[Ellipsis, None] + (bg_color * (1 - vis_alpha))[Ellipsis, None]

    # Remove the final row of background pixels.
    vis = vis[:-1]
    vis_alpha = vis_alpha[:-1]
    return vis, vis_alpha

def squarify(fig):
    w, h = fig.get_size_inches()
    if w > h:
        t = fig.subplotpars.top
        b = fig.subplotpars.bottom
        axs = h*(t-b)
        l = (1.-axs/w)/2
        fig.subplots_adjust(left=l, right=1-l)
    else:
        t = fig.subplotpars.right
        b = fig.subplotpars.left
        axs = w*(t-b)
        l = (1.-axs/h)/2
        fig.subplots_adjust(bottom=l, top=1-l)


# def draw_transients(gt, rendered, pixels_to_plot=[(64, 64), (50, 25), (110, 64)], weights=None, dists=None):
#     plotting_transients = []
#     plotting_transients_gt = []
#     for pixel in pixels_to_plot:
#         plotting_transients.append(rendered[pixel[0], pixel[1], :, 0])
#         plotting_transients_gt.append(gt[pixel[0], pixel[1], :, 0])

#     figure = plt.figure(figsize=((len(pixels_to_plot)+1), 4), dpi=250)

#     # plot the predicted intensity
#     gt_img = rendered.sum(-2)
#     gt_img = (gt_img/gt_img.max())**(1/2.2)
#     plt.subplot(2, (len(pixels_to_plot)+1)//2, 1)
#     plt.imshow(gt_img)
#     for i, pixel in enumerate(pixels_to_plot):
#         plt.plot(pixel[1], pixel[0], '.', markersize=10, color='red')
#         plt.text(pixel[1], pixel[0], str(i), color="yellow", fontsize=10)
#     plt.gca().set_aspect(1.0/plt.gca().get_data_ratio(), adjustable='box')
#     plt.title('gt intensity')

#     for i, pixel in enumerate(pixels_to_plot):
#         # plot transients on a log scale 
#         plt.subplot(2, (len(pixels_to_plot)+1)//2, i+2)
#         plt.plot(np.arange(700), plotting_transients[i], label='pred', linewidth=0.5)
#         plt.plot(np.arange(700), plotting_transients_gt[i], label='gt', linewidth=0.5)
#         # pdb.set_trace()
#         # plt.axvline(x = (plotting_transients_depth[i]/args.exposure_time).detach().cpu().numpy(), color = 'y')
#         plt.ylabel('intensity')
#         # plt.yscale('log')

#         if weights is not None and dists is not None:
#             # Create a twin Axes on the right y-axis
#             ax2 = plt.gca().twinx()
#             ax2.scatter(dists[pixel[0], pixel[1]],  weights[pixel[0], pixel[1]], label='pts', alpha = 0.5, s=1, color='green')
#             ax2.set_ylabel('weights')
#             ax2.legend(borderpad=0, labelspacing=0)
#             # plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable='box')

#         plt.legend(borderpad=0, labelspacing=0)

#         # plt.gca().set_aspect(1.0 / plt.gca().get_data_ratio(), adjustable='box')


#     plt.tight_layout()
        
def draw_transients(gt, rendered, pixels_to_plot=[(64, 64), (50, 25), (90, 64)], weights=None, dists=None, indirect = None, gamma=1/2):
    plotting_transients = []
    plotting_transients_gt = []
    if indirect is not None:
        plotting_transients_indirect = []
    for pixel in pixels_to_plot:
        plotting_transients.append(rendered[pixel[0], pixel[1], :, 0])
        plotting_transients_gt.append(gt[pixel[0], pixel[1], :, 0])
        if indirect is not None:
            plotting_transients_indirect.append(indirect[pixel[0], pixel[1], :, 0])

    # Create figure with adjusted size and DPI
    figure = plt.figure(figsize=(10, 6), dpi=100)

    # Plot the ground truth intensity image
    if indirect is not None:
        plotting_img = (rendered+indirect).sum(-2)
    else:
        plotting_img = (rendered).sum(-2)
    plotting_img = (plotting_img / plotting_img.max()) ** (1 / 2.2)
    plt.subplot(2, (len(pixels_to_plot) + 1) // 2, 1)
    plt.imshow(plotting_img, cmap='viridis')  # Adjust colormap as needed
    for i, pixel in enumerate(pixels_to_plot):
        plt.plot(pixel[1], pixel[0], '.', markersize=10, color='red')
        plt.text(pixel[1] + 2, pixel[0], str(i), color="yellow", fontsize=10)  # Adjust text position and size
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure aspect ratio is equal
    plt.title('Rendered Intensity')
    plt.colorbar(label='Intensity')  # Add colorbar with appropriate label

    for i, pixel in enumerate(pixels_to_plot):
        # Plot transients
        plt.subplot(2, (len(pixels_to_plot) + 1) // 2, i + 2)
        plt.plot(np.arange(plotting_transients[i].shape[0]), plotting_transients[i]**gamma, label='direct', linewidth=0.8, color='blue')
        plt.plot(np.arange(plotting_transients[i].shape[0]), plotting_transients_gt[i]**gamma, label='gt', linewidth=0.8, color='black')
        if indirect is not None:
            plt.plot(np.arange(plotting_transients[i].shape[0]), plotting_transients_indirect[i]**gamma, label='indirect', linewidth=0.8, color='orange')

        plt.ylabel('Intensity')
        plt.xlabel('Time')
        plt.legend(loc='upper right')  # Adjust legend position

        # Plot weights and distances if available
        if weights is not None and dists is not None:
            ax2 = plt.gca().twinx()
            ax2.scatter(dists[pixel[0], pixel[1]], weights[pixel[0], pixel[1]], label='pts', alpha=0.5, s=10, color='orange')  # Adjust marker size and color
            ax2.set_ylabel('Weights')
            ax2.legend(loc='lower right')

    plt.tight_layout()

    # Adjust space between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
   
    canvas = FigureCanvas(figure)
    canvas.draw()

    # Convert the canvas to a numpy array
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return img/255

def visualize_suite(rendering, config, xnp=np, vis_material=False, vis_secondary=False):
    """A wrapper around other visualizations for easy integration."""

    # pdb.set_trace()
    depth_curve_fn = lambda x: -xnp.log(x + xnp.finfo(xnp.float32).eps)

    rgb = image.linear_to_srgb(rendering["rgb"])

    if "rgb_variance" in rendering:
        rgb_var = xnp.abs(rendering["rgb_variance"]) * (config.var_scale / config.img_scale)
    else:
        rgb_var = xnp.zeros_like(rgb)

    cache_rgb = image.linear_to_srgb(rendering["cache_rgb"])
    cache_rgb0 = image.linear_to_srgb(rendering["cache_rgb"] / rendering["cache_rgb"].max())
    cache_diffuse_rgb = image.linear_to_srgb(rendering["cache_diffuse_rgb"])
    cache_specular_rgb = image.linear_to_srgb(rendering["cache_specular_rgb"])
    cache_occ = rendering["cache_occ"] * jnp.ones_like(rgb)
    cache_indirect_occ = rendering["cache_indirect_occ"] * jnp.ones_like(rgb)
    cache_direct_rgb = image.linear_to_srgb(rendering["cache_direct_rgb"])
    cache_indirect_rgb = image.linear_to_srgb(rendering["cache_indirect_rgb"])
    cache_ambient_rgb = image.linear_to_srgb(rendering["cache_ambient_rgb"])
    cache_irradiance_rgb = image.linear_to_srgb(rendering["cache_irradiance_rgb"])
    cache_albedo_rgb = image.linear_to_srgb(rendering["cache_albedo_rgb"])
    cache_direct_diffuse_rgb = image.linear_to_srgb(rendering["cache_direct_diffuse_rgb"])
    cache_direct_specular_rgb = image.linear_to_srgb(rendering["cache_direct_specular_rgb"])
    cache_indirect_diffuse_rgb = image.linear_to_srgb(rendering["cache_indirect_diffuse_rgb"])
    cache_indirect_specular_rgb = image.linear_to_srgb(rendering["cache_indirect_specular_rgb"])
    cache_ambient_diffuse_rgb = image.linear_to_srgb(rendering["cache_ambient_diffuse_rgb"])
    cache_ambient_specular_rgb = image.linear_to_srgb(rendering["cache_ambient_specular_rgb"])
    lossmult = rendering["lossmult"]

    if "material_rgb" in rendering:
        diffuse_rgb = image.linear_to_srgb(rendering["diffuse_rgb"])
        specular_rgb = image.linear_to_srgb(rendering["specular_rgb"])
        occ = rendering["occ"] * jnp.ones_like(rgb)
        indirect_occ = rendering["indirect_occ"] * jnp.ones_like(rgb)
        direct_rgb = image.linear_to_srgb(rendering["direct_rgb"])
        indirect_rgb = image.linear_to_srgb(rendering["indirect_rgb"])

        direct_diffuse_rgb = image.linear_to_srgb(rendering["direct_diffuse_rgb"])
        direct_specular_rgb = image.linear_to_srgb(rendering["direct_specular_rgb"])
        indirect_diffuse_rgb = image.linear_to_srgb(rendering["indirect_diffuse_rgb"])
        indirect_specular_rgb = image.linear_to_srgb(rendering["indirect_specular_rgb"])
    else:
        diffuse_rgb = xnp.zeros_like(rgb)
        specular_rgb = xnp.zeros_like(rgb)
        occ = xnp.zeros_like(rgb)
        indirect_occ = xnp.zeros_like(rgb)
        direct_rgb = xnp.zeros_like(rgb)
        indirect_rgb = xnp.zeros_like(rgb)

        direct_diffuse_rgb = jnp.ones_like(rgb)
        direct_specular_rgb = jnp.ones_like(rgb)
        indirect_diffuse_rgb = jnp.ones_like(rgb)
        indirect_specular_rgb = jnp.ones_like(rgb)

    if "irradiance_cache" in rendering:
        irradiance_cache = image.linear_to_srgb(
            rendering["irradiance_cache"]
        )
    else:
        irradiance_cache = xnp.zeros_like(rgb)

    if "lighting_irradiance" in rendering:
        lighting_irradiance = image.linear_to_srgb(
            rendering["lighting_irradiance"]
        )
    else:
        lighting_irradiance = xnp.zeros_like(rgb)

    if "cache_env_map_rgb" in rendering:
        cache_env_map_rgb = image.linear_to_srgb(rendering["cache_env_map_rgb"])
        cache_env_map_rgb0 = image.linear_to_srgb(rendering["cache_env_map_rgb"] / rendering["cache_rgb"].max())
    else:
        cache_env_map_rgb = xnp.zeros_like(rgb)
        cache_env_map_rgb0 = xnp.zeros_like(rgb)

    if "cache_incoming_rgb" in rendering:
        cache_incoming_rgb = image.linear_to_srgb(rendering["cache_incoming_rgb"])
        cache_incoming_rgb0 = image.linear_to_srgb(rendering["cache_incoming_rgb"] / rendering["cache_rgb"].max())
        cache_incoming_s_dist = rendering["cache_incoming_s_dist"] * xnp.ones_like(rgb)
        cache_incoming_acc = rendering["cache_incoming_acc"] * xnp.ones_like(rgb)
    else:
        cache_incoming_rgb = xnp.zeros_like(rgb)
        cache_incoming_rgb0 = xnp.zeros_like(rgb)
        cache_incoming_s_dist = xnp.zeros_like(rgb)
        cache_incoming_acc = xnp.zeros_like(rgb)
    
    if "material_albedo" in rendering:
        material_albedo = (rendering["material_albedo"] * xnp.ones_like(rgb)) ** (1.0 / 2.2)
        material_roughness = rendering["material_roughness"] * xnp.ones_like(rgb)
        material_metalness = rendering["material_metalness"] * xnp.ones_like(rgb)
        material_diffuseness = rendering["material_diffuseness"] * xnp.ones_like(rgb)
        material_mirrorness = rendering["material_mirrorness"] * xnp.ones_like(rgb)
        material_F_0 = rendering["material_F_0"] * xnp.ones_like(rgb)
    else:
        material_albedo = xnp.zeros_like(rgb)
        material_roughness = xnp.zeros_like(rgb)
        material_metalness = xnp.zeros_like(rgb)
        material_diffuseness = xnp.zeros_like(rgb)
        material_mirrorness = xnp.zeros_like(rgb)
        material_F_0 = xnp.zeros_like(rgb)

    acc = xnp.stack(rendering["acc"], axis=0)

    distance_mean = xnp.stack(rendering["distance_mean"], axis=0)
    distance_median = xnp.stack(rendering["distance_median"], axis=0)
    distance_p5 = xnp.stack(rendering["distance_percentile_5"], axis=0)
    distance_p95 = xnp.stack(rendering["distance_percentile_95"], axis=0)
    acc = xnp.where(xnp.isnan(distance_mean), xnp.zeros_like(acc), acc)

    p = 99.0
    lo, hi = weighted_percentile(distance_median, acc, [50 - p / 2, 50 + p / 2])

    def vis_fn(x):
        return visualize_cmap(jnp.array(x), jnp.array(acc), cm.get_cmap("turbo"), lo=lo, hi=hi, curve_fn=depth_curve_fn, xnp=jnp)

    vis_depth_mean, vis_depth_median = [vis_fn(x) for x in [distance_mean, distance_median]]
    
    
    # Render three depth percentiles directly to RGB channels, where the spacing
    # determines the color. delta == big change, epsilon = small change.
    #   Gray: A strong discontinuitiy, [x-epsilon, x, x+epsilon]
    #   Purple: A thin but even density, [x-delta, x, x+delta]
    #   Red: A thin density, then a thick density, [x-delta, x, x+epsilon]
    #   Blue: A thick density, then a thin density, [x-epsilon, x, x+delta]
    vis_depth_triplet = visualize_cmap(
        jnp.array(xnp.stack(
            [2 * distance_median - distance_p5, distance_median, distance_p95],
            axis=-1,
        )),
        jnp.array(acc),
        None,
        curve_fn=lambda x: xnp.log(x + xnp.finfo(xnp.float32).eps),
        xnp=jnp,
    )

    # Need to convert from array back to list for tensorboard logging.
    # This applies to all buffers except rgb and roughness.
    is_list = isinstance(rendering["acc"], list)
    unstack_fn = lambda z: list(z) if is_list else xnp.nan_to_num(z)
    unstack1_fn = lambda z: list(z)[-1:] if is_list else xnp.nan_to_num(z)
    vis = {
        "acc": unstack_fn(acc),
        "depth_mean": unstack1_fn(vis_depth_mean),
        "depth_median": unstack1_fn(vis_depth_median),
        "lossmult": unstack1_fn(lossmult),
        "color": unstack1_fn(rgb),
        "color_var": unstack1_fn(rgb_var),
        "color_cache": unstack1_fn(cache_rgb),
        "color_cache0": unstack1_fn(cache_rgb0),
        "cache_diffuse_color": unstack1_fn(cache_diffuse_rgb),
        "cache_specular_color": unstack1_fn(cache_specular_rgb),
        "cache_occ": unstack1_fn(cache_occ),
        "cache_indirect_occ": unstack1_fn(cache_indirect_occ),
        "cache_direct_color": unstack1_fn(cache_direct_rgb),
        "cache_indirect_color": unstack1_fn(cache_indirect_rgb),
        "cache_ambient_color": unstack1_fn(cache_ambient_rgb),
        "cache_irradiance_color": unstack1_fn(cache_irradiance_rgb),
        "cache_albedo_color": unstack1_fn(cache_albedo_rgb),
        "cache_direct_diffuse_color": unstack1_fn(cache_direct_diffuse_rgb),
        "cache_direct_specular_color": unstack1_fn(cache_direct_specular_rgb),
        "cache_indirect_diffuse_color": unstack1_fn(cache_indirect_diffuse_rgb),
        "cache_indirect_specular_color": unstack1_fn(cache_indirect_specular_rgb),
        "cache_ambient_diffuse_color": unstack1_fn(cache_ambient_diffuse_rgb),
        "cache_ambient_specular_color": unstack1_fn(cache_ambient_specular_rgb),
        "slf_rgb": unstack1_fn(cache_incoming_rgb),
        "slf_rgb0": unstack1_fn(cache_incoming_rgb0),
        "env_map_rgb": unstack1_fn(cache_env_map_rgb),
        "env_map_rgb0": unstack1_fn(cache_env_map_rgb0),
        "slf_depth": unstack1_fn(cache_incoming_s_dist),
        "slf_acc": unstack1_fn(cache_incoming_acc),
    }

    if vis_material:
        vis_material_dict = {
            "color_irradiance_cache": unstack1_fn(irradiance_cache),
            "material_albedo": unstack1_fn(matte(material_albedo, acc)),
            "material_roughness": unstack1_fn(matte(material_roughness, acc)),
            "material_F_0": unstack1_fn(matte(material_F_0, acc)),
            "material_metalness": unstack1_fn(matte(material_metalness, acc)),
            "material_diffuseness": unstack1_fn(matte(material_diffuseness, acc)),
            "material_mirrorness": unstack1_fn(matte(material_mirrorness, acc)),
            "material_diffuse_color": unstack1_fn(diffuse_rgb),
            "material_specular_color": unstack1_fn(specular_rgb),
            "material_occ": unstack1_fn(occ),
            "material_indirect_occ": unstack1_fn(indirect_occ),
            "material_direct_color": unstack1_fn(direct_rgb),
            "material_indirect_color": unstack1_fn(indirect_rgb),
            "material_direct_diffuse_color": unstack1_fn(direct_diffuse_rgb),
            "material_direct_specular_color": unstack1_fn(direct_specular_rgb),
            "material_indirect_diffuse_color": unstack1_fn(indirect_diffuse_rgb),
            "material_indirect_specular_color": unstack1_fn(indirect_specular_rgb),
            "material_lighting_irradiance": unstack1_fn(lighting_irradiance),
        }

        vis = dict(**vis, **vis_material_dict)

        
    # Render every item named "normals*".
    for key, val in rendering.items():
        if key.startswith("normals"):
            vis[key] = unstack_fn(matte(np.stack(val, axis=0) / 2.0 + 0.5, acc))

    return vis

def visualize_transient_suite(rendering, config, xnp=np, vis_material=False, vis_secondary=False):
    """A wrapper around other visualizations for easy integration."""
    depth_curve_fn = lambda x: -xnp.log(x + xnp.finfo(xnp.float32).eps)

    rgb = image.linear_to_srgb(xnp.clip(rendering["rgb"].sum(-2) / config.img_scale, 0, 1))

    if "rgb_variance" in rendering:
        rgb_var = xnp.abs(rendering["rgb_variance"]).sum(-2) * (config.var_scale / config.img_scale)
    else:
        rgb_var = xnp.zeros_like(rgb)

    cache_rgb_linear = rendering["cache_rgb"].sum(-2)
    cache_rgb = image.linear_to_srgb(cache_rgb_linear)
    cache_rgb0 = image.linear_to_srgb(np.clip(cache_rgb_linear / config.img_scale, 0, 1))
    direct_rgb_no_integration = image.linear_to_srgb(rendering["direct_rgb_viz"] / rendering["direct_rgb_viz"].max())

    if "color_gt" in rendering.keys() and not vis_secondary:
        w = rendering["weights"] if "weights" in rendering.keys() else None 
        d = rendering["dists"] if "dists" in rendering.keys() else None 
        indir = rendering["transient_indirect_viz"] if "transient_indirect_viz" in rendering.keys() else None 

        if indir is not None:
            transient_plot = draw_transients(rendering["color_gt"], rendering["transient_direct_viz"], weights=w, dists=d, indirect=indir)
        else:
            transient_plot = np.zeros((1000, 1000, 3))
    else:
        transient_plot = np.zeros((1000, 1000, 3))

    cache_diffuse_rgb = image.linear_to_srgb(rendering["cache_diffuse_rgb"] / config.img_scale)
    cache_specular_rgb = image.linear_to_srgb(rendering["cache_specular_rgb"] / config.img_scale)
    cache_indirect_occ = rendering["cache_indirect_occ"] * jnp.ones_like(rgb)
    vignette = rendering["vignette"] / rendering["vignette"].max()
    cache_direct_rgb = image.linear_to_srgb(rendering["cache_direct_rgb"] / config.img_scale)
    cache_indirect_rgb = image.linear_to_srgb(rendering["cache_indirect_rgb"] / config.img_scale)
    cache_ambient_rgb = image.linear_to_srgb(rendering["cache_ambient_rgb"])
    cache_irradiance_rgb = image.linear_to_srgb(rendering["cache_irradiance_rgb"] / np.max(rendering["cache_irradiance_rgb"]))
    cache_light_radiance_rgb = rendering["cache_light_radiance_rgb"] / np.max(rendering["cache_light_radiance_rgb"])
    cache_n_dot_l_rgb = image.linear_to_srgb(rendering["cache_n_dot_l_rgb"] / np.max(rendering["cache_n_dot_l_rgb"]))
    cache_albedo_rgb = image.linear_to_srgb(rendering["cache_albedo_rgb"])
    cache_direct_diffuse_rgb = image.linear_to_srgb(rendering["cache_direct_diffuse_rgb"] / config.img_scale)
    cache_direct_specular_rgb = image.linear_to_srgb(rendering["cache_direct_specular_rgb"] / config.img_scale)
    cache_indirect_diffuse_rgb = image.linear_to_srgb(rendering["cache_indirect_diffuse_rgb"] / config.img_scale)
    cache_indirect_specular_rgb = image.linear_to_srgb(rendering["cache_indirect_specular_rgb"] / config.img_scale)
    cache_ambient_diffuse_rgb = image.linear_to_srgb(rendering["cache_ambient_diffuse_rgb"])
    cache_ambient_specular_rgb = image.linear_to_srgb(rendering["cache_ambient_specular_rgb"])
    lossmult = rendering["lossmult"]

    if "cache_occ" in rendering:
        cache_occ = rendering["cache_occ"] * jnp.ones_like(rgb)
    else:
        cache_occ = xnp.zeros_like(rgb)

    if "material_rgb" in rendering:
        material_rgb = image.linear_to_srgb(rendering["material_rgb"])
        diffuse_rgb = image.linear_to_srgb(rendering["diffuse_rgb"]/config.img_scale)
        specular_rgb = image.linear_to_srgb(rendering["specular_rgb"]/config.img_scale)
        occ = rendering["occ"] * jnp.ones_like(rgb)
        indirect_occ = rendering["indirect_occ"] * jnp.ones_like(rgb)
        direct_rgb = image.linear_to_srgb(rendering["direct_rgb"]/config.img_scale)
        indirect_rgb = image.linear_to_srgb(rendering["indirect_rgb"]/config.img_scale)

        direct_diffuse_rgb = image.linear_to_srgb(rendering["direct_diffuse_rgb"] / config.img_scale)
        direct_specular_rgb = image.linear_to_srgb(rendering["direct_specular_rgb"] / config.img_scale)
        indirect_diffuse_rgb = image.linear_to_srgb(rendering["indirect_diffuse_rgb"] / config.img_scale)
        indirect_specular_rgb = image.linear_to_srgb(rendering["indirect_specular_rgb"] / config.img_scale)
    else:
        material_rgb = xnp.zeros_like(rgb)
        diffuse_rgb = jnp.ones_like(rgb)
        specular_rgb = jnp.ones_like(rgb)
        occ = jnp.ones_like(rgb)
        indirect_occ = jnp.ones_like(rgb)
        direct_rgb = jnp.ones_like(rgb)
        indirect_rgb = jnp.ones_like(rgb)

        direct_diffuse_rgb = jnp.ones_like(rgb)
        direct_specular_rgb = jnp.ones_like(rgb)
        indirect_diffuse_rgb = jnp.ones_like(rgb)
        indirect_specular_rgb = jnp.ones_like(rgb)

    if "relight_rgb" in rendering:
        relight_rgb = image.linear_to_srgb(rendering["relight_rgb"])
    else:
        relight_rgb = xnp.zeros_like(rgb)

    if "irradiance_cache" in rendering:
        irradiance_cache = image.linear_to_srgb(
            rendering["irradiance_cache"]
        )
    else:
        irradiance_cache = xnp.zeros_like(rgb)

    if "lighting_irradiance" in rendering:
        lighting_irradiance = image.linear_to_srgb(
            rendering["lighting_irradiance"] / np.max(rendering["cache_irradiance_rgb"])
        )
    else:
        lighting_irradiance = xnp.zeros_like(rgb)

    if "material_albedo" in rendering:
        material_albedo = (rendering["material_albedo"] * xnp.ones_like(rgb)) ** (1.0 / 2.2)
        material_roughness = rendering["material_roughness"] * xnp.ones_like(rgb)
        material_metalness = rendering["material_metalness"] * xnp.ones_like(rgb)
        material_diffuseness = rendering["material_diffuseness"] * xnp.ones_like(rgb)
        material_mirrorness = rendering["material_mirrorness"] * xnp.ones_like(rgb)
        material_F_0 = rendering["material_F_0"] * xnp.ones_like(rgb)
    else:
        material_albedo = xnp.zeros_like(rgb)
        material_roughness = xnp.zeros_like(rgb)
        material_metalness = xnp.zeros_like(rgb)
        material_diffuseness = xnp.zeros_like(rgb)
        material_mirrorness = xnp.zeros_like(rgb)
        material_F_0 = xnp.zeros_like(rgb)

    acc = xnp.stack(rendering["acc"], axis=0)

    distance_mean = xnp.stack(rendering["distance_mean"], axis=0)
    distance_median = xnp.stack(rendering["distance_median"], axis=0)
    distance_p5 = xnp.stack(rendering["distance_percentile_5"], axis=0)
    distance_p95 = xnp.stack(rendering["distance_percentile_95"], axis=0)
    acc = xnp.where(xnp.isnan(distance_mean), xnp.zeros_like(acc), acc)

    p = 99.0
    lo, hi = weighted_percentile(distance_median, acc, [50 - p / 2, 50 + p / 2])

    def vis_fn(x):
        return visualize_cmap(jnp.array(x), jnp.array(acc), cm.get_cmap("turbo"), lo=lo, hi=hi, curve_fn=depth_curve_fn, xnp=jnp)

    vis_depth_mean, vis_depth_median = [vis_fn(x) for x in [distance_mean, distance_median]]

    if "depth_gt" in rendering.keys():
        depth_gt = vis_fn(rendering["depth_gt"])

    # Render three depth percentiles directly to RGB channels, where the spacing
    # determines the color. delta == big change, epsilon = small change.
    #   Gray: A strong discontinuitiy, [x-epsilon, x, x+epsilon]
    #   Purple: A thin but even density, [x-delta, x, x+delta]
    #   Red: A thin density, then a thick density, [x-delta, x, x+epsilon]
    #   Blue: A thick density, then a thin density, [x-epsilon, x, x+delta]
    vis_depth_triplet = visualize_cmap(
        jnp.array(xnp.stack(
            [2 * distance_median - distance_p5, distance_median, distance_p95],
            axis=-1,
        )),
        jnp.array(acc),
        None,
        curve_fn=lambda x: xnp.log(x + xnp.finfo(xnp.float32).eps),
        xnp=jnp,
    )

    # Need to convert from array back to list for tensorboard logging.
    # This applies to all buffers except rgb and roughness.
    is_list = isinstance(rendering["acc"], list)
    unstack_fn = lambda z: list(z) if is_list else z
    unstack1_fn = lambda z: list(z)[-1:] if is_list else z
    vis = {
        "acc": unstack_fn(acc),
        "depth_mean": unstack1_fn(vis_depth_mean),
        "depth_median": unstack1_fn(vis_depth_median),
        "lossmult": unstack1_fn(lossmult),
        "vignette": unstack1_fn(vignette),
        "color": unstack1_fn(rgb),
        "color_var": unstack1_fn(rgb_var),
        "color_cache": unstack1_fn(cache_rgb),
        "color_cache0": unstack1_fn(cache_rgb0),
        "cache_diffuse_color": unstack1_fn(cache_diffuse_rgb),
        "cache_specular_color": unstack1_fn(cache_specular_rgb),
        "cache_occ": unstack1_fn(cache_occ),
        "cache_indirect_occ": unstack1_fn(cache_indirect_occ),
        "cache_direct_color": unstack1_fn(cache_direct_rgb),
        "cache_indirect_color": unstack1_fn(cache_indirect_rgb),
        "cache_ambient_color": unstack1_fn(cache_ambient_rgb),
        "cache_irradiance_color": unstack1_fn(cache_irradiance_rgb),
        "cache_light_radiance_color": unstack1_fn(cache_light_radiance_rgb),
        "cache_n_dot_l_color": unstack1_fn(cache_n_dot_l_rgb),
        "cache_albedo_color": unstack1_fn(cache_albedo_rgb),
        "cache_direct_diffuse_color": unstack1_fn(cache_direct_diffuse_rgb),
        "cache_direct_specular_color": unstack1_fn(cache_direct_specular_rgb),
        "cache_indirect_diffuse_color": unstack1_fn(cache_indirect_diffuse_rgb),
        "cache_indirect_specular_color": unstack1_fn(cache_indirect_specular_rgb),
        "cache_ambient_diffuse_color": unstack1_fn(cache_ambient_diffuse_rgb),
        "cache_ambient_specular_color": unstack1_fn(cache_ambient_specular_rgb),
        "transient_plot": unstack1_fn(transient_plot),
    }

    if vis_material:
        vis_material_dict = {
            "color_irradiance_cache": unstack1_fn(irradiance_cache),
            "material_albedo": unstack1_fn(matte(material_albedo, acc)),
            "material_roughness": unstack1_fn(matte(material_roughness, acc)),
            "material_F_0": unstack1_fn(matte(material_F_0, acc)),
            "material_metalness": unstack1_fn(matte(material_metalness, acc)),
            "material_diffuseness": unstack1_fn(matte(material_diffuseness, acc)),
            "material_mirrorness": unstack1_fn(matte(material_mirrorness, acc)),
            "material_diffuse_color": unstack1_fn(diffuse_rgb),
            "material_specular_color": unstack1_fn(specular_rgb),
            "material_occ": unstack1_fn(occ),
            "material_indirect_occ": unstack1_fn(indirect_occ),
            "material_direct_color": unstack1_fn(direct_rgb),
            "material_indirect_color": unstack1_fn(indirect_rgb),
            "material_direct_diffuse_color": unstack1_fn(direct_diffuse_rgb),
            "material_direct_specular_color": unstack1_fn(direct_specular_rgb),
            "material_indirect_diffuse_color": unstack1_fn(indirect_diffuse_rgb),
            "material_indirect_specular_color": unstack1_fn(indirect_specular_rgb),
            "material_lighting_irradiance": unstack1_fn(lighting_irradiance),
            "direct_rgb_no_integration": unstack1_fn(direct_rgb_no_integration)
        }

        vis = dict(**vis, **vis_material_dict)

    if "depth_gt" in rendering.keys():
        vis["depth_gt"] = unstack1_fn(depth_gt)

    # Render every item named "normals*".
    for key, val in rendering.items():
        if key.startswith("normals"):
            vis[key] = unstack_fn(matte(np.stack(val, axis=0) / 2.0 + 0.5, acc))

    return vis