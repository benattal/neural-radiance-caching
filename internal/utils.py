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
"""Utility functions."""

import concurrent
import enum
import os
import queue
import threading
import time
from typing import Any, Callable, Dict, Iterable, Optional, TypeVar, Union

import tensorflow as tf

gfile = tf.io.gfile
import cv2
import flax
import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax import random
from PIL import ExifTags, Image

_Array = Union[np.ndarray, jnp.ndarray]


import psutil

def log_memory_usage(msg=""):
    process = psutil.Process()
    memory_info = process.memory_info()
    logging.info(f"{msg}, memory usage: {memory_info.rss / (1024 * 1024)} MB")


def mse_to_psnr(mse):
  """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
  return -10.0 / np.log(10.0) * np.log(mse)


def get_sphere_directions(height, width, flip=False):
  # Returns coordinates (spherical and cartesian) for equirect points on
  # sphere, with half-pixel offsets (centers of equirect-spaced pixels).
  phi, theta = jnp.meshgrid(
      jnp.linspace(jnp.pi, -jnp.pi, width, endpoint=False)
      - 2.0 * jnp.pi / (2.0 * width),
      jnp.linspace(0.0, jnp.pi, height, endpoint=False)
      + jnp.pi / (2.0 * height),
  )

  # \delta theta * \delta phi (area of pixels in spherical coordinates),
  # for use in quadrature integration.
  dtheta_dphi = (2.0 * jnp.pi / width) * (jnp.pi / height)

  theta = theta.flatten()
  phi = phi.flatten()

  if flip:
    x = -jnp.cos(theta)
    y = jnp.sin(theta) * jnp.cos(phi)
    z = jnp.sin(theta) * jnp.sin(phi)
  else:
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)

  xyz = jnp.stack([x, y, z], axis=-1)

  return theta, phi, xyz, dtheta_dphi



def stopgrad_with_weight(x, weight):
    if not isinstance(x, jnp.ndarray) or jnp.issubdtype(x.dtype, jnp.integer) or (weight is None):
        return x
    elif not isinstance(weight, jnp.ndarray) and (weight == 1.0):
        return x
    elif not isinstance(weight, jnp.ndarray) and (weight == 0.0):
        return jax.lax.stop_gradient(x)
    else:
        return (x - jax.lax.stop_gradient(x)) * weight + jax.lax.stop_gradient(x)

def copy_tree(d):
    return jax.tree_util.tree_map(lambda x: x, d)

def partial_stopgrad_rays(rays, weight):
    """Applies stopgrad_with_weight if weight != (1.0, 1.0)."""
    if weight == (1.0, 1.0):
        return rays
    def _apply(x):
        return stopgrad_with_weight(x, weight[0]) if x is not None and weight is not None else x
    return jax.tree_util.tree_map(_apply, rays)

def apply_stopgrad_fields(results_dict, mapping):
    """Stopgrad multiple keys at once using provided weight mapping."""
    new_dict = {}
    for k, v in results_dict.items():
        if k in mapping:
            new_dict[k] = stopgrad_with_weight(v, mapping[k])
        else:
            new_dict[k] = v
    return new_dict

def random_split(rng):
    if rng is None:
        key = None
    else:
        key, rng = random.split(rng)
    return key, rng


@flax.struct.dataclass
class Pixels:
    """All tensors must have the same num_dims and first n-1 dims must match."""

    pix_x_int: _Array
    pix_y_int: _Array
    lossmult: _Array
    near: _Array
    far: _Array
    cam_idx: _Array
    light_idx: _Array
    exposure_idx: Optional[_Array] = None
    exposure_values: Optional[_Array] = None
    device_idx: Optional[_Array] = None


@flax.struct.dataclass
class Rays:
    """All tensors must have the same num_dims and first n-1 dims must match."""

    origins: _Array
    lights: _Array
    directions: _Array
    viewdirs: _Array
    radii: _Array
    imageplane: _Array
    look: _Array
    up: _Array
    cam_origins: _Array
    vcam_look: _Array
    vcam_up: _Array
    vcam_origins: _Array
    lossmult: _Array
    near: _Array
    far: _Array
    cam_idx: _Array
    light_idx: _Array
    normals: Optional[_Array] = None
    pix_x_int: Optional[_Array] = None
    pix_y_int: Optional[_Array] = None
    exposure_idx: Optional[_Array] = None
    exposure_values: Optional[_Array] = None
    device_idx: Optional[_Array] = None
    impulse_response: Optional[_Array] = None


def generate_random_rays(
    rng,
    n,
    origin_lo,
    origin_hi,
    radius_lo,
    radius_hi,
    near_lo,
    near_hi,
    far_lo,
    far_hi,
    include_exposure_idx=False,
    include_exposure_values=False,
    include_device_idx=False,
):
    """Generate a random Rays datastructure."""
    key, rng = random.split(rng)
    origins = random.uniform(key, shape=[n, 3], minval=origin_lo, maxval=origin_hi)

    key, rng = random.split(rng)
    directions = random.normal(key, shape=[n, 3])
    directions /= jnp.sqrt(
        jnp.maximum(
            jnp.finfo(jnp.float32).tiny,
            jnp.sum(directions**2, axis=-1, keepdims=True),
        )
    )

    viewdirs = directions

    key, rng = random.split(rng)
    radii = random.uniform(key, shape=[n, 1], minval=radius_lo, maxval=radius_hi)

    key, rng = random.split(rng)
    near = random.uniform(key, shape=[n, 1], minval=near_lo, maxval=near_hi)

    key, rng = random.split(rng)
    far = random.uniform(key, shape=[n, 1], minval=far_lo, maxval=far_hi)

    imageplane = jnp.zeros([n, 2])
    look = jnp.zeros([n, 3])
    up = jnp.zeros([n, 3])
    lossmult = jnp.ones([n, 1])

    cam_idx = jnp.int32(jnp.zeros([n, 1]))
    light_idx = jnp.int32(jnp.zeros([n, 1]))

    exposure_kwargs = {}
    if include_exposure_idx:
        exposure_kwargs["exposure_idx"] = jnp.zeros([n, 1], dtype=jnp.int32)
    if include_exposure_values:
        exposure_kwargs["exposure_values"] = jnp.zeros([n, 1])
    if include_device_idx:
        exposure_kwargs["device_idx"] = jnp.zeros([n, 1], dtype=jnp.int32)

    random_rays = Rays(
        origins=origins,
        lights=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        imageplane=imageplane,
        look=look,
        up=up,
        cam_origins=origins,
        vcam_look=look,
        vcam_up=up,
        vcam_origins=origins,
        lossmult=lossmult,
        near=near,
        far=far,
        cam_idx=cam_idx,
        light_idx=light_idx,
        pix_x_int=jnp.zeros_like(directions[Ellipsis, 0]),
        pix_y_int=jnp.zeros_like(directions[Ellipsis, 0]),
        **exposure_kwargs,
    )
    return random_rays


# Dummy Rays object that can be used to initialize NeRF model.
def dummy_rays(
    include_exposure_idx=False,
    include_exposure_values=False,
    include_device_idx=False,
):
    return generate_random_rays(
        random.PRNGKey(0),
        n=10,
        origin_lo=-1.5,
        origin_hi=1.5,
        radius_lo=1e-5,
        radius_hi=1e-3,
        near_lo=0.0,
        near_hi=1.0,
        far_lo=10,
        far_hi=10000,
        include_exposure_idx=include_exposure_idx,
        include_exposure_values=include_exposure_values,
        include_device_idx=include_device_idx,
    )


@flax.struct.dataclass
class Batch:
    """Data batch for NeRF training or testing."""

    rays: Union[Pixels, Rays]
    rgb: Optional[_Array] = None
    semantic: Optional[_Array] = None
    disps: Optional[_Array] = None
    normals: Optional[_Array] = None
    albedos: Optional[_Array] = None
    depth: Optional[_Array] = None
    alphas: Optional[_Array] = None
    masks: Optional[_Array] = None
    impulse_response: Optional[_Array] = None


class DataSplit(enum.Enum):
    """Dataset split."""

    TRAIN = "train"
    TEST = "test"


class BatchingMethod(enum.Enum):
    """Draw rays randomly from a single image or all images, in each batch."""

    ALL_IMAGES = "all_images"
    SINGLE_IMAGE = "single_image"


def open_file(pth, mode="r"):
    return gfile.GFile(pth, mode=mode)


def mv_file(src, dst, overwrite=True):
    return gfile.rename(src, dst, overwrite=overwrite)


def file_exists(pth):
    return gfile.exists(pth)


def listdir(pth):
    return gfile.listdir(pth)


def isdir(pth):
    return gfile.isdir(pth)


def makedirs(pth):
    gfile.makedirs(pth)


def device_is_tpu():
    return jax.local_devices()[0].platform == "tpu"


def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_util.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def unshard(x, padding=0):
    """Collect the sharded tensor to the shape before sharding."""
    y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
    if padding > 0:
        y = y[:-padding]
    return y


def load_img(pth, is_16bit=False):
    """Load an image and cast to float32."""
    with open_file(pth, "rb") as f:
        # Use OpenCV for reading 16-bit images, since PIL.Image.open() silently
        # casts those to 8-bit.
        if is_16bit:
            bytes_ = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Read bytes.
            image = np.array(cv2.imdecode(bytes_, cv2.IMREAD_UNCHANGED), dtype=np.float32)
        else:
            image = np.array(Image.open(f), dtype=np.float32)
    return image


def load_exr(pth):
    """Load an EXR image cast to float32."""
    with open_file(pth, "rb") as f:
        image = np.array(image_io.DecodeImage4f(f.read()), dtype=np.float32)
    return image


def load_npy(pth):
    """Load an numpy array cast to float32."""
    with open_file(pth, "rb") as f:
        image = np.load(f).astype(np.float32)
    return image


def load_exif(pth):
    """Load EXIF data for an image."""
    with open_file(pth, "rb") as f:
        image_pil = Image.open(f)
        exif_pil = image_pil._getexif()  # pylint: disable=protected-access
        if exif_pil is not None:
            exif = {ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS}
        else:
            exif = {}
    return exif

def filter_jax_dict(jax_dict):
    new_jax_dict = {}

    for key in jax_dict.keys():
        if isinstance(jax_dict[key], jnp.ndarray):
            new_jax_dict[key] = jax_dict[key]
    
    return new_jax_dict


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    img = Image.fromarray((np.round(np.clip(np.nan_to_num(img), 0.0, 1.0) * 255)).astype(np.uint8))
    tmp_pth = pth + ".tmp"
    with open_file(tmp_pth, "wb") as f:
        img.save(f, "PNG")
    mv_file(tmp_pth, pth)


def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    img = Image.fromarray(np.nan_to_num(depthmap).astype(np.float32))
    tmp_pth = pth + ".tmp"
    with open_file(tmp_pth, "wb") as f:
        img.save(f, "TIFF", compression="tiff_adobe_deflate")
    mv_file(tmp_pth, pth)


def assert_valid_stepfun(t, y):
    """Assert that step function (t, y) has a valid shape."""
    if t.shape[-1] != y.shape[-1] + 1:
        raise ValueError(f"Invalid shapes ({t.shape}, {y.shape}) for a step function.")


def assert_valid_linspline(t, y):
    """Assert that piecewise linear spline (t, y) has a valid shape."""
    if t.shape[-1] != y.shape[-1]:
        raise ValueError(f"Invalid shapes ({t.shape}, {y.shape}) for a linear spline.")


_FnT = TypeVar("_FnT", bound=Callable[Ellipsis, Iterable[Any]])


def iterate_in_separate_thread(
    queue_size=3,
):
    """Decorator factory that iterates a function in a separate thread.

    Args:
      queue_size: Keep at most queue_size elements in memory.

    Returns:
      Decorator that will iterate a function in a separate thread.
    """

    def decorator(
        fn,
    ):
        def result_fn(*args, **kwargs):
            results_queue = queue.Queue(queue_size)
            populating_data = True
            populating_data_lock = threading.Lock()

            def thread_fn():
                # Mark has_data as a variable that's outside of thread_fn
                # Otherwise, `populating_data = True` creates a local variable
                nonlocal populating_data
                try:
                    for item in fn(*args, **kwargs):
                        results_queue.put(item)
                finally:
                    # Set populating_data to False regardless of exceptions to stop
                    # iterations
                    with populating_data_lock:
                        populating_data = False

            # Use executor + futures instead of Thread to propagate exceptions
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                thread_fn_future = executor.submit(thread_fn)

                while True:
                    with populating_data_lock:
                        if not populating_data and results_queue.empty():
                            break
                    get_start = time.time()
                    try:
                        # Set timeout to allow for exceptions to be propagated.
                        next_value = results_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    logging.info("Got data in %0.3fs", time.time() - get_start)
                    yield next_value

                # Thread exception will be raised here
                thread_fn_future.result()

        return result_fn

    return decorator

def get_sphere_intersection(o, d, radius):
    dot_o_o = (o * o).sum(axis=-1, keepdims=True)
    dot_d_d = (d * d).sum(axis=-1, keepdims=True)
    dot_o_d = (o * d).sum(axis=-1, keepdims=True)

    a = dot_d_d
    b = 2 * dot_o_d
    c = dot_o_o - radius * radius
    disc = jnp.maximum(0.0, b * b - 4 * a * c)

    # Always return larger intersection
    return (-b + jnp.sqrt(disc + 1e-5)) / (2 * a)

# def render_envmap_sg(lgtSGs, viewdirs):
#     viewdirs = viewdirs.to(lgtSGs.device)
#     viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

#     # [M, 7] ---> [..., M, 7]
#     dots_sh = list(viewdirs.shape[:-2])
#     M = lgtSGs.shape[0]
#     lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
#     lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
#     lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
#     lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
#     # [..., M, 3]
#     rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
#         (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
#     rgb = torch.sum(rgb, dim=-2)  # [..., 3]
#     return rgb

# def fibonacci_sphere(samples=1):
#     points = []
#     phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
#     for i in range(samples):
#         z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
#         radius = np.sqrt(1 - z * z)  # radius at y

#         theta = phi * i  # golden angle increment

#         x = np.cos(theta) * radius
#         y = np.sin(theta) * radius

#         points.append([x, y, z])
#     points = np.array(points)
#     return points

# def init_sgs():
#     self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
#     self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))

#     # make sure lambda is not too close to zero
#     self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
#     # init envmap energy
#     energy = compute_energy(self.lgtSGs.data)
#     self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
#     energy = compute_energy(self.lgtSGs.data)
#     print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

#     # deterministicly initialize lobes
#     lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
#     self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
#     self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)

#     # rotation matrixs for incident light
#     self.light_rotation_matrix = []
#     for i in range(self.light_num):
#         horizontal_angle = torch.tensor(self.light_rotation[i] / 180 * torch.pi).to(torch.float32)
#         rotation_matrix = torch.tensor([[torch.cos(horizontal_angle), -torch.sin(horizontal_angle), 0], 
#                                         [torch.sin(horizontal_angle), torch.cos(horizontal_angle), 0], 
#                                         [0, 0, 1]]
#                                         ).to(torch.float32)

#         self.light_rotation_matrix.append(rotation_matrix)
#     self.light_rotation_matrix = torch.stack(self.light_rotation_matrix, dim=0) # [rotation_num, 3, 3]