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

import dataclasses
import inspect
import jax.numpy as jnp
import jax
from jax import vmap
from typing import Any
from types import FunctionType
import flax
from functools import partial
import gin
import numpy as np
from scipy.spatial.transform import Rotation as R

from internal import ref_utils
from internal import grid_utils
from internal import image
from internal import math as math_utils
from internal.inverse_render import math

# Make pytype skip this file. EnvmapSampler has Sampler1D as one of its fields,
# which breaks pytype for some reason...
# pytype: skip-file

DENOMINATOR_EPS = 1e-5

def get_directions(envmap_H, envmap_W):
  omega_phi, omega_theta = jnp.meshgrid(
      jnp.linspace(-jnp.pi, jnp.pi, envmap_W + 1)[:-1]
      + 2.0 * jnp.pi / (2.0 * envmap_W),
      jnp.linspace(0.0, jnp.pi, envmap_H + 1)[:-1] + jnp.pi / (2.0 * envmap_H),
  )

  dtheta_dphi = (omega_theta[1, 1] - omega_theta[0, 0]) * (
      omega_phi[1, 1] - omega_phi[0, 0]
  )

  omega_theta = omega_theta.flatten()
  omega_phi = omega_phi.flatten()

  omega_x = jnp.sin(omega_theta) * jnp.cos(omega_phi)
  omega_y = jnp.sin(omega_theta) * jnp.sin(omega_phi)
  omega_z = jnp.cos(omega_theta)
  omega_xyz = jnp.stack([omega_x, omega_y, omega_z], axis=-1)

  return omega_theta, omega_phi, omega_xyz, dtheta_dphi


def get_rays(H, W, focal, c2w, rand_ort=False, key=None):
  """c2w: 4x4 matrix

  output: two arrays of shape [H, W, 3]
  """
  j, i = jnp.meshgrid(
      jnp.arange(W, dtype=jnp.float32) + 0.5,
      jnp.arange(H, dtype=jnp.float32) + 0.5,
  )

  if rand_ort:
    k1, k2 = jax.random.split(key)

    i += jax.random.uniform(k1, shape=(H, W)) - 0.5
    j += jax.random.uniform(k2, shape=(H, W)) - 0.5

  dirs = jnp.stack(
      [
          (j.flatten() - 0.5 * W) / focal,
          -(i.flatten() - 0.5 * H) / focal,
          -jnp.ones((H * W,), dtype=jnp.float32),
      ],
      -1,
  )  # shape [HW, 3]

  rays_d = math.matmul(dirs, c2w[:3, :3].T)  # shape [HW, 3]
  rays_o = c2w[:3, -1:].T.repeat(H * W, 0)
  return rays_o.reshape(H, W, 3), rays_d.reshape(H, W, 3)


def get_rays_at_pixel_coords(pixel_coords, H, W, focal, c2w, rand_ort=False, key=None):
  """c2w: 4x4 matrix

  output: two arrays of shape [pixel_coords.shape[0], 3]
  """
  sh = pixel_coords.shape[0]
  i, j = jnp.unravel_index(pixel_coords, (H, W))
  i += 0.5
  j += 0.5

  if rand_ort:
    k1, k2 = jax.random.split(key)

    i += jax.random.uniform(k1, shape=i.shape) - 0.5
    j += jax.random.uniform(k2, shape=j.shape) - 0.5

  dirs = jnp.stack(
      [
          (j.flatten() - 0.5 * W) / focal,
          -(i.flatten() - 0.5 * H) / focal,
          -jnp.ones_like(i.flatten(), dtype=jnp.float32),
      ],
      -1,
  )  # shape [sh, 3]

  rays_d = math.matmul(dirs, c2w[:3, :3].T)  # shape [sh, 3]
  rays_o = c2w[:3, -1:].T.repeat(sh, 0)
  return rays_o.reshape(sh, 3), rays_d.reshape(sh, 3)


def get_random_ray_offsets(N, focal, c2w, key, randomness='uniform'):
  """c2w: 4x4 matrix

  output: N random vector shifts for d
  """

  if randomness == 'uniform':
    di, dj = jax.random.uniform(key, shape=(2, N)) - 0.5
  elif randomness == 'gaussian':
    di, dj = jax.random.normal(key, shape=(2, N)) * 0.5
  else:
    raise ValueError('Only uniform or gaussian')

  delta_dirs = jnp.stack(
      [dj / focal, -di / focal, jnp.zeros((N,), dtype=jnp.float32)], -1
  )  # shape [N, 3]

  return math.matmul(delta_dirs, c2w[:3, :3].T)  # shape [N, 3]


def get_rotation_matrix(normal, y_up=False):
  # Get rotation matrix mapping [0, 0, 1] to normal.
  # new_z = normal
  # old_x = jnp.array([1.0, 0.0, 0.0])
  # dp = normal[0] #normal.dot(old_x) # if this is 1 we're not going to be happy...
  # new_x = (old_x - dp * normal) / jnp.sqrt(1.0 - dp ** 2)
  # new_y = jnp.cross(new_z, new_x)

  old_z = jnp.array([0.0, 0.0, 1.0])[None]
  old_y = jnp.array([0.0, 1.0, 0.0])[None]

  if y_up:
    up = jnp.where(jnp.abs(normal[Ellipsis, 1:2]) < 0.9, old_y, old_z)
  else:
    up = jnp.where(jnp.abs(normal[Ellipsis, 2:3]) < 0.9, old_z, old_y)

  new_x = jnp.cross(up, normal)
  new_x = new_x / (jnp.linalg.norm(new_x, axis=-1, keepdims=True) + 1e-10)
  new_z = normal
  new_y = jnp.cross(new_z, new_x)
  new_y = new_y / (jnp.linalg.norm(new_y, axis=-1, keepdims=True) + 1e-10)

  R = jnp.stack([new_x, new_y, new_z], axis=-1)
  return R

def get_rotation_matrix_y_up(normal):
  # Get rotation matrix mapping [0, 0, 1] to normal.
  # new_z = normal
  # old_x = jnp.array([1.0, 0.0, 0.0])
  # dp = normal[0] #normal.dot(old_x) # if this is 1 we're not going to be happy...
  # new_x = (old_x - dp * normal) / jnp.sqrt(1.0 - dp ** 2)
  # new_y = jnp.cross(new_z, new_x)

  old_z = jnp.array([0.0, 0.0, 1.0])[None]
  old_y = jnp.array([0.0, 1.0, 0.0])[None]
  up = jnp.where(jnp.abs(normal[Ellipsis, 1:2]) < 0.9, old_y, old_z)
  new_x = jnp.cross(up, normal)
  new_x = new_x / (jnp.linalg.norm(new_x, axis=-1, keepdims=True) + 1e-10)
  new_z = normal
  new_y = jnp.cross(new_z, new_x)
  new_y = new_y / (jnp.linalg.norm(new_y, axis=-1, keepdims=True) + 1e-10)

  R = jnp.stack([new_x, new_y, new_z], axis=-1)
  return R


@gin.configurable
class EnvironmentSampler:
  global_dirs: bool = True
  return_rgb: bool = True
  samples_to_take: int = 256
  deterministic: bool = False

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    num_samples = u1.shape[-1]
    bs = wo.reshape(-1, num_samples, 3).shape[0]

    pdf = kwargs["env_map_pmf"]
    pdf_return = kwargs["env_map_pdf"]
    light_dirs = kwargs["env_map_dirs"]
    light_rgbs = kwargs["env_map"]

    # sampled the light directions
    if (bs * num_samples) % self.samples_to_take != 0:
      samples_to_take = bs * num_samples
      reps = 1
    else:
      samples_to_take = self.samples_to_take
      reps = (bs * num_samples // self.samples_to_take)

    key, rng = jax.random.split(rng)
    light_dir_idx = jax.random.categorical(
        key,
        math_utils.safe_log(pdf),
        axis=-2,
        shape=(
            pdf.shape[:-2] + (samples_to_take, pdf.shape[-1])
        )
    )

    # Importance sample
    print(self.samples_to_take, samples_to_take, bs, num_samples, bs * num_samples, (bs * num_samples) // (samples_to_take), light_dir_idx.shape, light_dirs.shape)
    light_dirs = jnp.repeat(
        jax.lax.stop_gradient(jnp.take_along_axis(light_dirs, light_dir_idx[..., None], axis=-3)),
        reps,
        0,
    ).reshape(u1.shape + (-1, 3,))
    light_pdf = jnp.repeat(
        jax.lax.stop_gradient(jnp.take_along_axis(pdf_return, light_dir_idx, axis=-2)),
        reps,
        0
    ).reshape(u1.shape + (-1,))
    light_rgbs = jnp.repeat(
        jax.lax.stop_gradient(jnp.take_along_axis(light_rgbs, light_dir_idx[..., None], axis=-3)),
        reps,
        0,
    ).reshape(u1.shape + (-1, 3,))

    # Take for current light
    light_idx = light_idx.reshape(u1.shape[:-1] + (1, 1))
    light_dirs = jnp.take_along_axis(light_dirs, light_idx[..., None], axis=-2)[..., 0, :]
    light_pdf = jnp.take_along_axis(light_pdf, light_idx, axis=-1)[..., 0]
    light_rgbs = jnp.take_along_axis(light_rgbs, light_idx[..., None], axis=-2)[..., 0, :]

    return light_dirs, light_pdf, light_rgbs

  def pdf(self, wo, wi, _, kwargs):
    pass


class MirrorSampler:
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    wi = math.reflect(wo)[None, :]
    pdf = jnp.ones_like(wi[Ellipsis, 0])  # jnp.ones_like(u1)
    return wi, pdf

  def pdf(self, wo, wi, _):
    return jnp.zeros_like(wi[Ellipsis, 2])


@flax.struct.dataclass
class QuadratureEnvmapSampler:
  sintheta: Any
  omega_xyz: Any
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  @classmethod
  # @partial(jax.jit, static_argnums=(0, 1, 2))
  def create(cls, envmap_H, envmap_W):
    omega_theta, _, omega_xyz, _ = get_directions(envmap_H, envmap_W)
    sintheta = jnp.sin(omega_theta)
    return cls(sintheta, omega_xyz)

  # @jax.jit
  def sample_directions(self, rng, _, __):
    pdf = 1.0 / (2.0 * jnp.pi**2 * self.sintheta)
    return self.omega_xyz, pdf

  # @jax.jit
  def pdf(self, directions):
    """This shouldn't really be called by multiple importance sampling, but I think it should still work?"""
    curr_sintheta = jnp.sqrt(1.0 - directions[Ellipsis, 2] ** 2)
    pdf = 1.0 / (2.0 * jnp.pi**2 * curr_sintheta)


@flax.struct.dataclass
class RandomGenerator2D:
  h_blocks: Any
  w_blocks: Any
  stratified: Any

  @classmethod
  def create(cls, n, stratified):
    h_blocks = int(2 ** jnp.int32(jnp.floor((jnp.log2(n) - 1) / 2.0)))
    w_blocks = h_blocks * 2
    h_shifts = (
        jnp.linspace(0.0, 1.0, w_blocks + 1)[:-1][None, :]
        .repeat(n // w_blocks, 0)
        .flatten()
    )
    w_shifts = (
        jnp.linspace(0.0, 1.0, h_blocks + 1)[:-1][:, None]
        .repeat(n // h_blocks, 1)
        .flatten()
    )
    return cls(h_blocks, w_blocks, h_shifts, w_shifts, stratified)

  # @functools.partial(jax.jit, static_argnames=['n', 'stratified'])
  def sample(self, rng, n, _):
    # Generate uniform samples on the top hemisphere
    key, rng = random_split(rng)
    u = jax.random.uniform(key, shape=(n, 2))
    uh = u[Ellipsis, 0]
    uw = u[Ellipsis, 1]

    if self.stratified:
      h_shifts = (
          jnp.linspace(0.0, 1.0, self.w_blocks + 1)[:-1][None, :]
          .repeat(n // self.w_blocks, 0)
          .flatten()
      )
      w_shifts = (
          jnp.linspace(0.0, 1.0, self.h_blocks + 1)[:-1][:, None]
          .repeat(n // self.h_blocks, 1)
          .flatten()
      )

      uh = jnp.clip(
          h_shifts + uh / self.w_blocks,
          0.0,
          1.0 - jnp.finfo(jnp.float32).eps,
      )
      uw = jnp.clip(
          w_shifts + uw / self.h_blocks,
          0.0,
          1.0 - jnp.finfo(jnp.float32).eps,
      )

    return uh, uw


@flax.struct.dataclass
class DummySampler2D:
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  def sample(self, _, __, ___):
    return None, None


@flax.struct.dataclass
class UniformSphereSampler:
  global_dirs: bool = True
  return_rgb: bool = False
  deterministic: bool = False

  # @jax.jit
  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    costheta = 1.0 - 2.0 * u1
    sintheta = jnp.sqrt((1.0 - u1) * 4.0 * u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = sintheta * jnp.cos(phi)
    wi_y = sintheta * jnp.sin(phi)
    wi_z = costheta
    pdf = 1 / 4.0 / jnp.pi * jnp.ones_like(phi)
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), pdf

  # @jax.jit
  def pdf(self, wo, wi, _, kwargs):
    return 1 / 4.0 / jnp.pi * jnp.ones_like(wi[Ellipsis, 2])


class UniformHemisphereSampler:
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    costheta = 1.0 - u1
    sintheta = jnp.sqrt((2.0 - u1) * u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = sintheta * jnp.cos(phi)
    wi_y = sintheta * jnp.sin(phi)
    wi_z = costheta
    pdf = 1 / 2.0 / jnp.pi * jnp.ones_like(phi)
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), pdf

  def pdf(self, wo, wi, _, kwargs):
    pdf = 1 / 2.0 / jnp.pi * jnp.ones_like(wi[Ellipsis, 2])

    pdf = jnp.where(
        wi[Ellipsis, 2] < 0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return jnp.maximum(pdf, 0.0)


class CosineSampler:
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    r = jnp.sqrt(u1)
    phi = u2 * 2.0 * jnp.pi - jnp.pi
    wi_x = r * jnp.cos(phi)
    wi_y = r * jnp.sin(phi)
    eps = DENOMINATOR_EPS
    wi_z = jnp.sqrt(jnp.maximum(eps, 1.0 - wi_x**2 - wi_y**2))
    pdf = wi_z / jnp.pi
    return jnp.stack([wi_x, wi_y, wi_z], axis=-1), jnp.maximum(pdf, 0.0)

  def pdf(self, wo, wi, _, kwargs):
    pdf = wi[Ellipsis, 2] / jnp.pi

    pdf = jnp.where(
        wi[Ellipsis, 2] < 0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return jnp.maximum(pdf, 0.0)


class IdentitySampler:
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = True

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    return wo, jnp.ones_like(wo[..., 0])

  def pdf(self, wo, wi, _, kwargs):
    return jnp.ones_like(wo[..., 0])


class ActiveSampler:
  global_dirs: bool = True
  return_rgb: bool = False
  deterministic: bool = True

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    light_offset = kwargs['lights'] - kwargs['origins']
    light_dists = jnp.linalg.norm(light_offset, axis=-1, keepdims=True)
    light_dirs = light_offset / jnp.maximum(light_dists, 1e-5)
    return light_dirs.reshape(wo.shape), jnp.ones_like(wo[..., 0])

  def pdf(self, wo, wi, _, kwargs):
    return jnp.ones_like(wo[..., 0])


def GGX_D(costheta, a):
  eps = jnp.finfo(jnp.float32).eps
  return a**2 / jnp.maximum(eps, jnp.pi * ((costheta ** 2 * (a ** 2 - 1.) + 1.)) ** 2)


@flax.struct.dataclass
class MicrofacetSampler:
  sample_visible: bool = False
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False

  def trowbridge_reitz_sample_11(self, u1, u2, costheta):
    pass

  def trowbridge_reitz_sample(self, u1, u2, alpha, wi):
    """
    https://github.com/mmp/pbrt-v3/blob/aaa552a4b9cbf9dccb71450f47b268e0ed6370e2/src/core/microfacet.cp284
    """
    pass

  def sample_normals(self, u1, u2, alpha):
    if self.sample_visible:
      raise NotImplementedError('')
    else:
      eps = jnp.finfo(jnp.float32).eps
      tantheta2 = alpha ** 2 * u1 / jnp.maximum(1.0 - u1, eps)
      costheta = 1.0 / jnp.sqrt(jnp.maximum(1.0 + tantheta2, eps))
      sintheta = jnp.sqrt(jnp.maximum(DENOMINATOR_EPS, 1.0 - costheta ** 2))
      phi = u2 * 2.0 * jnp.pi - jnp.pi
      nx = sintheta * jnp.cos(phi)
      ny = sintheta * jnp.sin(phi)
      nz = costheta

      pdf = GGX_D(costheta, alpha) * jnp.abs(costheta)
      return jnp.stack([nx, ny, nz], axis=-1), jnp.maximum(pdf, 0.0)

  def sample_directions(self, rng, u1, u2, wo, alpha, light_idx, kwargs):
    normals, normal_pdf = self.sample_normals(u1, u2, alpha[Ellipsis, 0])

    directions = jax.vmap(math.reflect, in_axes=(0, 0))(wo, normals)
    eps = jnp.finfo(jnp.float32).eps
    jac = 1.0 / jnp.maximum(4.0 * jnp.sum(wo * normals, axis=-1), eps)
    pdf = normal_pdf * jac

    pdf = jnp.where(
        jnp.sum(wo * normals, axis=-1) <= 0.0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return math.normalize(directions), jnp.maximum(pdf, 0.0)


  def pdf(self, wo, wi, alpha, kwargs):
    normals = math.normalize(wo + wi)
    eps = jnp.finfo(jnp.float32).eps
    jac = 1.0 / jnp.maximum(4.0 * jnp.sum(wo * normals, axis=-1), eps)
    pdf = GGX_D(normals[Ellipsis, 2], alpha[Ellipsis, 0]) * jnp.abs(normals[Ellipsis, 2]) * jac

    pdf = jnp.where(
        jnp.sum(wo * normals, axis=-1) <= 0.0,
        jnp.zeros_like(pdf),
        pdf,
    )

    return jnp.maximum(pdf, 0.0)


class DefensiveMicrofacetSampler:
  microfacet_sampler: MicrofacetSampler
  cosine_sampler: CosineSampler
  global_dirs: bool = False
  return_rgb: bool = False
  deterministic: bool = False


  def sample_directions(self, rng, u1, u2, wo, alpha, light_idx):
    pass


  def pdf(self, wo, wi, alpha):
    pass


# @functools.partial(jax.jit, static_argnums=(3,))
def get_lobe(wi, wo, normal, materials, brdf_correction, config):
  """Compute BRDF in local coordinates.

  wi: incoming light directions, shape [N, 3]
  wo: outgoing light direction, shape [3]
  materials: dictionary with elements of shapes [3]

  return values of BRDF evaluated at wi, wo. Shape is [N, 3]
  """
  # print(wi.shape, wo.shape, materials['specular_albedo'].shape)
  # assert opts.shading in ['lambertian', 'phong', 'blinnphong', 'mirror']

  if config.shading in ['mirror']:
    return 1.0

  lobe = 0.0

  if config.shading in ['lambertian', 'phong', 'blinnphong', 'microfacet']:
    lobe = (
        jnp.maximum(0.0, wi[Ellipsis, 2:])
        * materials['albedo'][Ellipsis, None, :]
        / jnp.pi
    )
  
  if 'microfacet' in config.shading:
    assert 'roughness' in materials.keys() and 'F_0' in materials.keys()
    eps = jnp.finfo(jnp.float32).eps 
    roughness = materials['roughness'][Ellipsis, None, :]
    F_0 = materials['F_0'][Ellipsis, None, :]

    albedo = materials['albedo'][Ellipsis, None, :]
    metalness = materials['metalness'][Ellipsis, None, :]

    if config.use_specular_albedo:
      specular_albedo = materials['specular_albedo'][Ellipsis, None, :]
    else:
      specular_albedo = albedo

    if config.use_mirrorness:
      mirrorness = materials['mirrorness'][Ellipsis, None, :]
    else:
      mirrorness = jnp.ones_like(metalness)

    if config.use_diffuseness:
      diffuseness = materials['diffuseness'][Ellipsis, None, :]

      if not config.use_mirrorness:
        mirrorness = (1.0 - diffuseness)
    else:
      diffuseness = (1.0 - metalness)

    F_0 = specular_albedo * metalness + F_0 * (1.0 - metalness)

    halfdirs = math.normalize(wi + wo)
    n_dot_v = jnp.maximum(0., math.dot(normal, wo))
    n_dot_l = jnp.maximum(0., math.dot(normal, wi))
    n_dot_h = jnp.maximum(0., math.dot(normal, halfdirs))
    l_dot_h = jnp.maximum(0., math.dot(wi, halfdirs))
    a = roughness

    def fresnel(cos_theta):
      return F_0 + (1. - F_0) * jnp.power(jnp.clip(1. - cos_theta, 0.0, 1.0), 5)

    D = GGX_D(n_dot_h, a)
    F = fresnel(l_dot_h)
    # k = (jnp.sqrt(roughness + 1e-5) + 1.) ** 2 / 8.
    k = a / 2
    # G = (n_dot_v / (1e-5 + n_dot_v * (1. - k) + k)) * (n_dot_l / (1e-5 + n_dot_l * (1. - k) + k))
    # ggx_lobe = (D * F * G / (1e-5 + 4. * n_dot_v))
    G = (n_dot_v / jnp.maximum(eps, n_dot_v * (1. - k) + k)) * (n_dot_l / jnp.maximum(eps, n_dot_l * (1. - k) + k))
    ggx_lobe = (D * F * G / jnp.maximum(eps, 4. * n_dot_v))
    lambertian_lobe = n_dot_l * albedo / jnp.pi

  if config.shading == 'microfacet':
    lobe = (
        (
            ggx_lobe
        ) * brdf_correction[Ellipsis, 0:1] * mirrorness
        + (
            lambertian_lobe
        ) * brdf_correction[Ellipsis, 1:2] * diffuseness
    )

  if config.shading == 'microfacet_diffuse':
    lobe = (
        (
            lambertian_lobe
        ) * brdf_correction[Ellipsis, 1:2]
    ) * diffuseness

  if config.shading == 'microfacet_specular':
    lobe = (
        (
            ggx_lobe
        ) * brdf_correction[Ellipsis, 0:1]
    ) * mirrorness

  if config.shading == 'phong':
    assert 'specular_albedo' in materials.keys()
    specular_albedo = materials['specular_albedo'][Ellipsis, None, :]
    exponent = materials['specular_exponent'][Ellipsis, None, :]
    refdir = math.reflect(wo)

    # No need to normalize because ||n|| = 1 and ||d|| = 1, so ||2(n.d)n - d|| = 1.
    print('Not normalizing here (because unnecessary, at least theoretically).')
    # refdirs /= (jnp.linalg.norm(refdirs, axis=-1, keepdims=True) + 1e-10)  # [N, HW, envmap_H, envmap_W, 3]

    lobe += (
        specular_albedo
        * jnp.maximum(0.0, (refdir * wi).sum(-1, keepdims=True)) ** exponent
    )

    # lobe += jnp.maximum(0.0, wi[..., 2:]) * materials['albedo'][..., None, :] / jnp.pi

  """
    if shading == 'blinnphong':
        assert 'specular_albedo' in materials.keys()
        specular_albedo = materials['specular_albedo'][:, None, None, :]
        exponent = materials['specular_exponent'][:, None, None, :]

        d_norm_sq = (rays_d ** 2).sum(-1, keepdims=True)
        rays_d_norm = -rays_d / jnp.sqrt(d_norm_sq + 1e-10)

        halfvectors = omega_xyz.reshape(1, envmap_H, envmap_W, 3) + rays_d_norm[:, None, None, :]
        halfvectors /= (jnp.linalg.norm(halfvectors, axis=-1, keepdims=True) + 1e-10)  # [N, envmap_H, envmap_W, 3]

        lobes += jnp.maximum(0.0, (halfvectors * normals[:, None, None, :]).sum(-1, keepdims=True)) ** exponent * specular_albedo

    """
  return lobe


def global_to_local(directions, R):
  return (
      directions[Ellipsis, 0:1] * R[Ellipsis, 0, :]
      + directions[Ellipsis, 1:2] * R[Ellipsis, 1, :]
      + directions[Ellipsis, 2:3] * R[Ellipsis, 2, :]
  )

def local_to_global(directions, R):
  return (
      directions[Ellipsis, 0:1] * R[Ellipsis, 0]
      + directions[Ellipsis, 1:2] * R[Ellipsis, 1]
      + directions[Ellipsis, 2:3] * R[Ellipsis, 2]
  )


def random_split(rng):
  if rng is None:
    key = None
  else:
    key, rng = jax.random.split(rng)

  return key, rng


def importance_sample_rays(
    rng,
    global_viewdirs,
    normal,
    material,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
    light_sampler_results=None,
):
  deterministic = all(sampler.deterministic for sampler, _ in samplers)

  rotation_mat = get_rotation_matrix(normal)
  local_viewdirs = global_to_local(global_viewdirs, rotation_mat)
  roughness = material.get('roughness', jnp.ones_like(local_viewdirs))

  if light_sampler_results is not None:
    light_idx = light_sampler_results.get('light_idx', jnp.ones_like(local_viewdirs[..., :1]).astype(jnp.int32))
  else:
    light_idx = jnp.ones_like(local_viewdirs[..., :1]).astype(jnp.int32)

  # Resample
  num_real_samples = sum(sample_count for _, sample_count in samplers)
  resample = num_real_samples > num_secondary_samples

  # Calculate MIS samples (directions)
  local_lightdirs = []
  pdf = []
  rgb = []
  weight = []
  del_rgb = False

  for sampler, sample_count in samplers:
    if resample:
      real_sample_count = sample_count
    else:
      real_sample_count = int(
          round(
              (float(sample_count) / num_real_samples) * num_secondary_samples
          )
      )

    # Get random samples in [0, 1)^2
    key, rng = random_split(rng)
    uh, uw = random_generator_2d.sample(
        key, local_viewdirs.shape[0] * real_sample_count, stratified_sampling
    )
    uh = uh.reshape(local_viewdirs.shape[0], real_sample_count)
    uw = uw.reshape(local_viewdirs.shape[0], real_sample_count)

    # Current inputs
    cur_local_viewdirs = jnp.repeat(
        local_viewdirs[Ellipsis, None, :], real_sample_count, axis=-2
    )
    cur_roughness = jnp.repeat(
        roughness[Ellipsis, None, :], real_sample_count, axis=-2
    )
    cur_light_idx = light_idx

    # Set sample rng
    key, rng = random_split(rng)

    # Importance sample
    if sampler.return_rgb:
      cur_local_lightdirs, cur_pdf, cur_rgb = sampler.sample_directions(
          key,
          uh,
          uw,
          cur_local_viewdirs,
          cur_roughness,
          cur_light_idx,
          light_sampler_results,
      )
    else:
      cur_local_lightdirs, cur_pdf = sampler.sample_directions(
          key,
          uh,
          uw,
          cur_local_viewdirs,
          cur_roughness,
          cur_light_idx,
          light_sampler_results
      )
      del_rgb = True

      cur_rgb = jnp.ones_like(cur_pdf)[..., None]

    if sampler.global_dirs:
      cur_local_lightdirs = global_to_local(
          cur_local_lightdirs, rotation_mat[Ellipsis, None, :, :]
      )

    # Calculate MIS weights
    if (
        use_mis
        and len(samplers) > 1
    ):
      denominator = 0.0

      for sampler_p, sample_count_p in samplers:
        if sampler_p.global_dirs:
          temp_viewdirs = local_to_global(
              cur_local_viewdirs, rotation_mat[Ellipsis, None, :, :]
          )
          temp_lightdirs = local_to_global(
              cur_local_lightdirs, rotation_mat[Ellipsis, None, :, :]
          )
        else:
          temp_viewdirs = cur_local_viewdirs
          temp_lightdirs = cur_local_lightdirs

        # Heuristic weight denominator
        denominator += jnp.square(
            sampler_p.pdf(
                temp_viewdirs,
                temp_lightdirs,
                cur_roughness,
                light_sampler_results
            ) * sample_count_p
        )

      # Heuristic weight
      cur_pdf = jnp.maximum(cur_pdf, 0.0)
      denominator = jnp.maximum(denominator, DENOMINATOR_EPS)
      cur_weight = jnp.square(sample_count * cur_pdf) / denominator

      # Correct total energy
      cur_weight = cur_weight * (
          float(num_real_samples) / float(sample_count)
      )
    else:
      cur_pdf = jnp.maximum(cur_pdf, 0.0)
      cur_weight = jnp.ones_like(cur_pdf)

    # Append
    local_lightdirs.append(cur_local_lightdirs)
    pdf.append(cur_pdf)
    weight.append(cur_weight)
    rgb.append(cur_rgb)

  # Concatenate
  local_lightdirs = jnp.concatenate(local_lightdirs, axis=-2)
  local_viewdirs = jnp.repeat(
      local_viewdirs[Ellipsis, None, :],
      num_secondary_samples,
      axis=-2
  )
  global_viewdirs = jnp.repeat(
      global_viewdirs[Ellipsis, None, :],
      num_secondary_samples,
      axis=-2
  )
  pdf = jnp.concatenate(pdf, axis=-1)[Ellipsis, None]
  weight = jnp.concatenate(weight, axis=-1)[Ellipsis, None]
  rgb = jnp.concatenate(rgb, axis=-2)

  # Global bounce directions
  global_lightdirs = local_to_global(
      local_lightdirs,
      rotation_mat[Ellipsis, None, :, :]
  )

  # Samples
  samples = {
      'local_lightdirs': local_lightdirs,
      'local_viewdirs': local_viewdirs,
      'global_lightdirs': global_lightdirs,
      'global_viewdirs': global_viewdirs,
      'pdf': jax.lax.stop_gradient(pdf),
      'rgb': jax.lax.stop_gradient(rgb),
      'weight': jax.lax.stop_gradient(weight),
  }

  if del_rgb:
    del samples['rgb']

  # Select one secondary sample
  if resample:
    probs = jnp.ones_like(pdf)

    key, rng = random_split(rng)
    inds = jax.random.categorical(
        key,
        math_utils.safe_log(probs),
        axis=-2,
        shape=(
            pdf.shape[:-2] + (num_secondary_samples,)
        )
    )[Ellipsis, None]

    samples = jax.tree_util.tree_map(
        lambda x: jnp.take_along_axis(x, inds, axis=-2),
        samples
    )

    samples['weight'] = (
        samples['weight']
        * float(num_real_samples) / float(num_secondary_samples)
    )

  return samples


def get_secondary_rays(
    rng,
    rays,
    means,
    viewdirs,
    normals,
    material,
    normal_eps=1e-2,
    refdir_eps=1e-2,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
    light_sampler_results=None,
    offset_origins=False,
    light_rotation=None,
    far=None,
):
  # Reflected ray origins
  ref_origins = means + jax.lax.stop_gradient(normals * normal_eps)
  ref_origins = jnp.repeat(
      ref_origins[Ellipsis, None, :], num_secondary_samples, axis=-2
  )

  # Reflected ray directions
  global_viewdirs = -viewdirs[Ellipsis, None, :] * jnp.ones_like(means)
  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )

  if light_sampler_results is not None and ("env_map" not in light_sampler_results):
    light_sampler_results = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[-2:]),
        light_sampler_results
    )

  key, rng = random_split(rng)
  ref_samples = importance_sample_rays(
      key,
      global_viewdirs.reshape(-1, 3),
      normals.reshape(-1, 3),
      material,
      random_generator_2d=random_generator_2d,
      stratified_sampling=stratified_sampling,
      use_mis=use_mis,
      samplers=samplers,
      num_secondary_samples=num_secondary_samples,
      light_sampler_results=light_sampler_results,
  )

  # Create reflect rays
  new_sh = (-1, num_secondary_samples, 3,)

  ref_rays = jax.tree_util.tree_map(lambda x: x, rays)
  # import pdb
  # pdb.set_trace()
  ref_rays = ref_rays.replace(
      near=(
          refdir_eps * jnp.ones_like(ref_origins[Ellipsis, :1])
      ).reshape(new_sh[:-1] + (1,)),
      far=(
          (rays.far[Ellipsis, None, None] if far is None else far) * jnp.ones_like(ref_origins[Ellipsis, :1])
      ).reshape(new_sh[:-1] + (1,)),
      cam_idx=(
          rays.cam_idx[Ellipsis, None, None] * jnp.ones_like(ref_origins[Ellipsis, :1]).astype(rays.cam_idx.dtype)
      ).reshape(new_sh[:-1] + (1,)),
      light_idx=(
          rays.light_idx[Ellipsis, None, None] * jnp.ones_like(ref_origins[Ellipsis, :1]).astype(rays.light_idx.dtype)
      ).reshape(new_sh[:-1] + (1,)),
      lights=(
          rays.lights[Ellipsis, None, None, :] * jnp.ones_like(ref_origins)
      ).reshape(new_sh[:-1] + (3,)),
      imageplane=(
          rays.imageplane[Ellipsis, None, None, :] * jnp.ones_like(ref_origins[..., :2])
      ).reshape(new_sh[:-1] + (2,)),
      look=(
          rays.look[Ellipsis, None, None, :] * jnp.ones_like(ref_origins)
      ).reshape(new_sh[:-1] + (3,)),
      up=(
          rays.up[Ellipsis, None, None, :] * jnp.ones_like(ref_origins)
      ).reshape(new_sh[:-1] + (3,)),
      cam_origins=(
          rays.cam_origins[Ellipsis, None, None, :] * jnp.ones_like(ref_origins[..., :3])
      ).reshape(new_sh[:-1] + (3,)),
      vcam_look=(
          rays.vcam_look[Ellipsis, None, None, :] * jnp.ones_like(ref_origins)
      ).reshape(new_sh[:-1] + (3,)),
      vcam_up=(
          rays.vcam_up[Ellipsis, None, None, :] * jnp.ones_like(ref_origins)
      ).reshape(new_sh[:-1] + (3,)),
      vcam_origins=(
          rays.vcam_origins[Ellipsis, None, None, :] * jnp.ones_like(ref_origins[..., :3])
      ).reshape(new_sh[:-1] + (3,)),
      origins=ref_origins.reshape(new_sh),
      directions=ref_samples['global_lightdirs'].reshape(new_sh),
      viewdirs=ref_samples['global_lightdirs'].reshape(new_sh),
  )

  ref_rays = ref_rays.replace(
      radii=jnp.ones_like(ref_rays.directions[Ellipsis, :1]),
      lossmult=(
          rays.lossmult[Ellipsis, None, None] * jnp.ones_like(ref_origins[Ellipsis, :1]).astype(rays.lossmult.dtype)
      ).reshape(new_sh[:-1] + (1,)),
  )

  if offset_origins:
    ref_rays = ref_rays.replace(
        origins=ref_rays.origins + ref_rays.directions * ref_rays.near,
        near=jnp.zeros_like(ref_rays.near),
    )

  if light_rotation is not None:
    ref_rays = ref_rays.replace(
        directions=local_to_global(
            ref_rays.directions, light_rotation.reshape(-1, 1, 3, 3)
        ),
        viewdirs=local_to_global(
            ref_rays.viewdirs, light_rotation.reshape(-1, 1, 3, 3)
        ),
    )

  # Reshape sample outputs
  ref_samples = jax.tree_util.tree_map(
      lambda x: x.reshape(new_sh[:-1] + (x.shape[-1],)),
      ref_samples
  )

  return ref_rays, ref_samples


def get_outgoing_rays(
    rng,
    rays,
    viewdirs,
    normals,
    material,
    random_generator_2d=None,
    stratified_sampling=False,
    use_mis=True,
    samplers=None,
    num_secondary_samples=None,
):
  # Reflected ray directions
  global_viewdirs = -viewdirs[Ellipsis, None, :] * jnp.ones_like(normals)
  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )

  key, rng = random_split(rng)
  ref_samples = importance_sample_rays(
      key,
      global_viewdirs.reshape(-1, 3),
      normals.reshape(-1, 3),
      material,
      random_generator_2d=random_generator_2d,
      stratified_sampling=stratified_sampling,
      use_mis=use_mis,
      samplers=samplers,
      num_secondary_samples=num_secondary_samples,
  )

  # Create reflect rays
  ref_rays = jax.tree_util.tree_map(lambda x: x, rays)
  ref_rays = ref_rays.replace(
      viewdirs=-ref_samples['global_lightdirs'].reshape(
          rays.viewdirs.shape
      )
  )

  return ref_rays


def integrate_reflect_rays(
    material_type,
    use_brdf_correction,
    material,
    samples,
    use_diffuseness=False,
    use_mirrorness=False,
    use_specular_albedo=False,
    max_radiance=float("inf"),
):
  eps = jnp.finfo(jnp.float32).eps
  config = type('', (), {})()
  config.shading = material_type
  config.use_brdf_correction = use_brdf_correction
  config.use_diffuseness = use_diffuseness
  config.use_mirrorness = use_mirrorness
  config.use_specular_albedo = use_specular_albedo

  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )
  local_normals = jnp.concatenate(
      [
          jnp.zeros_like(samples['local_lightdirs'][..., 0:1]),
          jnp.zeros_like(samples['local_lightdirs'][..., 0:1]),
          jnp.ones_like(samples['local_lightdirs'][..., 0:1]),
      ],
      axis=-1,
  )
  material_lobe = get_lobe(
      samples['local_lightdirs'],
      samples['local_viewdirs'],
      local_normals,
      material,
      samples['brdf_correction'],
      config
  )

  denominator = jnp.maximum(
      samples['pdf'],
      DENOMINATOR_EPS
  )
  weight = jnp.maximum(
      samples['weight'],
      0.0
  )
  weight = jnp.where(
      samples['local_lightdirs'][Ellipsis, 2:] > 0.0, weight, jnp.zeros_like(weight)
  )

  # Outgoing radiance
  radiance_out = (
      jnp.clip(samples['radiance_in'] * material_lobe, 0.0, max_radiance)
      * weight / denominator
  ).mean(1)

  indirect_occ = samples['indirect_occ'].mean(1)

  # Incoming irradiance
  diffuse_lobe = (
      jnp.maximum(0., samples['local_lightdirs'][Ellipsis, 2:]) / jnp.pi
  )
  irradiance = (
      jnp.clip(samples['radiance_in'] * diffuse_lobe, 0.0, max_radiance)
      * weight / denominator
  ).mean(1)

  # Multipliers
  if use_brdf_correction:
    integrated_multiplier = (
        samples['brdf_correction']
        * weight / denominator
    ).mean(1) / (2 * jnp.pi)

    integrated_multiplier_irradiance = (
        samples['brdf_correction'][Ellipsis, 1:2]
        * samples['radiance_in']
        * diffuse_lobe
        * weight / denominator
    ).mean(1)
  else:
    integrated_multiplier = samples['brdf_correction'][:, 0]
    integrated_multiplier_irradiance = samples['brdf_correction'][:, 0, :1]

  return dict(
      radiance_out=radiance_out,
      indirect_occ=indirect_occ,
      irradiance=irradiance,
      integrated_multiplier=integrated_multiplier,
      integrated_multiplier_irradiance=integrated_multiplier_irradiance,
  )

def transient_integrate_reflect_rays(
    material_type,
    use_brdf_correction,
    material,
    samples,
    use_diffuseness=False,
    use_mirrorness=False,
    use_specular_albedo=False,
    direct=True,
    max_radiance=float("inf"),
):
  import pdb
  eps = jnp.finfo(jnp.float32).eps
  config = type('', (), {})()
  config.shading = material_type
  config.use_brdf_correction = use_brdf_correction
  config.use_diffuseness = use_diffuseness
  config.use_mirrorness = use_mirrorness
  config.use_specular_albedo = use_specular_albedo

  material = jax.tree_util.tree_map(
      lambda x: x.reshape(-1, x.shape[-1]),
      material
  )
  local_normals = jnp.concatenate(
      [
          jnp.zeros_like(samples['local_lightdirs'][..., 0:1]),
          jnp.zeros_like(samples['local_lightdirs'][..., 0:1]),
          jnp.ones_like(samples['local_lightdirs'][..., 0:1]),
      ],
      axis=-1,
  )
  material_lobe = get_lobe(
      samples['local_lightdirs'],
      samples['local_viewdirs'],
      local_normals,
      material,
      samples['brdf_correction'],
      config
  )

  denominator = jnp.maximum(
      samples['pdf'],
      DENOMINATOR_EPS
  )
  weight = jnp.maximum(
      samples['weight'],
      0.0
  )
  weight = jnp.where(
      samples['local_lightdirs'][Ellipsis, 2:] > 0.0, weight, jnp.zeros_like(weight)
  )

  # Incoming irradiance
  diffuse_lobe = (
      jnp.maximum(0., samples['local_lightdirs'][Ellipsis, 2:]) / jnp.pi
  )

  if direct:
    # Outgoing radiance
    radiance_out = (
        jnp.clip(samples['radiance_in'] * material_lobe, 0.0, max_radiance)
        * weight / denominator
    ).mean(1)

    indirect_occ = None

    irradiance = (
        jnp.clip(samples['radiance_in'] * diffuse_lobe, 0.0, max_radiance)
        * weight / denominator
    ).mean(1)
  else:
    radiance_out = (
        jnp.clip(samples['radiance_in'] * material_lobe[..., None, :], 0.0, max_radiance)
        * weight[..., None, :] / denominator[..., None, :]
    ).mean(1)

    indirect_occ = samples['indirect_occ'].mean(1)

    irradiance = (
        jnp.clip(samples['radiance_in'] * diffuse_lobe[..., None, :], 0.0, max_radiance)
        * weight[..., None, :]/ denominator[..., None, :]
    ).mean(1)

  # Multipliers
  if use_brdf_correction:
    integrated_multiplier = (
        samples['brdf_correction']
        * weight / denominator
    ).mean(1) / (2 * jnp.pi)

    integrated_multiplier_irradiance = (
        samples['brdf_correction'][..., None, 1:2]
        * samples['radiance_in']
        * diffuse_lobe[..., None, :]
        * weight[..., None, :] / denominator[..., None, :]
    ).mean(1)
  else:
    integrated_multiplier = samples['brdf_correction'][:, 0]
    integrated_multiplier_irradiance = samples['brdf_correction'][:, 0, :1]

  return dict(
      radiance_out=radiance_out,
      indirect_occ=indirect_occ,
      irradiance=irradiance,
      integrated_multiplier=integrated_multiplier,
      integrated_multiplier_irradiance=integrated_multiplier_irradiance,
  )


def integrate_irradiance(
    samples,
):
  eps = jnp.finfo(jnp.float32).eps

  denominator = jnp.maximum(
      samples['pdf'],
      eps
  )
  weight = jnp.maximum(
      samples['weight'],
      0.0
  )
  weight = jnp.where(
      samples['local_lightdirs'][Ellipsis, 2:] > 0.0, weight, jnp.zeros_like(weight)
  )

  # Incoming irradiance
  diffuse_lobe = (
      jnp.maximum(0., samples['local_lightdirs'][Ellipsis, 2:]) / jnp.pi
  )
  irradiance = (
      samples['radiance_in']
      * diffuse_lobe
      * weight / denominator
  ).mean(1)

  return irradiance


def eval_vmf(x, means, kappa):
  # Evaluate vmf at directions x
  eps = jnp.finfo(jnp.float32).eps 
  vmf_vals = kappa * math.safe_exp(
      kappa * (jnp.sum(x * means, axis=-1))
  ) / (4 * jnp.pi * jnp.sinh(kappa))
  out = jnp.where(
      jnp.less_equal(kappa, eps),
      jnp.ones_like(means[Ellipsis, 0]) / (4. * jnp.pi),
      vmf_vals
  )
  return out


def expand_vmf_vars(vars, x):
  means, kappas, logits = vars

  means = jnp.repeat(means[None], x.shape[0], axis=0)
  kappas = jnp.repeat(kappas[None], x.shape[0], axis=0)

  return means, kappas, logits


def sample_vmf_vars(rng, vars, x):
  key, rng = jax.random.split(rng)
  latents = jax.random.categorical(
      key, logits=vars[2], axis=-1, shape=(x.shape[0],)
  )
  means = jnp.take_along_axis(
      vars[0], latents[Ellipsis, None, None], axis=-2
  )[Ellipsis, 0, :]
  kappas = jnp.take_along_axis(
      vars[1], latents[Ellipsis, None], axis=-1
  )[Ellipsis, 0]

  return means, kappas, vars[2]


def filter_vmf_vars(vars, sample_normals, t1=0.1, t2=0.09):
  means, kappas, logits = vars

  # Mask
  dotprod = (
      ref_utils.l2_normalize(means, grad_eps=1e-5) * sample_normals[Ellipsis, None, :]
  ).sum(axis=-1)

  new_logits = logits + jax.lax.stop_gradient(dotprod - t2) / (t1 - t2)

  logits = jnp.where(
      dotprod > t1, logits, new_logits
  )

  return means, kappas, logits


def sample_vmf(rng, vars, x, n_dirs):
  """Sample random directions from vmf distribution.

  Args:
    rng: jnp.ndarray, random generator key.
    mean: jnp.ndarray(float32), [..., 3].
    kappa: jnp.ndarray(float32), [...]., vmf kappa parameter.
    n_dirs: int.

  Returns:
    rand_dirs: jnp.ndarray(float32), [..., n_dirs, 3]
  """
  key, rng = jax.random.split(rng)
  mean, kappa, _ = sample_vmf_vars(key, vars, x)

  t_vec = jnp.stack([-mean[Ellipsis, 1], mean[Ellipsis, 0],
                     jnp.zeros_like(mean[Ellipsis, 0])], axis=-1)
  t_vec = ref_utils.l2_normalize(t_vec)
  b_vec = jnp.cross(mean, t_vec)
  b_vec = ref_utils.l2_normalize(b_vec)
  rotmat = jnp.stack([t_vec, b_vec, mean], axis=-1)

  key, rng = jax.random.split(rng)
  # vmf sampling (https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf)
  v = jax.random.normal(key, shape=mean.shape[:-1] + (n_dirs, 2))
  v = ref_utils.l2_normalize(v)

  key, rng = jax.random.split(rng)
  tmp = jax.random.uniform(key, shape=mean.shape[:-1] + (n_dirs,))
  eps = jnp.finfo(jnp.float32).eps
  w = 1. + (1. / jnp.maximum(kappa[Ellipsis, None], eps)) * math_utils.safe_log(
      tmp + (1. - tmp) * jnp.exp(-2. * kappa[Ellipsis, None])
  )
  rand_dirs = jnp.stack([math_utils.safe_sqrt(1. - w**2) * v[Ellipsis, 0],
                         math_utils.safe_sqrt(1. - w**2) * v[Ellipsis, 1],
                         w], axis=-1)
  rand_dirs = jnp.matmul(rotmat[Ellipsis, None, :, :], rand_dirs[Ellipsis, None])[Ellipsis, 0]

  return rand_dirs


class LightSampler:
  global_dirs: bool = True
  return_rgb: bool = False
  deterministic: bool = False

  def __init__(self):
    pass

  def sample_directions(self, rng, u1, u2, wo, _, light_idx, kwargs):
    vars = (
        kwargs['vmf_means'],
        kwargs['vmf_kappas'][Ellipsis, 0],
        kwargs['vmf_logits'][Ellipsis, 0]
    )
    means = ref_utils.l2_normalize(vars[0], grad_eps=1e-5)
    kappas = vars[1]
    logits = vars[2]
    weights = jax.nn.softmax(vars[2])

    key, rng = jax.random.split(rng)
    sample_dirs = sample_vmf(
        key,
        (means, kappas, logits),
        wo,
        n_dirs=u1.shape[-1],
    )

    # Get pdf
    pdf = jnp.sum(
        weights[Ellipsis, None, :] * eval_vmf(
            sample_dirs[Ellipsis, None, :],
            means[Ellipsis, None, :, :],
            kappas[Ellipsis, None, :],
        ),
        axis=-1
    )

    return sample_dirs, jnp.maximum(pdf, 0.0)

  def pdf(self, wo, wi, _, kwargs):
    vars = (
        kwargs['vmf_means'],
        kwargs['vmf_kappas'][Ellipsis, 0],
        kwargs['vmf_logits'][Ellipsis, 0]
    )
    means = ref_utils.l2_normalize(vars[0], grad_eps=DENOMINATOR_EPS)
    kappas = vars[1]
    logits = vars[2]
    weights = jax.nn.softmax(vars[2])

    pdf = jnp.sum(
        weights[Ellipsis, None, :] * eval_vmf(
            wi[Ellipsis, None, :],
            means[Ellipsis, None, :, :],
            kappas[Ellipsis, None, :],
        ),
        axis=-1
    )

    return jnp.maximum(pdf, 0.0)


def vmf_loss_fn(
    vars,
    sample_normals,
    sample_dirs,
    samples,
    function_vals,
    function_vals_nocorr,
    lossmult,
    linear_to_srgb=True,
):
  means = ref_utils.l2_normalize(vars[0], grad_eps=1e-5)
  kappas = vars[1][Ellipsis, 0]
  logits = vars[2][Ellipsis, 0]
  weights = math.safe_exp(vars[2][Ellipsis, 0])

  # KL Divergence Loss
  likelihood = jnp.sum(
      weights[Ellipsis, None, :] * eval_vmf(
          sample_dirs[Ellipsis, None, :],
          means[Ellipsis, None, :, :],
          kappas[Ellipsis, None, :]
      ),
      axis=-1
  )

  # Denominator
  denominator = jnp.maximum(
      samples['pdf'][Ellipsis, 0],
      1e-2
  )

  # Weight (MIS)
  dotprod = (
      sample_dirs * sample_normals[Ellipsis, None, :]
  ).sum(axis=-1)
  weight = jnp.clip(
      samples['weight'][Ellipsis, 0],
      0.0,
      10.0,
  )
  weight = jnp.where(
      dotprod > 0.0,
      weight,
      jnp.zeros_like(weight),
  )

  if linear_to_srgb:
    function_vals = image.linear_to_srgb(jnp.maximum(function_vals, 1e-5))
    function_vals_nocorr = image.linear_to_srgb(jnp.maximum(function_vals_nocorr, 1e-5))
    likelihood = image.linear_to_srgb(jnp.maximum(likelihood, 1e-5))

  return jnp.mean(
      (function_vals - likelihood)
      * jax.lax.stop_gradient(function_vals_nocorr - likelihood)
      * weight
      * lossmult
      / denominator
  )
  
def get_environment_color(
    ref_rays,
    env_map,
    env_map_w,
    env_map_h,
):
    x, y, z = (
        ref_rays.viewdirs[..., 0:1],
        ref_rays.viewdirs[..., 1:2],
        ref_rays.viewdirs[..., 2:3],
    )

    x, y, z = x, z, -y

    r = R.from_quat([0, np.sin(np.pi * 0.0), 0, np.cos(np.pi * 0.0)]).as_matrix()
    # r = R.from_quat([0, np.sin(np.pi * 0.5), 0, np.cos(np.pi * 0.5)]).as_matrix()
    new_x = (
        r[0, 0] * x + r[0, 1] * y + r[0, 2] * z
    )
    new_y = (
        r[1, 0] * x + r[1, 1] * y + r[1, 2] * z
    )
    new_z = (
        r[2, 0] * x + r[2, 1] * y + r[2, 2] * z
    )
    x, y, z = new_x, new_y, new_z

    sin_theta = jnp.sqrt(x * x + y * y + 1e-8)
    phi = jnp.arctan2(y / (sin_theta + 1e-8), x / (sin_theta + 1e-8))
    theta = jnp.arctan2(sin_theta, z)

    phi = ((-phi + jnp.pi) / (2 * jnp.pi)) * env_map_w
    theta = (theta / jnp.pi) * env_map_h
    locations = jnp.concatenate([theta, phi], axis=-1).reshape(1, -1, 2)

    values = grid_utils.jax_resample_2d(
        env_map.reshape(1, env_map_h, env_map_w, -1),
        locations,
        coordinate_order='yx',
    ).reshape(
        ref_rays.origins.shape[:-1] + (-1, 3)
    )

    light_idx = ref_rays.light_idx
    values = jnp.take_along_axis(values, light_idx[..., None], axis=-2)[..., 0, :]

    return values
  
def get_sl_color(
    pattern,
    pattern_width,
    pattern_height,
    up,
    look,
    points,
    lights,
    hfov=10.0,
    vfov=10.0,
    shift=(0.0, 0.0),
    mult=5.0,
    invert=False,
):
    # Project to camera pixel
    fx = pattern_width / (2.0 * np.tan(np.radians(hfov)))
    fy = pattern_height / (2.0 * np.tan(np.radians(vfov)))
    cx = pattern_width / 2.0
    cy = pattern_height / 2.0

    light_dirs = points - lights
    light_dirs = jnp.concatenate(
        [
            math.dot(light_dirs, jnp.cross(look, up)),
            -math.dot(light_dirs, up),
            math.dot(light_dirs, look),
        ],
        axis=-1
    )
    light_dirs = light_dirs / light_dirs[..., 2:3]

    light_pix = jnp.concatenate(
        [
            light_dirs[..., 1:2] * fy + cy + shift[1],
            light_dirs[..., 0:1] * fx + cx + shift[0],
        ],
        axis=-1
    )

    # Look up
    values = grid_utils.jax_resample_2d(
        pattern.reshape(1, pattern_height, pattern_width, -1),
        light_pix.reshape(1, -1, 2),
        coordinate_order='yx',
    ).reshape(points.shape[:-1] + (1,))

    return values * mult

def dtof_to_itof(dtof_data, frequency_phase_shifts, bin_to_total_dist):
    sh = dtof_data.shape

    dtof_data = dtof_data.reshape(-1, sh[-2], sh[-1])
    T = dtof_data.shape[-2]

    c = 299792458
    time_to_travel = (jnp.linspace(0, T * bin_to_total_dist, T, endpoint=False) / c)
    itof_data = []

    for frequency, phase_shift in frequency_phase_shifts:
      # Cos
      cur_W = jnp.cos(2 * np.pi * frequency * time_to_travel + phase_shift) + 1.0
      cur_itof_data = (cur_W[None, :, None] * dtof_data).sum(axis=-2, keepdims=True)
      itof_data.append(cur_itof_data)

      # Sin
      cur_W = jnp.sin(2 * np.pi * frequency * time_to_travel + phase_shift) + 1.0
      cur_itof_data = (cur_W[None, :, None] * dtof_data).sum(axis=-2, keepdims=True)
      itof_data.append(cur_itof_data)

    # Constant
    constant_itof_data = (dtof_data).sum(axis=-2, keepdims=True)
    itof_data.append(constant_itof_data / 2.0)
    
    itof_data = jnp.concatenate(itof_data, axis=-2)
    itof_data = itof_data.reshape(sh[:-2] + (-1, sh[-1]))

    return itof_data

def dtof_to_gauss(dtof_data, sigma_scales, constant_scale):
    sh = dtof_data.shape
    dtof_data = dtof_data.reshape(-1, sh[-2], sh[-1])
    conv_data = []

    for sigma, scale in sigma_scales:
      filter = jnp.arange(round(-4*sigma), round(4*sigma) + 1)
      filter = jnp.exp(-filter**2/(2*sigma**2))-jnp.exp(-8)
      cur_data = jax.scipy.signal.convolve(dtof_data, filter[None, :, None], mode="same")
      conv_data.append(cur_data * scale)

    # Constant
    constant_data = (dtof_data).sum(axis=-2, keepdims=True)
    conv_data.append(constant_data * constant_scale)
    
    conv_data = jnp.concatenate(conv_data, axis=-2)
    conv_data = conv_data.reshape(sh[:-2] + (-1, sh[-1]))

    return conv_data


def zero_invalid_bins(
    transient_indirect_diffuse,
    transient_indirect_specular,
    rays,
    means,
    config,
):
    shape_trans = transient_indirect_diffuse.shape
    bins = jnp.arange(config.n_bins).reshape(
        (1,) * (len(shape_trans) - 2) + (config.n_bins, 1)
    )

    # Bins too close (light cannot reach this point fast enough)
    hist_dists_light = (bins + config.bin_zero_threshold_light) * config.exposure_time
    light_dists = jnp.linalg.norm(
        rays.lights[..., None, :] - means, axis=-1, keepdims=True
    )

    transient_indirect_diffuse = jnp.where(
        hist_dists_light < light_dists[..., None, :],
        jnp.zeros_like(transient_indirect_diffuse),
        transient_indirect_diffuse,
    )
    transient_indirect_specular = jnp.where(
        hist_dists_light < light_dists[..., None, :],
        jnp.zeros_like(transient_indirect_specular),
        transient_indirect_specular,
    )

    # Bins too far (shifted out of measured bins at camera)
    hist_dists_cam = bins * config.exposure_time
    max_dists = (config.n_bins - 1) * config.exposure_time
    cam_dists = (
        jnp.linalg.norm(
            (rays.origins[..., None, :] - means), axis=-1, keepdims=True
        )
        + jnp.linalg.norm(
            (rays.origins[..., None, :] - rays.cam_origins[..., None, :]), axis=-1, keepdims=True
        )
    )

    transient_indirect_diffuse = jnp.where(
        (hist_dists_cam + cam_dists[..., None, :]) > max_dists,
        jnp.zeros_like(transient_indirect_diffuse),
        transient_indirect_diffuse,
    )
    transient_indirect_specular = jnp.where(
        (hist_dists_cam + cam_dists[..., None, :]) > max_dists,
        jnp.zeros_like(transient_indirect_specular),
        transient_indirect_specular,
    )

    # Light zero
    if config.light_zero:
        transient_indirect_diffuse = jnp.where(
            light_dists[..., None, :] < config.light_near,
            jnp.zeros_like(transient_indirect_diffuse),
            transient_indirect_diffuse,
        )
        transient_indirect_specular = jnp.where(
            light_dists[..., None, :] < config.light_near,
            jnp.zeros_like(transient_indirect_specular),
            transient_indirect_specular,
        )

    return (
        transient_indirect_diffuse,
        transient_indirect_specular,
    )


IMPORTANCE_SAMPLER_BY_NAME = {
    'light': LightSampler,
    'environment': EnvironmentSampler,
    'microfacet': MicrofacetSampler,
    'cosine': CosineSampler,
    'uniform': UniformHemisphereSampler,
    'uniform_sphere': UniformSphereSampler,
}