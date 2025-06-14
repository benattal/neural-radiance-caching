import time

import dataclasses
import functools
import gin

import numpy as np
import flax
import jax
import jax.numpy as jnp
from jax import random

import viser
import viser.transforms as vtf

from internal import camera_utils, configs, datasets, image, models, train_utils, utils, vis

@gin.configurable
@dataclasses.dataclass
class Viewer(object):
    render_every: int = 10
    port: int = 8082
    scene_scale_ratio: float = 10.0

    def setup(
        self,
        trainer,
    ):
        # Trainer
        self.trainer = trainer

        # Server
        self.server = viser.ViserServer(port=self.port)
        self.server.add_frame(
            "/frame",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            show_axes=True,
            axes_length=5.0,
        )
        self.server.on_client_connect(self.handle_new_client)

        # Rendering
        self.needs_update = False
        self.render_rngs = random.split(self.trainer.rng, jax.local_device_count())  # For pmapping RNG keys.

        self.pose = np.eye(4)

        def cast_fn(pose, h, w, focal, near, rng):
            key, rng = jax.random.split(rng)
            return camera_utils.cast_general_rays(
                pose,
                camera_utils.get_pixtocam(focal, w, h, xnp=jnp),
                h,
                w,
                near,
                self.trainer.config.far,
                camtype=self.trainer.dataset.camtype,
                rng=key,
                jitter=self.trainer.config.jitter_rays,
                jitter_scale=float(self.trainer.dataset.width) / float(w),
                xnp=jnp,
            )

        self.get_rays = cast_fn
        self.get_rays = jax.jit(
            self.get_rays,
            static_argnums=(
                1,
                2,
            ),
        )

        def render_rays(
            rays,
            variables,
            rng,
            train_frac=1.0,
            is_secondary=False,
        ):
            key, rng = jax.random.split(rng)
            r = self.trainer.model.apply(
                variables,
                key,
                rays,
                train_frac=train_frac,
                compute_extras=True,
                train=False,
                passes=("cache", "light", "material"),
                resample=True,
                mesh=self.trainer.dataset.mesh,
                is_secondary=is_secondary,
                cameras=self.trainer.cameras,
                camtype=self.trainer.dataset.camtype,
            )
            return r['render'], rng

        render_rays_p = jax.pmap(render_rays, in_axes=(0, 0, 0, None, None))

        def render_rays_batched(
            rays,
            variables,
            train_frac=1.0,
            is_secondary=False,
        ):
            out_sharded, self.render_rngs = render_rays_p(
                utils.shard(rays),
                variables,
                self.render_rngs,
                train_frac,
                is_secondary,
            )
            return jax.tree_map(utils.unshard, out_sharded)

        self.render_rays = render_rays_batched
    
    def handle_new_client(self, client: viser.ClientHandle):
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            self.needs_update = True

            R = vtf.SO3(wxyz=client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = R.as_matrix()
            pos = client.camera.position / self.scene_scale_ratio

            self.pose[:3, :4] = np.concatenate([R, pos[:, None]], axis=1)

    def render_image(
        self,
    ):
        if self.needs_update:
            # Get rays
            pose = jnp.array(self.pose)
            focal = 1111 * (400.0 / 800.0)

            _, key = jax.random.split(self.trainer.rng)
            rays = self.get_rays(
                pose,
                400,
                400,
                focal,
                self.trainer.config.near,
                key
            )

            # Render
            train_frac = jnp.clip(
                flax.jax_utils.unreplicate(self.trainer.state.step)
                / (self.trainer.config.max_steps - 1),
                0,
                1
            )

            eval_start_time = time.time()
            rendering = self.render_rays(
                rays,
                self.trainer.state.params,
                train_frac=train_frac,
            )
            eval_time = time.time() - eval_start_time
            print("Rendered frame in:", eval_time)

            # Display
            # rgb = np.array(rendering["rgb"])
            rgb = np.array(rendering["incoming_rgb"])
            self.server.set_background_image(rgb, format="png")

            # Needs update
            self.needs_update = False

            # Sleep
            time.sleep(0.01)