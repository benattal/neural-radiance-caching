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
"""Training script for mipNeRF360."""

import os

import gin
import jax
from absl import app

from internal import configs
from engine.trainer import Trainer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

configs.define_common_flags()
jax.config.parse_flags_with_absl()

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


def main(unused_argv):
    # Load initial config
    _ = configs.load_config()

    # Create and setup trainer
    trainer = Trainer()
    trainer.setup()

    # Train
    trainer.train()


if __name__ == "__main__":
    with gin.config_scope("train"):
        app.run(main)

