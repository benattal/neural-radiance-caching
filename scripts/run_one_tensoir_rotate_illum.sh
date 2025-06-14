#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCENE=$1

bash scripts/train.sh --scene $SCENE --stage cache_rotate_illum --early_exit_steps 100000
bash scripts/train.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample_rotate_illum --take_stage cache_rotate_illum --batch_size 1024 --render_chunk_size 1024 --early_exit_steps 40000
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample_rotate_illum --render_chunk_size 4096 --render_repeats 32 --albedo
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample_rotate_illum --render_chunk_size 1024 --render_repeats 32
