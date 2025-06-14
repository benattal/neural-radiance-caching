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

bash scripts/train.sh --scene $SCENE --stage cache --early_exit_steps 100000

# 8 samples
bash scripts/train.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _8 --take_stage cache --batch_size 1024 --render_chunk_size 1024 --early_exit_steps 40000 --num_resample 8 --sample_factor 1 --sample_render_factor 1 --render_chunk_size 1024
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _8 --render_chunk_size 4096 --num_resample 8 --render_repeats 32 --sample_render_factor 1 --albedo
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _8 --render_chunk_size 1024 --num_resample 8 --render_repeats 32 --sample_render_factor 1 

# 4 samples
bash scripts/train.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _4 --take_stage cache --batch_size 1024 --render_chunk_size 1024 --early_exit_steps 40000 --num_resample 4 --sample_factor 2 --sample_render_factor 2 --render_chunk_size 1024
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _4 --render_chunk_size 4096 --num_resample 4 --render_repeats 32 --sample_render_factor 2 --albedo
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _4 --render_chunk_size 1024 --num_resample 4 --render_repeats 32 --sample_render_factor 2

# 2 samples
bash scripts/train.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _2 --take_stage cache --batch_size 1024 --render_chunk_size 1024 --early_exit_steps 40000 --num_resample 2 --sample_factor 4 --sample_render_factor 4 --render_chunk_size 1024
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _2 --render_chunk_size 4096 --num_resample 2 --render_repeats 32 --sample_render_factor 4 --albedo
bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample --suffix _2 --render_chunk_size 1024 --num_resample 2 --render_repeats 32 --sample_render_factor 4 