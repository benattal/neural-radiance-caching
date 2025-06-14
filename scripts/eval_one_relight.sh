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

# Env maps for TensoIR are ['bridge', 'city', 'fireplace', 'forest', 'night']
# Env maps for OpenIllum are '001' through '013' (excluding '009', '011', and '013')
# Env maps for GlossSynthetic are ['corridor', 'golf', 'neon']

SCENE=$1
ENV_MAP=$2
SUFFIX=$3

bash scripts/eval.sh --scene $SCENE --stage material_surface_light_field_light_slf_variate_resample$SUFFIX --render_chunk_size 1024 --render_repeats 8 --env_map_name $ENV_MAP
# bash scripts/eval.sh --scene $SCENE --stage material_light_resample$SUFFIX --render_chunk_size 1024 --render_repeats 8 --env_map_name $ENV_MAP