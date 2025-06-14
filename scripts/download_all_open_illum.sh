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

bash scripts/download_one_open_illum.sh 02 # egg
bash scripts/download_one_open_illum.sh 04 # stone
bash scripts/download_one_open_illum.sh 05 # bird
bash scripts/download_one_open_illum.sh 17 # box
bash scripts/download_one_open_illum.sh 26 # pumpkin
bash scripts/download_one_open_illum.sh 29 # hat
bash scripts/download_one_open_illum.sh 35 # cup
bash scripts/download_one_open_illum.sh 36 # sponge
bash scripts/download_one_open_illum.sh 42 # banana
bash scripts/download_one_open_illum.sh 48 # bucket

cp -r ./generate_light_gt_sg ~/data/openillum/env_maps