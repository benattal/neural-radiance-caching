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

mkdir ~/data
mkdir ~/data/openillum
python download_open_illum.py --light lighting_patterns --obj_id $SCENE --local_dir ~/data/openillum

DIR=$(find ~/data/openillum/lighting_patterns -type d -name "*obj_$SCENE*" -print -quit)
cp -r $DIR/Lights/013/raw_undistorted $DIR/output/images