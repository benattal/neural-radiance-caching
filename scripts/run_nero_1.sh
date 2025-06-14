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

bash scripts/run_one_nero.sh nero_teapot
bash scripts/eval_one_relight.sh nero_teapot neon

bash scripts/run_one_nero.sh nero_angel
bash scripts/eval_one_relight.sh nero_angel golf

bash scripts/run_one_nero.sh nero_cat
# bash scripts/eval_one_relight.sh nero_cat corridor

bash scripts/run_one_nero.sh nero_potion
# bash scripts/eval_one_relight.sh nero_potion neon