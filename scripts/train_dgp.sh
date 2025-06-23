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

source ~/.bashrc
conda activate yobo
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib:/scratch/ondemand28/battal/miniconda3/envs/yobo/lib64:$LD_LIBRARY_PATH"



SCENE=""
STAGE=""

USE_TAKE_STAGE=0
TAKE_STAGE=""

USE_SUFFIX=0
SUFFIX=""

SAMPLE_FACTOR=8  # Default value
RENDER_REPEATS=1
SAMPLE_RENDER_FACTOR=8  # Default value
SAMPLE_RENDER_FACTOR_SET=0  # Flag to check if --sample_render_factor is set
NUM_RESAMPLE=1

BATCH_SIZE=16384
RENDER_CHUNK_SIZE=4096
train_length_mult=1
LR_FACTOR_MULT=1.0
GRAD_ACCUM_STEPS=1
SECONDARY_GRAD_ACCUM_STEPS=1

EARLY_EXIT_STEPS=200000
NO_VIS=0
NO_VIS_SECONDARY=0

SL_RELIGHT=0
FIXED_LIGHT=0
ROUND_ROUGHNESS=0
FILTER_MEDIAN=0
NO_GAUSSIAN=0
EVAL_TRAIN=0

LIGHT_IDX=0

for arg in "$@"; do
  if [[ $arg == "--stage" ]]; then
    nextArgIsStage=1
  elif [[ $nextArgIsStage == 1 ]]; then
    STAGE=$arg
    nextArgIsStage=0
    if [[ $STAGE == *"material"* ]]; then
      TAKE_STAGE="cache"
    fi
  fi
done

while (( "$#" )); do
  case "$1" in
    --no_vis)
      NO_VIS=1
      shift
      ;;
    --no_vis_secondary)
      NO_VIS_SECONDARY=1
      shift
      ;;
    --scene)
      SCENE=$2
      shift 2
      ;;
    --stage)
      STAGE=$2
      shift 2
      ;;
    --suffix)
      USE_SUFFIX=1
      SUFFIX=$2
      shift 2
      ;;
    --early_exit_steps)
      EARLY_EXIT_STEPS=$2
      shift 2
      ;;
    --take_stage)
      USE_TAKE_STAGE=1
      TAKE_STAGE=$2
      shift 2
      ;;
    --num_resample)
      NUM_RESAMPLE=$2
      shift 2
      ;;
    --sample_factor)
      SAMPLE_FACTOR=$2
      shift 2
      ;;
    --sample_render_factor)
      SAMPLE_RENDER_FACTOR=$2
      SAMPLE_RENDER_FACTOR_SET=1
      shift 2
      ;;
    --render_repeats)
      RENDER_REPEATS=$2
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --render_chunk_size)
      RENDER_CHUNK_SIZE=$2
      shift 2
      ;;
    --train_length_mult)
      train_length_mult=$2
      shift 2
      ;;
    --lr_factor_mult)
      LR_FACTOR_MULT=$2
      shift 2
      ;;
    --grad_accum_steps)
      GRAD_ACCUM_STEPS=$2
      shift 2
      ;;
    --secondary_grad_accum_steps)
      SECONDARY_GRAD_ACCUM_STEPS=$2
      shift 2
      ;;
    --sl_relight)
      SL_RELIGHT=1
      shift
      ;;
    --fixed_light)
      FIXED_LIGHT=1
      shift
      ;;
    --round_roughness)
      ROUND_ROUGHNESS=1
      shift
      ;;
    --filter_median)
      FILTER_MEDIAN=1
      shift
      ;;
    --no_gaussian)
      NO_GAUSSIAN=1
      shift
      ;;
    --eval_train)
      EVAL_TRAIN=1
      shift
      ;;
    --light_idx)
      LIGHT_IDX=$2
      shift 2
      ;;
    *)
      echo "Error: Invalid argument"
      exit 1
  esac
done

# If --sample_render_factor is not set, set it to the value of SAMPLE_FACTOR
if [ $SAMPLE_RENDER_FACTOR_SET -eq 0 ]; then
    SAMPLE_RENDER_FACTOR=$SAMPLE_FACTOR
fi

# Common part of the command
CMD="python scripts/train_one_stage.py --scene $SCENE --stage $STAGE --early_exit_steps $EARLY_EXIT_STEPS"
CMD="$CMD --batch_size $BATCH_SIZE --render_chunk_size $RENDER_CHUNK_SIZE --train_length_mult $train_length_mult --lr_factor_mult $LR_FACTOR_MULT --grad_accum_steps $GRAD_ACCUM_STEPS --secondary_grad_accum_steps $SECONDARY_GRAD_ACCUM_STEPS"
CMD="$CMD --sample_factor $SAMPLE_FACTOR --resample_render --sample_render_factor $SAMPLE_RENDER_FACTOR --num_resample $NUM_RESAMPLE --light_idx $LIGHT_IDX --render_repeats $RENDER_REPEATS"

# Append different parts based on the flags
if [ $USE_SUFFIX -eq 1 ]; then
    CMD="$CMD --suffix $SUFFIX"
fi

if [ $USE_TAKE_STAGE -eq 1 ]; then
    CMD="$CMD --take_stage $TAKE_STAGE"
fi 

if [ $NO_VIS -eq 1 ]; then
    CMD="$CMD --no_vis"
fi

if [ $NO_VIS_SECONDARY -eq 0 ]; then
    CMD="$CMD --vis_secondary --vis_surface_light_field --vis_light_sampler"
fi

if [ $SL_RELIGHT -eq 1 ]; then
    CMD="$CMD --sl_relight"
fi

if [ $ROUND_ROUGHNESS -eq 1 ]; then
    CMD="$CMD --round_roughness"
fi

if [ $FILTER_MEDIAN -eq 1 ]; then
    CMD="$CMD --filter_median"
fi

if [ $NO_GAUSSIAN -eq 1 ]; then
    CMD="$CMD --no_gaussian"
fi

if [ $EVAL_TRAIN -eq 1 ]; then
    CMD="$CMD --eval_train"
fi

if [ $FIXED_LIGHT -eq 1 ]; then
    CMD="$CMD --fixed_light"
fi

$CMD