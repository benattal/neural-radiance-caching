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

# source ~/.bashrc
# conda activate yobo
# export PATH="/usr/local/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib:/scratch/ondemand28/battal/miniconda3/envs/yobo/lib64:$LD_LIBRARY_PATH"

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

NOISY=0
ALBEDO=0
ALBEDO_LEAST=0
VIS_RENDER_PATH=0
VIS_EXTRA=0
FIXED_LIGHT=0
FIXED_CAMERA=0
VIS_RESTART=0
VIS_START=0
VIS_END=200
WRITE_TO_FILE=0

LIGHT_IDX=0

SAMPLE_FACTOR=8
RENDER_REPEATS=1
RENDER_CHUNK_SIZE=4096
NUM_RESAMPLE=1

RELIGHT=0
SL_RELIGHT=0
ROUND_ROUGHNESS=0
FILTER_MEDIAN=0
NO_GAUSSIAN=0
EVAL_TRAIN=0
EVAL_PATH=0
ENV_MAP_NAME="sunset"

# Parse command line arguments
while (( "$#" )); do
  case "$1" in
    --noisy)
      NOISY=1
      shift
      ;;
    --albedo)
      ALBEDO=1
      NOISY=1
      shift
      ;;
    --albedo_least)
      ALBEDO_LEAST=1
      NOISY=1
      shift
      ;;
    --write_to_file)
      WRITE_TO_FILE=1
      shift
      ;;
    --vis_render_path)
      VIS_RENDER_PATH=1
      shift
      ;;
    --vis_extra)
      VIS_EXTRA=1
      shift
      ;;
    --fixed_light)
      FIXED_LIGHT=1
      shift
      ;;
    --fixed_camera)
      FIXED_CAMERA=1
      shift
      ;;
    --vis_restart)
      VIS_RESTART=1
      shift
      ;;
    --sl_relight)
      SL_RELIGHT=1
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
    --eval_path)
      EVAL_PATH=1
      shift
      ;;
    --vis_start)
      VIS_START=$2
      shift 2
      ;;
    --vis_end)
      VIS_END=$2
      shift 2
      ;;
    --light_idx)
      LIGHT_IDX=$2
      shift 2
      ;;
    --scene)
      SCENE=$2
      shift 2
      ;;
    --suffix)
      USE_SUFFIX=1
      SUFFIX=$2
      shift 2
      ;;
    --stage)
      STAGE=$2
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
    --env_map_name)
      RELIGHT=1
      ENV_MAP_NAME=$2
      shift 2
      ;;
    --sample_render_factor)
      SAMPLE_FACTOR=$2
      shift 2
      ;;
    --render_repeats)
      RENDER_REPEATS=$2
      shift 2
      ;;
    --render_chunk_size)
      RENDER_CHUNK_SIZE=$2
      shift 2
      ;;
    *)
      echo "Error: Invalid argument"
      exit 1
  esac
done

# Common part of the command
CMD="python scripts/train_one_stage.py --scene $SCENE --stage $STAGE --vis_only --resample --sample_factor 1"
CMD="$CMD --batch_size 128 --render_chunk_size $RENDER_CHUNK_SIZE --vis_start $VIS_START --vis_end $VIS_END --light_idx $LIGHT_IDX"

# Append different parts based on the flags
if [ $NOISY -eq 1 ]; then
    CMD="$CMD --sample_render_factor 1 --render_repeats 1"
    OUTPUT="metrics/${SCENE}_${STAGE}_noisy_metrics.txt"
else
    CMD="$CMD --sample_render_factor $SAMPLE_FACTOR --render_repeats $RENDER_REPEATS --num_resample $NUM_RESAMPLE"
    OUTPUT="metrics/${SCENE}_${STAGE}_metrics.txt"
fi

if [ $ALBEDO -eq 1 ]; then
    SUFFIX="${SUFFIX}_albedo"
    USE_SUFFIX=1
elif [ $ALBEDO_LEAST -eq 1 ]; then
    SUFFIX="${SUFFIX}_albedo_least"
    USE_SUFFIX=1
else
    SUFFIX="${SUFFIX}_eval"
    USE_SUFFIX=1
fi

if [ $USE_TAKE_STAGE -eq 1 ]; then
    CMD="$CMD --take_stage $TAKE_STAGE"
else
    CMD="$CMD --take_stage $STAGE"
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

if [ $EVAL_PATH -eq 1 ]; then
    CMD="$CMD --eval_path"
fi

if [ $RELIGHT -eq 1 ]; then
    SUFFIX="${SUFFIX}_${ENV_MAP_NAME}"
    CMD="$CMD --env_map_name $ENV_MAP_NAME --relight"
    USE_SUFFIX=1
fi

if [ $FIXED_LIGHT -eq 1 ]; then
    CMD="$CMD --fixed_light"
    SUFFIX="${SUFFIX}_fixed_light"
    USE_SUFFIX=1
fi

if [ $FIXED_CAMERA -eq 1 ]; then
    CMD="$CMD --fixed_camera"
    SUFFIX="${SUFFIX}_fixed_camera"
    USE_SUFFIX=1
fi

if [ $USE_SUFFIX -eq 1 ]; then
    CMD="$CMD --suffix $SUFFIX"
fi

if [ $VIS_RENDER_PATH -eq 1 ]; then
    CMD="$CMD --vis_render_path"
fi

if [ $VIS_EXTRA -eq 1 ]; then
    CMD="$CMD --vis_secondary --vis_surface_light_field --vis_light_sampler --vis_extra"
fi

if [ $VIS_RESTART -eq 1 ]; then
    CMD="$CMD --vis_restart"
fi

# Run the command and optionally write the output to a file
if [ $WRITE_TO_FILE -eq 1 ]; then
    $CMD > $OUTPUT
else
    $CMD
fi