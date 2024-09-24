#!/bin/bash
#------------------------------------------------------------------------------#
# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
#------------------------------------------------------------------------------#
### Create folder based on Year-Month-Day
DATESTAMP=$(date +"%Y-%m-%d")
LOG_PREFIX="logs/$DATESTAMP"
# LOG_PREFIX="logs/use_100_samples" # for debugging
mkdir -p $LOG_PREFIX
### Create a timestamp variable
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
#------------------------------------------------------------------------------#

DATASET=ImageNet
ARCH=VGG16 ### For ImageNet, only VGG16 result is in the paper.
BATCH_SIZE=128
CURR_T_ALPHA=0.5

NUM_CALI_ITERS=3
NUM_CALI_SAMPLE_BATCHES=10

### relu threshold, 0 means use common relu; 3 is from the hint of VGG ckpt file
THRESH=3

DEVICE_ID=2
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

### Define the T values array
T_values=(128 64 32 16 8 4 2 1)

### Iterate over T values
for T in "${T_values[@]}"; do
    SHIFT_SNN=$T

    echo -e "${GREEN}Running simulation for T=${T} on device ${DEVICE_ID}...${NC}"
    python main_simulation_imagenet.py \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --arch $ARCH \
        --thresh $THRESH \
        --T $T \
        --shift_snn $SHIFT_SNN \
        --curr_t_alpha $CURR_T_ALPHA \
        --num_cali_iters $NUM_CALI_ITERS \
        --num_cali_sample_batches $NUM_CALI_SAMPLE_BATCHES \
        2>&1 | tee "${LOG_PREFIX}/${TIMESTAMP}_imagenet_vgg16_t${T}.log"
done
