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

#-----------------

### 1. 70.38%
# DATASET=CIFAR100
# ARCH=VGG16
# THRESH=2    # relu threshold, 0 means use common relu; 2 is from the hint of VGG ckpt file

#-----------------

### 2. 63.80 %; ours: 66.34 > baseline (T=16); 67.63 < baseline (T=32);  (T=64);  (T=128)
# DATASET=CIFAR100
# ARCH=ResNet20
# THRESH=2

#-----------------

# DATASET=CIFAR10
# ARCH=VGG16
# THRESH=1

#-----------------

DATASET=CIFAR10
ARCH=ResNet20
THRESH=1

#-----------------

################################################################################
BATCH_SIZE=32 # 8, 16, 32, 10

CURR_T_ALPHA=0.005
NUM_CALI_ITERS=50
NUM_CALI_SAMPLE_BATCHES=2

DEVICE_ID=1
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

### Define the T values array
T_values=(128 64 32 16)

### Iterate over T values
for T in "${T_values[@]}"; do
    SHIFT_SNN=$T

    echo -e "${GREEN}Running simulation for ${ARCH}, T=${T} on device ${DEVICE_ID}...${NC}"
    python main_simulation_cifar.py \
        --dataset $DATASET \
        --batch_size $BATCH_SIZE \
        --arch $ARCH \
        --thresh $THRESH \
        --T $T \
        --shift_snn $SHIFT_SNN \
        --curr_t_alpha $CURR_T_ALPHA \
        --num_cali_iters $NUM_CALI_ITERS \
        --num_cali_sample_batches $NUM_CALI_SAMPLE_BATCHES \
        2>&1 | tee "${LOG_PREFIX}/${TIMESTAMP}_${DATASET}_vgg16_T${T}.log"
done
