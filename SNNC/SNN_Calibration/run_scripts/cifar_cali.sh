#!/bin/bash

DATESTAMP=$(date +"%Y-%m-%d")
LOG_PREFIX="logs/$DATESTAMP"
mkdir -p $LOG_PREFIX

# Create a timestamp variable
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
#------------------------------------------------------------------------------#

DATASET=CIFAR100 # CIFAR10, CIFAR100
MODEL=res20 # res20, VGG16, mobilenet
USEBN="true"

CALIBRATION_METHOD=ours_wo_avg # ours_wo_avg; ours_w_avg; snn_cali_baseline
BASELINE_CALI_METHOD=light #'none', 'light', 'advanced'

NUM_CALI_ITERS=100 # num of iterations for calibration.
NUM_CALI_SAMPLE_BATCHES=5
CURR_T_ALPHA=0.5 # The alpha value for the current t bias, combined with NUM_CALI_SAMPLE_BATCHES.

CHANNEL_WISE_VTH="true" # set to true by defalut since the paper uses it in table 4.
DATASET_PATH=datasets

BATCH_SIZE=128

### Define the array of T values you want to iterate over
T_values=(1 2 4 8 16 32 64 128)

DEVICE_ID=3
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

if [ "$USEBN" = "true" ]; then
    USEBN_OPTION="--usebn"
fi

if [ "$CHANNEL_WISE_VTH" = "true" ]; then
    CHANNEL_WISE_VTH_OPTION="--channel_wise_vth"
fi

# Ours method: Loop through each T value and run the command
for T in "${T_values[@]}"
do
    echo -e "${YELLOW}Running ${DATASET}, ${MODEL} with T=$T at device ${DEVICE_ID}...${NC}"
    python main_cal_cifar.py \
        --dataset $DATASET \
        --arch $MODEL \
        $USEBN_OPTION \
        $CHANNEL_WISE_VTH_OPTION \
        --calibration_method $CALIBRATION_METHOD \
        --baseline_cali_method $BASELINE_CALI_METHOD \
        --batch_size $BATCH_SIZE \
        --dpath $DATASET_PATH \
        --T $T \
        --num_cali_iters $NUM_CALI_ITERS \
        --num_cali_sample_batches $NUM_CALI_SAMPLE_BATCHES \
        --curr_t_alpha $CURR_T_ALPHA
done
