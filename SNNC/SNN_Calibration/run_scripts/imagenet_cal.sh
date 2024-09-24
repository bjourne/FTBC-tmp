#!/bin/bash
DATESTAMP=$(date +"%Y-%m-%d")
LOG_PREFIX="logs/$DATESTAMP"
mkdir -p $LOG_PREFIX
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
#------------------------------------------------------------------------------#

ARCH=vgg16 # res34, vgg16, mobilenet
USEBN="true"

CALIBRATION_METHOD=ours_wo_avg # ours_wo_avg; ours_w_avg; snn_cali_baseline

NUM_CALI_ITERS=10
NUM_CALI_SAMPLE_BATCHES=10
CURR_T_ALPHA=0.5 # The alpha value for the current t bias, combined with NUM_CALI_SAMPLE_BATCHES.

TRAIN_BATCH_SIZE=128
VAL_BATCH_SIZE=128

### Define the array of T values you want to iterate over
T_values=(4 8 16 32 64 128)
DEVICE_ID=3

CHANNEL_WISE_VTH="true"

if [ "$USEBN" = "true" ]; then
    USEBN_OPTION="--usebn"
fi

if [ "$CHANNEL_WISE_VTH" = "true" ]; then
    CHANNEL_WISE_VTH_OPTION="--channel_wise_vth"
fi

# Loop through each T value and run the command
for T in "${T_values[@]}"
do
    echo -e "${YELLOW}Running with T=$T on device $DEVICE_ID...${NC}"
    export CUDA_VISIBLE_DEVICES=$DEVICE_ID
    python main_cal_imagenet.py \
        --arch $ARCH \
        $USEBN_OPTION \
        $CHANNEL_WISE_VTH_OPTION \
        --calibration_method $CALIBRATION_METHOD \
        --train_batch_size $TRAIN_BATCH_SIZE \
        --val_batch_size $VAL_BATCH_SIZE \
        --T $T \
        --num_cali_iters $NUM_CALI_ITERS \
        --num_cali_sample_batches $NUM_CALI_SAMPLE_BATCHES \
        --curr_t_alpha $CURR_T_ALPHA
done
