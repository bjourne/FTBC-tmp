mkdir -p logs

MODEL=resnet34 # vgg16, resnet34
T=6
BATCH_SIZE=32
DEVICE_ID=0 # 0,1,2,3

if [ "$MODEL" = "vgg16" ]; then
    ID="ImageNet-VGG16-t16"
    L=16
elif [ "$MODEL" = "resnet34" ]; then
    ID="ImageNet-ResNet34-t8"
    L=8
else
    echo "Invalid MODEL: $MODEL"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
python main_cali.py \
    --model $MODEL --batch_size $BATCH_SIZE --L $L -id=$ID -data=imagenet \
    -T=$T \
    2>&1 | tee logs/${ID}_${MODEL}_T_${T}.log

