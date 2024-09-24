DEVICE_ID=0,1,2,3
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

MODEL=resnet34 # vgg16, resnet34
T=128
BATCH_SIZE=4

if [ "$MODEL" = "vgg16" ]; then
    ID="ImageNet-VGG16-t16"
elif [ "$MODEL" = "resnet34" ]; then
    ID="ImageNet-ResNet34-t8"
else
    echo "Invalid MODEL: $MODEL"
    exit 1
fi

python main_test.py \
    --model $MODEL --batch_size $BATCH_SIZE -L8 -id=$ID -data=imagenet \
    -T=$T
