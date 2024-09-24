# python main_test.py -id=vgg16_wd[0.0005] -data=cifar10 -T=8 -dev=0

python main_test.py \
    --model resnet20 -L4 -id=resnet20_L[4] -data=cifar10 -dev=0 \
    -T=4
