DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

# python main_train_cifar.py --dataset CIFAR10 --arch VGG16 --usebn # 95.59
python main_train_cifar.py --dataset CIFAR10 --arch VGG16

# python main_train_cifar.py --dataset CIFAR100 --arch VGG16 --usebn
# python main_train_cifar.py --dataset CIFAR100 --arch VGG16

### pass!
# python main_train_cifar.py --dataset CIFAR10 --arch res20 --usebn # 96.98
# python main_train_cifar.py --dataset CIFAR10 --arch res20

# python main_train_cifar.py --dataset CIFAR100 --arch res20 --usebn # 96.98
# python main_train_cifar.py --dataset CIFAR100 --arch res20

# python main_train_cifar.py --dataset CIFAR10 --usebn --arch mobilenet # 94.17
# python main_train_cifar.py --dataset CIFAR100 --arch mobilenet --usebn