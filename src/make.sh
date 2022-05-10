python make.py --mode teacher --data MNIST --model linear --run train --round 8
python make.py --mode teacher --data MNIST --model linear --run test --round 8

python make.py --mode teacher --data CIFAR10 --model linear --run train --round 8
python make.py --mode teacher --data CIFAR10 --model linear --run test --round 8

python make.py --mode teacher --data MNIST --model conv --run train --round 8
python make.py --mode teacher --data MNIST --model conv --run test --round 8

python make.py --mode teacher --data CIFAR10 --model conv --run train --round 8
python make.py --mode teacher --data CIFAR10 --model conv --run test --round 8

python make.py --mode teacher --data MNIST --model resnet18 --run train --round 8
python make.py --mode teacher --data MNIST --model resnet18 --run test --round 8

python make.py --mode teacher --data CIFAR10 --model resnet18 --run train --round 8
python make.py --mode teacher --data CIFAR10 --model resnet18 --run test --round 8