#!/bin/bash
#start with inference tests and then with training
for bsize in 1000 100 10
do
	python3 mlswift_playground.py --dataset mnist --model convnet --batch_size $bsize >> explog
	python3 mlswift_playground.py --dataset cifar10 --model cifarnet --batch_size $bsize >> explog
	python3 mlswift_playground.py --dataset cifar10 --model resnet50 --batch_size $bsize >> explog
	python3 mlswift_playground.py --dataset cifar10 --model resnet152 --batch_size $bsize >> explog
	python3 mlswift_playground.py --dataset imagenet --model resnet50 --batch_size $bsize >> explog
	python3 mlswift_playground.py --dataset imagenet --model resnet152 --batch_size $bsize >> explog
done
