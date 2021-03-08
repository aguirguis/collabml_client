#!/bin/bash
#start with inference tests and then with training
for bsize in 1000 100 10
do
	python3 mlswift_playground.py --dataset mnist --model convnet --batch_size $bsize >> explog
        echo "============================================================================" >> explog
	python3 mlswift_playground.py --dataset cifar10 --model cifarnet --batch_size $bsize >> explog
        echo "============================================================================" >> explog
	python3 mlswift_playground.py --dataset cifar10 --model resnet50 --batch_size $bsize >> explog
        echo "============================================================================" >> explog
	python3 mlswift_playground.py --dataset cifar10 --model resnet152 --batch_size $bsize >> explog
        echo "============================================================================" >> explog
	python3 mlswift_playground.py --dataset imagenet --model resnet50 --batch_size $bsize >> explog
        echo "============================================================================" >> explog
	python3 mlswift_playground.py --dataset imagenet --model resnet152 --batch_size $bsize >> explog
        echo "============================================================================" >> explog
done

#now training....requires also number of epochs to be passed
for epoch in 10 50
do
	for bsize in 1000 100 10
	do
      		python3 mlswift_playground.py --dataset mnist --model convnet --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
        	python3 mlswift_playground.py --dataset cifar10 --model cifarnet --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
        	python3 mlswift_playground.py --dataset cifar10 --model resnet50 --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
        	python3 mlswift_playground.py --dataset cifar10 --model resnet152 --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
        	python3 mlswift_playground.py --dataset imagenet --model resnet50 --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
        	python3 mlswift_playground.py --dataset imagenet --model resnet152 --batch_size $bsize --task training --num_epoch $epoch >> explog
        	echo "============================================================================" >> explog
	done
done
