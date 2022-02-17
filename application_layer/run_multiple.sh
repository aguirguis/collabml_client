#!/bin/bash
#This script runs parallel requests to the server (to simulate paralell clients)
n=$1
for i in $(seq 1 $n);
do
	python3 main.py --dataset imagenet --model myalexnet --num_epochs 1 --batch_size 2000 \
		 --freeze_idx 17 --freeze --use_intermediate > temp_output/output_$i &
done
