#!/bin/bash

c=50
until [ $c -ge 60 ]
do
	python evaluate_gen.py -mode vd_eval_val -start_path checkpoints/iter1/x88/single_prob03/vd_train_88_$c.ckpt -save_path results -save_name eval_88_$c.txt -gpu_ids 4 5
	echo $c
	c=$(expr $c + 1)
done

