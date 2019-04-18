#!/bin/bash

for i in 32;
do 
	cd "C"$i;
	mkdir csvDump
	/home/chaitanya/Documents/CUDA/pbmOnGPUs/cuda_cpp/./model.out PBM_Input.in 128 200 0.0 > output_$i.txt

	
	cd ..
done


	       	
