#!/bin/bash

for i in 1 2 4
do
	cd "core_"$i;
	#bash "mpi_"$i"_runs.sh";
	for j  in 1 2 4 8 16 32
	do 
		cd "C"$j;
		echo $i $j:x
		tail "output_"$j".txt";
		cd ..
	done

	cd .. ;
done

