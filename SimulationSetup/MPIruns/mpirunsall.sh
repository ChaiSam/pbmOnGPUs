#!/bin/bash

for i in 2 4
do
	cd "core_"$i;
	bash "mpi_"$i"_runs.sh";
	cd .. ;
done

