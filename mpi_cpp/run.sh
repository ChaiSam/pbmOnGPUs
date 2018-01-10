#!/bin/bash

rm -f *.txt
rm -f csvDump/*.csv
#make clean

#make -j 7

echo "compilation & linkin done"

echo "Running model.out"

#export OMP_NUM_THREADS=2

mpirun -n 4 ./model.out /home/chai/Documents/git/CyberManufacturing/src/twoway_PBM/PBM_Input.in 128 200 0.0 > output_test.txt

echo "DONE"

#matlab -nodisplay -nosplash -r "d50andRatioplt;quit";
#matlab -nodisplay -nosplash  exit;

#echo "Plots saved in csvDump folder"
