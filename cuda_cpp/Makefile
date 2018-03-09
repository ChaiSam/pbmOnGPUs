# -*_ Makefile _*_

#target: dependencies
#<tab>action

CC=nvcc -std=c++11 -arch=sm_52 -G -Xcompiler -rdynamic -rdc=true --expt-relaxed-constexpr 
#CC=nvcc -std=c++11 -arch=sm_52 -O3 -Xcompiler -rdynamic -rdc=true --expt-relaxed-constexpr 
#ARCH= -arch=sm_52

all: model.out

model.out: model.o compartment.o liggghtsData.o atomFileParser.o parameterData.o utility.o
	$(CC) model.o compartment.o liggghtsData.o atomFileParser.o parameterData.o utility.o -o model.out

model.o: model.cu liggghtsData.h compartment.cuh utility.cuh  parameterData.h
	$(CC) -c model.cu

compartment.o: compartment.cu compartment.cuh liggghtsData.h utility.cuh parameterData.h
	$(CC) -c compartment.cu

liggghtsData.o: liggghtsData.cpp liggghtsData.h atomFileParser.h utility.cuh parameterData.h
	$(CC) -c liggghtsData.cpp

atomFileParser.o: atomFileParser.cpp atomFileParser.h
	$(CC) -c atomFileParser.cpp

parameterData.o: parameterData.cpp parameterData.h utility.cuh parameters.h
	$(CC) -c parameterData.cpp

utility.o: utility.cu utility.cuh
	$(CC) -c utility.cu

clean:
	rm -f *.o *.out
