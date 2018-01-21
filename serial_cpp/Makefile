# -*_ Makefile _*_

#target: dependencies
#<tab>action
CC=mpic++ -std=c++11 -O3 -fPIC
#CC=mpic++ -std=c++11 -g

all: model.out

model.out: model.o compartment.o kernel.o timeStamp.o liggghtsData.o atomFileParser.o parameterData.o utility.o
	$(CC) model.o compartment.o kernel.o timeStamp.o liggghtsData.o atomFileParser.o parameterData.o utility.o -o model.out

model.o: model.cpp liggghtsData.h compartment.h utility.h timeStamp.h parameterData.h
	$(CC) -Wall -c model.cpp

compartment.o: compartment.cpp compartment.h liggghtsData.h utility.h parameterData.h
	$(CC) -Wall -c compartment.cpp

kernel.o: kernel.cpp kernel.h compartment.h liggghtsData.h parameterData.h
	$(CC) -Wall -c kernel.cpp

liggghtsData.o: liggghtsData.cpp liggghtsData.h atomFileParser.h utility.h parameterData.h
	$(CC) -Wall -c liggghtsData.cpp

atomFileParser.o: atomFileParser.cpp atomFileParser.h
	$(CC) -Wall -c atomFileParser.cpp

timeStamp.o: timeStamp.cpp timeStamp.h
	$(CC) -Wall -c timeStamp.cpp

parameterData.o: parameterData.cpp parameterData.h utility.h parameters.h
	$(CC) -Wall -c parameterData.cpp

utility.o: utility.cpp utility.h
	$(CC) -Wall -c utility.cpp

clean:
	rm -f *.o *.out


