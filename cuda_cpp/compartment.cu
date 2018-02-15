#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>

#include "parameterData.h"
#include "utility.cuh"
#include "compartment.cuh"

using namespace std;

__global__ void performAggCalculations(PreviousCompartmentIn *prevCompIn, CompartmentIn *compartmentIn, CompartmentDEMIn *compartmentDEMIn, 
                                        CompartmentOut *compartmentOut, CompartmentVar *d_compVar, AggregationCompVar *d_aggCompVar, 
                                        double time, double timeStep, double initialTime)
{
    
}
