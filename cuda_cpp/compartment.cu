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
                                        double time, double timeStep, double initialTime, double demTimeStep)
{
    int bix = blockIdx.x;
    int biy = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;

    int tix = threadIdx.x;
    int tiy = threadIdx.y;
    int dimx = gridDim.x;
    int dimy = gridDim.y;

    int idx4 = tiy * bdx + tix;

    double criticalExternalLiquid = 0.2;
    bool flag1 = (compartmentIn->fAll[tix] >= 0.0) && (compartmentIn->fAll[tiy] >= 0.0);
    bool flag2 = ((d_compVar->externalLiquid[tix] + d_compVar->externalLiquid[tiy]) / (compartmentIn->fAll[tiy] * compartmentIn->vs[tiy % 16] + compartmentIn->fAll[tix] * compartmentIn->vss[tix % 16]));
    bool flag3 = (compartmentDEMIn->velocityCol[tiy % 16] < compartmentDEMIn->uCriticalCol);
    if (flag1 && flag2 && flag3)
    {
        compartmentDEMIn->colEfficiency[idx4] = compartmentDEMIn->colProbability[tix % 16];
    }
    else
        compartmentDEMIn->colEfficiency[idx4] = 0.0;
    compartmentDEMIn->colFrequency[idx4] = (compartmentDEMIn->DEMCollisionData[tix] * timeStep) / demTimeStep;

    compartmentOut->aggregationKernel[idx4] = d_aggCompVar->aggKernelConst * compartmentDEMIn->colFrequency[idx4] * compartmentDEMIn->colEfficiency[idx4];
    printf("Value of kernel at %d and %d is %f ", tix, tiy, compartmentOut->aggregationKernel[idx4]);
}
