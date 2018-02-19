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

    compartmentOut->aggregationKernel[idx4] = d_aggCompVar->aggKernelConst[0] * compartmentDEMIn->colFrequency[idx4] * compartmentDEMIn->colEfficiency[idx4];
    printf("Value of kernel at %d and %d is %f ", tix, tiy, compartmentOut->aggregationKernel[idx4]);
}


CompartmentVar :: CompartmentVar(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        internalLiquid = alloc_double_vector(nX2);
        externalLiquid = alloc_double_vector(nX2);
        externalLiquidContent = alloc_double_vector(nX2);
        volumeBins = alloc_double_vector(nX2);
        aggregationRate = alloc_double_vector(nX2);
        breakageRate = alloc_double_vector(nX2);
        particleMovement = alloc_double_vector(nX2);
        liquidMovement = alloc_double_vector(nX2);
        gasMovement = alloc_double_vector(nX2);
        liquidBins = alloc_double_vector(nX2);
        gasBins = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        internalLiquid = device_alloc_double_vector(nX2);
        externalLiquid = device_alloc_double_vector(nX2);
        externalLiquidContent = device_alloc_double_vector(nX2);
        volumeBins = device_alloc_double_vector(nX2);
        aggregationRate = device_alloc_double_vector(nX2);
        breakageRate = device_alloc_double_vector(nX2);
        particleMovement = device_alloc_double_vector(nX2);
        liquidMovement = device_alloc_double_vector(nX2);
        gasMovement = device_alloc_double_vector(nX2);
        liquidBins = device_alloc_double_vector(nX2);
        gasBins = device_alloc_double_vector(nX2);
    }

    else
        printf("\n Wrong Value of check passed in Compartment Var call \n");
}

CompartmentIn :: CompartmentIn (unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        fAll = alloc_double_vector(nX2);
        fLiquid = alloc_double_vector(nX2);
        fGas = alloc_double_vector(nX2);
        liquidAdditionRate = 0.0;
        vs = alloc_double_vector(nX2);
        vss = alloc_double_vector(nX2);
        sMeshXY = alloc_double_vector(nX2);
        ssMeshXY = alloc_double_vector(nX2);
        sAggregationCheck = alloc_integer_vector(nX2);
        ssAggregationCheck = alloc_integer_vector(nX2);;
        sInd = alloc_integer_vector(nX2);;
        ssInd = alloc_integer_vector(nX2);;
        sIndB = alloc_integer_vector(nX2);;
        ssIndB = alloc_integer_vector(nX2);;
        sLow = alloc_double_vector(nX2);
        sHigh = alloc_double_vector(nX2);
        ssLow = alloc_double_vector(nX2);
        ssHigh = alloc_double_vector(nX2);
        sCheckB = alloc_integer_vector(nX2);;
        ssCheckB = alloc_integer_vector(nX2);;
        diameter = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        liquidAdditionRate = 0.0;
        fAll = device_alloc_double_vector(nX2);
        fLiquid = device_alloc_double_vector(nX2);
        fGas = device_alloc_double_vector(nX2);
        vs = device_alloc_double_vector(nX2);
        vss = device_alloc_double_vector(nX2);
        sMeshXY = device_alloc_double_vector(nX2);
        ssMeshXY = device_alloc_double_vector(nX2);
        sAggregationCheck = device_alloc_integer_vector(nX2);
        ssAggregationCheck = device_alloc_integer_vector(nX2);;
        sInd = device_alloc_integer_vector(nX2);;
        ssInd = device_alloc_integer_vector(nX2);;
        sIndB = device_alloc_integer_vector(nX2);;
        ssIndB = device_alloc_integer_vector(nX2);;
        sLow = device_alloc_double_vector(nX2);
        sHigh = device_alloc_double_vector(nX2);
        ssLow = device_alloc_double_vector(nX2);
        ssHigh = device_alloc_double_vector(nX2);
        sCheckB = device_alloc_integer_vector(nX2);;
        ssCheckB = device_alloc_integer_vector(nX2);;
        diameter = device_alloc_double_vector(nX2);
    }

    else
        printf("\n Wrong Value of check passed in CompartmentIn  call \n");
}

PreviousCompartmentIn :: PreviousCompartmentIn(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        fAllPreviousCompartment = alloc_double_vector(nX2);
        flPreviousCompartment = alloc_double_vector(nX2);
        fgPreviousCompartment = alloc_double_vector(nX2);
        fAllComingIn = alloc_double_vector(nX2);
        fgComingIn = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        fAllPreviousCompartment = device_alloc_double_vector(nX2);
        flPreviousCompartment = device_alloc_double_vector(nX2);
        fgPreviousCompartment = device_alloc_double_vector(nX2);
        fAllComingIn = device_alloc_double_vector(nX2);
        fgComingIn = device_alloc_double_vector(nX2);
    }
    else
        printf("\n Wrong Value of check passed in PreviousCompartmentIn  call \n");

}

CompartmentDEMIn :: CompartmentDEMIn(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        DEMDiameter = alloc_double_vector(nX2);
        DEMCollisionData = alloc_double_vector(nX2);
        DEMImpactData = alloc_double_vector(nX2);
        colProbability = alloc_double_vector(nX2);
        brProbability = alloc_double_vector(nX2);
        colEfficiency = alloc_double_vector(nX2);
        colFrequency = alloc_double_vector(nX2);
        velocityCol = alloc_double_vector(nX2);
        uCriticalCol = 0.0;
    }

    else if (check == 1)
    {
        DEMDiameter = device_alloc_double_vector(nX2);
        DEMCollisionData = device_alloc_double_vector(nX2);
        DEMImpactData = device_alloc_double_vector(nX2);
        colProbability = device_alloc_double_vector(nX2);
        brProbability = device_alloc_double_vector(nX2);
        colEfficiency = device_alloc_double_vector(nX2);
        colFrequency = device_alloc_double_vector(nX2);
        velocityCol = device_alloc_double_vector(nX2);
        uCriticalCol = 0.0;
    }

    else 
        printf("\n Wrong Value of check passed in CompartmentDEMIn call \n");

}

CompartmentOut :: CompartmentOut(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        dfAlldt = alloc_double_vector(nX2);
        dfLiquiddt = alloc_double_vector(nX2);
        dfGasdt = alloc_double_vector(nX2);
        liquidBins = alloc_double_vector(nX2);
        gasBins = alloc_double_vector(nX2);
        internalVolumeBins = alloc_double_vector(nX2);
        externalVolumeBins = alloc_double_vector(nX2);
        aggregationKernel = alloc_double_vector(nX2);
        breakageKernel = alloc_double_vector(nX2);
        collisionFrequency = alloc_double_vector(nX2);
        formationThroughAggregation = 0.0;
        depletionThroughAggregation = 0.0;
        formationThroughBreakage = 0.0;
        depletionThroughBreakage = 0.0;
    }

    else if (check == 1)
    {
        dfAlldt = device_alloc_double_vector(nX2);
        dfLiquiddt = device_alloc_double_vector(nX2);
        dfGasdt = device_alloc_double_vector(nX2);
        liquidBins = device_alloc_double_vector(nX2);
        gasBins = device_alloc_double_vector(nX2);
        internalVolumeBins = device_alloc_double_vector(nX2);
        externalVolumeBins = device_alloc_double_vector(nX2);
        aggregationKernel = device_alloc_double_vector(nX2);
        breakageKernel = device_alloc_double_vector(nX2);
        collisionFrequency = device_alloc_double_vector(nX2);
        formationThroughAggregation = 0.0;
        depletionThroughAggregation = 0.0;
        formationThroughBreakage = 0.0;
        depletionThroughBreakage = 0.0;
    }

    else
        printf("\n Wrong Value of check passed in CompartmentOut call \n");
}