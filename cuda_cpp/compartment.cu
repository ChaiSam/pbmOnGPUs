#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>

#include "parameterData.h"
#include "utility.cuh"
#include "compartment.cuh"

using namespace std;

__global__ void performAggCalculations(PreviousCompartmentIn *d_prevCompIn, CompartmentIn *d_compartmentIn, CompartmentDEMIn *d_compartmentDEMIn, 
                                        CompartmentOut *d_compartmentOut, CompartmentVar *d_compVar, AggregationCompVar *d_aggCompVar, 
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

    int idx4 = tix * bdx + tix;

    double criticalExternalLiquid = 0.2;
    bool flag1 = (d_compartmentIn->fAll[tix] >= 0.0) && (d_compartmentIn->fAll[tiy] >= 0.0);
    bool flag2 = ((d_compVar->externalLiquid[tix] + d_compVar->externalLiquid[tiy]) / (d_compartmentIn->fAll[tiy] * d_compartmentIn->vs[tiy % 16] + d_compartmentIn->fAll[tix] * d_compartmentIn->vss[tix % 16]));
    bool flag3 = (d_compartmentDEMIn->velocityCol[tiy % 16] < d_compartmentDEMIn->uCriticalCol);
    if (flag1 && flag2 && flag3)
    {
        d_compartmentDEMIn->colEfficiency[idx4] = d_compartmentDEMIn->colProbability[tix % 16];
    }
    else
        d_compartmentDEMIn->colEfficiency[idx4] = 0.0;
    d_compartmentDEMIn->colFrequency[idx4] = (d_compartmentDEMIn->DEMCollisionData[tix] * timeStep) / demTimeStep;

    d_compartmentOut->aggregationKernel[idx4] = d_aggCompVar->aggKernelConst * d_compartmentDEMIn->colFrequency[idx4] * d_compartmentDEMIn->colEfficiency[idx4];
    printf("Value of kernel at %d and %d is %f \n", tix, tiy, d_compartmentOut->aggregationKernel[idx4]);
}



// ============ Constructors for the Classes =================


CompartmentVar :: CompartmentVar(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        internalLiquid = alloc_double_vector(nX2);
        externalLiquid = alloc_double_vector(nX2);
        externalLiquidContent = alloc_double_vector(nX2);
        volumeBins = alloc_double_vector(nX2);
        aggregationRate = alloc_double_vector(nX4);
        breakageRate = alloc_double_vector(nX4);
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
        aggregationRate = device_alloc_double_vector(nX4);
        breakageRate = device_alloc_double_vector(nX4);
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
        DEMDiameter = alloc_double_vector(sqrt(nX2));
        DEMCollisionData = alloc_double_vector(nX2);
        DEMImpactData = alloc_double_vector(sqrt(nX2));
        colProbability = alloc_double_vector(sqrt(nX2));
        brProbability = alloc_double_vector(sqrt(nX2));
        colEfficiency = alloc_double_vector(nX4);
        colFrequency = alloc_double_vector(nX4);
        velocityCol = alloc_double_vector(sqrt(nX2));
        uCriticalCol = 0.0;
    }

    else if (check == 1)
    {
        DEMDiameter = device_alloc_double_vector(sqrt(nX2));
        DEMCollisionData = device_alloc_double_vector(nX2);
        DEMImpactData = device_alloc_double_vector(sqrt(nX2));
        colProbability = device_alloc_double_vector(sqrt(nX2));
        brProbability = device_alloc_double_vector(sqrt(nX2));
        colEfficiency = device_alloc_double_vector(nX4);
        colFrequency = device_alloc_double_vector(nX4);
        velocityCol = device_alloc_double_vector(sqrt(nX2));
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
        aggregationKernel = alloc_double_vector(nX4);
        breakageKernel = alloc_double_vector(nX4);
        collisionFrequency = alloc_double_vector(nX4);
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
        aggregationKernel = device_alloc_double_vector(nX4);
        breakageKernel = device_alloc_double_vector(nX4);
        collisionFrequency = device_alloc_double_vector(nX4);
        formationThroughAggregation = 0.0;
        depletionThroughAggregation = 0.0;
        formationThroughBreakage = 0.0;
        depletionThroughBreakage = 0.0;
    }

    else
        printf("\n Wrong Value of check passed in CompartmentOut call \n");
}

BreakageCompVar :: BreakageCompVar(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        birthThroughBreakage1 = alloc_double_vector(nX2);
        birthThroughBreakage2 = alloc_double_vector(nX2);
        firstSolidBirthThroughBreakage = alloc_double_vector(nX2);
        secondSolidBirthThroughBreakage = alloc_double_vector(nX2);
        liquidBirthThroughBreakage1 = alloc_double_vector(nX2);
        gasBirthThroughBreakage1 = alloc_double_vector(nX2);
        liquidBirthThroughBreakage2 = alloc_double_vector(nX2);
        gasBirthThroughBreakage2 = alloc_double_vector(nX2);
        firstSolidVolumeThroughBreakage = alloc_double_vector(nX2);
        secondSolidVolumeThroughBreakage = alloc_double_vector(nX2);
        fractionBreakage00 = alloc_double_vector(nX2);
        fractionBreakage01 = alloc_double_vector(nX2);
        fractionBreakage10 = alloc_double_vector(nX2);
        fractionBreakage11 = alloc_double_vector(nX2);
        formationThroughBreakageCA = alloc_double_vector(nX2);
        formationOfLiquidThroughBreakageCA = alloc_double_vector(nX2);
        formationOfGasThroughBreakageCA = alloc_double_vector(nX2);
        transferThroughLiquidAddition = alloc_double_vector(nX2);
        transferThroughConsolidation = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        birthThroughBreakage1 = device_alloc_double_vector(nX2);
        birthThroughBreakage2 = device_alloc_double_vector(nX2);
        firstSolidBirthThroughBreakage = device_alloc_double_vector(nX2);
        secondSolidBirthThroughBreakage = device_alloc_double_vector(nX2);
        liquidBirthThroughBreakage1 = device_alloc_double_vector(nX2);
        gasBirthThroughBreakage1 = device_alloc_double_vector(nX2);
        liquidBirthThroughBreakage2 = device_alloc_double_vector(nX2);
        gasBirthThroughBreakage2 = device_alloc_double_vector(nX2);
        firstSolidVolumeThroughBreakage = device_alloc_double_vector(nX2);
        secondSolidVolumeThroughBreakage = device_alloc_double_vector(nX2);
        fractionBreakage00 = device_alloc_double_vector(nX2);
        fractionBreakage01 = device_alloc_double_vector(nX2);
        fractionBreakage10 = device_alloc_double_vector(nX2);
        fractionBreakage11 = device_alloc_double_vector(nX2);
        formationThroughBreakageCA = device_alloc_double_vector(nX2);
        formationOfLiquidThroughBreakageCA = device_alloc_double_vector(nX2);
        formationOfGasThroughBreakageCA = device_alloc_double_vector(nX2);
        transferThroughLiquidAddition = device_alloc_double_vector(nX2);
        transferThroughConsolidation = device_alloc_double_vector(nX2);
    }

    else
        printf("\n Wrong Value of check passed in BreakageCompVar call \n");
}

AggregationCompVar :: AggregationCompVar(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        aggKernelConst = 0.0;
        depletionOfGasThroughAggregation = alloc_double_vector(nX2);
        depletionOfLiquidThroughAggregation = alloc_double_vector(nX2);
        birthThroughAggregation = alloc_double_vector(nX2);
        firstSolidBirthThroughAggregation = alloc_double_vector(nX2);
        secondSolidBirthThroughAggregation = alloc_double_vector(nX2);
        liquidBirthThroughAggregation = alloc_double_vector(nX2);
        gasBirthThroughAggregation = alloc_double_vector(nX2);
        firstSolidVolumeThroughAggregation = alloc_double_vector(nX2);
        secondSolidVolumeThroughAggregation = alloc_double_vector(nX2);
        birthAggLowLow = alloc_double_vector(nX2);
        birthAggHighHigh = alloc_double_vector(nX2);
        birthAggLowHigh = alloc_double_vector(nX2);
        birthAggHighLow = alloc_double_vector(nX2);
        birthAggLowLowLiq = alloc_double_vector(nX2);
        birthAggHighHighLiq = alloc_double_vector(nX2);
        birthAggLowHighLiq = alloc_double_vector(nX2);
        birthAggHighLowLiq = alloc_double_vector(nX2);
        birthAggLowLowGas = alloc_double_vector(nX2);
        birthAggHighHighGas = alloc_double_vector(nX2);
        birthAggLowHighGas = alloc_double_vector(nX2);
        birthAggHighLowGas = alloc_double_vector(nX2);
        formationThroughAggregationCA = alloc_double_vector(nX2);
        formationOfLiquidThroughAggregationCA = alloc_double_vector(nX2);
        formationOfGasThroughAggregationCA = alloc_double_vector(nX2);
        depletionThroughAggregation = alloc_double_vector(nX2);
        depletionThroughBreakage = alloc_double_vector(nX2);
        depletionOfGasThroughBreakage = alloc_double_vector(nX2);
        depletionOfLiquidthroughBreakage = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        aggKernelConst = 0.0;
        depletionOfGasThroughAggregation = device_alloc_double_vector(nX2);
        depletionOfLiquidThroughAggregation = device_alloc_double_vector(nX2);
        birthThroughAggregation = device_alloc_double_vector(nX2);
        firstSolidBirthThroughAggregation = device_alloc_double_vector(nX2);
        secondSolidBirthThroughAggregation = device_alloc_double_vector(nX2);
        liquidBirthThroughAggregation = device_alloc_double_vector(nX2);
        gasBirthThroughAggregation = device_alloc_double_vector(nX2);
        firstSolidVolumeThroughAggregation = device_alloc_double_vector(nX2);
        secondSolidVolumeThroughAggregation = device_alloc_double_vector(nX2);
        birthAggLowLow = device_alloc_double_vector(nX2);
        birthAggHighHigh = device_alloc_double_vector(nX2);
        birthAggLowHigh = device_alloc_double_vector(nX2);
        birthAggHighLow = device_alloc_double_vector(nX2);
        birthAggLowLowLiq = device_alloc_double_vector(nX2);
        birthAggHighHighLiq = device_alloc_double_vector(nX2);
        birthAggLowHighLiq = device_alloc_double_vector(nX2);
        birthAggHighLowLiq = device_alloc_double_vector(nX2);
        birthAggLowLowGas = device_alloc_double_vector(nX2);
        birthAggHighHighGas = device_alloc_double_vector(nX2);
        birthAggLowHighGas = device_alloc_double_vector(nX2);
        birthAggHighLowGas = device_alloc_double_vector(nX2);
        formationThroughAggregationCA = device_alloc_double_vector(nX2);
        formationOfLiquidThroughAggregationCA = device_alloc_double_vector(nX2);
        formationOfGasThroughAggregationCA = device_alloc_double_vector(nX2);
        depletionThroughAggregation = device_alloc_double_vector(nX2);
        depletionThroughBreakage = device_alloc_double_vector(nX2);
        depletionOfGasThroughBreakage = device_alloc_double_vector(nX2);
        depletionOfLiquidthroughBreakage = device_alloc_double_vector(nX2);
    }
    else
        printf("\n Wrong Value of check passed in BreakageCompVar call \n");
}