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
                                        double time, double timeStep, double initialTime, double demTimeStep, int blx, int tlx, int mbdx, int nFirstSolidBins, int nSecondSolidBins)
{
    int bix = blockIdx.x;
    int bdx = blockDim.x;

    int tix = threadIdx.x;
    int dimx = gridDim.x;

    int idx4 = blx * mbdx * bdx + tlx * bdx + tix;
    int idx3 = blx * mbdx + tlx;
    int idx3s = blx * bdx + tix;

    double criticalExternalLiquid = 0.2;
    bool flag1 = (d_compartmentIn->fAll[idx3] >= 0.0) && (d_compartmentIn->fAll[idx3] >= 0.0);
    bool flag2 = ((d_compVar->externalLiquid[idx3] + d_compVar->externalLiquid[idx3]) / (d_compartmentIn->fAll[idx3] * d_compartmentIn->vs[idx3 % nFirstSolidBins] + d_compartmentIn->fAll[idx3] * d_compartmentIn->vss[idx3 % 16]));
    bool flag3 = (d_compartmentDEMIn->velocityCol[idx3 % 16] < d_compartmentDEMIn->uCriticalCol);
    if (flag1 && flag2 && flag3)
    {
        d_compartmentDEMIn->colEfficiency[idx4] = d_compartmentDEMIn->colProbability[tix % 16];
    }
    else
        d_compartmentDEMIn->colEfficiency[idx4] = 0.0;
    d_compartmentDEMIn->colFrequency[idx4] = (d_compartmentDEMIn->DEMCollisionData[tix] * timeStep) / demTimeStep;

    d_compartmentOut->aggregationKernel[idx4] = d_aggCompVar->aggKernelConst * d_compartmentDEMIn->colFrequency[idx4] * d_compartmentDEMIn->colEfficiency[idx4];
    // printf("Value of kernel at %d and %d is %f \n", tix, tix, d_compartmentOut->aggregationKernel[idx4]);

    d_compVar->aggregationRate[idx4] = d_compartmentIn->sAggregationCheck[idx3] * d_compartmentIn->ssAggregationCheck[idx3s] * d_compartmentOut->aggregationKernel[idx4] * d_compartmentIn->fAll[idx3] * d_compartmentIn->fAll[idx3s];

    d_aggCompVar->depletionThroughAggregation[idx3] += d_compVar->aggregationRate[idx4];
    d_aggCompVar->depletionThroughAggregation[idx3s] += d_compVar->aggregationRate[idx4];
    d_aggCompVar->depletionOfGasThroughAggregation[idx3] = d_aggCompVar->depletionThroughAggregation[idx3] * d_compartmentOut->gasBins[idx3];
    d_aggCompVar->depletionOfLiquidThroughAggregation[idx3] = d_aggCompVar->depletionThroughAggregation[idx3] * d_compartmentOut->liquidBins[idx3];
    __syncthreads();

    for (int i = 0; i < nFirstSolidBins; i++)
    {
        for(int j = 0; j < nSecondSolidBins; j++)
        {
            int s12 = (idx3 % nFirstSolidBins) * (bdx / nFirstSolidBins) + (idx3 % nFirstSolidBins);
            int ss12 = (idx3s % nFirstSolidBins) * (bdx / nFirstSolidBins) + (idx3s % nFirstSolidBins);
            if (d_compartmentIn->sInd[s12] == (i+1) && d_compartmentIn->ssInd[ss12] == (j+1))
            {
                int a = i * nFirstSolidBins + j;
                d_aggCompVar->birthThroughAggregation[idx3] += d_compVar->aggregationRate[idx4];
                d_aggCompVar->firstSolidBirthThroughAggregation[idx3] += (d_compartmentIn->vs[idx3 % nFirstSolidBins] + d_compartmentIn->vs[idx3 % nFirstSolidBins]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->secondSolidBirthThroughAggregation[idx3] += (d_compartmentIn->vs[idx3s % nFirstSolidBins] + d_compartmentIn->vs[idx3s % nFirstSolidBins]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->liquidBirthThroughAggregation[idx3] += (d_compartmentOut->liquidBins[idx3] + d_compartmentOut->liquidBins[idx3s]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->gasBirthThroughAggregation[idx3] += (d_compartmentOut->gasBins[idx3] + d_compartmentOut->gasBins[idx3s]) * d_compVar->aggregationRate[idx4];
            }
        }
    }

    __syncthreads();

    if (fabs(d_aggCompVar->birthThroughAggregation[idx3]) > 1e-16)
    {
        d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] = d_aggCompVar->firstSolidBirthThroughAggregation[idx3] / d_aggCompVar->birthThroughAggregation[idx3];;
        d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] = d_aggCompVar->secondSolidBirthThroughAggregation[idx3] / d_aggCompVar->birthThroughAggregation[idx3];
    }
    else
    {
        d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] = 0.0;
        d_aggCompVar->secondSolidBirthThroughAggregation[idx3s] = 0.0;
    }

    int val1 = idx3 % nFirstSolidBins; // s
    int val2 = idx3s % nSecondSolidBins; // ss
    // int s3 = val1 * nFirstSolidBins + val2;

    if (val1 == nFirstSolidBins - 1 && val2 == nSecondSolidBins - 1)
    {
        d_aggCompVar->birthAggHighHigh[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
        d_aggCompVar->birthAggHighHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggHighHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighHighLiq[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
        d_aggCompVar->birthAggHighHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggHighHighLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighHighGas[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
        d_aggCompVar->birthAggHighHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggHighHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
    }

    else if (val2 == nSecondSolidBins - 1)
    {
        d_aggCompVar->birthAggLowHigh[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggLowHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggLowHighLiq[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggLowHighLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggLowHighGas[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2] - d_compartmentIn->vs[val2 -1]);
        d_aggCompVar->birthAggLowHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
    }

    else if (val1 == nFirstSolidBins -1)
    {
        d_aggCompVar->birthAggHighLow[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
        d_aggCompVar->birthAggHighLow[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLow[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighLowLiq[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
        d_aggCompVar->birthAggHighLowLiq[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLowLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighLowGas[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
        d_aggCompVar->birthAggHighLowGas[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLowGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

    }

    else
    {
        d_aggCompVar->birthAggLowLow[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggLowLow[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowLow[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighHigh[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1] ) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2 + 1]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggLowHigh[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighLow[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighLow[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLow[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];



        d_aggCompVar->birthAggLowLowLiq[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggLowLowLiq[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowLowLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighHighLiq[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1] ) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2 + 1]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighHighLiq[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

        d_aggCompVar->birthAggLowHighLiq[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowHighLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighLowLiq[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighLowLiq[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLowLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];


        d_aggCompVar->birthAggLowLowGas[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggLowLowGas[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowLowGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighHighGas[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2 + 1]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggLowHighGas[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
        d_aggCompVar->birthAggLowHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggLowHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

        d_aggCompVar->birthAggHighLowGas[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 + 1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
        d_aggCompVar->birthAggHighLowGas[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val2 + 1] - d_compartmentIn->vs[val2]);
        d_aggCompVar->birthAggHighLowGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
    }
    
    __syncthreads();

    d_aggCompVar->formationThroughAggregationCA[tlx] = d_aggCompVar->birthAggHighHigh[tlx] + d_aggCompVar->birthAggHighLow[tlx] + d_aggCompVar->birthAggLowHigh[tlx] + d_aggCompVar->birthAggLowLow[tlx];
    d_aggCompVar->formationOfLiquidThroughAggregationCA[tlx] = d_aggCompVar->birthAggHighHighLiq[tlx] + d_aggCompVar->birthAggHighLowLiq[tlx] + d_aggCompVar->birthAggLowHighLiq[tlx] + d_aggCompVar->birthAggLowLowLiq[tlx];
    d_aggCompVar->formationOfGasThroughAggregationCA[tlx] = d_aggCompVar->birthAggHighHighGas[tlx] + d_aggCompVar->birthAggHighLowGas[tlx] + d_aggCompVar->birthAggLowHighGas[tlx] + d_aggCompVar->birthAggLowLowGas[tlx];

}

// ==================== BREAKAGE COMPARTMENT CALCULATIONS ===========================================

__global__ void performBreakageCalculations(PreviousCompartmentIn *d_prevCompIn, CompartmentIn *d_compartmentIn, CompartmentDEMIn *d_compartmentDEMIn, 
                                        CompartmentOut *d_compartmentOut, CompartmentVar *d_compVar, BreakageCompVar *d_brCompVar, 
                                        double time, double timeStep, double initialTime, double demTimeStep, int blx, int tlx, int mbdx, int nFirstSolidBins, int nSecondSolidBins)
{
    int tix = threadIdx.x;
    int bdx = blockDim.x;

    int val1 = tlx % nFirstSolidBins; // s
    int val2 = tix % nSecondSolidBins; // ss
    int s3 = val1 * nFirstSolidBins + val2;

    d_compartmentDEMIn->impactFrequency[tix % nFirstSolidBins] = (d_compartmentDEMIn->DEMImpactData[tix % nFirstSolidBins] * timeStep) / demTimeStep;
    int idx4 = blx * mbdx * bdx + tlx * bdx + tix;

    d_compartmentOut->breakageKernel[idx4] = d_compartmentDEMIn->impactFrequency[tix % nFirstSolidBins] * d_compartmentDEMIn->brProbability[tix % nFirstSolidBins] * d_brCompVar->brkKernelConst;

    d_compVar->breakageRate[idx4] = d_compartmentIn->sCheckB[tlx] * d_compartmentIn->ssCheckB[tix] * d_compartmentOut->breakageKernel[idx4] * d_compartmentIn->fAll[tlx];
    
    __syncthreads();
    
    int idx3 = blx * mbdx + tlx; 
    d_brCompVar->depletionThroughBreakage[idx3] += d_compVar->breakageRate[idx4];
    d_brCompVar->depletionOfLiquidthroughBreakage[idx3] = d_brCompVar->depletionThroughBreakage[idx3] * d_compartmentOut->liquidBins[idx3];
    d_brCompVar->depletionOfGasThroughBreakage[idx3] = d_brCompVar->depletionThroughBreakage[idx3] * d_compartmentOut->gasBins[idx3];

    d_brCompVar->birthThroughBreakage1[idx3] = d_compVar->breakageRate[idx4];

    for (int i = 0; i < nFirstSolidBins; i++)
    {
        for(int j = 0; j < nSecondSolidBins; j++)
        {
            int s12 = (tlx % nFirstSolidBins) * (bdx / nFirstSolidBins) + (tlx % nFirstSolidBins);
            int ss12 = (tix % nFirstSolidBins) * (bdx / nFirstSolidBins) + (tix % nFirstSolidBins);
            if (d_compartmentIn->sIndB[s12] == (i+1) && d_compartmentIn->ssIndB[ss12] == (j+1))
            {
                int a = i * nFirstSolidBins + j;
                d_brCompVar->birthThroughBreakage2[a] += d_compVar->aggregationRate[idx4];
                d_brCompVar->firstSolidBirthThroughBreakage[a] += (d_compartmentIn->vs[tlx % nFirstSolidBins] + d_compartmentIn->vs[tlx % nFirstSolidBins]) * d_compVar->breakageRate[idx4];
                d_brCompVar->secondSolidBirthThroughBreakage[a] += (d_compartmentIn->vs[tix % nFirstSolidBins] + d_compartmentIn->vs[tix % nFirstSolidBins]) * d_compVar->breakageRate[idx4];
                d_brCompVar->liquidBirthThroughBreakage2[a] += (d_compartmentOut->liquidBins[tlx] + d_compartmentOut->liquidBins[tix]) * d_compVar->breakageRate[idx4];
                d_brCompVar->gasBirthThroughBreakage2[a] += (d_compartmentOut->gasBins[tlx] + d_compartmentOut->gasBins[tix]) * d_compVar->breakageRate[idx4];

                if (fabs(d_brCompVar->birthThroughBreakage2[a]) > 1e-16)
                {
                    d_brCompVar->firstSolidVolumeThroughBreakage[a] = d_brCompVar->firstSolidBirthThroughBreakage[a] / d_brCompVar->birthThroughBreakage2[a];
                    d_brCompVar->secondSolidVolumeThroughBreakage[a] = d_brCompVar->secondSolidBirthThroughBreakage[a] / d_brCompVar->birthThroughBreakage2[a];
                }
            }
        }
    }
    __syncthreads();
    d_brCompVar->liquidBirthThroughBreakage1[tix] += (d_compartmentOut->liquidBins[tlx] * (d_compVar->volumeBins[tix] / d_compVar->volumeBins[tlx])) * d_compVar->breakageRate[idx4];
    d_brCompVar->gasBirthThroughBreakage1[tix] += (d_compartmentOut->gasBins[tlx] * (d_compVar->volumeBins[tix] / d_compVar->volumeBins[tlx])) * d_compVar->breakageRate[idx4];

    double value1 = 0.0;
    double value2 = 0.0;

    value1 = fabs(d_compartmentIn->sLow[tix] - d_brCompVar->firstSolidBirthThroughBreakage[tix]);
    value1 = d_compartmentIn->sHigh[tix] - d_compartmentIn->sLow[tix] - value1;
    value1 /= d_compartmentIn->sHigh[tix] - d_compartmentIn->sLow[tix];

    value2 = fabs(d_compartmentIn->ssLow[tix] - d_brCompVar->secondSolidVolumeThroughBreakage[tix]);
    value2 = d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix] - value2;
    value2 /= d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix];

    d_brCompVar->fractionBreakage00[tix] = value1 / value2;

    value2 = fabs(d_compartmentIn->ssHigh[tix] - d_brCompVar->secondSolidVolumeThroughBreakage[tix]);
    value2 = d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix] - value2;
    value2 /= d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix];
    
    d_brCompVar->fractionBreakage01[tix] = value1 / value2;

    value1 = fabs(d_compartmentIn->sHigh[tix] - d_brCompVar->firstSolidVolumeThroughBreakage[tix]);
    value1 = d_compartmentIn->sHigh[tix] - d_compartmentIn->sLow[tix] - value1;
    value1 /= d_compartmentIn->sHigh[tix] - d_compartmentIn->sLow[tix];

    d_brCompVar->fractionBreakage11[tix] = value1 / value2;

    value2 = fabs(d_compartmentIn->ssLow[tix] - d_brCompVar->secondSolidVolumeThroughBreakage[tix]);
    value2 = d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix] - value2;
    value2 /= d_compartmentIn->ssHigh[tix] - d_compartmentIn->ssLow[tix];

    d_brCompVar->fractionBreakage10[tix] = value1 / value2;

    __syncthreads();

    if (val1 == nFirstSolidBins - 1 && val2 == nSecondSolidBins - 1)
    {
            d_brCompVar->formationThroughBreakageCA[s3] += d_brCompVar->birthThroughBreakage2[s3] * d_brCompVar->fractionBreakage11[s3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[s3] += d_brCompVar->liquidBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage11[s3];
            d_brCompVar->formationOfGasThroughBreakageCA[s3] += d_brCompVar->gasBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage11[s3];

    }

    else if (val2 == nSecondSolidBins - 1)
    {
        d_brCompVar->formationThroughBreakageCA[s3] += d_brCompVar->birthThroughBreakage2[s3] * d_brCompVar->fractionBreakage01[s3];
        d_brCompVar->formationOfLiquidThroughBreakageCA[s3] += d_brCompVar->liquidBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage01[s3];
        d_brCompVar->formationOfGasThroughBreakageCA[s3] += d_brCompVar->gasBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage01[s3];

    }

    else if (val1 == nFirstSolidBins -1)
    {
        d_brCompVar->formationThroughBreakageCA[s3] += d_brCompVar->birthThroughBreakage2[s3] * d_brCompVar->fractionBreakage10[s3];
        d_brCompVar->formationOfLiquidThroughBreakageCA[s3] += d_brCompVar->liquidBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage10[s3];
        d_brCompVar->formationOfGasThroughBreakageCA[s3] += d_brCompVar->gasBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage10[s3];

    }

    else
    {
        d_brCompVar->formationThroughBreakageCA[s3] += d_brCompVar->birthThroughBreakage2[s3] * d_brCompVar->fractionBreakage00[s3];
        d_brCompVar->formationOfLiquidThroughBreakageCA[s3] += d_brCompVar->liquidBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage00[s3];
        d_brCompVar->formationOfGasThroughBreakageCA[s3] += d_brCompVar->gasBirthThroughBreakage2[s3] * d_brCompVar->fractionBreakage00[s3];
    }
}

// ============ Constructors for the Classes =================


CompartmentVar :: CompartmentVar(unsigned int nX2, unsigned int nX5, unsigned int check)
{
    if (check == 0)
    {
        internalLiquid = alloc_double_vector(nX2);
        externalLiquid = alloc_double_vector(nX2);
        externalLiquidContent = alloc_double_vector(nX2);
        volumeBins = alloc_double_vector(nX2);
        aggregationRate = alloc_double_vector(nX5);
        breakageRate = alloc_double_vector(nX5);
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
        aggregationRate = device_alloc_double_vector(nX5);
        breakageRate = device_alloc_double_vector(nX5);
        particleMovement = device_alloc_double_vector(nX2);
        liquidMovement = device_alloc_double_vector(nX2);
        gasMovement = device_alloc_double_vector(nX2);
        liquidBins = device_alloc_double_vector(nX2);
        gasBins = device_alloc_double_vector(nX2);
    }

    else
        printf("\n Wrong Value of check passed in Compartment Var call \n");
}

CompartmentIn :: CompartmentIn (unsigned int nX2, unsigned int nX5, unsigned int check)
{
    if (check == 0)
    {
        fAll = alloc_double_vector(nX5 / nX2);
        fLiquid = alloc_double_vector(nX5 / nX2);
        fGas = alloc_double_vector(nX5 / nX2);
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
        fAll = device_alloc_double_vector(nX5 / nX2);
        fLiquid = device_alloc_double_vector(nX5 / nX2);
        fGas = device_alloc_double_vector(nX5 / nX2);
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
        fAllPreviousCompartment = alloc_double_vector(nX5 / nX2);
        flPreviousCompartment = alloc_double_vector(nX5 / nX2);
        fgPreviousCompartment = alloc_double_vector(nX5 / nX2);
        fAllComingIn = alloc_double_vector(nX5 / nX2);
        fgComingIn = alloc_double_vector(nX5 / nX2);
    }

    else if (check == 1)
    {
        fAllPreviousCompartment = device_alloc_double_vector(nX5 / nX2);
        flPreviousCompartment = device_alloc_double_vector(nX5 / nX2);
        fgPreviousCompartment = device_alloc_double_vector(nX5 / nX2);
        fAllComingIn = device_alloc_double_vector(nX5 / nX2);
        fgComingIn = device_alloc_double_vector(nX5 / nX2);
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
        impactFrequency = alloc_double_vector(sqrt(nX2));
        uCriticalCol = 0.0;
        ubreak = 0.0;
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
        impactFrequency = device_alloc_double_vector(sqrt(nX2));
        uCriticalCol = 0.0;
        ubreak = 0.0;
    }

    else 
        printf("\n Wrong Value of check passed in CompartmentDEMIn call \n");

}

CompartmentOut :: CompartmentOut(unsigned int nX2, unsigned int nX5, unsigned int check)
{
    if (check == 0)
    {
        dfAlldt = alloc_double_vector(nX2);
        dfLiquiddt = alloc_double_vector(nX2);
        dfGasdt = alloc_double_vector(nX2);
        liquidBins = alloc_double_vector(nX5 / nX2);
        gasBins = alloc_double_vector(nX5 / nX2);
        internalVolumeBins = alloc_double_vector(nX5 / nX2);
        externalVolumeBins = alloc_double_vector(nX5 / nX2);
        aggregationKernel = alloc_double_vector(nX5);
        breakageKernel = alloc_double_vector(nX5);
        collisionFrequency = alloc_double_vector(nX5);
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
        liquidBins = device_alloc_double_vector(nX5 / nX2);
        gasBins = device_alloc_double_vector(nX5 / nX2);
        internalVolumeBins = device_alloc_double_vector(nX5 / nX2);
        externalVolumeBins = device_alloc_double_vector(nX5 / nX2);
        aggregationKernel = device_alloc_double_vector(nX5);
        breakageKernel = device_alloc_double_vector(nX5);
        collisionFrequency = device_alloc_double_vector(nX5);
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
        brkKernelConst = 0.0;
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
        depletionThroughBreakage = alloc_double_vector(nX2);
        depletionOfGasThroughBreakage = alloc_double_vector(nX2);
        depletionOfLiquidthroughBreakage = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        brkKernelConst = 0.0;
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
        depletionThroughBreakage = device_alloc_double_vector(nX2);
        depletionOfGasThroughBreakage = device_alloc_double_vector(nX2);
        depletionOfLiquidthroughBreakage = device_alloc_double_vector(nX2);
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
    }
    else
        printf("\n Wrong Value of check passed in BreakageCompVar call \n");
}