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
                                        double time, double timeStep, double initialTime, double demTimeStep, int blx, int tlx, int mbdx, int nFirstSolidBins, 
                                        int nSecondSolidBins, int nCompartments, double aggKernelConst)
{
    int bix = blockIdx.x;
    int bdx = blockDim.x;

    int tix = threadIdx.x;

    int idx4 = blx * mbdx * bdx + tlx * bdx + tix;
    int idx3 = blx * mbdx + tlx;
    int idx3s = blx * bdx + tix;
    int s1 = (int) floorf(tlx / nFirstSolidBins);
    int ss1 = tlx % nSecondSolidBins;
    int s2 = (int) floorf(tix / nFirstSolidBins);
    int ss2 = tix % nSecondSolidBins;
    

    double criticalExternalLiquid = 0.2;
    bool flag1 = (d_compartmentIn->fAll[idx3] >= 0.0) && (d_compartmentIn->fAll[idx3s] >= 0.0);
    bool flag2 = ((d_compVar->externalLiquid[idx3] + d_compVar->externalLiquid[idx3s]) / (d_compartmentIn->fAll[idx3] * d_compartmentIn->vs[s1] + d_compartmentIn->fAll[idx3s] * d_compartmentIn->vss[ss1]));
    bool flag3 = (d_compartmentDEMIn->velocityCol[s1] < d_compartmentDEMIn->uCriticalCol[0]);
    if (flag1 && flag2 && flag3)
    {
        d_compartmentDEMIn->colEfficiency[idx4] = d_compartmentDEMIn->colProbability[ss2];
    }
    else
        d_compartmentDEMIn->colEfficiency[idx4] = 0.0;

    d_compartmentDEMIn->colFrequency[idx4] = (d_compartmentDEMIn->DEMCollisionData[tix] * timeStep) / demTimeStep;

    d_compartmentOut->aggregationKernel[idx4] = d_aggCompVar->aggKernelConst[0] * d_compartmentDEMIn->colFrequency[idx4] * d_compartmentDEMIn->colEfficiency[idx4];
    d_compVar->aggregationRate[idx4] = d_compartmentIn->sAggregationCheck[s1 * nFirstSolidBins + s2] * d_compartmentIn->ssAggregationCheck[ss1 * nFirstSolidBins + ss2] * d_compartmentOut->aggregationKernel[idx4] * d_compartmentIn->fAll[idx3] * d_compartmentIn->fAll[idx3s];
    __syncthreads();
    d_aggCompVar->depletionThroughAggregation[idx3] += d_compVar->aggregationRate[idx4];
    __syncthreads();
    d_aggCompVar->depletionThroughAggregation[idx3s] += d_compVar->aggregationRate[idx4];
    __syncthreads();
    d_aggCompVar->depletionOfGasThroughAggregation[idx3] = d_aggCompVar->depletionThroughAggregation[idx3] * d_compartmentOut->gasBins[idx3];
    d_aggCompVar->depletionOfLiquidThroughAggregation[idx3] = d_aggCompVar->depletionThroughAggregation[idx3] * d_compartmentOut->liquidBins[idx3];

    for (int i = 0; i < nFirstSolidBins; i++)
    {
        for(int j = 0; j < nSecondSolidBins; j++)
        {
            if (d_compartmentIn->sInd[s1 * nFirstSolidBins + s2] == (i+1) && d_compartmentIn->ssInd[ss1 * nSecondSolidBins + ss2] == (j+1))
            {
                d_aggCompVar->birthThroughAggregation[blx * mbdx + i * nFirstSolidBins + j] += d_compVar->aggregationRate[idx4];
                d_aggCompVar->firstSolidBirthThroughAggregation[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentIn->vs[(s1)] + d_compartmentIn->vs[s2]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->secondSolidBirthThroughAggregation[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentIn->vss[(ss1)] + d_compartmentIn->vss[ss2]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->liquidBirthThroughAggregation[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentOut->liquidBins[idx3] + d_compartmentOut->liquidBins[idx3s]) * d_compVar->aggregationRate[idx4];
                d_aggCompVar->gasBirthThroughAggregation[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentOut->gasBins[idx3] + d_compartmentOut->gasBins[idx3s]) * d_compVar->aggregationRate[idx4];
            }
        }
    }

    __syncthreads();

    if (tix == 0)
    {
        if (fabs(d_aggCompVar->birthThroughAggregation[idx3]) > 1e-16)
        {
            d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] = d_aggCompVar->firstSolidBirthThroughAggregation[idx3] / d_aggCompVar->birthThroughAggregation[idx3];
            d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] = d_aggCompVar->secondSolidBirthThroughAggregation[idx3] / d_aggCompVar->birthThroughAggregation[idx3];
        }
        else
        {
            d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] = 0.0;
            d_aggCompVar->secondSolidBirthThroughAggregation[idx3] = 0.0;
        }
        int val1 = s1; // s
        int val2 = ss1; // ss
        // int s3 = val1 * nFirstSolidBins + val2;
        // this is only defined to account for loops going nsolidBins - 1

        if (val1 == nFirstSolidBins - 1 && val2 == nSecondSolidBins - 1)
        {
            d_aggCompVar->birthAggHighHigh[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 - 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighLiq[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1- 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHighLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighGas[idx3]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1- 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
        }

        else if (val2 == nSecondSolidBins - 1)
        {
            d_aggCompVar->birthAggLowHigh[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHigh[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2 - 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggLowHigh[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins)]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins)] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins)] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowHighLiq[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHighLiq[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2 - 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggLowHighLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins)]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins)] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins)] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowHighGas[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHighGas[idx3] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2 - 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggLowHighGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins)]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins)] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2- 1]) / (d_compartmentIn->vss[val2] - d_compartmentIn->vss[val2 -1]);
            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins)] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
        }

        else if (val1 == nFirstSolidBins - 1)
        {
            d_aggCompVar->birthAggHighLow[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 - 1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
            d_aggCompVar->birthAggHighLow[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLow[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHigh[idx3 + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 - 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHigh[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHigh[idx3 + 1] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighLowLiq[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 - 1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
            d_aggCompVar->birthAggHighLowLiq[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLowLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighLiq[idx3 + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1- 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + 1] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighLowGas[idx3] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1 - 1] ) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 - 1]);
            d_aggCompVar->birthAggHighLowGas[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLowGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighGas[idx3 + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1- 1]) / (d_compartmentIn->vs[val1] - d_compartmentIn->vs[val1 -1]);
            d_aggCompVar->birthAggHighHighGas[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHighGas[idx3 + 1] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

        }

        else
        {
            d_aggCompVar->birthAggLowLow[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggLowLow[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowLow[idx3] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins) + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins) + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHigh[idx3 + (nFirstSolidBins) + 1] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowHigh[idx3 + 1] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHigh[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowHigh[idx3 + 1] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighLow[idx3 + (nFirstSolidBins)] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighLow[idx3 + (nFirstSolidBins)] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLow[idx3 + (nFirstSolidBins)] *= d_aggCompVar->birthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowLowLiq[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggLowLowLiq[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowLowLiq[idx3] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins) + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins) + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHighLiq[idx3 + (nFirstSolidBins) + 1] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowHighLiq[idx3 + 1] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHighLiq[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowHighLiq[idx3 + 1] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighLowLiq[idx3 + (nFirstSolidBins)] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighLowLiq[idx3 + (nFirstSolidBins)] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLowLiq[idx3 + (nFirstSolidBins)] *= d_aggCompVar->liquidBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowLowGas[idx3] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggLowLowGas[idx3] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowLowGas[idx3] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins) + 1]  = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins) + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] -d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighHighGas[idx3 + (nFirstSolidBins) + 1] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggLowHighGas[idx3 + 1] = (d_compartmentIn->vs[val1 + 1] - d_aggCompVar->firstSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vs[val1 + 1] - d_compartmentIn->vs[val1]); 
            d_aggCompVar->birthAggLowHighGas[idx3 + 1] *= (d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vss[val2]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggLowHighGas[idx3 + 1] *= d_aggCompVar->gasBirthThroughAggregation[idx3];

            d_aggCompVar->birthAggHighLowGas[idx3 + (nFirstSolidBins)] = (d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] - d_compartmentIn->vs[val1]) / (d_compartmentIn->vs[val1 +1] - d_compartmentIn->vs[val1]);
            d_aggCompVar->birthAggHighLowGas[idx3 + (nFirstSolidBins)] *= (d_compartmentIn->vss[val2 + 1] - d_aggCompVar->secondSolidVolumeThroughAggregation[idx3]) / (d_compartmentIn->vss[val2 + 1] - d_compartmentIn->vss[val2]);
            d_aggCompVar->birthAggHighLowGas[idx3 + (nFirstSolidBins)] *= d_aggCompVar->gasBirthThroughAggregation[idx3];
        }
    }
    __syncthreads();
    if (tix == 0)
    {
    d_aggCompVar->formationThroughAggregationCA[idx3] = d_aggCompVar->birthAggHighHigh[idx3] + d_aggCompVar->birthAggHighLow[idx3] + d_aggCompVar->birthAggLowHigh[idx3] + d_aggCompVar->birthAggLowLow[idx3];
    d_aggCompVar->formationOfLiquidThroughAggregationCA[idx3] = d_aggCompVar->birthAggHighHighLiq[idx3] + d_aggCompVar->birthAggHighLowLiq[idx3] + d_aggCompVar->birthAggLowHighLiq[idx3] + d_aggCompVar->birthAggLowLowLiq[idx3];
    d_aggCompVar->formationOfGasThroughAggregationCA[idx3] = d_aggCompVar->birthAggHighHighGas[idx3] + d_aggCompVar->birthAggHighLowGas[idx3] + d_aggCompVar->birthAggLowHighGas[idx3] + d_aggCompVar->birthAggLowLowGas[idx3];
    }
    __syncthreads();
}

// ==================== BREAKAGE COMPARTMENT CALCULATIONS ===========================================

__global__ void performBreakageCalculations(PreviousCompartmentIn *d_prevCompIn, CompartmentIn *d_compartmentIn, CompartmentDEMIn *d_compartmentDEMIn, 
                                        CompartmentOut *d_compartmentOut, CompartmentVar *d_compVar, BreakageCompVar *d_brCompVar, 
                                        double time, double timeStep, double initialTime, double demTimeStep, int blx, int tlx, int mbdx, int nFirstSolidBins, int nSecondSolidBins, double brkKernelConst)
{
    int tix = threadIdx.x;
    int bdx = blockDim.x;

    // int s3 = val1 * nFirstSolidBins + val2;
    int s1 = (int) floorf(tlx / nFirstSolidBins);
    int ss1 = tlx % nSecondSolidBins;
    int s2 = (int) floorf(tix / nFirstSolidBins);
    int ss2 = tix % nSecondSolidBins;
    int idx4 = blx * mbdx * bdx + tlx * bdx + tix;
    int idx3 = blx * mbdx + tlx; 
    int idx3s = blx * bdx + tix; 
    
    int val1 = s1; // s
    int val2 = ss1; // ss
    
    d_compartmentDEMIn->impactFrequency[s1] = (d_compartmentDEMIn->DEMImpactData[s1] * timeStep) / demTimeStep;
    __syncthreads();
    d_compartmentOut->breakageKernel[idx4] = d_compartmentDEMIn->impactFrequency[s1] * d_compartmentDEMIn->brProbability[ss2] * d_brCompVar->brkKernelConst[0];
    d_compVar->breakageRate[idx4] = d_compartmentIn->sCheckB[s1 * nFirstSolidBins + s2] * d_compartmentIn->ssCheckB[ss1 * nFirstSolidBins + ss2] * d_compartmentOut->breakageKernel[idx4] * d_compartmentIn->fAll[blx * mbdx + s1 * nFirstSolidBins + s2];
    
    d_brCompVar->depletionThroughBreakage[idx3] += d_compVar->breakageRate[idx4];
    d_brCompVar->depletionOfLiquidthroughBreakage[idx3] = d_brCompVar->depletionThroughBreakage[idx3] * d_compartmentOut->liquidBins[idx3];
    d_brCompVar->depletionOfGasThroughBreakage[idx3] = d_brCompVar->depletionThroughBreakage[idx3] * d_compartmentOut->gasBins[idx3];
    d_brCompVar->birthThroughBreakage1[idx3] += d_compVar->breakageRate[idx4];
    
    d_compartmentOut->internalVolumeBins[idx3] = d_compartmentIn->sMeshXY[tlx] + d_compartmentIn->ssMeshXY[tix] + d_compVar->internalLiquid[idx3] + d_compVar->gasBins[idx3];
    d_compVar->externalLiquid[idx3] = d_compartmentIn->sMeshXY[tlx] + d_compartmentIn->ssMeshXY[tix] + d_compVar->liquidBins[idx3] + d_compVar->gasBins[idx3];
    d_compVar->volumeBins[idx3] = d_compartmentIn->sMeshXY[tlx] + d_compartmentIn->ssMeshXY[tix];
    __syncthreads();

    for (int i = 0; i < nFirstSolidBins; i++)
    {
        for(int j = 0; j < nSecondSolidBins; j++)
        {
            // int s12 = (idx3 % nFirstSolidBins) * (bdx / nFirstSolidBins) + (idx3 % nFirstSolidBins);
            // int s22 = (idx3s % nFirstSolidBins) * (bdx / nFirstSolidBins) + (idx3s % nFirstSolidBins);
            if (d_compartmentIn->sIndB[s1 * nFirstSolidBins + s2] == (i+1) && d_compartmentIn->ssIndB[ss1 * nFirstSolidBins + ss2] == (j+1))
            {
                d_brCompVar->birthThroughBreakage2[blx * mbdx + i * nFirstSolidBins + j] += d_compVar->breakageRate[idx4];
                d_brCompVar->firstSolidBirthThroughBreakage[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentIn->vs[s1] - d_compartmentIn->vs[s2]) * d_compVar->breakageRate[idx4];
                d_brCompVar->secondSolidBirthThroughBreakage[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentIn->vss[ss1] - d_compartmentIn->vss[ss2]) * d_compVar->breakageRate[idx4];
                d_brCompVar->liquidBirthThroughBreakage2[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentOut->liquidBins[idx3] * (1 - (d_compVar->volumeBins[idx3s] / d_compVar->volumeBins[idx3]))) * d_compVar->breakageRate[idx4];
                d_brCompVar->gasBirthThroughBreakage2[blx * mbdx + i * nFirstSolidBins + j] += (d_compartmentOut->gasBins[idx3] * (1 - (d_compVar->volumeBins[idx3s] / d_compVar->volumeBins[idx3]))) * d_compVar->breakageRate[idx4];
                if (fabs(d_brCompVar->birthThroughBreakage2[blx * mbdx + i * nFirstSolidBins + j]) > 1e-16)
                {
                    d_brCompVar->firstSolidVolumeThroughBreakage[blx * mbdx + i * nFirstSolidBins + j] = d_brCompVar->firstSolidBirthThroughBreakage[idx3] / d_brCompVar->birthThroughBreakage2[idx3];
                    d_brCompVar->secondSolidVolumeThroughBreakage[blx * mbdx + i * nFirstSolidBins + j] = d_brCompVar->secondSolidBirthThroughBreakage[idx3] / d_brCompVar->birthThroughBreakage2[idx3];
                }
            }
        }
    }
    __syncthreads();
    d_brCompVar->liquidBirthThroughBreakage1[idx3s] += (d_compartmentOut->liquidBins[idx3] * (d_compVar->volumeBins[idx3s] / d_compVar->volumeBins[idx3])) * d_compVar->breakageRate[idx4];
    d_brCompVar->gasBirthThroughBreakage1[idx3s] += (d_compartmentOut->gasBins[idx3] * (d_compVar->volumeBins[idx3s] / d_compVar->volumeBins[idx3])) * d_compVar->breakageRate[idx4];
    __syncthreads();
    double value1 = 0.0;
    double value2 = 0.0;
    if (tix == 0)
    {
        value1 = fabs(d_compartmentIn->sLow[tlx] - d_brCompVar->firstSolidBirthThroughBreakage[idx3]);
        value1 = d_compartmentIn->sHigh[tlx] - d_compartmentIn->sLow[tlx] - value1;
        value1 /= (d_compartmentIn->sHigh[tlx] - d_compartmentIn->sLow[tlx]);

        value2 = fabs(d_compartmentIn->ssLow[tlx] - d_brCompVar->secondSolidVolumeThroughBreakage[idx3]);
        value2 = d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx] - value2;
        value2 /= (d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx]);

        d_brCompVar->fractionBreakage00[idx3] = value1 / value2;

        value2 = fabs(d_compartmentIn->ssHigh[tlx] - d_brCompVar->secondSolidVolumeThroughBreakage[idx3]);
        value2 = d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx] - value2;
        value2 /= (d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx]);
        
        d_brCompVar->fractionBreakage01[idx3] = value1 / value2;

        value1 = fabs(d_compartmentIn->sHigh[tlx] - d_brCompVar->firstSolidVolumeThroughBreakage[idx3]);
        value1 = d_compartmentIn->sHigh[tlx] - d_compartmentIn->sLow[tlx] - value1;
        value1 /= (d_compartmentIn->sHigh[tlx] - d_compartmentIn->sLow[tlx]);

        d_brCompVar->fractionBreakage11[idx3] = value1 / value2;

        value2 = fabs(d_compartmentIn->ssLow[tlx] - d_brCompVar->secondSolidVolumeThroughBreakage[idx3]);
        value2 = d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx] - value2;
        value2 /= (d_compartmentIn->ssHigh[tlx] - d_compartmentIn->ssLow[tlx]);

        d_brCompVar->fractionBreakage10[idx3] = value1 / value2;
    

        if (val1 == nFirstSolidBins - 1 && val2 == nSecondSolidBins - 1)
        {
            d_brCompVar->formationThroughBreakageCA[idx3] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];

        }

        else if (val2 == nSecondSolidBins - 1)
        {
            d_brCompVar->formationThroughBreakageCA[idx3] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];

            d_brCompVar->formationThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];


        }

        else if (val1 == nFirstSolidBins -1)
        {
            d_brCompVar->formationThroughBreakageCA[idx3] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];

            d_brCompVar->formationThroughBreakageCA[idx3 + 1] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3 + 1] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3 + 1] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];


        }

        else
        {
            d_brCompVar->formationThroughBreakageCA[idx3 + nFirstSolidBins + 1] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3 + nFirstSolidBins + 1] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3 + nFirstSolidBins + 1] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage11[idx3];

            d_brCompVar->formationThroughBreakageCA[idx3 + 1] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3 + 1] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3 + 1] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage01[idx3];

            d_brCompVar->formationThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3 + nFirstSolidBins] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage10[idx3];

            d_brCompVar->formationThroughBreakageCA[idx3] += d_brCompVar->birthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage00[idx3];
            d_brCompVar->formationOfLiquidThroughBreakageCA[idx3] += d_brCompVar->liquidBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage00[idx3];
            d_brCompVar->formationOfGasThroughBreakageCA[idx3] += d_brCompVar->gasBirthThroughBreakage2[idx3] * d_brCompVar->fractionBreakage00[idx3];
        }
    }
}

// ============ Constructors for the Classes =================


CompartmentVar :: CompartmentVar(unsigned int nX2, unsigned int nX5, unsigned int check)
{
    if (check == 0)
    {
        internalLiquid = alloc_double_vector(nX5 / nX2);
        externalLiquid = alloc_double_vector(nX5 / nX2);
        externalLiquidContent = alloc_double_vector(nX2);
        volumeBins = alloc_double_vector(nX5 / nX2);
        aggregationRate = alloc_double_vector(nX5);
        breakageRate = alloc_double_vector(nX5);
        particleMovement = alloc_double_vector(nX2);
        liquidMovement = alloc_double_vector(nX2);
        gasMovement = alloc_double_vector(nX2);
        liquidBins = alloc_double_vector(nX2);
        gasBins = alloc_double_vector(nX2);
        meshXYSum = alloc_double_vector(nX2);
    }

    else if (check == 1)
    {
        internalLiquid = device_alloc_double_vector(nX5 / nX2);
        externalLiquid = device_alloc_double_vector(nX5 / nX2);
        externalLiquidContent = device_alloc_double_vector(nX2);
        volumeBins = device_alloc_double_vector(nX5 / nX2);
        aggregationRate = device_alloc_double_vector(nX5);
        breakageRate = device_alloc_double_vector(nX5);
        particleMovement = device_alloc_double_vector(nX2);
        liquidMovement = device_alloc_double_vector(nX2);
        gasMovement = device_alloc_double_vector(nX2);
        liquidBins = device_alloc_double_vector(nX2);
        gasBins = device_alloc_double_vector(nX2);
        meshXYSum = device_alloc_double_vector(nX2);
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
        liquidAdditionRate = alloc_double_vector((nX5 / (nX2 ^ 2)));
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
        liquidAdditionRate = device_alloc_double_vector((nX5 / (nX2 ^ 2)));
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

PreviousCompartmentIn :: PreviousCompartmentIn(unsigned int nX2, unsigned int nX5, unsigned int check)
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
        uCriticalCol = alloc_double_vector(1);
        ubreak = alloc_double_vector(1);
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
        uCriticalCol = device_alloc_double_vector(1);
        ubreak = device_alloc_double_vector(1);
    }

    else 
        printf("\n Wrong Value of check passed in CompartmentDEMIn call \n");

}

CompartmentOut :: CompartmentOut(unsigned int nX2, unsigned int nX5, unsigned int check)
{
    if (check == 0)
    {
        dfAlldt = alloc_double_vector(nX5 / nX2);
        dfLiquiddt = alloc_double_vector(nX5 / nX2);
        dfGasdt = alloc_double_vector(nX5 / nX2);
        liquidBins = alloc_double_vector(nX5 / nX2);
        gasBins = alloc_double_vector(nX5 / nX2);
        internalVolumeBins = alloc_double_vector(nX5 / nX2);
        externalVolumeBins = alloc_double_vector(nX5 / nX2);
        aggregationKernel = alloc_double_vector(nX5);
        breakageKernel = alloc_double_vector(nX5);
        collisionFrequency = alloc_double_vector(nX5);
        formationThroughAggregation = alloc_double_vector((nX5 / (nX2 ^ 2)));
        depletionThroughAggregation = alloc_double_vector((nX5 / (nX2 ^ 2)));
        formationThroughBreakage = alloc_double_vector((nX5 / (nX2 ^ 2)));
        depletionThroughBreakage = alloc_double_vector((nX5 / (nX2 ^ 2)));
    }

    else if (check == 1)
    {
        dfAlldt = device_alloc_double_vector(nX5 / nX2);
        dfLiquiddt = device_alloc_double_vector(nX5 / nX2);
        dfGasdt = device_alloc_double_vector(nX5 / nX2);
        liquidBins = device_alloc_double_vector(nX5 / nX2);
        gasBins = device_alloc_double_vector(nX5 / nX2);
        internalVolumeBins = device_alloc_double_vector(nX5 / nX2);
        externalVolumeBins = device_alloc_double_vector(nX5 / nX2);
        aggregationKernel = device_alloc_double_vector(nX5);
        breakageKernel = device_alloc_double_vector(nX5);
        collisionFrequency = device_alloc_double_vector(nX5);
        formationThroughAggregation = device_alloc_double_vector((nX5 / (nX2 ^ 2)));
        depletionThroughAggregation = device_alloc_double_vector((nX5 / (nX2 ^ 2)));
        formationThroughBreakage = device_alloc_double_vector((nX5 / (nX2 ^ 2)));
        depletionThroughBreakage = device_alloc_double_vector((nX5 / (nX2 ^ 2)));
    }

    else
        printf("\n Wrong Value of check passed in CompartmentOut call \n");
}

BreakageCompVar :: BreakageCompVar(unsigned int nX2, unsigned int nX4, unsigned int check)
{
    if (check == 0)
    {
        brkKernelConst = alloc_double_vector(1);
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
        brkKernelConst = device_alloc_double_vector(1);
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
        aggKernelConst = alloc_double_vector(1);
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
        aggKernelConst = device_alloc_double_vector(1);
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