#include <vector>
#include <cmath>
#include <float.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#include "utility.cuh"
#include "parameterData.h"
#include "liggghtsData.h"
#include "compartment.cuh"

using namespace std;

#define TWOWAYCOUPLING false

// MACROS 
// Calling macros for error check and dump data to files to VaribleName.txt

#define DUMP(varName) dumpData(varName, #varName)
#define DUMP2D(varName) dump2DData(varName, #varName)
#define DUMP3D(varName) dump3DData(varName, #varName)

#define DUMPCSV(varName) dumpCSV(varName, #varName)
#define DUMP2DCSV(varName) dump2DCSV(varName, #varName)
#define DUMP3DCSV(varName) dump3DCSV(varName, #varName)

#define DUMPDIACSV(time, dia) dumpDiaCSV(time, dia, #dia)

#define DUMP2DCSV4MATLAB(varName) dump2DCSV4Matlab(varName, #varName)

// extern __shared__ double *d_sMeshXY, *d_ssMeshXY;


// ==================================== INITIALIZATION KERNEL ===================================================

__global__ void initialization_kernel(double *d_vs, double *d_vss, size_t size2D, double fsVolCoeff, double ssVolCoeff, double fsVolBase, double ssVolBase, double *d_sAgg, 
                                      double *d_ssAgg, int *d_sAggregationCheck, int *d_ssAggregationCheck, double *d_sLow, double *d_ssLow, double *d_sHigh, double *d_ssHigh, 
                                      double *d_sMeshXY, double *d_ssMeshXY, int *d_sLoc, int *d_ssLoc, int *d_sInd, int *d_ssInd, double *d_sBreak, double *d_ssBreak, 
                                      int *d_sLocBreak, int *d_ssLocBreak, int *d_sCheckB, int*d_ssCheckB, int  *d_sIndB, int *d_ssIndB)
{
    int idx = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;

    // __shared__ double d_sMeshXY[256], d_ssMeshXY[256];

    d_sMeshXY[bdx * bix + idx] = d_vs[bix];
    d_ssMeshXY[bdx * bix + idx] = d_vss[idx];
    d_sAgg[bdx * bix + idx] = d_vs[idx] + d_vs[bix];
    d_ssAgg[bdx * bix + idx] = d_vss[idx] + d_vss[bix];
    d_sAggregationCheck[bdx * bix + idx] = d_sAgg[bdx * bix + idx] <= d_vs[bdx - 1] ? 1 : 0;
    d_ssAggregationCheck[bdx * bix + idx] = d_ssAgg[bdx * bix + idx] <= d_vss[bdx - 1] ? 1 : 0;
    d_sLow [bdx * bix + idx] = d_sMeshXY[bdx * bix + idx];
    d_ssLow[bdx * bix + idx] = d_ssMeshXY[bdx * bix + idx];
    __syncthreads();
    if (bix < bdx -1)
        d_sHigh[bdx * bix + idx] = d_sMeshXY[bdx * (bix + 1) + idx];

    d_ssHigh[bdx * bix + idx] = d_ssMeshXY[bdx * (bix) + idx + 1];
    d_sHigh[bdx * (bdx -1) + idx] = 0.0;
    d_ssHigh[bdx * bix + bdx - 1] = 0.0;
    d_sLoc[bdx * bix + idx] = floor(log(d_sAgg[bdx * bix + idx] / fsVolCoeff) / log(fsVolBase) + 1);
    d_ssLoc[bdx * bix + idx] = floor(log(d_ssAgg[bdx * bix + idx] / ssVolCoeff) / log(ssVolBase) + 1);
    d_sInd[bdx * bix + idx] = (idx <= bix) ? (bix + 1) : (idx + 1);
    d_ssInd[bdx * bix + idx] = (idx <= bix) ? (bix + 1) : (idx + 1);
    __syncthreads();
    double value = d_vs[idx] - d_vs[bix];
    double value1 = d_vss[idx] - d_vss[bix];
    d_sBreak[bdx * bix + idx] = value < 0.0 ? 0.0 : value;
    d_ssBreak[bdx * bix + idx] = value1 < 0.0 ? 0.0 : value1;
    __syncthreads();
}
__global__ void initialization_kernel2(double *d_vs, double *d_vss, size_t size2D, double fsVolCoeff, double ssVolCoeff, double fsVolBase, double ssVolBase, double *d_sAgg, 
                                      double *d_ssAgg, int *d_sAggregationCheck, int *d_ssAggregationCheck, double *d_sLow, double *d_ssLow, double *d_sHigh, double *d_ssHigh, 
                                      double *d_sMeshXY, double *d_ssMeshXY, int *d_sLoc, int *d_ssLoc, int *d_sInd, int *d_ssInd, double *d_sBreak, double *d_ssBreak, 
                                      int *d_sLocBreak, int *d_ssLocBreak, int *d_sCheckB, int*d_ssCheckB, int  *d_sIndB, int *d_ssIndB)
{
    int idx = threadIdx.x;
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    d_sLocBreak[bdx * bix + idx] = (d_sBreak[bdx * idx + bix] == 0) ? 0 : (floor(log(d_sBreak[bdx * idx + bix] / fsVolCoeff) / log(fsVolBase) + 1));
    d_ssLocBreak[bdx * bix + idx] = (d_ssBreak[bdx * idx + bix] == 0) ? 0 : (floor(log(d_ssBreak[bdx * idx + bix] / ssVolCoeff) / log(ssVolBase) + 1));
    __syncthreads();
    d_sCheckB[bdx * bix + idx] = d_sLocBreak[bdx * bix + idx] >= 1 ? 1 : 0;
    d_ssCheckB[bdx * bix + idx] = d_ssLocBreak[bdx * bix + idx] >= 1 ? 1 : 0;
    d_sIndB[bdx * bix + idx] = d_sLocBreak[bdx * bix + idx];
    d_ssIndB[bdx * bix + idx] = d_ssLocBreak[bdx * bix + idx];
    if (d_sIndB[bdx * bix + idx] < 1)
        d_sIndB[bdx * bix + idx] = bdx + 1;
    if (d_ssIndB[bdx * bix + idx] < 1)
        d_ssIndB[bdx * bix + idx] = bdx + 1;

    __syncthreads();
}

// ================================= COMPARTMENT LAUNCH KERNEL ============================================================

__global__ void launchCompartment(CompartmentIn *d_compartmentIn, PreviousCompartmentIn *d_prevCompInData, CompartmentOut *d_compartmentOut, CompartmentDEMIn *d_compartmentDEMIn,
                                  CompartmentVar *d_compVar, AggregationCompVar *d_aggCompVar, BreakageCompVar *d_brCompVar, double time, double timeStep, double initialTime,
                                  double *d_fAllCompartments, double *d_flAllCompartments, double *d_fgAllCompartments, double *d_liquidAdditionRateAllCompartments,
                                  unsigned int size2D, unsigned int size3D, unsigned int size4D, double *d_fIn, double initPorosity, double demTimeStep, int nFirstSolidBins, int nSecondSolidBins,
                                  double granulatorLength, double partticleResTime, double premixTime, double liqAddTime, double consConst, double minPorosity, int nCompartments, 
                                  double granSatFactor, double aggKernelConst, double brkKernelConst)
{
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    int tix = threadIdx.x;
    int idx3 = bix * bdx + tix;
    int s1 = (int) floorf(tix / nFirstSolidBins);
    int ss1 = tix % nSecondSolidBins;

    if (tix ==0)
    {    
        d_compartmentOut->formationThroughAggregation[bix] = 0.0;
        d_compartmentOut->depletionThroughAggregation[bix] = 0.0;
        d_compartmentOut->formationThroughBreakage[bix] = 0.0;
        d_compartmentOut->depletionThroughBreakage[bix] = 0.0;
    }
    // int tiy = threadIdx.y;
    d_compartmentOut->dfAlldt[idx3] = 0.0;
    d_compartmentOut->dfLiquiddt[idx3] = 0.0;
    d_compartmentOut->dfGasdt[idx3] = 0.0;
    d_compartmentOut->liquidBins[idx3] = 0.0;
    d_compartmentOut->gasBins[idx3] = 0.0;
    d_compartmentOut->internalVolumeBins[idx3] = 0.0;
    d_compartmentOut->externalVolumeBins[idx3] = 0.0;
    // int idx = bix * bdx * bdy + tiy * bdx + tix;
    //if (tiy == 0)
   
    d_compartmentIn->fAll[idx3] = d_fAllCompartments[idx3];
    d_compartmentIn->fLiquid[idx3] = d_flAllCompartments[idx3];
    d_compartmentIn->fGas[idx3] = d_fgAllCompartments[idx3];
    d_compartmentIn->liquidAdditionRate[bix] = d_liquidAdditionRateAllCompartments[bix];

    if (bix == 0)
    {
        d_prevCompInData->fAllPreviousCompartment[idx3] = 0.0;
        d_prevCompInData->flPreviousCompartment[idx3] = 0.0;
        d_prevCompInData->fgPreviousCompartment[idx3] = 0.0;
        d_prevCompInData->fAllComingIn[idx3] = d_fIn[tix];
        d_prevCompInData->fgComingIn[idx3] = 0.0;
        double value = initPorosity * timeStep;
        d_prevCompInData->fgComingIn[idx3] = d_fIn[tix] * (d_compartmentIn->vs[s1] + d_compartmentIn->vss[ss1]) * value;
    }
    else
    {
        d_prevCompInData->fAllPreviousCompartment[idx3] = d_fAllCompartments[(bix - 1) * bdx + tix];
        d_prevCompInData->flPreviousCompartment[idx3] = d_flAllCompartments[(bix - 1) * bdx + tix];
        d_prevCompInData->fgPreviousCompartment[idx3] = d_fgAllCompartments[(bix - 1) * bdx + tix];
        d_prevCompInData->fAllComingIn[idx3] = 0.0;
        d_prevCompInData->fgComingIn[idx3] = 0.0;
    }
    __syncthreads();

    if (fabs(d_compartmentIn->fAll[idx3]) > 1e-16)
    {
        d_compartmentOut->liquidBins[idx3] = d_compartmentIn->fLiquid[idx3] / d_compartmentIn->fAll[idx3];
        d_compartmentOut->gasBins[idx3] = d_compartmentIn->fGas[idx3] / d_compartmentIn->fAll[idx3];
    }
    else
    {
        d_compartmentOut->liquidBins[idx3] = 0.0;
        d_compartmentOut->gasBins[idx3] = 0.0;
    }

    d_aggCompVar->depletionThroughAggregation[idx3] = 0.0;
    d_aggCompVar->depletionOfGasThroughAggregation[idx3] = 0.0;
    d_aggCompVar->depletionOfLiquidThroughAggregation[idx3] = 0.0;
    d_aggCompVar->birthThroughAggregation[idx3] = 0.0;
    d_aggCompVar->firstSolidBirthThroughAggregation[idx3] = 0.0;
    d_aggCompVar->secondSolidBirthThroughAggregation[idx3] = 0.0;
    d_aggCompVar->liquidBirthThroughAggregation[idx3] = 0.0;
    d_aggCompVar->gasBirthThroughAggregation[idx3] = 0.0;
    d_aggCompVar->firstSolidVolumeThroughAggregation[idx3] = 0.0;
    d_aggCompVar->secondSolidVolumeThroughAggregation[idx3] = 0.0;
    d_aggCompVar->birthAggLowHigh[idx3] = 0.0;
    d_aggCompVar->birthAggLowHighLiq[idx3] = 0.0;
    d_aggCompVar->birthAggLowHighGas[idx3] = 0.0;
    d_aggCompVar->birthAggHighLow[idx3] = 0.0;
    d_aggCompVar->birthAggHighLowLiq[idx3] = 0.0;
    d_aggCompVar->birthAggHighLowGas[idx3] = 0.0;
    d_aggCompVar->birthAggLowLow[idx3] = 0.0;
    d_aggCompVar->birthAggLowLowLiq[idx3] = 0.0;
    d_aggCompVar->birthAggLowLowGas[idx3] = 0.0;
    d_aggCompVar->birthAggHighHigh[idx3] = 0.0;
    d_aggCompVar->birthAggHighHighLiq[idx3] = 0.0;
    d_aggCompVar->birthAggHighHighGas[idx3] = 0.0;
    d_aggCompVar->formationThroughAggregationCA[idx3] = 0.0;
    d_aggCompVar->formationOfLiquidThroughAggregationCA[idx3] = 0.0;
    d_aggCompVar->formationOfGasThroughAggregationCA[idx3] = 0.0;


    d_brCompVar->fractionBreakage00[idx3] = 0.0;
    d_brCompVar->fractionBreakage01[idx3] = 0.0;
    d_brCompVar->fractionBreakage10[idx3] = 0.0;
    d_brCompVar->fractionBreakage11[idx3] = 0.0;
    d_brCompVar->transferThroughLiquidAddition[idx3] = 0.0;
    d_brCompVar->transferThroughConsolidation[idx3] = 0.0;
    d_brCompVar->depletionThroughBreakage[idx3] = 0.0;
    d_brCompVar->depletionOfLiquidthroughBreakage[idx3] = 0.0;
    d_brCompVar->depletionOfGasThroughBreakage[idx3] = 0.0;
    d_brCompVar->birthThroughBreakage1[idx3] = 0.0;
    d_brCompVar->birthThroughBreakage2[idx3] = 0.0;
    d_brCompVar->firstSolidBirthThroughBreakage[idx3] = 0.0;
    d_brCompVar->secondSolidBirthThroughBreakage[idx3] = 0.0;
    d_brCompVar->liquidBirthThroughBreakage2[idx3] = 0.0;
    d_brCompVar->liquidBirthThroughBreakage1[idx3] = 0.0;
    d_brCompVar->gasBirthThroughBreakage1[idx3] = 0.0;
    d_brCompVar->gasBirthThroughBreakage2[idx3] = 0.0;
    d_brCompVar->firstSolidVolumeThroughBreakage[idx3] = 0.0;
    d_brCompVar->secondSolidVolumeThroughBreakage[idx3] = 0.0;
    d_brCompVar->formationThroughBreakageCA[idx3] = 0.0;
    d_brCompVar->formationOfLiquidThroughBreakageCA[idx3] = 0.0;
    d_brCompVar->formationOfGasThroughBreakageCA[idx3] = 0.0;

    d_compVar->internalLiquid[idx3] = 0.0;
    d_compVar->externalLiquid[idx3] = 0.0;
    d_compVar->externalLiquidContent[idx3] = 0.0;
    d_compVar->volumeBins[tix] = 0.0;
    d_compVar->particleMovement[idx3] = 0.0;
    d_compVar->liquidMovement[idx3] = 0.0;
    d_compVar->gasMovement[idx3] = 0.0;
    d_compVar->meshXYSum[tix] = 0.0;
    d_compVar->totalSolidvolume[bix] = 0.0;
    d_compVar->particleMovement[idx3] = 0.0;
    d_compVar->liquidMovement[idx3] = 0.0;
    d_compVar->gasMovement[idx3] = 0.0;
    
    d_compVar->internalLiquid[idx3] = min((granSatFactor * d_compartmentOut->gasBins[idx3]), d_compartmentOut->liquidBins[idx3]);
    d_compVar->externalLiquid[idx3] = max(0.0, (d_compartmentOut->liquidBins[idx3] - d_compVar->internalLiquid[idx3]));
    
    d_compartmentOut->internalVolumeBins[idx3] = d_compartmentIn->sMeshXY[tix] + d_compartmentIn->ssMeshXY[tix] + d_compVar->internalLiquid[idx3] + d_compartmentOut->gasBins[idx3];
    d_compVar->meshXYSum[tix] = d_compartmentIn->sMeshXY[tix] + d_compartmentIn->ssMeshXY[tix];
    __syncthreads();
}
    // printf("d_compartmentOut->liquidBins  = %f \n", d_compartmentOut->liquidBins[tix]);
__global__ void  consolidationAndMovementCalcs(CompartmentIn *d_compartmentIn, PreviousCompartmentIn *d_prevCompInData, CompartmentOut *d_compartmentOut, CompartmentDEMIn *d_compartmentDEMIn,
                                CompartmentVar *d_compVar, AggregationCompVar *d_aggCompVar, BreakageCompVar *d_brCompVar, int nCompartments, double granulatorLength, double partticleResTime,
                                double time, double timeStep, double premixTime, double liqAddTime, double initialTime, int nFirstSolidBins, int nSecondSolidBins, double consConst, double minPorosity)
{
    int bix = blockIdx.x;
    int bdx = blockDim.x;
    int tix = threadIdx.x;
    int idx3 = bix * bdx + tix;
    int s1 = (int) floorf(tix / nFirstSolidBins);
    int ss1 = tix % nSecondSolidBins;

    

    double maxValue = -DBL_MAX;
    for (size_t d1 = 0; d1 < bdx; d1++)
    {
        maxValue = max(maxValue, d_compVar->meshXYSum[d1]);
    }
    __syncthreads();


    double valueMeshXY = 1 - (d_compartmentIn->sMeshXY[tix] + d_compartmentIn->ssMeshXY[tix]) / maxValue;
    double distanceBetweenCompartments = granulatorLength / nCompartments;
    double particleAverageVelocity = granulatorLength /  partticleResTime;
    double distanceMoved = particleAverageVelocity * timeStep / distanceBetweenCompartments;// value renamed as distanceMoved

    d_compVar->particleMovement[idx3] = d_prevCompInData->fAllComingIn[idx3];
    d_compVar->particleMovement[idx3] += d_prevCompInData->fAllPreviousCompartment[idx3] * distanceMoved * valueMeshXY;
    d_compVar->particleMovement[idx3] -= d_compartmentIn->fAll[idx3] * distanceMoved;

    d_compVar->liquidMovement[idx3] = d_prevCompInData->flPreviousCompartment[idx3] * distanceMoved * valueMeshXY;
    d_compVar->liquidMovement[idx3] -= d_compartmentIn->fLiquid[idx3] * distanceMoved;

    d_compVar->gasMovement[idx3] = d_prevCompInData->fgComingIn[idx3];
    d_compVar->gasMovement[idx3] += d_prevCompInData->fgPreviousCompartment[idx3] * distanceMoved * valueMeshXY;
    d_compVar->gasMovement[idx3] -= d_compartmentIn->fGas[idx3] * distanceMoved;

    double finalTime = premixTime + liqAddTime + initialTime;
    if (tix == 0)
    {
        if (time >= premixTime && time <= finalTime)
            d_compartmentIn->liquidAdditionRate[bix] *= timeStep;
        else
            d_compartmentIn->liquidAdditionRate[bix] = 0.0;

        for (int i = bix * bdx; i < (bix+1) * bdx; i++)
            d_compVar->totalSolidvolume[bix] += d_compartmentIn->fAll[i] * (d_compartmentIn->vs[(int) floorf((i - bix * bdx) / nFirstSolidBins)] + d_compartmentIn->vss[(i - bix * bdx) % nSecondSolidBins]);
    }

    // d_compVar->totalSolidvolume[bix] += d_compartmentIn->fAll[idx3] * (d_compartmentIn->vs[s1] + d_compartmentIn->vss[ss1]);
    __syncthreads();
    d_compartmentOut->dfAlldt[idx3] = d_compVar->particleMovement[idx3];
    d_compartmentOut->dfAlldt[idx3] += d_aggCompVar->formationThroughAggregationCA[idx3] - d_aggCompVar->depletionThroughAggregation[idx3];
    d_compartmentOut->dfAlldt[idx3] += d_brCompVar->birthThroughBreakage1[idx3] + d_brCompVar->formationThroughBreakageCA[idx3] - d_brCompVar->depletionThroughBreakage[idx3];

    if (d_compVar->totalSolidvolume[bix] > 1.0e-16)
        d_brCompVar->transferThroughLiquidAddition[idx3] = d_compartmentIn->liquidAdditionRate[bix] * ((d_compartmentIn->vs[s1] + d_compartmentIn->vss[ss1]) / d_compVar->totalSolidvolume[bix]);

    d_compartmentOut->dfLiquiddt[idx3] = d_compVar->liquidMovement[idx3];
    d_compartmentOut->dfLiquiddt[idx3] += d_compartmentIn->fAll[idx3] * d_brCompVar->transferThroughLiquidAddition[idx3];
    d_compartmentOut->dfLiquiddt[idx3] += d_aggCompVar->formationOfLiquidThroughAggregationCA[idx3] - d_aggCompVar->depletionOfLiquidThroughAggregation[idx3];
    d_compartmentOut->dfLiquiddt[idx3] += d_brCompVar->liquidBirthThroughBreakage1[idx3] + d_brCompVar->formationOfLiquidThroughBreakageCA[idx3];
    d_compartmentOut->dfLiquiddt[idx3] -= d_brCompVar->depletionOfLiquidthroughBreakage[idx3];


    if(d_compartmentIn->fGas[idx3] > 1.0e-16)
    {
        d_brCompVar->transferThroughConsolidation[idx3] = consConst * d_compartmentOut->internalVolumeBins[idx3] * ((1 - minPorosity) / (d_compartmentIn->vs[s1] + d_compartmentIn->vss[ss1]));
        d_brCompVar->transferThroughConsolidation[idx3] *= (d_compartmentOut->gasBins[idx3] - (minPorosity / (1-minPorosity)) * (d_compartmentIn->vs[s1] + d_compartmentIn->vss[ss1]) + d_compVar->internalLiquid[idx3]);
    }
    else
        d_brCompVar->transferThroughConsolidation[idx3] = 0.0;

    d_compartmentOut->dfGasdt[idx3] = d_compVar->gasMovement[idx3];
    d_compartmentOut->dfGasdt[idx3] += d_compartmentIn->fAll[idx3] * d_brCompVar->transferThroughConsolidation[idx3];
    d_compartmentOut->dfGasdt[idx3] += d_aggCompVar->formationOfGasThroughAggregationCA[idx3] - d_aggCompVar->depletionOfGasThroughAggregation[idx3];
    d_compartmentOut->dfGasdt[idx3] += d_brCompVar->gasBirthThroughBreakage1[idx3] + d_brCompVar->formationOfGasThroughBreakageCA[idx3];
    d_compartmentOut->dfGasdt[idx3] -= d_brCompVar->depletionOfGasThroughBreakage[idx3];


    __syncthreads();

    if (tix == 0)
    {
        for (int i = bix * bdx; i < ((bix +1) * bdx); i++)
        {
            d_compartmentOut->formationThroughAggregation[bix] +=  d_aggCompVar->formationThroughAggregationCA[i];
            d_compartmentOut->depletionThroughAggregation[bix] += d_aggCompVar->depletionThroughAggregation[i];
            d_compartmentOut->formationThroughBreakage[bix] += d_brCompVar->formationThroughBreakageCA[i] + d_brCompVar->birthThroughBreakage1[i];
            d_compartmentOut->depletionThroughBreakage[bix] += d_brCompVar->depletionThroughBreakage[i];
        }
    }
}


// ===================================== MAIN FUNCTION ======================================================

int main(int argc, char *argv[])
{
    cout << "Code begins..." << endl;
    // Read passed arguments
    string startTimeStr;
    double startTime = 0.0;
    liggghtsData *lData = nullptr;
    parameterData *pData = nullptr;

    string coreVal;
    string diaVal;
    string pbmInFilePath;
    string timeVal;

    if (argc <5)
    {
        cout << "All values are not available as imput parameters " << endl;
        return 1;
    }

    pbmInFilePath = string(argv[1]);
    coreVal = string(argv[2]);
    diaVal = string(argv[3]);
    timeVal = string(argv[4]);

    pData = parameterData::getInstance();
    pData->readPBMInputFile(pbmInFilePath);

    int nCompartments = pData->nCompartments;
   
    unsigned int nFirstSolidBins = pData->nFirstSolidBins;
    unsigned int nSecondSolidBins = pData->nSecondSolidBins;

    size_t size1D = nFirstSolidBins;
    size_t size2D = size1D * nSecondSolidBins;
    size_t size3D = size2D * nCompartments;
    size_t size4D = size2D * size2D;
    size_t size5D = size4D * nCompartments;

    CompartmentIn compartmentIn(size2D, size5D, 0), x_compartmentIn(size2D, size5D, 1), *d_compartmentIn;
    PreviousCompartmentIn prevCompInData(size2D, size5D, 0), x_prevCompInData(size2D, size5D, 1), *d_prevCompInData;
    CompartmentOut compartmentOut(size2D, size5D, 0), x_compartmentOut(size2D, size5D, 1), *d_compartmentOut;
    CompartmentDEMIn compartmentDEMIn(size2D, size5D, 0), x_compartmentDEMIn(size2D, size5D, 1), *d_compartmentDEMIn;


    vector<double> h_vs(size1D, 0.0);
    vector<double> h_vss(size1D, 0.0);
    
    // Bin Initiation
    double fsVolCoeff = pData->fsVolCoeff;
    double fsVolBase = pData->fsVolBase;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        h_vs[i] = fsVolCoeff * pow(fsVolBase, i); // m^3

    double ssVolCoeff = pData->ssVolCoeff;
    double ssVolBase = pData->ssVolBase;
    for (size_t i = 0; i < nSecondSolidBins; i++) 
        h_vss[i] = ssVolCoeff * pow(ssVolBase, i); // m^3

    arrayOfDouble2D diameter1 = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    for (size_t s = 0; s < nFirstSolidBins; s++)
        for (size_t ss = 0; ss < nSecondSolidBins; ss++)
            diameter1[s][ss] = cbrt((6/M_PI) * (h_vs[s] + h_vss[ss]));

    vector<double> diameter = linearize2DVector(diameter1);
    
    vector<double> particleIn;
    particleIn.push_back(726657587.0);
    particleIn.push_back(286654401.0);
    particleIn.push_back(118218011.0);
    particleIn.push_back(50319795.0);
    particleIn.push_back(20954036.0);
    particleIn.push_back(7345998.0);
    particleIn.push_back(1500147.0);
    particleIn.push_back(76518.0);
    particleIn.push_back(149.0);
    
    vector<double> h_fIn(size2D, 0.0);
    for (size_t i = 0; i < particleIn.size(); i++)
        h_fIn[i * size1D + i] = particleIn[i];
    
    // allocation of memory for the matrices that will be copied onto the device from the host
    double *d_vs = device_alloc_double_vector(size1D);
    double *d_vss = device_alloc_double_vector(size1D);
    
    double *d_sMeshXY = device_alloc_double_vector(size2D);
    double *d_ssMeshXY = device_alloc_double_vector(size2D);

    double *d_sAgg = device_alloc_double_vector(size2D);
    double *d_ssAgg = device_alloc_double_vector(size2D);

    int *d_sAggregationCheck = device_alloc_integer_vector(size2D);
    int *d_ssAggregationCheck = device_alloc_integer_vector(size2D);

    double *d_sLow = device_alloc_double_vector(size2D);
    double *d_ssLow = device_alloc_double_vector(size2D);

    double *d_sHigh = device_alloc_double_vector(size2D);
    double *d_ssHigh = device_alloc_double_vector(size2D);

    int *d_sLoc = device_alloc_integer_vector(size2D);
    int *d_ssLoc = device_alloc_integer_vector(size2D);

    int *d_sInd = device_alloc_integer_vector(size2D);
    int *d_ssInd = device_alloc_integer_vector(size2D);

    double *d_sBreak = device_alloc_double_vector(size2D);
    double *d_ssBreak = device_alloc_double_vector(size2D);

    int *d_sLocBreak = device_alloc_integer_vector(size2D);
    int *d_ssLocBreak = device_alloc_integer_vector(size2D);

    int *d_sCheckB = device_alloc_integer_vector(size2D);
    int *d_ssCheckB = device_alloc_integer_vector(size2D);

    int *d_sIndB = device_alloc_integer_vector(size2D);
    int *d_ssIndB = device_alloc_integer_vector(size2D);

    // defining vectors for data required for compartment calculations
    vector<double> h_sMeshXY(size2D, 0.0);
    vector<double> h_ssMeshXY(size2D, 0.0);

    vector<int> h_sAggregationCheck(size2D, 0);
    vector<int> h_ssAggregationCheck(size2D, 0);

    vector<double> h_sLow(size2D, 0.0);
    vector<double> h_ssLow(size2D, 0.0);

    vector<double> h_sHigh(size2D, 0.0);
    vector<double> h_ssHigh(size2D, 0.0);

    vector<int> h_sInd(size2D, 0);
    vector<int> h_ssInd(size2D, 0);

    vector<int> h_sLoc(size2D, 0);
    vector<int> h_ssLoc(size2D, 0);

    vector<int> h_sCheckB(size2D, 0);
    vector<int> h_ssCheckB(size2D, 0);

    vector<int> h_sIndB(size2D, 0.0);
    vector<int> h_ssIndB(size2D, 0.0);

    vector<int> h_sLocBreak(size2D, 0.0);
    vector<int> h_ssLocBreak(size2D, 0.0);

    vector<double> h_sBreak(size2D, 0.0);
    vector<double> h_ssBreak(size2D, 0.0);


    copy_double_vector_fromHtoD(d_vs, h_vs.data(), size1D);
    copy_double_vector_fromHtoD(d_vss, h_vss.data(), size1D);

    int nBlocks = nFirstSolidBins;
    int nThreads = nSecondSolidBins;

    initialization_kernel<<<nBlocks,nThreads>>>(d_vs, d_vss, size2D, fsVolCoeff, ssVolCoeff, fsVolBase, ssVolBase, d_sAgg,d_ssAgg, d_sAggregationCheck, d_ssAggregationCheck, 
                                    d_sLow, d_ssLow, d_sHigh, d_ssHigh, d_sMeshXY, d_ssMeshXY, d_sLoc, d_ssLoc, d_sInd, d_ssInd, d_sBreak, d_ssBreak, d_sLocBreak, d_ssLocBreak,
                                    d_sCheckB, d_ssCheckB, d_sIndB, d_ssIndB);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    initialization_kernel2<<<nBlocks,nThreads>>>(d_vs, d_vss, size2D, fsVolCoeff, ssVolCoeff, fsVolBase, ssVolBase, d_sAgg,d_ssAgg, d_sAggregationCheck, d_ssAggregationCheck, 
                                    d_sLow, d_ssLow, d_sHigh, d_ssHigh, d_sMeshXY, d_ssMeshXY, d_sLoc, d_ssLoc, d_sInd, d_ssInd, d_sBreak, d_ssBreak, d_sLocBreak, d_ssLocBreak,
                                    d_sCheckB, d_ssCheckB, d_sIndB, d_ssIndB);
    err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    
    cout << "Initialization complete" << endl;

    // copy back data required for the compartment calculations

    copy_double_vector_fromDtoH(h_vs.data(), d_vs, size1D);
    copy_double_vector_fromDtoH(h_vss.data(), d_vss, size1D);

    copy_double_vector_fromDtoH(h_sMeshXY.data(), d_sMeshXY, size2D);
    copy_double_vector_fromDtoH(h_ssMeshXY.data(), d_ssMeshXY, size2D);

    copy_integer_vector_fromDtoH(h_sAggregationCheck.data(), d_sAggregationCheck, size2D);
    copy_integer_vector_fromDtoH(h_ssAggregationCheck.data(), d_ssAggregationCheck, size2D);

    copy_double_vector_fromDtoH(h_sLow.data(), d_sLow, size2D);
    copy_double_vector_fromDtoH(h_ssLow.data(), d_ssLow, size2D);

    copy_double_vector_fromDtoH(h_sHigh.data(), d_sHigh, size2D);
    copy_double_vector_fromDtoH(h_ssHigh.data(), d_ssHigh, size2D);

    copy_integer_vector_fromDtoH(h_sInd.data(), d_sInd, size2D);
    copy_integer_vector_fromDtoH(h_ssInd.data(), d_ssInd, size2D);

    copy_integer_vector_fromDtoH(h_sLoc.data(), d_sLoc, size2D);
    copy_integer_vector_fromDtoH(h_ssLoc.data(), d_ssLoc, size2D);

    copy_integer_vector_fromDtoH(h_sCheckB.data(), d_sCheckB, size2D);
    copy_integer_vector_fromDtoH(h_ssCheckB.data(), d_ssCheckB, size2D);

    copy_integer_vector_fromDtoH(h_sIndB.data(), d_sIndB, size2D);
    copy_integer_vector_fromDtoH(h_ssIndB.data(), d_ssIndB, size2D);

    copy_integer_vector_fromDtoH(h_sLocBreak.data(), d_sLocBreak, size2D);
    copy_integer_vector_fromDtoH(h_ssLocBreak.data(), d_ssLocBreak, size2D);

    copy_double_vector_fromDtoH(h_sBreak.data(), d_sBreak, size2D);
    copy_double_vector_fromDtoH(h_ssBreak.data(), d_ssBreak, size2D);
    cudaDeviceSynchronize();

    
    // DUMP(h_sMeshXY);
    // DUMP(h_ssMeshXY);
    // DUMP(h_sAggregationCheck);
    // DUMP(h_ssAggregationCheck);
    // DUMP(h_sLow);
    // DUMP(h_ssLow);
    // DUMP(h_sHigh);
    // DUMP(h_ssHigh);
    // DUMP(h_sInd);
    // DUMP(h_ssInd);
    // DUMP(h_sCheckB);
    // DUMP(h_ssCheckB);
    // DUMP(h_sIndB);
    // DUMP(h_ssIndB);
    // DUMP(h_sLocBreak);
    // DUMP(h_ssLocBreak);
    // DUMP(h_sBreak);
    // DUMP(h_ssBreak);
    // DUMP(h_sLoc);
    // DUMP(h_ssLoc);

    vector<double> h_fAllCompartments(size3D, 0.0);
    vector<double> h_flAllCompartments(size3D, 0.0);
    vector<double> h_fgAllCompartments(size3D, 0.0);

    vector<double> h_dfdtAllCompartments(size3D, 0.0);
    vector<double> h_dfldtAllCompartments(size3D, 0.0);
    vector<double> h_dfgdtAllCompartments(size3D, 0.0);
    
    vector<double> h_externalVolumeBinsAllCompartments(size3D, 0.0);
    vector<double> h_internalVolumeBinsAllCompartments(size3D, 0.0);
    vector<double> h_liquidBinsAllCompartments(size3D, 0.0);
    vector<double> h_gasBinsAllCompartments(size3D, 0.0);
    vector<double> h_totalVolumeBinsAllCompartments(size3D, 0.0);
    
    vector<double> h_internalLiquidAllCompartments(size3D, 0.0);
    vector<double> h_externalLiquidAllCompartments(size3D, 0.0);
    
    vector<double> h_internalVolumeBins(size2D, 0.0);
    vector<double> h_externalVolumeBins(size2D, 0.0);

    lData = liggghtsData::getInstance();
    lData->readLiggghtsDataFiles(coreVal, diaVal);

    vector<double> DEMDiameter = lData->getDEMParticleDiameters();
    if ((DEMDiameter).size() == 0)
    {
        cout << "Diameter data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        return 1;
    }

    vector<double> DEMImpactData = lData->getFinalDEMImpactData();
    if ((DEMImpactData).size() == 0)
    {
        cout << "Impact data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        return 1;
    }

    arrayOfDouble2D DEMCollisionData = lData->getFinalDEMCollisionData();
    if (DEMCollisionData.size() == 0)
    {
        cout << "Collision data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        return 1;
    }
    vector<double> velocity = lData->getFinalDEMImpactVelocity();
    if (velocity.size() == 0)
    {
        cout << "Velocity is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
	    return 1;
    }
    vector<double> colVelocity = lData->getFinalDEMCollisionVelocity();
    if (colVelocity.size() == 0)
    {
        cout << "Velocity is missing in LIGGGHTS collision output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        return 1;
    }

    // moved velocity  based probability calculation to the model from kernel.cpp to reduce computation

    double demTimeStep = pData->demTimeStep;

    copy_double_vector_fromHtoD(x_compartmentDEMIn.velocityCol, colVelocity.data(), size1D);
    

    double inverseDiameterSum = 0.0;
    double inverseMassSum = 0.0;
    int sized = DEMDiameter.size();
    double solDensity = pData->solDensity;
    for (int i = 0; i < sized; i++)
    {
        inverseDiameterSum += (1 / DEMDiameter[i]);
        inverseMassSum += (1 / ((4 / 3) * M_PI * pow((DEMDiameter[i] / 2), 3) * solDensity));
    }

    double coefOfRest = pData->coefOfRest;
    double liqThick = pData->liqThick;
    double surfAsp = pData->surfAsp;
    double bindVisc = pData->bindVisc;
    double sumVelo = 0.0;

    double harmonic_diameter = sized / inverseDiameterSum;
    double harmonic_mass = sized / inverseMassSum;
    double uCritical = (10 + (1 / coefOfRest)) * log((liqThick / surfAsp)) * (3 * M_PI * pow(harmonic_diameter, 2) * bindVisc) / (8 * harmonic_mass);
    // x_compartmentDEMIn.uCriticalCol[0] = uCritical;
    copy_double_vector_fromHtoD(x_compartmentDEMIn.uCriticalCol, &uCritical, 1);
    // cout << "Critical velocity for agg is " << uCritical << endl;

    int veloSize = colVelocity.size();
    for (int i = 0; i < veloSize; i++)
        sumVelo += colVelocity[i];

    unsigned int nDEMBins = pData->nDEMBins;
    double averageVelocity = sumVelo / nDEMBins;
    double stdDevVelocity = 0.0;
    double varianceVelocity = 0.0;

    for (int i = 0; i < veloSize; ++i)
        varianceVelocity += pow((colVelocity[i] - averageVelocity), 2) / nDEMBins;

    stdDevVelocity = sqrt(varianceVelocity);
    //double intVelocity = 0.0;
    vector<double> colProbablityOfVelocity(veloSize, 0.0);
    for (int i = 0; i < veloSize; i++)
    {
        colProbablityOfVelocity[i] = (1 / (colVelocity[i] * sqrt(2 * M_PI) * stdDevVelocity)) * exp(-((log(colVelocity[i]) - averageVelocity) / (2 * pow(varianceVelocity, 2))));
        // cout << "Probability at " << colVelocity[i] << "is " << colProbablityOfVelocity[i] << endl;
    }

    copy_double_vector_fromHtoD(x_compartmentDEMIn.colProbability, colProbablityOfVelocity.data(), size1D);

    // vector<double> impactFrequency = DEMImpactData;
    // for (int s = 0; s < nFirstSolidBins; s++)
    //     for (int ss = 0; ss < nSecondSolidBins; ss++)
    //         for (int i = 0; i < nDEMBins; i++)
    //         {
    //             if (fAll[n2] > 0.0)
    //                 impactFrequency[i] = (DEMImpactData[i] * timeStep) / demTimeStep;
    //         }

    double critStDefNum = pData->critStDefNum;
    double initPorosity = pData->initPorosity;
    // cout << critStDefNum << "\t" << solDensity << "\t" << initPorosity << "\t" << bindVisc << endl;
    double Ubreak = (2 * critStDefNum / solDensity) * (9 / 8.0) * (pow((1 - initPorosity), 2) / pow(initPorosity, 2)) * (9 / 16.0) * (bindVisc / DEMDiameter[0]);
    // x_compartmentDEMIn.ubreak[0] = Ubreak;
    copy_double_vector_fromHtoD(x_compartmentDEMIn.ubreak, &Ubreak, 1);
    // cout << "Critical velocity for breakage is " << Ubreak << endl;
    copy_double_vector_fromHtoD(x_compartmentDEMIn.impVelocity, velocity.data(),size1D);

    int size1 = velocity.size();
    double sum = 0.0;

    for (int i = 0; i < size1; i++)
        sum += velocity[i];

    double averageVelocityBr = sum / nDEMBins;
    double stdDevVelocityBr = 0.0;
    double varianceVelocityBr = 0.0;
    for (int i = 0; i < size1; ++i)
    {
        varianceVelocityBr += pow((velocity[i] - averageVelocityBr), 2) / nDEMBins;
    }

    stdDevVelocityBr = sqrt(varianceVelocityBr);
    //double intVelocity = 0.0;
    // cout << "Std Dev. of Velocity = " << stdDevVelocity << endl;

    vector<double> breakageProbablityOfVelocity(size1, 0.0);
    for (int i = 0; i < size1; i++)
    {
        if (velocity[i] != 0)
        {
            breakageProbablityOfVelocity[i] = (1 / (velocity[i] * sqrt(2 * M_PI) * stdDevVelocityBr)) * exp(-((log(velocity[i]) - averageVelocityBr) / (2 * pow(varianceVelocityBr, 2))));
            // cout << "Probability at " << velocity[i] << "is " << breakageProbablityOfVelocity[i] << endl;
        }
    }

    copy_double_vector_fromHtoD(x_compartmentDEMIn.brProbability, breakageProbablityOfVelocity.data(), size1D);

    DUMP2D(DEMCollisionData);
    DUMP(DEMDiameter);
    DUMP(DEMImpactData);
    DUMP(velocity);

    //Initialize DEM data for compartment
    copy_double_vector_fromHtoD(x_compartmentDEMIn.DEMDiameter, DEMDiameter.data(), size1D);
    copy_double_vector_fromHtoD(x_compartmentDEMIn.DEMCollisionData, (linearize2DVector(DEMCollisionData)).data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentDEMIn.DEMImpactData, DEMImpactData.data(), size1D);

    vector<double> liquidAdditionRateAllCompartments(nCompartments, 0.0);
    double liqSolidRatio = pData->liqSolidRatio;
    double throughput = pData->throughput;
    double liqDensity = pData->liqDensity;
    double liquidAddRate = (liqSolidRatio * throughput) / (liqDensity * 3600);
    liquidAdditionRateAllCompartments[0] = liquidAddRate;
    
    arrayOfDouble2D h_fAllCompartmentsOverTime;
    arrayOfDouble2D h_externalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble2D h_internalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble2D h_liquidBinsAllCompartmentsOverTime;
    arrayOfDouble2D h_gasBinsAllCompartmentsOverTime;

    double granulatorLength = pData->granulatorLength;
    double partticleResTime = pData->partticleResTime;
    double particleAveVelo = granulatorLength /  partticleResTime;
    vector<double> particleAverageVelocity(nCompartments, particleAveVelo);

    
    //Initialize input data for compartment

    copy_double_vector_fromHtoD(x_compartmentIn.vs, h_vs.data(), size1D);
    copy_double_vector_fromHtoD(x_compartmentIn.vss, h_vss.data(), size1D);
    copy_double_vector_fromHtoD(x_compartmentIn.diameter, diameter.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.sMeshXY, h_sMeshXY.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.ssMeshXY, h_ssMeshXY.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.sLow, h_sLow.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.sHigh, h_sHigh.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.ssLow, h_ssLow.data(), size2D);
    copy_double_vector_fromHtoD(x_compartmentIn.ssHigh, h_ssHigh.data(), size2D);
    

    copy_integer_vector_fromHtoD(x_compartmentIn.sAggregationCheck, h_sAggregationCheck.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.ssAggregationCheck, h_ssAggregationCheck.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.sInd, h_sInd.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.ssInd, h_ssInd.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.sCheckB, h_sCheckB.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.ssCheckB, h_ssCheckB.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.sIndB, h_sIndB.data(), size2D);
    copy_integer_vector_fromHtoD(x_compartmentIn.ssIndB, h_ssIndB.data(), size2D);


    vector<int> sieveGrid;
    sieveGrid.push_back(38);
    sieveGrid.push_back(63);
    sieveGrid.push_back(90);
    sieveGrid.push_back(125);
    sieveGrid.push_back(250);
    sieveGrid.push_back(355);
    sieveGrid.push_back(500);
    sieveGrid.push_back(710);
    sieveGrid.push_back(850);
    sieveGrid.push_back(1000);
    sieveGrid.push_back(1400);
    sieveGrid.push_back(2000);
    sieveGrid.push_back(2380);
    sieveGrid.push_back(4000);
    size_t nSieveGrid = sieveGrid.size();

    arrayOfDouble2D d10OverTime;
    arrayOfDouble2D d50OverTime;
    arrayOfDouble2D d90OverTime;

    double time = stod(timeVal); // initial time to start PBM
    double timeStep = 0.5; //1.0e-1;
    vector<double> Time;

    // double lastTime = time;
    int timeIdxCount = 0;
    // int lastTimeIdxCount = 0;

    double premixTime = pData->premixTime;
    double liqAddTime = pData->liqAddTime;
    double postMixTime = pData->postMixTime;
    double finalTime = premixTime + liqAddTime + postMixTime + stod(timeVal);

    vector<double *> formationThroughAggregationOverTime;
    vector<double *> depletionThroughAggregationOverTime;
    vector<double *> formationThroughBreakageOverTime;
    vector<double *> depletionThroughBreakageOverTime;
    cout << "time" << endl;

    // defining compartment varibale pointers

    CompartmentVar compVar(size3D, size5D, 0), d_compVarCpy(size3D, size5D, 1), *d_compVar;
    AggregationCompVar aggCompVar(size3D, size5D, 0), x_aggCompVar(size3D, size5D, 1), *d_aggCompVar;
    BreakageCompVar brCompVar(size3D, size5D, 0), x_brCompVar(size3D, size5D, 1), *d_brCompVar;

    // allocating memory for structures used for compartment calculations

    err = cudaMalloc(&d_compartmentIn, sizeof(CompartmentIn));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : CompartmentIn (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_prevCompInData, sizeof(PreviousCompartmentIn));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : prevCompInData (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_compartmentDEMIn, sizeof(CompartmentDEMIn));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : compartmentDEMIn (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_compVar, sizeof(CompartmentVar));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : CompartmentVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_aggCompVar, sizeof(AggregationCompVar));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : AggregationCompVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_brCompVar, sizeof(BreakageCompVar));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : BreakageCompVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc(&d_compartmentOut, sizeof(CompartmentOut));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to  cudaMalloc : d_compartmentOut (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // copying data to the allocated GPU

    cudaMemcpy(d_compartmentIn, &x_compartmentIn, sizeof(CompartmentIn), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : CompartmentIn (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cudaMemcpy(d_prevCompInData, &x_prevCompInData, sizeof(PreviousCompartmentIn), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : PreviousCompartmentIn (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_compartmentDEMIn, &x_compartmentDEMIn, sizeof(CompartmentDEMIn), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : CompartmentDEMIn (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_compVar, &d_compVarCpy, sizeof(CompartmentVar), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : CompartmentVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    double aggKernelConst = pData->aggKernelConst;
    // x_aggCompVar.aggKernelConst[0] = aggKernelConst;
    copy_double_vector_fromHtoD(x_aggCompVar.aggKernelConst, &aggKernelConst, 1);

    double brkKernelConst = pData->brkKernelConst;
    // x_brCompVar.brkKernelConst[0] = brkKernelConst;
    copy_double_vector_fromHtoD(x_brCompVar.brkKernelConst, &brkKernelConst, 1);

    cudaMemcpy(d_aggCompVar, &x_aggCompVar, sizeof(AggregationCompVar), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : AggregationCompVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_brCompVar, &x_brCompVar, sizeof(BreakageCompVar), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : BreakageCompVar (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemcpy(d_compartmentOut, &x_compartmentOut, sizeof(CompartmentOut), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMemcpy : compartmentOut (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // vector<double> h_formationThroughAggregation(nCompartments, 0.0);
    // vector<double> h_depletionThroughAggregation(nCompartments, 0.0);
    // vector<double> h_formationThroughBreakage(nCompartments, 0.0);
    // vector<double> h_depletionThroughBreakage(nCompartments, 0.0);

    double *d_formationThroughAggregation = device_alloc_double_vector(nCompartments);
    double *d_depletionThroughAggregation = device_alloc_double_vector(nCompartments);
    double *d_formationThroughBreakage = device_alloc_double_vector(nCompartments);
    double *d_depletionThroughBreakage = device_alloc_double_vector(nCompartments);

    double *d_fAllCompartments = device_alloc_double_vector(size3D);
    double *d_flAllCompartments = device_alloc_double_vector(size3D);
    double *d_fgAllCompartments = device_alloc_double_vector(size3D);
    double *d_liquidAdditionRateAllCompartments = device_alloc_double_vector(nCompartments);

    double *d_fIn = device_alloc_double_vector(size2D);

    copy_double_vector_fromHtoD(d_liquidAdditionRateAllCompartments, liquidAdditionRateAllCompartments.data(), nCompartments);
    copy_double_vector_fromHtoD(d_fIn, h_fIn.data(), size2D);



    // dim3 compKernel_nblocks, compKernel_nthreads;
    // compKernel_nblocks = dim3(nCompartments,1,1);
    // compKernel_nthreads = dim3(size2D, size2D,1);
    // int compKernel_nblocks = 16;
    // int compKernel_nthreads = size2D * size2D;
    // cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 1792);
    // double granulatorLength = pData->granulatorLength;
    // double partticleResTime = pData->partticleResTime;
    // double premixTime = pData->premixTime;
    // double liqAddTime = pData->liqAddTime;
    double consConst = pData->consConst;
    double minPorosity = pData->minPorosity;
    double granSatFactor = pData->granSatFactor;
    int threads = size2D;
    double initialTime = stod(timeVal);
    CompartmentOut h_results(size2D, size5D, 1);
    // cudaDeviceSynchronize();
    while (time <= finalTime)
    {
        copy_double_vector_fromHtoD(d_fAllCompartments, h_fAllCompartments.data(), size3D);
        copy_double_vector_fromHtoD(d_flAllCompartments, h_flAllCompartments.data(), size3D);
        copy_double_vector_fromHtoD(d_fgAllCompartments, h_fgAllCompartments.data(), size3D);
        cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 0);
        launchCompartment<<<nCompartments,threads>>>(d_compartmentIn, d_prevCompInData, d_compartmentOut, d_compartmentDEMIn, d_compVar, d_aggCompVar, d_brCompVar,
                                                    time, timeStep, initialTime, d_fAllCompartments, d_flAllCompartments, d_fgAllCompartments, 
                                                    d_liquidAdditionRateAllCompartments, size2D, size3D, size4D, d_fIn, initPorosity, demTimeStep, nFirstSolidBins, nSecondSolidBins,
                                                    granulatorLength, partticleResTime, premixTime, liqAddTime, consConst, minPorosity, nCompartments, granSatFactor, aggKernelConst, brkKernelConst);

        // cudaDeviceSynchronize();
        err = cudaSuccess; // check kernel launach
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch launchCompartment kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        dim3 compKernel_nblocks, compKernel_nthreads;
        cudaDeviceSynchronize();
        
        // vector<int> h_idx4(size5D, 0);
        // int *d_idx4 = device_alloc_integer_vector(size5D);

        // copy_integer_vector_fromHtoD(d_idx4, h_idx4.data(), size5D);

        performAggCalculations<<<nCompartments,threads>>>(d_prevCompInData, d_compartmentIn, d_compartmentDEMIn, d_compartmentOut, d_compVar, d_aggCompVar, time, timeStep, initialTime, demTimeStep, nFirstSolidBins, nSecondSolidBins, nCompartments, aggKernelConst/* , d_idx4 */);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Failed to launch agg kernel (error code %s)!\n", cudaGetErrorString(err));
        }

        // copy_integer_vector_fromDtoH(h_idx4.data(), d_idx4, size5D);
        cudaDeviceSynchronize();
        performBreakageCalculations<<<nCompartments,threads>>>(d_prevCompInData, d_compartmentIn, d_compartmentDEMIn, d_compartmentOut, d_compVar, d_brCompVar, time, timeStep, initialTime, demTimeStep, nFirstSolidBins, nSecondSolidBins, brkKernelConst);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Failed to launch breakage kernel (error code %s)!\n", cudaGetErrorString(err));
        }
        cudaDeviceSynchronize();
        consolidationAndMovementCalcs<<<nCompartments,threads>>>(d_compartmentIn, d_prevCompInData, d_compartmentOut, d_compartmentDEMIn, d_compVar, d_aggCompVar, d_brCompVar, nCompartments, granulatorLength, partticleResTime,
                                time, timeStep, premixTime, liqAddTime, initialTime, nFirstSolidBins, nSecondSolidBins, consConst, minPorosity);
        
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Failed to launch breakage kernel (error code %s)!\n", cudaGetErrorString(err));
        }
        // cout << "Compartment ended " << endl;
        cudaDeviceSynchronize();

        // Copying data strcutres reqd for calculation
        err = cudaMemcpy(&h_results, d_compartmentOut, sizeof(CompartmentOut), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMemcpy : CompartmentOut D to Hmake (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // copy necessary variables back to the CPU
        copy_double_vector_fromDtoH(compartmentOut.dfAlldt, h_results.dfAlldt, size3D);
        copy_double_vector_fromDtoH(compartmentOut.dfLiquiddt, h_results.dfLiquiddt, size3D);
        copy_double_vector_fromDtoH(compartmentOut.dfGasdt, h_results.dfGasdt, size3D);

        copy_double_vector_fromDtoH(compartmentOut.liquidBins, h_results.liquidBins, size3D);
        copy_double_vector_fromDtoH(compartmentOut.gasBins, h_results.gasBins, size3D);


        copy_double_vector_fromDtoH(compartmentOut.formationThroughAggregation, h_results.formationThroughAggregation, size1D);
        copy_double_vector_fromDtoH(compartmentOut.depletionThroughAggregation, h_results.depletionThroughAggregation, size1D);
        copy_double_vector_fromDtoH(compartmentOut.formationThroughBreakage, h_results.formationThroughBreakage, size1D);
        copy_double_vector_fromDtoH(compartmentOut.depletionThroughBreakage, h_results.depletionThroughBreakage, size1D);
        
        // copy_double_vector_fromDtoH(h_fAllCompartments.data(), d_fAllCompartments, size3D);
        // copy_double_vector_fromDtoH(h_flAllCompartments.data(), d_flAllCompartments, size3D);
        // copy_double_vector_fromDtoH(h_fgAllCompartments.data(), d_fgAllCompartments, size3D);

        formationThroughAggregationOverTime.push_back(compartmentOut.formationThroughAggregation); 
        depletionThroughAggregationOverTime.push_back(compartmentOut.depletionThroughAggregation); 
        formationThroughBreakageOverTime.push_back(compartmentOut.formationThroughBreakage); 
        depletionThroughBreakageOverTime.push_back(compartmentOut.depletionThroughBreakage); 

        // for (int w = 0; w < nCompartments; w++)
        // {
        //     cout << "Compartment Number = " << w +1 << endl;
        //     cout << "compartmentOut.formationThroughAggregation  = " << compartmentOut.formationThroughAggregation[w] <<  endl;
        //     cout << "compartmentOut.depletionThroughAggregation  = " << compartmentOut.depletionThroughAggregation[w] <<  endl;
        //     cout <<  "Agg Ratio = " << compartmentOut.formationThroughAggregation[w] / compartmentOut.depletionThroughAggregation[w] << endl;
        //     cout << "compartmentOut.formationThroughBreakage  = " << compartmentOut.formationThroughBreakage[w] <<  endl;
        //     cout << "compartmentOut.depletionThroughBreakage  = " << compartmentOut.depletionThroughBreakage[w] <<  endl;
        //     cout << "Breakage Ratio = " << compartmentOut.formationThroughBreakage[w] / compartmentOut.depletionThroughBreakage[w] << endl;
        // }

        double maxofthree = -DBL_MAX;
        double maxAll = -DBL_MAX;
        double maxLiquid = -DBL_MAX;
        double maxGas = -DBL_MAX;

        for (size_t i = 0; i < size3D; i++)
        {
            // cout << "compartmentOut.dfAlldt[" << i << "] is " << compartmentOut.dfAlldt[i] <<  endl;
            if (fabs(h_fAllCompartments[i]) > 1.0e-16)
                maxAll = max(maxAll, -compartmentOut.dfAlldt[i] / h_fAllCompartments[i]);
            if (fabs(h_flAllCompartments[i]) > 1.0e-16)
                maxLiquid = max(maxLiquid, -compartmentOut.dfLiquiddt[i] / h_flAllCompartments[i]);
            if (fabs(h_fgAllCompartments[i]) > 1.0e-16)
                maxGas = max(maxGas, -compartmentOut.dfGasdt[i] / h_fgAllCompartments[i]);
            maxofthree = max(maxofthree, max(maxAll, max(maxLiquid, maxGas)));
        }

        cout << "maxAll = " << maxAll << endl;
        cout << "maxLiquid = " << maxLiquid << endl;
        cout << "maxGas = " << maxGas << endl;
        cout << "maxofthree = " << maxofthree << endl;

        while (maxofthree < 0.1 / timeStep && timeStep < 0.25)
            timeStep *= 2.0;

        while (maxofthree > 0.1 / timeStep && timeStep > 5.0e-5)
            timeStep /= 2.0;

        int nanCount = 0;
        double minfAll = -DBL_MAX;
        for (size_t i = 0; i < size3D; i++)
        {
            double value = 0.0;
            h_fAllCompartments[i] = h_fAllCompartments[i] + compartmentOut.dfAlldt[i] * timeStep;
            // cout << " h_fAllCompartments[" << i <<"] is   " << h_fAllCompartments[i] << endl;
            if (std::isnan(h_fAllCompartments[i]))
                nanCount++;

            value = h_flAllCompartments[i] + compartmentOut.dfLiquiddt[i] * timeStep;
            h_flAllCompartments[i] = value > 0.0 ? value : 0.0;
            value = h_fgAllCompartments[i] + compartmentOut.dfGasdt[i] * timeStep;
            h_fgAllCompartments[i] = value > 0.0 ? value : 0.0;
        }

        if (nanCount)
        {
            cout << endl << "*****fAllCompartments has " << nanCount << "nan values******" << endl << endl;
            DUMPCSV(h_fAllCompartments);
            exit(EXIT_FAILURE);

        }

        int countnegfAll = 0;

        minfAll = getMinimumOfArray(h_fAllCompartments);
        if (minfAll < -1.0e-16 &&  countnegfAll > 0.1 * nCompartments * nFirstSolidBins * nSecondSolidBins)
        {
            //int mpi_err = 0;
            cout << endl;
            //DUMP3DCSV(dfdtAllCompartments);
            //DUMP3DCSV(fAllCompartments);
            //cout << "My process id = " << mpi_id << endl;
            cout << "minfAll" << minfAll << endl;
            cout << "******fAllCompartments has negative values********" << endl;
            cout << "Number of negative values = " << countnegfAll << endl;
            DUMPCSV(h_fAllCompartments);
            cout << "Aborting..." << endl;
            return 1;
        }

        // BIN recalculation 

        for (int c = 0; c < nCompartments; c++)
        {
            vector<double> liquidBins(size2D, 0.0);
            vector<double> gasBins(size2D, 0.0);
            vector<double> internalLiquid(size2D, 0.0);
            vector<double> externalLiquid(size2D, 0.0);

            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                {
                    int m = c * nFirstSolidBins * nSecondSolidBins + s * nSecondSolidBins + ss;
                    int n2 = s * nSecondSolidBins + ss;
                    if (fabs(h_fAllCompartments[m]) > 1.0e-16)
                    {
                        liquidBins[n2] = h_flAllCompartments[m] / h_fAllCompartments[m];
                        gasBins[n2] = h_fgAllCompartments[m] / h_fAllCompartments[m];
                    }
                    internalLiquid[n2] = min(granSatFactor * gasBins[n2], liquidBins[n2]);
                    externalLiquid[n2] = max(0.0, liquidBins[n2] - internalLiquid[n2]);

                    double value = compartmentIn.sMeshXY[n2] + compartmentIn.ssMeshXY[n2] + gasBins[n2];
                    h_internalVolumeBins[n2] = value + internalLiquid[n2];
                    h_externalVolumeBins[n2] = value + liquidBins[n2];
            
                    h_liquidBinsAllCompartments[m] = liquidBins[n2];
                    h_gasBinsAllCompartments[m] = gasBins[n2];
                    h_externalVolumeBinsAllCompartments[m] = h_externalVolumeBins[n2];
                    h_internalVolumeBinsAllCompartments[m] = h_internalVolumeBins[n2];
                }
        }

        vector<double> d10OverCompartment(nCompartments, 0.0);
        vector<double> d50OverCompartment(nCompartments, 0.0);
        vector<double> d90OverCompartment(nCompartments, 0.0);

        for (int c = 0; c < nCompartments; c++)
        {
            arrayOfDouble2D diameter = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                {
                    int m = c * nFirstSolidBins * nSecondSolidBins + s * nSecondSolidBins + ss;
                    diameter[s][ss] = cbrt((6 / M_PI) * h_externalVolumeBinsAllCompartments[m]) * 1.0e6;
                }


            vector<double> totalVolumeGrid(nSieveGrid, 0.0);
            for (size_t d = 0; d < nSieveGrid - 1; d++)
                for (size_t s = 0; s < nFirstSolidBins; s++)
                    for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                    {
                        int m = c * nFirstSolidBins * nSecondSolidBins + s * nSecondSolidBins + ss;
                        if (diameter[s][ss] < sieveGrid[d + 1] && diameter[s][ss] >= sieveGrid[d])
                            totalVolumeGrid[d] += h_fAllCompartments[m] * h_externalVolumeBinsAllCompartments[m];
                    }

            double sum = 0.0;
            for (size_t d = 0; d < nSieveGrid; d++)
                sum += totalVolumeGrid[d];

            vector<double> volumeDistribution(nSieveGrid, 0.0);
            for (size_t d = 0; d < nSieveGrid; d++)
                if(sum > 1e-16)
                    volumeDistribution[d] = totalVolumeGrid[d] / sum;

            vector<double> cumulativeVolumeDistribution(nSieveGrid, 0.0);
            sum = 0.0;
            for (size_t d = 0; d < nSieveGrid; d++)
            {
                sum += volumeDistribution[d];
                cumulativeVolumeDistribution[d] = sum;
            }
            double d10 = 0.1 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);
            double d50 = 0.5 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);
            double d90 = 0.9 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);

            for (size_t d = 1; d < nSieveGrid; d++)
            {
                double value1 = (sieveGrid[d] - sieveGrid[d - 1]) / (cumulativeVolumeDistribution[d] - cumulativeVolumeDistribution[d - 1]);
                double value2 = sieveGrid[d - 1];
                if (cumulativeVolumeDistribution[d - 1] < 0.5 && cumulativeVolumeDistribution[d] >= 0.5)
                {
                    double value = 0.5 - cumulativeVolumeDistribution[d - 1];
                    d50 = value * value1 + value2;
                }
                if (cumulativeVolumeDistribution[d - 1] < 0.1 && cumulativeVolumeDistribution[d] >= 0.1)
                {
                    double value = 0.1 - cumulativeVolumeDistribution[d - 1];
                    d10 = value * value1 + value2;
                }
                if (cumulativeVolumeDistribution[d - 1] < 0.1 && cumulativeVolumeDistribution[d] >= 0.1)
                {
                    double value = 0.9 - cumulativeVolumeDistribution[d - 1];
                    d90 = value * value1 + value2;
                }
            }
            
            d10OverCompartment[c] = d10;
            d50OverCompartment[c] = d50;
            d10OverCompartment[c] = d90;
        }

        Time.push_back(time);
        d10OverTime.push_back(d10OverCompartment);
        d50OverTime.push_back(d50OverCompartment);
        d90OverTime.push_back(d90OverCompartment);
        //SAVING OVER TIME
        //cout << endl <<  "************Saving over time" << endl << endl;
        h_fAllCompartmentsOverTime.push_back(h_fAllCompartments);
        h_externalVolumeBinsAllCompartmentsOverTime.push_back(h_externalVolumeBinsAllCompartments);
        h_internalVolumeBinsAllCompartmentsOverTime.push_back(h_internalVolumeBinsAllCompartments);
        h_liquidBinsAllCompartmentsOverTime.push_back(h_liquidBinsAllCompartments);
        h_gasBinsAllCompartmentsOverTime.push_back(h_gasBinsAllCompartments);

        cout << "time = " << time << endl;
        cout << "timeStep = " << timeStep << endl;
        cout << endl;
        timeIdxCount++;
        time += timeStep;

        // free_double_vector_device(h_results.dfAlldt);
        // free_double_vector_device(h_results.dfLiquiddt);
        // free_double_vector_device(h_results.dfGasdt);
        // free_double_vector_device(h_results.liquidBins);
        // free_double_vector_device(h_results.gasBins);
        // free_double_vector_device(h_results.formationThroughAggregation);
        // free_double_vector_device(h_results.depletionThroughAggregation);
        // free_double_vector_device(h_results.formationThroughBreakage);
        // free_double_vector_device(h_results.depletionThroughBreakage);
    }

    size_t nTimeSteps = Time.size();
    cout << endl
         << "nTimeSteps = " << nTimeSteps << endl
         << endl;

        //dump values for ratio plots
    dumpDiaCSVpointer(Time, formationThroughAggregationOverTime, Time.size() * nCompartments, string("FormationThroughAggregation"));
    dumpDiaCSVpointer(Time, depletionThroughAggregationOverTime, Time.size() * nCompartments, string("DepletionThroughAggregation"));
    dumpDiaCSVpointer(Time, formationThroughBreakageOverTime, Time.size() * nCompartments, string("FormationThroughBreakage"));
    dumpDiaCSVpointer(Time, depletionThroughBreakageOverTime, Time.size() * nCompartments, string("DepletionThroughBreakage"));

    dumpDiaCSV(Time, d50OverTime, string(("d50OverTime")));
    

    double endTime = static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
    cout << "That took " << endTime - startTime << " seconds" << endl;
    cout << "Code End" << endl;
    return 0;
    // vector<double> h(size4D, 0.0);
    // for (int i = 0; i < size5D; i++)
    // {
    //     cout << "At i = " << i << "  kernel = " << compartmentOut.aggregationKernel[i] << endl;
    // }
    // cudaFree(d_vs);
    // cudaFree(d_vss);
    // cudaFree(d_sMeshXY);
    // cudaFree(d_ssMeshXY);
    // cudaFree(d_compartmentIn);
}