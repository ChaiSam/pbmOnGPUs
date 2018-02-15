#ifndef COMPARTMENT_CUH
#define COMPARTMENT_CUH

#include <vector>
#include "utility.cuh"

typedef struct
{
    double *fAll;
    double *fLiquid;
    double *fGas;
    double liquidAdditionRate;

    double *vs;
    double *vss;

    double *sMeshXY;
    double *ssMeshXY;

    int *sAggregationCheck;
    int *ssAggregationCheck;

    int *sInd;
    int *ssInd;

    int *sIndB;
    int *ssIndB;

    double *sLow;
    double *sHigh;

    double *ssLow;
    double *ssHigh;

    int *sCheckB;
    int *ssCheckB;

    double *diameter;

} CompartmentIn;

typedef struct
{
    double *dfAlldt;
    double *dfLiquiddt;
    double *dfGasdt;
    double *liquidBins;
    double *gasBins;
    double *internalVolumeBins;
    double *externalVolumeBins;
    double *aggregationKernel;
    double *breakageKernel;
    double formationThroughAggregation;
    double depletionThroughAggregation;
    double formationThroughBreakage;
    double depletionThroughBreakage;
} CompartmentOut;

typedef struct
{
    double *DEMDiameter;
    double *DEMCollisionData;
    double *DEMImpactData;
} CompartmentDEMIn;

typedef struct
{
    double *fAllPreviousCompartment;
    double *flPreviousCompartment;
    double *fgPreviousCompartment;
    double *fAllComingIn;
    double *fgComingIn;
} PreviousCompartmentIn;

typedef struct
{
    double *internalLiquid;
    double *externalLiquid;
    double *externalLiquidContent;
    double *volumeBins;
    double *aggregationRate;
    double *breakageRate;
    double *particleMovement;
    double *liquidMovement;
    double *gasMovement;
    double *liquidBins;
    double *gasBins;
} CompartmentVar;


typedef struct
{
    double *depletionOfGasThroughAggregation;
    double *depletionOfLiquidThroughAggregation;
    double *birthThroughAggregation;
    double *firstSolidBirthThroughAggregation;
    double *secondSolidBirthThroughAggregation;
    double *liquidBirthThroughAggregation;
    double *gasBirthThroughAggregation;
    double *firstSolidVolumeThroughAggregation;
    double *secondSolidVolumeThroughAggregation;
    double *birthAggLowLow;
    double *birthAggHighHigh;
    double *birthAggLowHigh;
    double *birthAggHighLow;
    double *birthAggLowLowLiq;
    double *birthAggHighHighLiq;
    double *birthAggLowHighLiq;
    double *birthAggHighLowLiq;
    double *birthAggLowLowGas;
    double *birthAggHighHighGas;
    double *birthAggLowHighGas;
    double *birthAggHighLowGas;
    double *formationThroughAggregationCA;
    double *formationOfLiquidThroughAggregationCA;
    double *formationOfGasThroughAggregationCA;
    double *depletionThroughAggregation;
    double *depletionThroughBreakage;
    double *depletionOfGasThroughBreakage;
    double *depletionOfLiquidthroughBreakage;

} AggregationCompVar;

typedef struct
{
    double *birthThroughBreakage1;
    double *birthThroughBreakage2;
    double *firstSolidBirthThroughBreakage;
    double *secondSolidBirthThroughBreakage;
    double *liquidBirthThroughBreakage1;
    double *gasBirthThroughBreakage1;
    double *liquidBirthThroughBreakage2;
    double *gasBirthThroughBreakage2;
    double *firstSolidVolumeThroughBreakage;
    double *secondSolidVolumeThroughBreakage;
    double *fractionBreakage00;
    double *fractionBreakage01;
    double *fractionBreakage10;
    double *fractionBreakage11;
    double *formationThroughBreakageCA;
    double *formationOfLiquidThroughBreakageCA;
    double *formationOfGasThroughBreakageCA;
    double *transferThroughLiquidAddition;
    double *transferThroughConsolidation;


} BreakageCompVar;

// CompartmentOut performCompartmentCalculations(PreviousCompartmentIn prevCompIn, CompartmentIn compartmentIn, CompartmentDEMIn compartmentDEMIn, double time, double timeStep, double initialTime = 0.0);

__global__ void performAggCalculations(PreviousCompartmentIn *, CompartmentIn *, CompartmentDEMIn *, CompartmentOut *, CompartmentVar *, AggregationCompVar *, double, double, double);

#endif // COMPARTMENT_CUH
