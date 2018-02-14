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

// CompartmentOut performCompartmentCalculations(PreviousCompartmentIn prevCompIn, CompartmentIn compartmentIn, CompartmentDEMIn compartmentDEMIn, double time, double timeStep, double initialTime = 0.0);

__global__ void performAggCalculations(PreviousCompartmentIn *, CompartmentIn *, CompartmentDEMIn *, double, double, double);

#endif // COMPARTMENT_CUH
