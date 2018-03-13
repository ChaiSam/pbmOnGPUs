#ifndef COMPARTMENT_CUH
#define COMPARTMENT_CUH

#include <vector>
#include "utility.cuh"


class CompartmentIn
{
public:
    double *fAll;
    double *fLiquid;
    double *fGas;
    double *liquidAdditionRate;
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

    CompartmentIn(unsigned int, unsigned int, unsigned int);

};

class CompartmentOut
{
public:
    double *dfAlldt;
    double *dfLiquiddt;
    double *dfGasdt;
    double *liquidBins;
    double *gasBins;
    double *internalVolumeBins;
    double *externalVolumeBins;
    double *aggregationKernel;
    double *breakageKernel;
    double *collisionFrequency;
    double *formationThroughAggregation;
    double *depletionThroughAggregation;
    double *formationThroughBreakage;
    double *depletionThroughBreakage;

    CompartmentOut(unsigned int, unsigned int, unsigned int);
};

class CompartmentDEMIn
{
public:
    double *DEMDiameter;
    double *DEMCollisionData;
    double *DEMImpactData;
    double *colProbability;
    double *brProbability;
    double *colEfficiency;
    double *colFrequency;
    double *velocityCol;
    double *impactFrequency;
    double *uCriticalCol;
    double *ubreak;

    CompartmentDEMIn(unsigned int, unsigned int, unsigned int);
};

class PreviousCompartmentIn
{
public:
    double *fAllPreviousCompartment;
    double *flPreviousCompartment;
    double *fgPreviousCompartment;
    double *fAllComingIn;
    double *fgComingIn;

    PreviousCompartmentIn(unsigned int, unsigned int, unsigned int);
};

class CompartmentVar
{
public:
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
    double *meshXYSum;

    CompartmentVar(unsigned int, unsigned int, unsigned int);
};

class AggregationCompVar
{
public:
    double *aggKernelConst;
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
    AggregationCompVar(unsigned int, unsigned int, unsigned int);

};

class BreakageCompVar
{
public:
    double *brkKernelConst;
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
    double *depletionThroughBreakage;
    double *depletionOfGasThroughBreakage;
    double *depletionOfLiquidthroughBreakage;
    BreakageCompVar(unsigned int, unsigned int, unsigned int);
};

__global__ void performAggCalculations(PreviousCompartmentIn *, CompartmentIn *, CompartmentDEMIn *, CompartmentOut *, CompartmentVar *, AggregationCompVar *, double, double, double, double, int, int, int, int, int, int, double);
__global__ void performBreakageCalculations(PreviousCompartmentIn *, CompartmentIn *, CompartmentDEMIn *, CompartmentOut *, CompartmentVar *, BreakageCompVar *, double, double, double, double, int, int, int, int, int, double);
#endif // COMPARTMENT_CUH
