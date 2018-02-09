#ifndef COMPARTMENT_CUH
#define COMPARTMENT_CUH

#include <vector>
#include "utility.cuh"

typedef struct
{
    std::vector<double> fAll;
    std::vector<double> fLiquid;
    std::vector<double> fGas;
    double liquidAdditionRate;

    std::vector<double> vs;
    std::vector<double> vss;

    std::vector<double> sMeshXY;
    std::vector<double> ssMeshXY;

    std::vector<int> sAggregationCheck;
    std::vector<int> ssAggregationCheck;

    std::vector<int> sInd;
    std::vector<int> ssInd;

    std::vector<int> sIndB;
    std::vector<int> ssIndB;

    std::vector<double> sLow;
    std::vector<double> sHigh;

    std::vector<double> ssLow;
    std::vector<double> ssHigh;

    std::vector<int> sCheckB;
    std::vector<int> ssCheckB;

    std::vector<double> diameter;

} CompartmentIn;

typedef struct
{
    std::vector<double> dfAlldt;
    std::vector<double> dfLiquiddt;
    std::vector<double> dfGasdt;
    std::vector<double> liquidBins;
    std::vector<double> gasBins;
    std::vector<double> internalVolumeBins;
    std::vector<double> externalVolumeBins;
    std::vector<double> aggregationKernel;
    std::vector<double> breakageKernel;
    double formationThroughAggregation;
    double depletionThroughAggregation;
    double formationThroughBreakage;
    double depletionThroughBreakage;
} CompartmentOut;

typedef struct
{
    std::vector<double> DEMDiameter;
    std::vector<double> DEMCollisionData;
    std::vector<double> DEMImpactData;
} CompartmentDEMIn;

typedef struct
{
    std::vector<double> fAllPreviousCompartment;
    std::vector<double> flPreviousCompartment;
    std::vector<double> fgPreviousCompartment;
    std::vector<double> fAllComingIn;
    std::vector<double> fgComingIn;
} PreviousCompartmentIn;

CompartmentOut performCompartmentCalculations(PreviousCompartmentIn prevCompIn, CompartmentIn compartmentIn, CompartmentDEMIn compartmentDEMIn, double time, double timeStep, double initialTime = 0.0);

#endif // COMPARTMENT_CUH
