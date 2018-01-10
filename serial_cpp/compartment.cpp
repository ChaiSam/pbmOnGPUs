#include <iostream>
#include <cmath>

#include "parameters.h"
#include "utility.h"
#include "kernel.h"
#include "compartment.h"

using namespace std;

#define DUMP2D(varName) dump2DData(varName, #varName)
#define DUMP3D(varName) dump3DData(varName, #varName)
#define DUMP4D(varName) dump4DData(varName, #varName)

#define DUMP2DCSV(varName) dump2DCSV(varName, #varName)
#define DUMP3DCSV(varName) dump3DCSV(varName, #varName)
#define DUMP4DCSV(varName) dump4DCSV(varName, #varName)

CompartmentOut performCompartmentCalculations(PreviousCompartmentIn prevCompIn, CompartmentIn compartmentIn, CompartmentDEMIn compartmentDEMIn, double time, double timeStep)
{
    CompartmentOut compartmentOut;

    //Declaration of arrays required initially for OMP implementation
    arrayOfDouble2D internalLiquid = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D externalLiquid = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D externalLiquidContent = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D externalVolumeBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D volumeBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble4D aggregationRate = getArrayOfDouble4D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS,
                                                         NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D depletionOfGasThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D depletionOfLiquidThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D firstSolidBirthThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D secondSolidBirthThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D liquidBirthThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D gasBirthThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D firstSolidVolumeThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D secondSolidVolumeThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggLowLow = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighHigh = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggLowHigh = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighLow = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggLowLowLiq = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighHighLiq = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggLowHighLiq = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighLowLiq = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D birthAggLowLowGas = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighHighGas = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggLowHighGas = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthAggHighLowGas = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationThroughAggregationCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationOfLiquidThroughAggregationCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationOfGasThroughAggregationCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble4D breakageRate = getArrayOfDouble4D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS,
                                                      NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D depletionThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D depletionOfGasThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D depletionOfLiquidthroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D birthThroughBreakage1 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D birthThroughBreakage2 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D firstSolidBirthThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D secondSolidBirthThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D liquidBirthThroughBreakage1 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D gasBirthThroughBreakage1 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D liquidBirthThroughBreakage2 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D gasBirthThroughBreakage2 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D firstSolidVolumeThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D secondSolidVolumeThroughBreakage = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D fractionBreakage00 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D fractionBreakage01 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D fractionBreakage10 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D fractionBreakage11 = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationThroughBreakageCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationOfLiquidThroughBreakageCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D formationOfGasThroughBreakageCA = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D transferThroughLiquidAddition = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D transferThroughConsolidation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D particleMovement = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D liquidMovement = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D gasMovement = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //Calculation of liquid and gas bins
    //cout << "Begin liquidBins & gasBins" << endl;
    compartmentOut.liquidBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    compartmentOut.gasBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            if (fabs(compartmentIn.fAll[s][ss]) > EPSILON) //If there are particles in the bin...
            {
                //New liquid bins are calculated as total amount liquid in that size class divided by
                //the number of particle in that size class
                compartmentOut.liquidBins[s][ss] = compartmentIn.fLiquid[s][ss] / compartmentIn.fAll[s][ss];
                // New gas bins are calculated as total amount liquid in that size class divided by
                //the number of particle in that size class
                compartmentOut.gasBins[s][ss] = compartmentIn.fGas[s][ss] / compartmentIn.fAll[s][ss];
            }
            else
            {
                compartmentOut.liquidBins[s][ss] = 0.0; // No particles in the bin -> particles in that size class has no liquid
                compartmentOut.gasBins[s][ss] = 0.0;    // No particles in the bin -> particles in that size class has no gas
            }
        }
    //cout << "End liquidBins & gasBins" << endl;

    //Internal and external liquid demarcation
    //cout << "Begin Internal & External liquid" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            internalLiquid[s][ss] = min(GRANULESATURATIONFACTOR * compartmentOut.gasBins[s][ss], compartmentOut.liquidBins[s][ss]);
            externalLiquid[s][ss] = max(0.0, compartmentOut.liquidBins[s][ss] - internalLiquid[s][ss]);
            externalLiquidContent[s][ss] = externalLiquid[s][ss] / compartmentOut.liquidBins[s][ss];
        }

    //cout << "End Internal & External liquid" << endl;

    //cout << "Begin Internal, External & Volume Bins" << endl;

    compartmentOut.internalVolumeBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            compartmentOut.internalVolumeBins[s][ss] = compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss];
            compartmentOut.internalVolumeBins[s][ss] += internalLiquid[s][ss] + compartmentOut.gasBins[s][ss];

            externalVolumeBins[s][ss] = compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss];
            externalVolumeBins[s][ss] += compartmentOut.liquidBins[s][ss] + compartmentOut.gasBins[s][ss];

            volumeBins[s][ss] = compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss];
        }

    //cout << "End Internal, External & Volume Bins" << endl;

    //cout << "Begin Aggregation Kernel" << endl;
    //    compartmentOut.aggregationKernel = getArrayOfDouble4D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS,
    //                                       NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    //    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
    //        for (int ss1  = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
    //            for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
    //                for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
    //                {
    //                    double expr1 = externalVolumeBins[s1][ss1]+externalVolumeBins[s2][ss2]-externalLiquid[s1][ss1]-externalLiquid[s2][ss2];
    //                    expr1 = pow(expr1, AGGREGATIONKERNELCONSTANTGAMMA);
    //
    //                    double expr2 = (externalLiquid[s1][ss1]/externalVolumeBins[s1][ss1] + externalLiquid[s1][ss1]/externalVolumeBins[s1][ss1])/2.0;
    //                    expr2 = pow(expr2, AGGREGATIONKERNELCONSTANTALPHA);
    //
    //                    double expr3 = 1-(externalLiquid[s1][ss1]/externalVolumeBins[s1][ss1] + externalLiquid[s2][ss2]/externalVolumeBins[s2][ss2])/2.0;
    //                    expr3 = pow(expr3, AGGREGATIONKERNELCONSTANTDELTA);
    //
    //                    compartmentOut.aggregationKernel[s1][ss1][s2][ss2] = AGGREGATIONKERNELCONSTANT*expr1*pow(expr2*expr3, AGGREGATIONKERNELCONSTANTALPHA);
    //                    // compartmentOut.aggregationKernel[s1][ss1][s2][ss2] = DEMAGGREGATIONKERNELCONST*DEMAGGREGATIONKERNELVALUE;
    //                }
    compartmentOut.aggregationKernel = DEMDependentAggregationKernel(compartmentIn, compartmentDEMIn, externalLiquidContent, timeStep);

    //cout << "End Aggregation Kernel" << endl;

    //Breakage kernel Calculation inside time loop becasue gas and liquid bins change with time
    //cout << "Begin Breakage Kernel" << endl;
    //compartmentOut.breakageKernel = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);//, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    //compartmentOut.breakageKernel = getArrayOfDouble4D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    //    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
    //        for (int ss  = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
    //        {
    //            double expr1 = sqrt(4/(15*M_PI));
    //            double expr2 = pow(SHEARRATE, 2)* cbrt(3*externalVolumeBins[s][ss]/(4*M_PI));
    //            compartmentOut.breakageKernel[s][ss] = expr1*SHEARRATE*exp(-BREAKAGEKERNELCONSTANT/expr2);
    //            //compartmentOut.breakageKernel[s][ss] = DEMBREAKAGEKERNELCONST * DEMBREAKAGEKERNELVALUE;
    //        }
    //    //reset to zero values
    //    compartmentOut.breakageKernel = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    compartmentOut.breakageKernel = DEMDependentBreakageKernel(compartmentIn, compartmentDEMIn, timeStep);

    //cout << "End Breakage Kernel" << endl;

    //AGGREGATION
    //cout << "Begin aggregationRate" << endl;

    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
            for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
                for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
                    aggregationRate[s1][ss1][s2][ss2] = compartmentIn.sAggregationCheck[s1][ss1] * compartmentIn.ssAggregationCheck[s2][ss2] * compartmentOut.aggregationKernel[s1][ss1][s2][ss2] * compartmentIn.fAll[s1][ss1] * compartmentIn.fAll[s2][ss2];

    //cout << "End aggregationRate" << endl;

    //cout << "Begin depletionThroughAggregation" << endl;
    arrayOfDouble2D depletionThroughAggregation = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
            for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
                for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
                    for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
                        for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
                        {
                            if (s == s1 && ss == ss1)
                                depletionThroughAggregation[s][ss] += aggregationRate[s1][ss1][s2][ss2];
                            if (s == s2 && ss == ss2)
                                depletionThroughAggregation[s][ss] += aggregationRate[s1][ss1][s2][ss2];
                        }

    //cout << "End depletionThroughAggregation" << endl;

    //cout << "Begin depletionOfGasThroughAggregation & depletionOfLiquidThroughAggregation" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            depletionOfGasThroughAggregation[s][ss] = depletionThroughAggregation[s][ss] * compartmentOut.gasBins[s][ss];
            depletionOfLiquidThroughAggregation[s][ss] = depletionThroughAggregation[s][ss] * compartmentOut.liquidBins[s][ss];
        }
    //cout << "End depletionOfGasThroughAggregation & depletionOfLiquidThroughAggregation" << endl << endl;

    //cout << "Begin birthThroughAggregation, firstSolidBirthThroughAggregation, secondSolidBirthThroughAggregation, ";
    //cout << "liquidBirthThroughAggregation & gasBirthThroughAggregation" << endl;

    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
            for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
                for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
                    for (int a = 0; a < NUMBEROFFIRSTSOLIDBINS; a++)
                        for (int b = 0; b < NUMBEROFSECONDSOLIDBINS; b++)
                        {
                            if (compartmentIn.sInd[s1][s2] == (a + 1) && compartmentIn.ssInd[ss1][ss2] == (b + 1))
                            {
                                birthThroughAggregation[a][b] += aggregationRate[s1][ss1][s2][ss2];
                                firstSolidBirthThroughAggregation[a][b] += (compartmentIn.vs[s1] + compartmentIn.vs[s2]) * aggregationRate[s1][ss1][s2][ss2];
                                secondSolidBirthThroughAggregation[a][b] += (compartmentIn.vss[ss1] + compartmentIn.vss[ss2]) * aggregationRate[s1][ss1][s2][ss2];
                                liquidBirthThroughAggregation[a][b] += (compartmentOut.liquidBins[s1][ss1] + compartmentOut.liquidBins[s2][ss2]) * aggregationRate[s1][ss1][s2][ss2];
                                gasBirthThroughAggregation[a][b] += (compartmentOut.gasBins[s1][ss1] + compartmentOut.gasBins[s2][ss2]) * aggregationRate[s1][ss1][s2][ss2];
                            }
                        }
    //cout << "End birthThroughAggregation, firstSolidBirthThroughAggregation, secondSolidBirthThroughAggregation, ";
    //cout << "liquidBirthThroughAggregation & gasBirthThroughAggregation" << endl << endl;

    //cout << "Begin firstSolidVolumeThroughAggregation & secondSolidVolumeThroughAggregation" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            if (fabs(birthThroughAggregation[s][ss]) > EPSILON)
            {
                firstSolidVolumeThroughAggregation[s][ss] = firstSolidBirthThroughAggregation[s][ss] / birthThroughAggregation[s][ss];
                secondSolidVolumeThroughAggregation[s][ss] = secondSolidBirthThroughAggregation[s][ss] / birthThroughAggregation[s][ss];
            }
            else
            {
                firstSolidVolumeThroughAggregation[s][ss] = 0.0;
                secondSolidVolumeThroughAggregation[s][ss] = 0.0;
            }
        }
    //cout << "End firstSolidVolumeThroughAggregation & secondSolidVolumeThroughAggregation" << endl << endl;

    //cout << "Begin birth_agg_low_low, birth_agg_high_high, birth_agg_low_high & birth_agg_high_low" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
        {
            birthAggLowLow[s][ss] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowLow[s][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowLow[s][ss] *= birthThroughAggregation[s][ss];

            birthAggHighHigh[s + 1][ss + 1] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighHigh[s + 1][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighHigh[s + 1][ss + 1] *= birthThroughAggregation[s][ss];

            birthAggLowHigh[s][ss + 1] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowHigh[s][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowHigh[s][ss + 1] *= birthThroughAggregation[s][ss];

            birthAggHighLow[s + 1][ss] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighLow[s + 1][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighLow[s + 1][ss] *= birthThroughAggregation[s][ss];
        }

    //cout << "End birth_agg_low_low, birth_agg_high_high, birth_agg_low_high & birth_agg_high_low" << endl << endl;

    //cout << "Begin birth_agg_low_low_liq, birth_agg_high_high_liq, birth_agg_low_high_liq & birth_agg_high_low_liq" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
        {
            birthAggLowLowLiq[s][ss] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowLowLiq[s][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowLowLiq[s][ss] *= liquidBirthThroughAggregation[s][ss];

            birthAggHighHighLiq[s + 1][ss + 1] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighHighLiq[s + 1][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighHighLiq[s + 1][ss + 1] *= liquidBirthThroughAggregation[s][ss];

            birthAggLowHighLiq[s][ss + 1] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowHighLiq[s][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowHighLiq[s][ss + 1] *= liquidBirthThroughAggregation[s][ss];

            birthAggHighLowLiq[s + 1][ss] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighLowLiq[s + 1][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighLowLiq[s + 1][ss] *= liquidBirthThroughAggregation[s][ss];
        }
    //cout << "End birth_agg_low_low_liq, birth_agg_high_high_liq, birth_agg_low_high_liq & birth_agg_high_low_liq" << endl << endl;

    //cout << "Begin birth_agg_low_low_gas, birth_agg_high_high_gas, birth_agg_low_high_gas & birth_agg_high_low_gas" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
        {
            birthAggLowLowGas[s][ss] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowLowGas[s][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowLowGas[s][ss] *= gasBirthThroughAggregation[s][ss];

            birthAggHighHighGas[s + 1][ss + 1] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighHighGas[s + 1][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighHighGas[s + 1][ss + 1] *= gasBirthThroughAggregation[s][ss];

            birthAggLowHighGas[s][ss + 1] = (compartmentIn.vs[s + 1] - firstSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggLowHighGas[s][ss + 1] *= (secondSolidVolumeThroughAggregation[s][ss] - compartmentIn.vss[ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggLowHighGas[s][ss + 1] *= gasBirthThroughAggregation[s][ss];

            birthAggHighLowGas[s + 1][ss] = (firstSolidVolumeThroughAggregation[s][ss] - compartmentIn.vs[s]) / (compartmentIn.vs[s + 1] - compartmentIn.vs[s]);
            birthAggHighLowGas[s + 1][ss] *= (compartmentIn.vss[ss + 1] - secondSolidVolumeThroughAggregation[s][ss]) / (compartmentIn.vss[ss + 1] - compartmentIn.vss[ss]);
            birthAggHighLowGas[s + 1][ss] *= gasBirthThroughAggregation[s][ss];
        }
    //cout << "End birth_agg_low_low_gas, birth_agg_high_high_gas, birth_agg_low_high_gas & birth_agg_high_low_gas" << endl << endl;

    //cout << "Begin formationThroughAggregationCA, formationOfLiquidThroughAggregationCA & formationOfGasThroughAggregationCA" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            formationThroughAggregationCA[s][ss] = birthAggLowLow[s][ss] + birthAggHighHigh[s][ss] + birthAggLowHigh[s][ss] + birthAggHighLow[s][ss];
            formationOfLiquidThroughAggregationCA[s][ss] = birthAggLowLowLiq[s][ss] + birthAggHighHighLiq[s][ss] + birthAggLowHighLiq[s][ss] + birthAggHighLowLiq[s][ss];
            formationOfGasThroughAggregationCA[s][ss] = birthAggLowLowGas[s][ss] + birthAggHighHighGas[s][ss] + birthAggLowHighGas[s][ss] + birthAggHighLowGas[s][ss];
        }
    //cout << "End formationThroughAggregationCA, formationOfLiquidThroughAggregationCA & formationOfGasThroughAggregationCA" << endl << endl;

    //cout << "************End of Aggregation**************" << endl << endl;

    //cout << "Begin breakageRate, depletionThroughBreakage, depletionOfLiquidthroughBreakage & depletionOfGasThroughBreakage" << endl;

    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
            for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
                for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
                {
                    breakageRate[s1][ss1][s2][ss2] = compartmentIn.sCheckB[s1][s2] * compartmentIn.ssCheckB[ss1][ss2];
                    //[s1][ss1][s2][ss2] *= compartmentOut.breakageKernel[s1][ss1] * compartmentIn.fAll[s1][ss1];
                    breakageRate[s1][ss1][s2][ss2] *= compartmentOut.breakageKernel[s1][ss1][s2][ss2] * compartmentIn.fAll[s1][ss1];

                    depletionThroughBreakage[s1][ss1] += breakageRate[s1][ss1][s2][ss2];
                    depletionOfLiquidthroughBreakage[s1][ss1] = depletionThroughBreakage[s1][ss1] * compartmentOut.liquidBins[s1][ss1];
                    depletionOfGasThroughBreakage[s1][ss1] = depletionThroughBreakage[s1][ss1] * compartmentOut.gasBins[s1][ss1];

                    birthThroughBreakage1[s1][ss1] += breakageRate[s1][ss1][s2][ss2];
                }
    //cout << "End breakageRate, depletionThroughBreakage, depletionOfLiquidthroughBreakage & depletionOfGasThroughBreakage" << endl << endl;

    //cout << "Begin birthThroughBreakage2, firstSolidBirthThroughBreakage, secondSolidBirthThroughBreakage, ";
    //cout << "liquidBirthThroughBreakage1, gasBirthThroughBreakage1, liquidBirthThroughBreakage2, ";
    //cout << "gasBirthThroughBreakage2, firstSolidVolumeThroughBreakage & secondSolidVolumeThroughBreakage" << endl;

    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
            for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
                for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
                {
                    for (int a = 0; a < NUMBEROFFIRSTSOLIDBINS; a++)
                        for (int b = 0; b < NUMBEROFSECONDSOLIDBINS; b++)
                        {
                            if (compartmentIn.sIndB[s1][s2] == (a + 1) && compartmentIn.ssIndB[ss1][ss2] == (b + 1))
                            {
                                birthThroughBreakage2[a][b] += breakageRate[s1][ss1][s2][ss2];

                                firstSolidBirthThroughBreakage[a][b] += (compartmentIn.vs[s1] - compartmentIn.vs[s2]) * breakageRate[s1][ss1][s2][ss2];
                                secondSolidBirthThroughBreakage[a][b] += (compartmentIn.vss[ss1] - compartmentIn.vss[ss2]) * breakageRate[s1][ss1][s2][ss2];

                                liquidBirthThroughBreakage2[a][b] += (compartmentOut.liquidBins[s1][ss1] * (1 - (volumeBins[s2][ss2] / volumeBins[s1][ss1]))) * breakageRate[s1][ss1][s2][ss2];
                                gasBirthThroughBreakage2[a][b] += (compartmentOut.gasBins[s1][ss1] * (1 - (volumeBins[s2][ss2] / volumeBins[s1][ss1]))) * breakageRate[s1][ss1][s2][ss2];
                                if (fabs(birthThroughBreakage2[a][b]) > 1e-16)
                                {
                                    firstSolidVolumeThroughBreakage[a][b] = firstSolidBirthThroughBreakage[a][b] / birthThroughBreakage2[a][b];
                                    secondSolidVolumeThroughBreakage[a][b] = secondSolidBirthThroughBreakage[a][b] / birthThroughBreakage2[a][b];
                                }
                            }
                        }
                    liquidBirthThroughBreakage1[s2][ss2] += (compartmentOut.liquidBins[s1][ss1] * (volumeBins[s2][ss2] / volumeBins[s1][ss1])) * breakageRate[s1][ss1][s2][ss2];
                    gasBirthThroughBreakage1[s2][ss2] += (compartmentOut.gasBins[s1][ss1] * (volumeBins[s2][ss2] / volumeBins[s1][ss1])) * breakageRate[s1][ss1][s2][ss2];
                }
    //cout << "End birthThroughBreakage2, firstSolidBirthThroughBreakage, secondSolidBirthThroughBreakage, ";
    //cout << "liquidBirthThroughBreakage1, gasBirthThroughBreakage1, liquidBirthThroughBreakage2, ";
    //cout << "gasBirthThroughBreakage2, firstSolidVolumeThroughBreakage & secondSolidVolumeThroughBreakage" << endl << endl;

    //cout << "Begin fractionBreakage00, fractionBreakage01, fractionBreakage10 & fractionBreakage11" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
    {
        double value1 = 0.0;
        double value2 = 0.0;
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            value1 = fabs(compartmentIn.sLow[s][ss] - firstSolidVolumeThroughBreakage[s][ss]);
            value1 = compartmentIn.sHigh[s][ss] - compartmentIn.sLow[s][ss] - value1;
            value1 /= (compartmentIn.sHigh[s][ss] - compartmentIn.sLow[s][ss]);
            value2 = fabs(compartmentIn.ssLow[s][ss] - secondSolidVolumeThroughBreakage[s][ss]);
            value2 = compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss] - value2;
            value2 /= (compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss]);
            fractionBreakage00[s][ss] = value1 / value2;

            value2 = fabs(compartmentIn.ssHigh[s][ss] - secondSolidVolumeThroughBreakage[s][ss]);
            value2 = compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss] - value2;
            value2 /= (compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss]);
            fractionBreakage01[s][ss] = value1 / value2;

            value1 = fabs(compartmentIn.sHigh[s][ss] - firstSolidVolumeThroughBreakage[s][ss]);
            value1 = compartmentIn.sHigh[s][ss] - compartmentIn.sLow[s][ss] - value1;
            value1 /= (compartmentIn.sHigh[s][ss] - compartmentIn.sLow[s][ss]);
            fractionBreakage11[s][ss] = value1 / value2;

            value2 = fabs(compartmentIn.ssLow[s][ss] - secondSolidVolumeThroughBreakage[s][ss]);
            value2 = compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss] - value2;
            value2 /= (compartmentIn.ssHigh[s][ss] - compartmentIn.ssLow[s][ss]);
            fractionBreakage10[s][ss] = value1 / value2;
        }
    }

    //cout << "End fractionBreakage00, fractionBreakage01, fractionBreakage10 & fractionBreakage11" << endl << endl;

    //cout << "Begin formationThroughBreakageCA, formationOfLiquidThroughBreakageCA & formationOfGasThroughBreakageCA" << endl;

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
        {
            formationThroughBreakageCA[s][ss] += birthThroughBreakage2[s][ss] * fractionBreakage00[s][ss];
            formationThroughBreakageCA[s][ss + 1] += birthThroughBreakage2[s][ss] * fractionBreakage01[s][ss];
            formationThroughBreakageCA[s + 1][ss] += birthThroughBreakage2[s][ss] * fractionBreakage10[s][ss];
            formationThroughBreakageCA[s + 1][ss + 1] += birthThroughBreakage2[s][ss] * fractionBreakage11[s][ss];

            formationOfLiquidThroughBreakageCA[s][ss] += liquidBirthThroughBreakage2[s][ss] * fractionBreakage00[s][ss];
            formationOfLiquidThroughBreakageCA[s][ss + 1] += liquidBirthThroughBreakage2[s][ss] * fractionBreakage01[s][ss];
            formationOfLiquidThroughBreakageCA[s + 1][ss] += liquidBirthThroughBreakage2[s][ss] * fractionBreakage10[s][ss];
            formationOfLiquidThroughBreakageCA[s + 1][ss + 1] += liquidBirthThroughBreakage2[s][ss] * fractionBreakage11[s][ss];

            formationOfGasThroughBreakageCA[s][ss] += gasBirthThroughBreakage2[s][ss] * fractionBreakage00[s][ss];
            formationOfGasThroughBreakageCA[s][ss + 1] += gasBirthThroughBreakage2[s][ss] * fractionBreakage01[s][ss];
            formationOfGasThroughBreakageCA[s + 1][ss] += gasBirthThroughBreakage2[s][ss] * fractionBreakage10[s][ss];
            formationOfGasThroughBreakageCA[s + 1][ss + 1] += gasBirthThroughBreakage2[s][ss] * fractionBreakage11[s][ss];
        }
    //cout << "End formationThroughBreakageCA, formationOfLiquidThroughBreakageCA & formationOfGasThroughBreakageCA" << endl << endl;

    //cout << "**************End of Breakage****************" << endl << endl;

    //cout << "Begin Particle Transfer" << endl;
    //cout << "Finding Max of s_meshxy+ss_meshxy " << endl;
    arrayOfDouble2D meshXYSum = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
            meshXYSum[s][ss] = compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss];

    double maxMeshXY = getMaximumOf2DArray(meshXYSum);
    double value = PARTICLEAVERAGEVELOCITY * timeStep / DISTANCEBETWEENCOMPARTMENTS;
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
        {
            double valueMeshXY = 1 - (compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss]) / maxMeshXY;

            particleMovement[s][ss] = prevCompIn.fAllComingIn[s][ss];
            particleMovement[s][ss] += prevCompIn.fAllPreviousCompartment[s][ss] * value * valueMeshXY;
            particleMovement[s][ss] -= compartmentIn.fAll[s][ss] * value;

            liquidMovement[s][ss] = prevCompIn.flPreviousCompartment[s][ss] * value * valueMeshXY;
            liquidMovement[s][ss] -= compartmentIn.fLiquid[s][ss] * value;

            gasMovement[s][ss] = prevCompIn.fgComingIn[s][ss];
            gasMovement[s][ss] += prevCompIn.fgPreviousCompartment[s][ss] * value * valueMeshXY;
            gasMovement[s][ss] -= compartmentIn.fGas[s][ss] * value;
        }
    //cout << "End Particle Transfer" << endl;

    //cout << "Begin rate calculations" << endl << endl;
    if (time >= PREMIXINGTIME && time <= PREMIXINGTIME + LIQUIDADDITIONTIME)
        compartmentIn.liquidAdditionRate *= timeStep;
    else
        compartmentIn.liquidAdditionRate = 0.0;

    double totalSolidVolume = 0.0;
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
            totalSolidVolume += compartmentIn.fAll[s][ss] * (compartmentIn.vs[s] + compartmentIn.vss[ss]);

    compartmentOut.dfAlldt = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    compartmentOut.dfLiquiddt = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    compartmentOut.dfGasdt = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS - 1; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS - 1; ss++)
        {
            compartmentOut.dfAlldt[s][ss] = particleMovement[s][ss];
            compartmentOut.dfAlldt[s][ss] += formationThroughAggregationCA[s][ss] - depletionThroughAggregation[s][ss];
            compartmentOut.dfAlldt[s][ss] += birthThroughBreakage1[s][ss] + formationThroughBreakageCA[s][ss] - depletionThroughBreakage[s][ss];

            if (totalSolidVolume > EPSILON)
            {
                double value = (compartmentIn.vs[s] + compartmentIn.vss[ss]) / totalSolidVolume;
                transferThroughLiquidAddition[s][ss] = compartmentIn.liquidAdditionRate * value;
            }

            compartmentOut.dfLiquiddt[s][ss] = liquidMovement[s][ss];
            compartmentOut.dfLiquiddt[s][ss] += compartmentIn.fAll[s][ss] * transferThroughLiquidAddition[s][ss];
            compartmentOut.dfLiquiddt[s][ss] += formationOfLiquidThroughAggregationCA[s][ss] - depletionOfLiquidThroughAggregation[s][ss];
            compartmentOut.dfLiquiddt[s][ss] += liquidBirthThroughBreakage1[s][ss] + formationOfLiquidThroughBreakageCA[s][ss];
            compartmentOut.dfLiquiddt[s][ss] -= depletionOfLiquidthroughBreakage[s][ss];

            compartmentOut.dfGasdt[s][ss] = gasMovement[s][ss];
            compartmentOut.dfGasdt[s][ss] += compartmentIn.fAll[s][ss] * transferThroughConsolidation[s][ss];
            compartmentOut.dfGasdt[s][ss] += formationOfGasThroughAggregationCA[s][ss] - depletionOfGasThroughAggregation[s][ss];
            compartmentOut.dfGasdt[s][ss] += gasBirthThroughBreakage1[s][ss] + formationOfGasThroughBreakageCA[s][ss];
            compartmentOut.dfGasdt[s][ss] -= depletionOfGasThroughBreakage[s][ss];
        }

    //cout << "**************End of Rate Calculations****************" << endl << endl;

    //cout << "Return to Model code" << endl << endl;

    return compartmentOut;
}
