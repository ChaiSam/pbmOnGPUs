#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <float.h>
#include <string>
#include <random>
#include <ctime>

#include "parameters.h"
#include "utility.h"
#include "compartment.h"
#include "liggghtsData.h"

using namespace std;

//MACROS
//Call to these macros will dump values to file name as variableName.txt
#define DUMP(varName) dumpData(varName, #varName)
#define DUMP2D(varName) dump2DData(varName, #varName)
#define DUMP3D(varName) dump3DData(varName, #varName)

#define DUMPCSV(varName) dumpCSV(varName, #varName)
#define DUMP2DCSV(varName) dump2DCSV(varName, #varName)
#define DUMP3DCSV(varName) dump3DCSV(varName, #varName)

#define DUMPDIACSV(time, dia) dumpDiaCSV(time, dia, #dia)

#define DUMP2DCSV4MATLAB(varName) dump2DCSV4Matlab(varName, #varName)

int main()
{
    //cout << "CodeBegins" << endl << endl;
    double startTime = static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
    //Read Dump Atom Files
    liggghtsData *lData = liggghtsData::getInstance();
    lData->readLiggghtsDataFiles();

    //return 0;//to stop execution after reading liggghts data

    CompartmentIn compartmentIn;          //Input for compartment call
    PreviousCompartmentIn prevCompInData; //Input data from previous compartment; 2nd compartment onwards
    CompartmentDEMIn compartmentDEMIn;
    CompartmentOut compartmentOut; //Output for compartment call

    compartmentIn.vs = array<double, NUMBEROFFIRSTSOLIDBINS>{{0.0}};
    compartmentIn.vss = array<double, NUMBEROFSECONDSOLIDBINS>{{0.0}};

    //Bin initialization
    //cout << "Begin assign value to vsArray" << endl;
    int s = 1;
    for (auto &vs : compartmentIn.vs)
    {
        vs = SCOEF * pow(SBASE, s - 1); // m^3
        s++;
    }

    //cout << "End assign value to vsArray" << endl << endl;

    ////cout << "Begin assign value to vssArray" << endl;
    s = 1;
    for (auto &vss : compartmentIn.vss)
    {
        vss = SSCOEF * pow(SSBASE, s - 1); // m^3
        s++;
    }

    //cout << "End assign value to vssArray" << endl << endl;

    compartmentIn.diameter = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
        for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
            compartmentIn.diameter[s][ss] = cbrt((6 / M_PI) * (compartmentIn.vs[s] + compartmentIn.vss[ss]));

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

    //cout << "Creating fIn and assigning zeros" << endl << endl;
    arrayOfDouble2D fIn = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    cout << "Begin initialize hard coded values to fIn" << endl;
    // for (size_t i = 0; i < particleIn.size(); i++)
    //    for (size_t j = 0; j < particleIn.size(); j++)
    //        fIn[i][j] = sqrt(particleIn[i] * particleIn[j]);
    for (size_t i = 0; i < particleIn.size(); i++)
        fIn[i][i] = particleIn[i];
    //cout << "End initialize hard coded values to fIn" << endl << endl;

    //MATLAB ndgrid
    //cout << "Creating sMeshXY & ssMeshXY and assigning zeros" << endl;
    compartmentIn.sMeshXY = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    compartmentIn.ssMeshXY = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sMeshXY & ssMeshXY (MATLAB ndgrid)" << endl;
    for (size_t i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (size_t j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
        {
            compartmentIn.sMeshXY[i][j] = compartmentIn.vs[i];
            compartmentIn.ssMeshXY[i][j] = compartmentIn.vss[j];
        }

    //cout << "End initializing values to sMeshXY & ssMeshXY (MATLAB ndgrid)" << endl << endl;

    //bsxfun @plus for sAgg & ssAgg
    //cout << "Creating sAgg & ssAgg and assigning zeros" << endl;
    arrayOfDouble2D sAgg = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    arrayOfDouble2D ssAgg = getArrayOfDouble2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sAgg (MATLAB bsxfun @plus)" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
            sAgg[i][j] = compartmentIn.vs[j] + compartmentIn.vs[i];
    //cout << "End initializing values to sAgg (MATLAB bsxfun @plus)" << endl << endl;

    //cout << "Begin initializing values to ssAgg (MATLAB bsxfun @plus)" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
            ssAgg[i][j] = compartmentIn.vss[j] + compartmentIn.vss[i];
    //cout << "End initializing values to ssAgg (MATLAB bsxfun @plus)" << endl << endl;

    //repmat for  sAggregationCheck & ssAggregationCheck
    //cout << "Creating sAggregationCheck & ssAggregationCheck and assigning zeros" << endl;
    compartmentIn.sAggregationCheck = getArrayOfInt2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    compartmentIn.ssAggregationCheck = getArrayOfInt2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sAggregationCheck" << endl;
    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
            compartmentIn.sAggregationCheck[s1][s2] = sAgg[s1][s2] <= compartmentIn.vs[NUMBEROFFIRSTSOLIDBINS - 1] ? 1 : 0;
    //cout << "End initializing values to sAggregationCheck" << endl;

    //cout << "Begin initializing values to ssAggregationCheck" << endl;
    for (int ss1 = 0; ss1 < NUMBEROFSECONDSOLIDBINS; ss1++)
        for (int ss2 = 0; ss2 < NUMBEROFSECONDSOLIDBINS; ss2++)
            compartmentIn.ssAggregationCheck[ss1][ss2] = ssAgg[ss1][ss2] <= compartmentIn.vss[NUMBEROFSECONDSOLIDBINS - 1] ? 1 : 0;
    //cout << "End initializing values to sAggregationCheck" << endl;

    // end of repmat for  sAggregationCheck & ssAggregationCheck

    compartmentIn.sLow = compartmentIn.sMeshXY;
    compartmentIn.sHigh = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS - 1; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
            compartmentIn.sHigh[i][j] = compartmentIn.sMeshXY[i + 1][j];
    for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
        compartmentIn.sHigh[NUMBEROFFIRSTSOLIDBINS - 1][j] = 0.0;

    compartmentIn.ssLow = compartmentIn.ssMeshXY;
    compartmentIn.ssHigh = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS - 1; j++)
            compartmentIn.ssHigh[i][j] = compartmentIn.ssMeshXY[i][j + 1];
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        compartmentIn.ssHigh[i][NUMBEROFSECONDSOLIDBINS - 1] = 0.0;

    //cout << "Creating sLoc & ssLoc and assigning zeros" << endl;
    arrayOfInt2D sLoc = getArrayOfInt2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    arrayOfInt2D ssLoc = getArrayOfInt2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sLoc" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
            sLoc[i][j] = floor(log(sAgg[i][j] / SCOEF) / log(SBASE) + 1);

    //cout << "End initializing values to sLoc" << endl;

    //cout << "Begin initializing values to ssLoc" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
            ssLoc[i][j] = floor(log(ssAgg[i][j] / SSCOEF) / log(SSBASE) + 1);

    //cout << "End initializing values to ssLoc" << endl;

    //repmat/reshape for sInd & ssInd
    //cout << "Creating sInd & ssInd and assigning zeros" << endl;
    compartmentIn.sInd = getArrayOfInt2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    compartmentIn.ssInd = getArrayOfInt2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    //cout << "Begin initializing values to sInd" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
            compartmentIn.sInd[i][j] = (j <= i) ? (i + 1) : (j + 1);

    //cout << "End initializing values to sInd" << endl;

    //cout << "Begin initializing values to ssInd" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
            compartmentIn.ssInd[i][j] = (j <= i) ? (i + 1) : (j + 1);

    //cout << "End initializing values to ssInd" << endl;

    //bsxfun @minus for sBreak & ssBreak
    //cout << "Creating sBreak & ssBreak and assigning zeros" << endl;
    arrayOfDouble2D sBreak = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    arrayOfDouble2D ssBreak = getArrayOfDouble2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sBreak (MATLAB bsxfun @minus)" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
        {
            double value = compartmentIn.vs[j] - compartmentIn.vs[i];
            sBreak[i][j] = value < 0.0 ? 0.0 : value;
        }

    //cout << "End initializing values to sBreak (MATLAB bsxfun @minus)" << endl << endl;

    //cout << "Begin initializing values to ssBreak (MATLAB bsxfun @minus)" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
        {
            double value = compartmentIn.vss[j] - compartmentIn.vss[i];
            ssBreak[i][j] = value < 0.0 ? 0.0 : value;
        }

    //cout << "End initializing values to ssBreak (MATLAB bsxfun @minus)" << endl << endl;
    // end of bsxfun @minus for sBreak & ssBreak

    //cout << "Creating sLocBreak & ssLocBreak and assigning zeros" << endl;
    arrayOfInt2D sLocBreak = getArrayOfInt2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    arrayOfInt2D ssLocBreak = getArrayOfInt2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sLocBreak" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
            sLocBreak[i][j] = (sBreak[j][i] == 0) ? 0 : (floor(log(sBreak[j][i] / SCOEF) / log(SBASE) + 1));

    //cout << "End initializing values to sLocBreak" << endl;

    //cout << "Begin initializing values to ssLocBreak" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
            ssLocBreak[i][j] = (ssBreak[j][i] == 0) ? 0 : (floor(log(ssBreak[j][i] / SSCOEF) / log(SSBASE) + 1));

    //cout << "End initializing values to ssLocBreak" << endl;

    // repmat for  sCheckB & ssCheckB
    //cout << "Creating sCheckB & ssCheckB and assigning zeros" << endl;
    compartmentIn.sCheckB = getArrayOfInt2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFFIRSTSOLIDBINS);
    compartmentIn.ssCheckB = getArrayOfInt2D(NUMBEROFSECONDSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    //cout << "Begin initializing values to sCheckB" << endl;
    for (int s1 = 0; s1 < NUMBEROFFIRSTSOLIDBINS; s1++)
        for (int s2 = 0; s2 < NUMBEROFFIRSTSOLIDBINS; s2++)
            compartmentIn.sCheckB[s1][s2] = sLocBreak[s1][s2] >= 1 ? 1 : 0;

    //cout << "End initializing values to sCheckB" << endl;

    //cout << "Begin initializing values to ssCheckB" << endl;
    for (int s1 = 0; s1 < NUMBEROFSECONDSOLIDBINS; s1++)
        for (int s2 = 0; s2 < NUMBEROFSECONDSOLIDBINS; s2++)
            ;
    //cout << "End initializing values to ssCheckB" << endl;

    //repmat/reshape for sIndB & ssIndB
    //cout << "Creating sIndB & ssIndB and assigning zeros" << endl;
    compartmentIn.sIndB = sLocBreak;
    compartmentIn.ssIndB = ssLocBreak;

    //cout << "Begin initializing values to sIndB" << endl;
    for (int i = 0; i < NUMBEROFFIRSTSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFFIRSTSOLIDBINS; j++)
        {
            if (compartmentIn.sIndB[i][j] < 1) // = i+1;
                compartmentIn.sIndB[i][j] = NUMBEROFFIRSTSOLIDBINS + 1;
        }

    //cout << "End initializing values to sIndB" << endl;

    //cout << "Begin initializing values to ssIndB" << endl;
    for (int i = 0; i < NUMBEROFSECONDSOLIDBINS; i++)
        for (int j = 0; j < NUMBEROFSECONDSOLIDBINS; j++)
        {
            if (compartmentIn.ssIndB[i][j] < 1) // = i+1;
                compartmentIn.ssIndB[i][j] = NUMBEROFSECONDSOLIDBINS + 1;
        }

    //cout << "End initializing values to ssIndB" << endl;
    //cout << endl << endl;

    //cout << "To be implemented" << endl;
    //cout << "sMeshBreak: Can be handled through vs(s1) - vs(s2)...by Anik" << endl;
    //cout << "ssMeshBreak: Can be handled through vss(ss1) - vss(ss2)...by Anik" << endl;

    //cout << endl << endl;
    //cout<< "Time Loop" << endl;
    //cout << "Initializations..." << endl;

    arrayOfDouble3D fAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D flAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D fgAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble3D dfdtAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D dfldtAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D dfgdtAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble3D externalVolumeBinsAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D internalVolumeBinsAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D liquidBinsAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D gasBinsAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D totalVolumeBinsAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble3D internalLiquidAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble3D externalLiquidAllCompartments = getArrayOfDouble3D(NUMBEROFCOMPARTMENTS, NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    arrayOfDouble2D internalVolumeBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
    arrayOfDouble2D externalVolumeBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

    compartmentDEMIn.DEMDiameter = vector <double>(NUMBEROFDEMBINS, 0.0);
    //compartmentDEMIn.numberOfCollisions = getArrayOfDouble2D(NUMBEROFDEMBINS, NUMBEROFDEMBINS,0.0);
    //compartmentDEMIn.numberOfImpacts = vector <double> (NUMBEROFDEMBINS, 0.0);

    //default_random_engine generator;
    // uniform_real_distribution<double> distribution(0.0,1.0);

    for (int i = 0; i < NUMBEROFDEMBINS; i++)
    {
      compartmentDEMIn.DEMDiameter[i] = (i+1)*1.0e-3; //distribution(generator)*1.0e3;
    // // used (i+1)*1.0e-3 bec distrib fxn give rand values in non size order which not work with DEM based scaled data kernel
    //// also note that we may have to do dem data converison in liggghts side then send over to PBM -subhodh
    // compartmentDEMIn.numberOfImpacts[i] = 1.0; //distribution(generator);
    // for(int j = 0; j < NUMBEROFDEMBINS; j++)
    //   compartmentDEMIn.numberOfCollisions[i][j] = 1.0; //distribution(generator);
    }
    // ***************** synthetic ligghts data  end ****************

    // ***************** read liggghts files START ******************
    // uncomment this section if you want to read acgtualy ligggghts files
    // use dem data ... make sure to comment section before this one which makes syn data

    // compartmentDEMIn.DEMDiameter = lData->getParticleDiameters();
    // if ((compartmentDEMIn.DEMDiameter).size() == 0)
    //     cout << "Diameter data is missing in LIGGGHTS output file" << endl;

    compartmentDEMIn.numberOfImpacts = lData->getFinalNumberOfImpacts();
    if ((compartmentDEMIn.numberOfImpacts).size() == 0)
        cout << "Impact data is missing in LIGGGHTS output file" << endl;

    compartmentDEMIn.numberOfCollisions = lData->getFinalNumberOfCollisions();
    if ((compartmentDEMIn.numberOfCollisions)[0].size() == 0)
        cout << "Collision data is missing in LIGGGHTS output file" << endl;
    // ************ read liggghts files end ******************

    double maxDiameter = getMaximumOf2DArray(compartmentIn.diameter);
    double maxDEMDiameter = getMaximumOfArray(compartmentDEMIn.DEMDiameter);

    vector<double> scaledDEMDiameter(NUMBEROFDEMBINS, 0.0);
    for (int i = 0; i < NUMBEROFDEMBINS; i++)
        scaledDEMDiameter[i] = compartmentDEMIn.DEMDiameter[i] * (maxDiameter / maxDEMDiameter);

    DUMP2D(compartmentDEMIn.numberOfCollisions);
    DUMP(compartmentDEMIn.DEMDiameter);
    DUMP(compartmentDEMIn.numberOfImpacts);
    DUMP(scaledDEMDiameter);

    vector<double> liquidAdditionRateAllCompartments(NUMBEROFCOMPARTMENTS, 0.0);
    liquidAdditionRateAllCompartments[0] = LIQUIDADDITIONRATE;
    double time = 0.0;
    double timeStep = 0.5; //1.0e-1;
    vector<double> Time;
    arrayOfDouble4D fAllCompartmentsOverTime;
    arrayOfDouble4D externalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble4D internalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble4D liquidBinsAllCompartmentsOverTime;
    arrayOfDouble4D gasBinsAllCompartmentsOverTime;
    vector<double> particleAverageVelocity(NUMBEROFCOMPARTMENTS, PARTICLEAVERAGEVELOCITY);

    while (time <= FINALTIME)
    {
        if (time > PREMIXINGTIME + LIQUIDADDITIONTIME)
            fIn = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
        for (int c = 0; c < NUMBEROFCOMPARTMENTS; c++)
        {

            compartmentIn.fAll = fAllCompartments[c];
            compartmentIn.fLiquid = flAllCompartments[c];
            compartmentIn.fGas = fgAllCompartments[c];
            compartmentIn.liquidAdditionRate = liquidAdditionRateAllCompartments[c];

            if (c == 0)
            {
                prevCompInData.fAllPreviousCompartment = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
                prevCompInData.flPreviousCompartment = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
                prevCompInData.fgPreviousCompartment = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
                prevCompInData.fAllComingIn = fIn;
                prevCompInData.fgComingIn = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
                double value = INITIALPOROSITY * timeStep;
                for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                    for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                        prevCompInData.fgComingIn[s][ss] = fIn[s][ss] * (compartmentIn.vs[ss] + compartmentIn.vss[ss]) * value;
            }

            else
            {
                prevCompInData.fAllPreviousCompartment = fAllCompartments[c - 1];
                prevCompInData.flPreviousCompartment = flAllCompartments[c - 1];
                prevCompInData.fgPreviousCompartment = fgAllCompartments[c - 1];
                prevCompInData.fAllComingIn = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
                prevCompInData.fgComingIn = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
            }

            compartmentOut = performCompartmentCalculations(prevCompInData, compartmentIn, compartmentDEMIn, time, timeStep);

            dfdtAllCompartments[c] = compartmentOut.dfAlldt;
            dfldtAllCompartments[c] = compartmentOut.dfLiquiddt;
            dfgdtAllCompartments[c] = compartmentOut.dfGasdt;

            liquidBinsAllCompartments[c] = compartmentOut.liquidBins;
            gasBinsAllCompartments[c] = compartmentOut.gasBins;
        }

        double maxofthree = -DBL_MAX;
        double maxAll = -DBL_MAX;
        double maxLiquid = -DBL_MAX;
        double maxGas = -DBL_MAX;

        for (int c = 0; c < NUMBEROFCOMPARTMENTS; c++)
            for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                {
                    if (fabs(fAllCompartments[c][s][ss]) > EPSILON)
                        maxAll = max(maxAll, -dfdtAllCompartments[c][s][ss] / fAllCompartments[c][s][ss]);
                    if (fabs(flAllCompartments[c][s][ss]) > EPSILON)
                        maxLiquid = max(maxLiquid, -dfldtAllCompartments[c][s][ss] / flAllCompartments[c][s][ss]);
                    if (fabs(fgAllCompartments[c][s][ss]) > EPSILON)
                        maxGas = max(maxGas, -dfgdtAllCompartments[c][s][ss] / fgAllCompartments[c][s][ss]);
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
        for (int c = 0; c < NUMBEROFCOMPARTMENTS; c++)
            for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                {
                    double value = 0.0;
                    fAllCompartments[c][s][ss] += dfdtAllCompartments[c][s][ss] * timeStep;
                    //fAllCompartments[c][s][ss] = value > 0.0 ? value : 0.0;
                    if (std::isnan(fAllCompartments[c][s][ss]))
                        nanCount++;

                    value = flAllCompartments[c][s][ss] + dfldtAllCompartments[c][s][ss] * timeStep;
                    flAllCompartments[c][s][ss] = value > 0.0 ? value : 0.0;
                    value = fgAllCompartments[c][s][ss] + dfgdtAllCompartments[c][s][ss] * timeStep;
                    fgAllCompartments[c][s][ss] = value > 0.0 ? value : 0.0;
                }

        if (nanCount)
            cout << endl
                 << "*****fAllCompartments has " << nanCount << "nan values******" << endl
                 << endl;
        minfAll = getMinimumOf3DArray(fAllCompartments);
        if (minfAll < 0.0 /*&& fabs(minfAll) > 1e-16*/)
        {
            cout << "minfAll" << minfAll << endl;
            cout << endl
                 << "******fAllCompartments has negative values********" << endl
                 << endl;
            cout << "Aborting..." << endl;
            return 0;
        }

        //BIN RECALCULATION
        //cout << "********Begin bin recalculation***********" << endl;

        for (int c = 0; c < NUMBEROFCOMPARTMENTS; c++)
        {
            arrayOfDouble2D liquidBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
            arrayOfDouble2D gasBins = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
            arrayOfDouble2D internalLiquid = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
            arrayOfDouble2D externalLiquid = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);
            for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                {
                    if (fabs(fAllCompartments[c][s][ss]) > EPSILON)
                    {
                        liquidBins[s][ss] = flAllCompartments[c][s][ss] / fAllCompartments[c][s][ss];
                        gasBins[s][ss] = fgAllCompartments[c][s][ss] / fAllCompartments[c][s][ss];
                    }
                    internalLiquid[s][ss] = min(GRANULESATURATIONFACTOR * gasBins[s][ss], liquidBins[s][ss]);
                    externalLiquid[s][ss] = max(0.0, liquidBins[s][ss] - internalLiquid[s][ss]);

                    double value = compartmentIn.sMeshXY[s][ss] + compartmentIn.ssMeshXY[s][ss] + gasBins[s][ss];
                    internalVolumeBins[s][ss] = value + internalLiquid[s][ss];
                    externalVolumeBins[s][ss] = value + liquidBins[s][ss];
                }
            liquidBinsAllCompartments[c] = liquidBins;
            gasBinsAllCompartments[c] = gasBins;
            externalVolumeBinsAllCompartments[c] = externalVolumeBins;
            internalVolumeBinsAllCompartments[c] = internalVolumeBins;
        }
        //SAVING OVER TIME
        //cout << endl <<  "************Saving over time" << endl << endl;
        fAllCompartmentsOverTime.push_back(fAllCompartments);
        externalVolumeBinsAllCompartmentsOverTime.push_back(externalVolumeBinsAllCompartments);
        internalVolumeBinsAllCompartmentsOverTime.push_back(internalVolumeBinsAllCompartments);
        liquidBinsAllCompartmentsOverTime.push_back(liquidBinsAllCompartments);
        gasBinsAllCompartmentsOverTime.push_back(gasBinsAllCompartments);

        cout << "time = " << time << endl;
        cout << "timeStep = " << timeStep << endl;
        cout << endl;
        Time.push_back(time);
        time += timeStep;
    }

    size_t nTimeSteps = Time.size();
    cout << endl
         << "nTimeSteps = " << nTimeSteps << endl
         << endl;

    arrayOfDouble3D dumpedLastValue = *(fAllCompartmentsOverTime.end() - 1);

    DUMP3DCSV(dumpedLastValue);
    //    string fileName = string("last_f_for_") + to_string(DEMAGGREGATIONKERNELVALUE) + string("_");
    //    fileName += to_string(DEMAGGREGATIONKERNELCONST) + string("_");
    //    fileName += to_string(DEMBREAKAGEKERNELVALUE) + string("_");
    //    fileName += to_string(DEMBREAKAGEKERNELCONST);
    //    dumpTestCSV(dumpedLastValue, fileName, DEMAGGREGATIONKERNELVALUE, DEMAGGREGATIONKERNELCONST, DEMBREAKAGEKERNELVALUE, DEMBREAKAGEKERNELCONST);

    // D10, D50, D90
    //cout << "Begin computing D10, D50, D90" << endl;
    arrayOfDouble2D d10OverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D d50OverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D d90OverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);

    arrayOfDouble2D totalVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D totalSolidVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D totalPoreVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D totalLiquidVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);
    arrayOfDouble2D totalGasVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, NUMBEROFCOMPARTMENTS);

    //sieveGrid=[38, 63, 90, 125, 250, 355, 500, 710, 850, 1000, 1400, 2000, 2380, 4000]; %Sieve (in micron)
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

    array<double, NUMBEROFFIRSTSOLIDBINS> vs = compartmentIn.vs;
    array<double, NUMBEROFFIRSTSOLIDBINS> vss = compartmentIn.vss;
    arrayOfDouble3D cumulativeVolumeDistributionAllCompartmentsOverTime = getArrayOfDouble3D(nTimeSteps, NUMBEROFCOMPARTMENTS, nSieveGrid);
    arrayOfDouble3D totalVolumeGridAllCompartmentsOverTime = getArrayOfDouble3D(nTimeSteps, NUMBEROFCOMPARTMENTS, nSieveGrid);

    vector<double> totalStuffLeavingOverTime(nTimeSteps, 0.0);
    vector<double> totalSolidLeavingOverTime(nTimeSteps, 0.0);
    vector<double> totalLiquidLeavingOverTime(nTimeSteps, 0.0);

    for (size_t n = 1; n < nTimeSteps; n++)
    {
        for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
            for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
            {
                timeStep = Time[n] - Time[n - 1];
                double value1 = fAllCompartmentsOverTime[n][NUMBEROFCOMPARTMENTS - 1][s][ss];
                value1 *= (particleAverageVelocity[NUMBEROFCOMPARTMENTS - 1] / DISTANCEBETWEENCOMPARTMENTS) * timeStep;
                double value2 = value1 * externalVolumeBinsAllCompartmentsOverTime[n][NUMBEROFCOMPARTMENTS - 1][s][ss];
                totalStuffLeavingOverTime[n] += value2;

                value2 = value1 * (vs[s] + vss[ss]);
                totalSolidLeavingOverTime[n] += value2;

                value2 = value1 * (externalVolumeBinsAllCompartmentsOverTime[n][NUMBEROFCOMPARTMENTS - 1][s][ss] - vs[s] - vss[ss]);
                totalLiquidLeavingOverTime[n] += value2;
            }

        for (int c = 0; c < NUMBEROFCOMPARTMENTS; c++)
        {
            for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                {
                    double value1 = fAllCompartmentsOverTime[n][c][s][ss];
                    double value2 = value1 * externalVolumeBinsAllCompartmentsOverTime[n][NUMBEROFCOMPARTMENTS - 1][s][ss];
                    totalVolumeAllCompartmentsOverTime[n][c] += value2;

                    value2 = value1 * (vs[s] + vss[ss]);
                    totalSolidVolumeAllCompartmentsOverTime[n][c] += value2;

                    value2 = value1 * (externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss] - vs[s] - vss[ss]);
                    totalPoreVolumeAllCompartmentsOverTime[n][c] += value2;

                    value2 = value1 * liquidBinsAllCompartmentsOverTime[n][c][s][ss];
                    totalLiquidVolumeAllCompartmentsOverTime[n][c] += value2;

                    value2 = value1 * gasBinsAllCompartmentsOverTime[n][c][s][ss];
                    totalGasVolumeAllCompartmentsOverTime[n][c] += value2;
                }

            arrayOfDouble2D fAll = fAllCompartmentsOverTime[n][c];
            externalVolumeBins = externalVolumeBinsAllCompartmentsOverTime[n][c];
            arrayOfDouble2D diameter = getArrayOfDouble2D(NUMBEROFFIRSTSOLIDBINS, NUMBEROFSECONDSOLIDBINS);

            for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                    diameter[s][ss] = cbrt((6 / M_PI) * externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss]) * 1.0e6;

            vector<double> totalVolumeGrid(nSieveGrid, 0.0);
            for (size_t d = 0; d < nSieveGrid - 1; d++)
                for (int s = 0; s < NUMBEROFFIRSTSOLIDBINS; s++)
                    for (int ss = 0; ss < NUMBEROFSECONDSOLIDBINS; ss++)
                    {
                        if (diameter[s][ss] < sieveGrid[d + 1] && diameter[s][ss] >= sieveGrid[d])
                            totalVolumeGrid[d] += fAll[s][ss] * externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss];
                    }

            totalVolumeGridAllCompartmentsOverTime[n][c] = totalVolumeGrid;

            double sum = 0.0;
            for (size_t d = 0; d < nSieveGrid; d++)
                sum += totalVolumeGrid[d];

            vector<double> volumeDistribution(nSieveGrid, 0.0);
            for (size_t d = 0; d < nSieveGrid; d++)
                volumeDistribution[d] = totalVolumeGrid[d] / sum;

            vector<double> cumulativeVolumeDistribution(nSieveGrid, 0.0);
            sum = 0.0;
            for (size_t d = 0; d < nSieveGrid; d++)
            {
                sum += volumeDistribution[d];
                cumulativeVolumeDistribution[d] = sum;
            }
            cumulativeVolumeDistributionAllCompartmentsOverTime[n][c] = cumulativeVolumeDistribution;

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
                    d10 = value * value1  + value2;
                }
                if (cumulativeVolumeDistribution[d - 1] < 0.1 && cumulativeVolumeDistribution[d] >= 0.1)
                {
                    double value = 0.9 - cumulativeVolumeDistribution[d - 1];
                    d90 = value * value1  + value2;
                }
            }
            d10OverTime[n][c] = d10;
            d50OverTime[n][c] = d50;
            d90OverTime[n][c] = d90;
        }
    }
    //cout << "End computing D10, D50, D90" << endl;
    //   DUMP2D(d10OverTime);
    //   DUMP2DCSV(d10OverTime);
    //   DUMP2D(d50OverTime);
    //   DUMP2DCSV(d50OverTime);
    //   DUMP2D(d90OverTime);
    //   DUMP2DCSV(d90OverTime);

    //DUMPDIACSV(Time, d10OverTime);
    //DUMPDIACSV(Time, d50OverTime);
    //DUMPDIACSV(Time, d90OverTime);

    //    string appendFileName = string("_") + to_string(DEMAGGREGATIONKERNELVALUE);
    //    appendFileName += string("_") + to_string(DEMAGGREGATIONKERNELCONST);
    //    appendFileName += string("_") + to_string(DEMBREAKAGEKERNELVALUE);
    //    appendFileName += string("_") + to_string(DEMBREAKAGEKERNELCONST);
    //
    //
    //    dumpDiaCSV(Time, d10OverTime, string("d10OverTime") + appendFileName);
    //    dumpDiaCSV(Time, d50OverTime, string("d50OverTime") + appendFileName);
    //    dumpDiaCSV(Time, d90OverTime, string("d90OverTime") + appendFileName);
    double endTime = static_cast<double>(clock()) / static_cast<double>(CLOCKS_PER_SEC);
    cout << "That took " << endTime - startTime << " seconds" << endl;
    cout << "Code End" << endl;
    return 0;
}
