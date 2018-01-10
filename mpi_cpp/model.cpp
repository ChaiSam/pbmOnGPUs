#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include <string>
#include <memory>

#include "utility.h"
#include "compartment.h"
#include "liggghtsData.h"
#include "timeStamp.h"
#include "parameterData.h"

#include <mpi.h>

using namespace std;

#define TWOWAYCOUPLE true
#define MASTER 0

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

int main(int argc, char *argv[])
{
    //cout << "CodeBegins" << endl << endl;
    //Read Dump Atom Files
    string startTimeStr;
    double startTime = 0.0;
    liggghtsData *lData = nullptr;
    parameterData *pData = nullptr;

    string coreVal;
    string diaVal;
    string pbmInFilePath;
    string timeVal;

    //**************************************************************************************************
    int num_mpi = 0;
    int mpi_id = 0;
    // spawn MPI processes
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_mpi);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);

    if (mpi_id == MASTER)
    {
        startTimeStr = string(timestring());
        startTime = MPI_Wtime();
        if (argc < 5)
        {
            cout << "All values aren't available as input parameters" << endl;
            int mpi_err = 0;
            MPI_Abort(MPI_COMM_WORLD, mpi_err);
        }
    }
    pbmInFilePath = string(argv[1]); 
    coreVal = string(argv[2]);
    diaVal = string(argv[3]);
    timeVal = string(argv[4]);

    MPI_Barrier(MPI_COMM_WORLD);

    //MPI HELLO World Test!
    //cout << "Hello, I am mpi process # = " << mpi_id << endl;

    pData = parameterData::getInstance();
    pData->readPBMInputFile(pbmInFilePath);

    int nCompartments = pData->nCompartments;
    if (mpi_id == MASTER)
    {
        cout << "Number of Compartments = " << nCompartments << endl;
        cout << "Number of Processes = " << num_mpi << endl;
    }

    //**************** MPI load Array Start *******************
    vector<int> load(2 * num_mpi, 0);
    vector<int> load_rev(2 * num_mpi, 0);

    int load_xi = 0;

    //MPI_Barrier(MPI_COMM_WORLD);

    //if (print == 1 && mpi_id == MASTER)
    //   cout << "Start to fill load with values..." << endl;

    for (int i = 0; i < nCompartments; i++)
    {
        load[load_xi] += 1;

        //if (print_load == 1 && mpi_id == MASTER)
        //    cout << "load[" << load_xi << "] = " << load[load_xi] << endl;

        load_xi++;

        if (load_xi == num_mpi)
            load_xi = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //if (print == 1)
    //   cout << "Start 2nd part of load calcs..." << endl;
    //    cout << "Start 2nd part of load calcs..." << endl;

    for (int i = 0; i < num_mpi + 1; i++)
    {
        load_rev[num_mpi - i] = load[i];
        //if (print_load && mpi_id == MASTER)
        //    cout << "load_rev[" << num_mpi - i << "] = " << load_rev[num_mpi - i] << endl;
    }

    //if (print == 1)
    //    cout << "Now copying load_rev into load..." << endl;
    //   cout << "Now copying load_rev into load..." << endl;

    load = load_rev;
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < num_mpi + 1; i++)
    {
        if (i != MASTER)
            load[i] = load[i] + load[i - 1];

        //if (print_load && mpi_id == MASTER)
            //cout << "load[" << i << "] = " << load[i] << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    /*dist lower and upper chunks to cores*/
    //if (print && mpi_id == MASTER)
    //    cout << "now distribute core_low and core_up to each process" << endl;

    int core_low = 0, core_up = 0;

    core_low = load[mpi_id];
    core_up = load[mpi_id + 1];

    //MPI workload distribute end

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "my process id = " << mpi_id << ", core_low = " << core_low << ", core_up = " << core_up << endl;

    //*************** MPI load Array End   ********************

    //**************************************************************************************************
    CompartmentIn compartmentIn;          //Input for compartment call
    PreviousCompartmentIn prevCompInData; //Input data from previous compartment; 2nd compartment onwards
    CompartmentDEMIn compartmentDEMIn;
    CompartmentOut compartmentOut; //Output for compartment call

    unsigned int nFirstSolidBins = pData->nFirstSolidBins;
    unsigned int nSecondSolidBins = pData->nSecondSolidBins;

    vector<double> vs(nFirstSolidBins, 0.0);
    vector<double> vss(nSecondSolidBins, 0.0);

    //Bin initialization
    //cout << "Begin assign value to vsArray" << endl;
    double fsVolCoeff = pData->fsVolCoeff;
    double fsVolBase = pData->fsVolBase;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        vs[i] = fsVolCoeff * pow(fsVolBase, i); // m^3

    //cout << "End assign value to vsArray" << endl << endl;

    ////cout << "Begin assign value to vssArray" << endl;
    double ssVolCoeff = pData->ssVolCoeff;
    double ssVolBase = pData->ssVolBase;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        vss[i] = ssVolCoeff * pow(ssVolBase, i); // m^3

    //cout << "End assign value to vssArray" << endl << endl;

    arrayOfDouble2D diameter = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    for (size_t s = 0; s < nFirstSolidBins; s++)
        for (size_t ss = 0; ss < nSecondSolidBins; ss++)
            diameter[s][ss] = cbrt((6 / M_PI) * (vs[s] + vss[ss]));

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
    arrayOfDouble2D fIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    //cout << "Begin initialize hard coded values to fIn" << endl;
    // for (size_t i = 0; i < particleIn.size(); i++)
    //    for (size_t j = 0; j < particleIn.size(); j++)
    //        fIn[i][j] = sqrt(particleIn[i] * particleIn[j]);
    for (size_t i = 0; i < particleIn.size(); i++)
        fIn[i][i] = particleIn[i];

    //cout << "End initialize hard coded values to fIn" << endl << endl;

    //MATLAB ndgrid
    //cout << "Creating sMeshXY & ssMeshXY and assigning zeros" << endl;
    arrayOfDouble2D sMeshXY = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble2D ssMeshXY = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sMeshXY & ssMeshXY (MATLAB ndgrid)" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
        {
            sMeshXY[i][j] = vs[i];
            ssMeshXY[i][j] = vss[j];
        }

    //cout << "End initializing values to sMeshXY & ssMeshXY (MATLAB ndgrid)" << endl << endl;

    //bsxfun @plus for sAgg & ssAgg
    //cout << "Creating sAgg & ssAgg and assigning zeros" << endl;
    arrayOfDouble2D sAgg = getArrayOfDouble2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfDouble2D ssAgg = getArrayOfDouble2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sAgg (MATLAB bsxfun @plus)" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
            sAgg[i][j] = vs[j] + vs[i];
    //cout << "End initializing values to sAgg (MATLAB bsxfun @plus)" << endl << endl;

    //cout << "Begin initializing values to ssAgg (MATLAB bsxfun @plus)" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
            ssAgg[i][j] = vss[j] + vss[i];
    //cout << "End initializing values to ssAgg (MATLAB bsxfun @plus)" << endl << endl;

    //repmat for  sAggregationCheck & ssAggregationCheck
    //cout << "Creating sAggregationCheck & ssAggregationCheck and assigning zeros" << endl;
    arrayOfInt2D sAggregationCheck = getArrayOfInt2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfInt2D ssAggregationCheck = getArrayOfInt2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sAggregationCheck" << endl;
    for (size_t s1 = 0; s1 < nFirstSolidBins; s1++)
        for (size_t s2 = 0; s2 < nFirstSolidBins; s2++)
            sAggregationCheck[s1][s2] = sAgg[s1][s2] <= vs[nFirstSolidBins - 1] ? 1 : 0;
    //cout << "End initializing values to sAggregationCheck" << endl;

    //cout << "Begin initializing values to ssAggregationCheck" << endl;
    for (size_t ss1 = 0; ss1 < nSecondSolidBins; ss1++)
        for (size_t ss2 = 0; ss2 < nSecondSolidBins; ss2++)
            ssAggregationCheck[ss1][ss2] = ssAgg[ss1][ss2] <= vss[nSecondSolidBins - 1] ? 1 : 0;
    //cout << "End initializing values to sAggregationCheck" << endl;

    // end of repmat for  sAggregationCheck & ssAggregationCheck

    arrayOfDouble2D sLow = sMeshXY;
    arrayOfDouble2D sHigh = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    for (size_t i = 0; i < nSecondSolidBins - 1; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
            sHigh[i][j] = sMeshXY[i + 1][j];

    for (size_t j = 0; j < nSecondSolidBins; j++)
        sHigh[nFirstSolidBins - 1][j] = 0.0;

    arrayOfDouble2D ssLow = ssMeshXY;
    arrayOfDouble2D ssHigh = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins - 1; j++)
            ssHigh[i][j] = ssMeshXY[i][j + 1];
    for (size_t i = 0; i < nSecondSolidBins; i++)
        ssHigh[i][nSecondSolidBins - 1] = 0.0;

    //cout << "Creating sLoc & ssLoc and assigning zeros" << endl;
    arrayOfInt2D sLoc = getArrayOfInt2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfInt2D ssLoc = getArrayOfInt2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sLoc" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
            sLoc[i][j] = floor(log(sAgg[i][j] / fsVolCoeff) / log(fsVolBase) + 1);

    //cout << "End initializing values to sLoc" << endl;

    //cout << "Begin initializing values to ssLoc" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
            ssLoc[i][j] = floor(log(ssAgg[i][j] / ssVolCoeff) / log(ssVolBase) + 1);

    //cout << "End initializing values to ssLoc" << endl;

    //repmat/reshape for sInd & ssInd
    //cout << "Creating sInd & ssInd and assigning zeros" << endl;
    arrayOfInt2D sInd = getArrayOfInt2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfInt2D ssInd = getArrayOfInt2D(nSecondSolidBins, nSecondSolidBins);
    //cout << "Begin initializing values to sInd" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
            sInd[i][j] = (j <= i) ? (i + 1) : (j + 1);

    //cout << "End initializing values to sInd" << endl;

    //cout << "Begin initializing values to ssInd" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
            ssInd[i][j] = (j <= i) ? (i + 1) : (j + 1);

    //cout << "End initializing values to ssInd" << endl;

    //bsxfun @minus for sBreak & ssBreak
    //cout << "Creating sBreak & ssBreak and assigning zeros" << endl;
    arrayOfDouble2D sBreak = getArrayOfDouble2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfDouble2D ssBreak = getArrayOfDouble2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sBreak (MATLAB bsxfun @minus)" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
        {
            double value = vs[j] - vs[i];
            sBreak[i][j] = value < 0.0 ? 0.0 : value;
        }

    //cout << "End initializing values to sBreak (MATLAB bsxfun @minus)" << endl << endl;

    //cout << "Begin initializing values to ssBreak (MATLAB bsxfun @minus)" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
        {
            double value = vss[j] - vss[i];
            ssBreak[i][j] = value < 0.0 ? 0.0 : value;
        }

    //cout << "End initializing values to ssBreak (MATLAB bsxfun @minus)" << endl << endl;
    // end of bsxfun @minus for sBreak & ssBreak

    //cout << "Creating sLocBreak & ssLocBreak and assigning zeros" << endl;
    arrayOfInt2D sLocBreak = getArrayOfInt2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfInt2D ssLocBreak = getArrayOfInt2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sLocBreak" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
            sLocBreak[i][j] = (sBreak[j][i] == 0) ? 0 : (floor(log(sBreak[j][i] / fsVolCoeff) / log(fsVolBase) + 1));

    //cout << "End initializing values to sLocBreak" << endl;

    //cout << "Begin initializing values to ssLocBreak" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
            ssLocBreak[i][j] = (ssBreak[j][i] == 0) ? 0 : (floor(log(ssBreak[j][i] / ssVolCoeff) / log(ssVolBase) + 1));

    //cout << "End initializing values to ssLocBreak" << endl;

    // repmat for  sCheckB & ssCheckB
    //cout << "Creating sCheckB & ssCheckB and assigning zeros" << endl;
    arrayOfInt2D sCheckB = getArrayOfInt2D(nFirstSolidBins, nFirstSolidBins);
    arrayOfInt2D ssCheckB = getArrayOfInt2D(nSecondSolidBins, nSecondSolidBins);

    //cout << "Begin initializing values to sCheckB" << endl;
    for (size_t s1 = 0; s1 < nFirstSolidBins; s1++)
        for (size_t s2 = 0; s2 < nFirstSolidBins; s2++)
            sCheckB[s1][s2] = sLocBreak[s1][s2] >= 1 ? 1 : 0;

    //cout << "End initializing values to sCheckB" << endl;

    //cout << "Begin initializing values to ssCheckB" << endl;
    for (size_t ss1 = 0; ss1 < nSecondSolidBins; ss1++)
        for (size_t ss2 = 0; ss2 < nSecondSolidBins; ss2++)
            ssCheckB[ss1][ss2] = ssLocBreak[ss1][ss2] >= 1 ? 1 : 0;
    ;
    //cout << "End initializing values to ssCheckB" << endl;

    //repmat/reshape for sIndB & ssIndB
    //cout << "Creating sIndB & ssIndB and assigning zeros" << endl;
    arrayOfInt2D sIndB = sLocBreak;
    arrayOfInt2D ssIndB = ssLocBreak;

    //cout << "Begin initializing values to sIndB" << endl;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        for (size_t j = 0; j < nFirstSolidBins; j++)
        {
            if (sIndB[i][j] < 1) // = i+1;
                sIndB[i][j] = nFirstSolidBins + 1;
        }

    //cout << "End initializing values to sIndB" << endl;

    //cout << "Begin initializing values to ssIndB" << endl;
    for (size_t i = 0; i < nSecondSolidBins; i++)
        for (size_t j = 0; j < nSecondSolidBins; j++)
        {
            if (ssIndB[i][j] < 1) // = i+1;
                ssIndB[i][j] = nSecondSolidBins + 1;
        }

    //cout << "End initializing values to ssIndB" << endl;
    //cout << endl << endl;

    //cout << "To be implemented" << endl;
    //cout << "sMeshBreak: Can be handled through vs(s1) - vs(s2)...by Anik" << endl;
    //cout << "ssMeshBreak: Can be handled through vss(ss1) - vss(ss2)...by Anik" << endl;

    //cout << endl << endl;
    //cout<< "Time Loop" << endl;
    //cout << "Initializations..." << endl;

    arrayOfDouble3D fAllCompartments;
    arrayOfDouble3D flAllCompartments;
    arrayOfDouble3D fgAllCompartments;

    if (TWOWAYCOUPLE && fabs(stod(timeVal)) > 1.0e-16)
    {
        fAllCompartments = pData->readCompartmentInputFile(/*timeValueString*/ timeVal, string("particles"));
        if (fAllCompartments.empty())
        {
            cout << "particle file missing for time = " << timeVal << endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }

        flAllCompartments = pData->readCompartmentInputFile(/*timeValueString*/ timeVal, string("liquid"));
        if (flAllCompartments.empty())
        {
            cout << "liquid file missing for time = " << timeVal << endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        fgAllCompartments = pData->readCompartmentInputFile(/*timeValueString*/ timeVal, string("gas"));
        if (fgAllCompartments.empty())
        {
            cout << "gas file missing for time = " << timeVal << endl;
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        
        // cout << "dumping test data" << endl;
        // if(mpi_id == MASTER)
        // {
        //     DUMP3DCSV(fAllCompartments);
        //     DUMP3DCSV(flAllCompartments);
        //     DUMP3DCSV(fgAllCompartments);
        // }
    }
    else
    {
        fAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
        flAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
        fgAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    }

    arrayOfDouble3D dfdtAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D dfldtAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D dfgdtAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);

    arrayOfDouble3D externalVolumeBinsAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D internalVolumeBinsAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D liquidBinsAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D gasBinsAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D totalVolumeBinsAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);

    arrayOfDouble3D internalLiquidAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble3D externalLiquidAllCompartments = getArrayOfDouble3D(nCompartments, nFirstSolidBins, nSecondSolidBins);

    arrayOfDouble2D internalVolumeBins = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    arrayOfDouble2D externalVolumeBins = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);

    // **************** snythetic liggghts data ******************************
    //**** uncommetn this section to use make snythetic dem data
    // in place of next section which actually will read available ligghts files

    //vector<double> DEMDiameter(NUMBEROFDEMBINS,0.0);
    //arrayOfDouble2D numberOfCollisions = getArrayOfDouble2D(NUMBEROFDEMBINS, NUMBEROFDEMBINS,0.0);
    //vector <double> numberOfImpacts(NUMBEROFDEMBINS, 0.0);

    //default_random_engine generator;
    //uniform_real_distribution<double> distribution(0.0,1.0);

    //for (int i = 0; i < NUMBEROFDEMBINS; i++)
    //{
    //compartmentDEMIn.DEMDiameter[i] = (i+1)*1.0e-3; //distribution(generator)*1.0e3;
    // // used (i+1)*1.0e-3 bec distrib fxn give rand values in non size order which not work with DEM based scaled data kernel
    //// also note that we may have to do dem data converison in liggghts side then send over to PBM -subhodh
    // compartmentDEMIn.numberOfImpacts[i] = 1.0; //distribution(generator);
    // for(int j = 0; j < NUMBEROFDEMBINS; j++)
    //   compartmentDEMIn.numberOfCollisions[i][j] = 1.0; //distribution(generator);
    //}
    // ***************** synthetic ligghts data  end ****************

    // ***************** read liggghts files START ******************
    // uncomment this section if you want to read acgtualy ligggghts files
    // use dem data ... make sure to comment section before this one which makes syn data

    lData = liggghtsData::getInstance();

    if (!lData)
        cout << "mpi_id = " << mpi_id << ", fis null" << endl;
    lData->readLiggghtsDataFiles(coreVal, diaVal);

    vector<double> DEMDiameter = lData->getDEMParticleDiameters();
    if ((DEMDiameter).size() == 0)
    {
        cout << "My process id = " << mpi_id << endl;
        cout << "Diameter data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        int mpi_err = 0;
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }

    vector<double> DEMImpactData = lData->getFinalDEMImpactData();
    if ((DEMImpactData).size() == 0)
    {
        cout << "My process id = " << mpi_id << endl;
        cout << "Impact data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        int mpi_err = 0;
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }

    arrayOfDouble2D DEMCollisionData = lData->getFinalDEMCollisionData();
    if (DEMCollisionData.size() == 0)
    {
        cout << "My process id = " << mpi_id << endl;
        cout << "Collision data is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        int mpi_err = 0;
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }
    vector<double> velocity = lData->getFinalDEMImpactVelocity();
    if (velocity.size() == 0)
    {
        cout << "My process id = " << mpi_id << endl;
        cout << "Velocity is missing in LIGGGHTS output file" << endl;
        cout << "Input parameters for DEM core and diameter aren't matching with LIGGGHTS output file" << endl;
        int mpi_err = 0;
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }

    // ************ read liggghts files end ******************

    if (mpi_id == MASTER)
    {
        DUMP2D(DEMCollisionData);
        DUMP(DEMDiameter);
        DUMP(DEMImpactData);
        DUMP(velocity);
    }

    vector<double> liquidAdditionRateAllCompartments(nCompartments, 0.0);
    double liqSolidRatio = pData->liqSolidRatio;
    double throughput = pData->throughput;
    double liqDensity = pData->liqDensity;
    double liquidAddRate = (liqSolidRatio * throughput) / (liqDensity * 3600);
    liquidAdditionRateAllCompartments[0] = liquidAddRate;
    
    arrayOfDouble4D fAllCompartmentsOverTime;
    arrayOfDouble4D externalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble4D internalVolumeBinsAllCompartmentsOverTime;
    arrayOfDouble4D liquidBinsAllCompartmentsOverTime;
    arrayOfDouble4D gasBinsAllCompartmentsOverTime;

    double granulatorLength = pData->granulatorLength;
    double partticleResTime = pData->partticleResTime;
    double particleAveVelo = granulatorLength /  partticleResTime;
    vector<double> particleAverageVelocity(nCompartments, particleAveVelo);

    //Initialize input data for compartment

    compartmentIn.vs = vs;
    compartmentIn.vss = vss;

    compartmentIn.diameter = diameter;

    compartmentIn.sMeshXY = sMeshXY;
    compartmentIn.ssMeshXY = ssMeshXY;

    compartmentIn.sAggregationCheck = sAggregationCheck;
    compartmentIn.ssAggregationCheck = ssAggregationCheck;

    compartmentIn.sLow = sLow;
    compartmentIn.sHigh = sHigh;

    compartmentIn.ssLow = ssLow;
    compartmentIn.ssHigh = ssHigh;

    compartmentIn.sInd = sInd;
    compartmentIn.ssInd = sInd;

    compartmentIn.sCheckB = sCheckB;
    compartmentIn.ssCheckB = ssCheckB;

    compartmentIn.sIndB = sIndB;
    compartmentIn.ssIndB = ssIndB;

    //Initialize DEM data for compartment
    compartmentDEMIn.DEMDiameter = DEMDiameter;
    compartmentDEMIn.DEMCollisionData = DEMCollisionData;
    compartmentDEMIn.DEMImpactData = DEMImpactData;

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

    arrayOfDouble2D d10OverTime;
    arrayOfDouble2D d50OverTime;
    arrayOfDouble2D d90OverTime;

    double time = stod(timeVal); // initial time to start PBM
    double timeStep = 0.5; //1.0e-1;
    vector<double> Time;

    double lastTime = time;
    int timeIdxCount = 0;
    int lastTimeIdxCount = 0;

    double premixTime = pData->premixTime;
    double liqAddTime = pData->liqAddTime;
    double postMixTime = pData->postMixTime;
    double finalTime = premixTime + liqAddTime + postMixTime + stod(timeVal);
    double initPorosity = pData->initPorosity;
    
    arrayOfDouble2D formationThroughAggregationOverTime;
    arrayOfDouble2D depletionThroughAggregationOverTime;
    arrayOfDouble2D formationThroughBreakageOverTime;
    arrayOfDouble2D depletionThroughBreakageOverTime;
   
    
    while (time <= finalTime)
    {
        // if (time > premixTime + liqAddTime + stod(timeVal))
        //     fIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
        vector<double> formationThroughAggregation(nCompartments, 0.0);
        vector<double> depletionThroughAggregation(nCompartments, 0.0);
        vector<double> formationThroughBreakage(nCompartments, 0.0);
        vector<double> depletionThroughBreakage(nCompartments, 0.0);

        for (int c = core_low; c < core_up; c++) //for (int c = 0; c < nCompartments; c++)
        {
            compartmentIn.fAll = fAllCompartments[c];
            compartmentIn.fLiquid = flAllCompartments[c];
            compartmentIn.fGas = fgAllCompartments[c];
            compartmentIn.liquidAdditionRate = liquidAdditionRateAllCompartments[c];

            if (c == 0)
            {
                prevCompInData.fAllPreviousCompartment = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
                prevCompInData.flPreviousCompartment = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
                prevCompInData.fgPreviousCompartment = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
                prevCompInData.fAllComingIn = fIn;
                prevCompInData.fgComingIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
                double value = initPorosity * timeStep;
                for (size_t s = 0; s < nFirstSolidBins; s++)
                    for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                        prevCompInData.fgComingIn[s][ss] = fIn[s][ss] * (compartmentIn.vs[ss] + compartmentIn.vss[ss]) * value;
            }

            else
            {
                prevCompInData.fAllPreviousCompartment = fAllCompartments[c - 1];
                prevCompInData.flPreviousCompartment = flAllCompartments[c - 1];
                prevCompInData.fgPreviousCompartment = fgAllCompartments[c - 1];
                prevCompInData.fAllComingIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
                prevCompInData.fgComingIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            }

            compartmentOut = performCompartmentCalculations(prevCompInData, compartmentIn, compartmentDEMIn, time, timeStep, stod(timeVal));

            dfdtAllCompartments[c] = compartmentOut.dfAlldt;
            dfldtAllCompartments[c] = compartmentOut.dfLiquiddt;
            dfgdtAllCompartments[c] = compartmentOut.dfGasdt;

            liquidBinsAllCompartments[c] = compartmentOut.liquidBins;
            gasBinsAllCompartments[c] = compartmentOut.gasBins;

            formationThroughAggregation[c] = compartmentOut.formationThroughAggregation;
            depletionThroughAggregation[c] = compartmentOut.depletionThroughAggregation;
            formationThroughBreakage[c] = compartmentOut.formationThroughBreakage;
            depletionThroughBreakage[c] = compartmentOut.depletionThroughBreakage;
        }


        // *************************************************************
        //************** START MPI Send Recv ***************************
        // *************************************************************
        MPI_Status status;
        vector<double> buff_sr_lin(nCompartments, 0.0);

        if (mpi_id != MASTER)
        {
            /* send */
            //printf("msg pre dfdt sent i=%d\n",i);
            buff_sr_lin = formationThroughAggregation; //linearize2DVector(formationThroughAggregation); //memcpy(&buff_sr, &dfdtAllCompartments,sizeof(dfdtAllCompartments));
            MPI_Send(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
            //printf("msg dfdt sent i=%d\n",i);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    formationThroughAggregation[c] = buff_sr_lin[c];
            }
        }
        if (mpi_id != MASTER)
        {
            /* send */
            //printf("msg pre dfdt sent i=%d\n",i);
            buff_sr_lin = depletionThroughAggregation; //memcpy(&buff_sr, &dfdtAllCompartments,sizeof(dfdtAllCompartments));
            MPI_Send(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
            //printf("msg dfdt sent i=%d\n",i);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    depletionThroughAggregation[c] = buff_sr_lin[c];
            }
        }
        if (mpi_id != MASTER)
        {
            /* send */
            //printf("msg pre dfdt sent i=%d\n",i);
            buff_sr_lin = formationThroughBreakage; //memcpy(&buff_sr, &dfdtAllCompartments,sizeof(dfdtAllCompartments));
            MPI_Send(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
            //printf("msg dfdt sent i=%d\n",i);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    formationThroughBreakage[c] = buff_sr_lin[c];
            }
        }
        if (mpi_id != MASTER)
        {
            /* send */
            //printf("msg pre dfdt sent i=%d\n",i);
            buff_sr_lin = depletionThroughBreakage; //memcpy(&buff_sr, &dfdtAllCompartments,sizeof(dfdtAllCompartments));
            MPI_Send(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
            //printf("msg dfdt sent i=%d\n",i);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr_lin.data(), static_cast<int>(buff_sr_lin.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    depletionThroughBreakage[c] = buff_sr_lin[c];
            }
        }

        formationThroughAggregationOverTime.push_back(formationThroughAggregation);
        depletionThroughAggregationOverTime.push_back(depletionThroughAggregation);
        formationThroughBreakageOverTime.push_back(formationThroughBreakage);
        depletionThroughBreakageOverTime.push_back(depletionThroughBreakage);


        MPI_Barrier(MPI_COMM_WORLD);
        vector<double> buff_sr(nCompartments * nFirstSolidBins * nSecondSolidBins, 0.0);


        // MPI SR dFdT_AllComaprments START

        //for (int i = 1; i < num_mpi; i++)
        //{
        if (mpi_id != MASTER)
        {
            /* send */
            //printf("msg pre dfdt sent i=%d\n",i);
            buff_sr = linearize3DVector(dfdtAllCompartments); //memcpy(&buff_sr, &dfdtAllCompartments,sizeof(dfdtAllCompartments));
            MPI_Send(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
            //printf("msg dfdt sent i=%d\n",i);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    for (size_t i0 = 0; i0 < nFirstSolidBins; i0++)
                        for (size_t ix = 0; ix < nSecondSolidBins; ix++)
                            dfdtAllCompartments[c][i0][ix] = buff_sr[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_id != MASTER)
        {
            buff_sr = linearize3DVector(dfldtAllCompartments);
            MPI_Send(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    for (size_t i0 = 0; i0 < nFirstSolidBins; i0++)
                        for (size_t ix = 0; ix < nSecondSolidBins; ix++)
                            dfldtAllCompartments[c][i0][ix] = buff_sr[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_id != MASTER)
        {
            buff_sr = linearize3DVector(dfgdtAllCompartments);
            MPI_Send(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, MASTER, mpi_id, MPI_COMM_WORLD);
        }

        if (mpi_id == MASTER)
        {
            for (int i = 1; i < num_mpi; i++)
            {
                MPI_Recv(buff_sr.data(), static_cast<int>(buff_sr.size()), MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);

                for (int c = load[i]; c < load[i + 1]; c++)
                    for (size_t i0 = 0; i0 < nFirstSolidBins; i0++)
                        for (size_t ix = 0; ix < nSecondSolidBins; ix++)
                            dfgdtAllCompartments[c][i0][ix] = buff_sr[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
            }
        }

        //*************************MPI MSG PASSING End  ********************************************************

        // ************************ MPI BCAST START     ********************************************************

        MPI_Barrier(MPI_COMM_WORLD);

        vector<double> buff_dfdt = linearize3DVector(dfdtAllCompartments);
        vector<double> buff_dfldt = linearize3DVector(dfldtAllCompartments);
        vector<double> buff_dfgdt = linearize3DVector(dfgdtAllCompartments);
        /* now doing MPI_Bcast all of the fvalues needed to all cores .... */
        //printf("early bcast dfdt\n");
        MPI_Bcast(buff_dfdt.data(), buff_dfdt.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //printf("early bcast dfl dt\n");
        MPI_Bcast(buff_dfldt.data(), buff_dfldt.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //printf("early bcast dfg dt\n");
        MPI_Bcast(buff_dfgdt.data(), buff_dfgdt.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        /*ierr = MPI_Bcast(&timeStep,sizeof(double),MPI_DOUBLE,0,MPI_COMM_WORLD);*/
        MPI_Barrier(MPI_COMM_WORLD);

        if (mpi_id != MASTER)
        {
            for (int c = 0; c < nCompartments; c++)
                for (size_t i0 = 0; i0 < nFirstSolidBins; i0++)
                    for (size_t ix = 0; ix < nSecondSolidBins; ix++)
                    {
                        dfdtAllCompartments[c][i0][ix] = buff_dfdt[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
                        dfldtAllCompartments[c][i0][ix] = buff_dfldt[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
                        dfgdtAllCompartments[c][i0][ix] = buff_dfgdt[c * nFirstSolidBins * nSecondSolidBins + i0 * nSecondSolidBins + ix];
                    }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // *************************************************************
        //************** end   MPI Send Recv ***************************
        // *************************************************************

        //***************************************** START time step calcs **************************************************
        double maxofthree = -DBL_MAX;
        double maxAll = -DBL_MAX;
        double maxLiquid = -DBL_MAX;
        double maxGas = -DBL_MAX;

        for (int c = 0; c < nCompartments; c++)
            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                {
                    if (fabs(fAllCompartments[c][s][ss]) > 1.0e-16)
                        maxAll = max(maxAll, -dfdtAllCompartments[c][s][ss] / fAllCompartments[c][s][ss]);
                    if (fabs(flAllCompartments[c][s][ss]) > 1.0e-16)
                        maxLiquid = max(maxLiquid, -dfldtAllCompartments[c][s][ss] / flAllCompartments[c][s][ss]);
                    if (fabs(fgAllCompartments[c][s][ss]) > 1.0e-16)
                        maxGas = max(maxGas, -dfgdtAllCompartments[c][s][ss] / fgAllCompartments[c][s][ss]);
                    maxofthree = max(maxofthree, max(maxAll, max(maxLiquid, maxGas)));
                }
        if (mpi_id == MASTER)
        {
            cout << "maxAll = " << maxAll << endl;
            cout << "maxLiquid = " << maxLiquid << endl;
            cout << "maxGas = " << maxGas << endl;
            cout << "maxofthree = " << maxofthree << endl;
        }

        while (maxofthree < 0.1 / timeStep && timeStep < 0.25)
            timeStep *= 2.0;

        while (maxofthree > 0.1 / timeStep && timeStep > 5.0e-5)
            timeStep /= 2.0;

        int nanCount = 0;
        double minfAll = -DBL_MAX;
        for (int c = 0; c < nCompartments; c++)
            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
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
        {
            int mpi_err = 0;
            cout << endl;
            cout << "My process id = " << mpi_id << endl;
            cout << "*****fAllCompartments has " << nanCount << " nan values******" << endl;
            cout << " Aborting..." << endl;
            MPI_Abort(MPI_COMM_WORLD, mpi_err);
        }
        int countnegfAll = 0;
        minfAll = getMinimumOf3DArray(fAllCompartments, countnegfAll);
        if (minfAll < -1.0e-16 &&  countnegfAll > 0.1 * nCompartments * nFirstSolidBins * nSecondSolidBins)
        {
            int mpi_err = 0;
            cout << endl;
            //DUMP3DCSV(dfdtAllCompartments);
            //DUMP3DCSV(fAllCompartments);
            cout << "My process id = " << mpi_id << endl;
            cout << "minfAll" << minfAll << endl;
            cout << "******fAllCompartments has negative values********" << endl;
            cout << "Number of negative values = " << countnegfAll << endl;
            DUMP3DCSV(fAllCompartments);
            cout << " Aborting..." << endl;
            MPI_Abort(MPI_COMM_WORLD, mpi_err);
        }

        //******************************** END time step calcs ****************************************************

        //BIN RECALCULATION
        //cout << "********Begin bin recalculation***********" << endl;

        double granSatFactor = pData->granSatFactor;
        for (int c = 0; c < nCompartments; c++)
        {
            arrayOfDouble2D liquidBins = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            arrayOfDouble2D gasBins = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            arrayOfDouble2D internalLiquid = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            arrayOfDouble2D externalLiquid = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                {
                    if (fabs(fAllCompartments[c][s][ss]) > 1.0e-16)
                    {
                        liquidBins[s][ss] = flAllCompartments[c][s][ss] / fAllCompartments[c][s][ss];
                        gasBins[s][ss] = fgAllCompartments[c][s][ss] / fAllCompartments[c][s][ss];
                    }
                    internalLiquid[s][ss] = min(granSatFactor * gasBins[s][ss], liquidBins[s][ss]);
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

        MPI_Barrier(MPI_COMM_WORLD);
        // Calculate d10, d50, d90
        vector<double> d10OverCompartment(nCompartments, 0.0);
        vector<double> d50OverCompartment(nCompartments, 0.0);
        vector<double> d90OverCompartment(nCompartments, 0.0);

        for (int c = 0; c < nCompartments; c++)
        {
            arrayOfDouble2D diameter = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
            for (size_t s = 0; s < nFirstSolidBins; s++)
                for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                    diameter[s][ss] = cbrt((6 / M_PI) * externalVolumeBinsAllCompartments[c][s][ss]) * 1.0e6;

            vector<double> totalVolumeGrid(nSieveGrid, 0.0);
            for (size_t d = 0; d < nSieveGrid - 1; d++)
                for (size_t s = 0; s < nFirstSolidBins; s++)
                    for (size_t ss = 0; ss < nSecondSolidBins; ss++)
                    {
                        if (diameter[s][ss] < sieveGrid[d + 1] && diameter[s][ss] >= sieveGrid[d])
                            totalVolumeGrid[d] += fAllCompartments[c][s][ss] * externalVolumeBinsAllCompartments[c][s][ss];
                    }

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
        
        //Saving d10, d50 & d90 over time
        // d10OverTime.push_back(d10OverCompartment);
        d50OverTime.push_back(d50OverCompartment);
        // d90OverTime.push_back(d90OverCompartment);

        // End of calculate d10, d50, d90

        //SAVING OVER TIME
        //cout << endl <<  "************Saving over time" << endl << endl;
        // fAllCompartmentsOverTime.push_back(fAllCompartments);
        // externalVolumeBinsAllCompartmentsOverTime.push_back(externalVolumeBinsAllCompartments);
        // internalVolumeBinsAllCompartmentsOverTime.push_back(internalVolumeBinsAllCompartments);
        // liquidBinsAllCompartmentsOverTime.push_back(liquidBinsAllCompartments);
        // gasBinsAllCompartmentsOverTime.push_back(gasBinsAllCompartments);

        if (mpi_id == MASTER)
        {
            cout << "time = " << time << endl;
            cout << "timeStep = " << timeStep << endl;
            cout << endl;
        }
        

        Time.push_back(time);
        if (mpi_id == MASTER)
        {
            if (time - lastTime >= 0.2)
            {
                //cout << "dumping d50 etc. & particles for time = " << time << endl;
                string timeStr = to_string(floorf(time * 1.0e6) / 1.0e6);// + string("sec"); //moreSigs(time, 2);
                size_t dumpVecSize = timeIdxCount - lastTimeIdxCount + 1;
                arrayOfDouble2D d50ToDump (dumpVecSize);
                vector<double> timeToDump (dumpVecSize);
                copy(d50OverTime.begin()+lastTimeIdxCount, d50OverTime.end(), d50ToDump.begin());
                copy(Time.begin()+lastTimeIdxCount, Time.end(), timeToDump.begin());
                //string timeStr = moreSigs(time, 6); //get time to 2 sig digits, traiing zero removed
                string appendFileName = string("_") + timeStr;
                dumpDiaCSV(timeToDump, d50ToDump, string("d50") + appendFileName);
                dump3DCSV(fAllCompartments, string("particles") + appendFileName);
                dump3DCSV(flAllCompartments, string("liquid") + appendFileName);
                dump3DCSV(fgAllCompartments, string("gas") + appendFileName);
                lastTime = time;
                lastTimeIdxCount = timeIdxCount;
            }            
        }  
        time += timeStep;
        timeIdxCount++;
    }

    size_t nTimeSteps = Time.size();
    if (mpi_id == MASTER)
    {
        cout << endl;
        cout << "Number of time steps = " << nTimeSteps << endl;
        cout << endl;

        //dump values for ratio plots
        dumpDiaCSV(Time, formationThroughAggregationOverTime, string("FormationThroughAggregation"));
        dumpDiaCSV(Time, depletionThroughAggregationOverTime, string("DepletionThroughAggregation"));
        dumpDiaCSV(Time, formationThroughBreakageOverTime, string("FormationThroughBreakage"));
        dumpDiaCSV(Time, depletionThroughBreakageOverTime, string("DepletionThroughBreakage"));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // arrayOfDouble3D dumpedLastValue = *(fAllCompartmentsOverTime.end() - 1);

    // DUMP3DCSV(dumpedLastValue);
    //    string fileName = string("last_f_for_") + to_string(DEMAGGREGATIONKERNELVALUE) + string("_");
    //    fileName += to_string(DEMAGGREGATIONKERNELCONST) + string("_");
    //    fileName += to_string(DEMBREAKAGEKERNELVALUE) + string("_");
    //    fileName += to_string(DEMBREAKAGEKERNELCONST);
    //    dumpTestCSV(dumpedLastValue, fileName, DEMAGGREGATIONKERNELVALUE, DEMAGGREGATIONKERNELCONST, DEMBREAKAGEKERNELVALUE, DEMBREAKAGEKERNELCONST);

    // D10, D50, D90
    //cout << "Begin computing D10, D50, D90" << endl;
    // arrayOfDouble2D d10OverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D d50OverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D d90OverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);

    // arrayOfDouble2D totalVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D totalSolidVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D totalPoreVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D totalLiquidVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);
    // arrayOfDouble2D totalGasVolumeAllCompartmentsOverTime = getArrayOfDouble2D(nTimeSteps, nCompartments);

    // //sieveGrid=[38, 63, 90, 125, 250, 355, 500, 710, 850, 1000, 1400, 2000, 2380, 4000]; %Sieve (in micron)
    // vector<int> sieveGrid;
    // sieveGrid.push_back(38);
    // sieveGrid.push_back(63);
    // sieveGrid.push_back(90);
    // sieveGrid.push_back(125);
    // sieveGrid.push_back(250);
    // sieveGrid.push_back(355);
    // sieveGrid.push_back(500);
    // sieveGrid.push_back(710);
    // sieveGrid.push_back(850);
    // sieveGrid.push_back(1000);
    // sieveGrid.push_back(1400);
    // sieveGrid.push_back(2000);
    // sieveGrid.push_back(2380);
    // sieveGrid.push_back(4000);
    // size_t nSieveGrid = sieveGrid.size();

    // arrayOfDouble3D cumulativeVolumeDistributionAllCompartmentsOverTime = getArrayOfDouble3D(nTimeSteps, nCompartments, nSieveGrid);
    // arrayOfDouble3D totalVolumeGridAllCompartmentsOverTime = getArrayOfDouble3D(nTimeSteps, nCompartments, nSieveGrid);

    // vector<double> totalStuffLeavingOverTime(nTimeSteps, 0.0);
    // vector<double> totalSolidLeavingOverTime(nTimeSteps, 0.0);
    // vector<double> totalLiquidLeavingOverTime(nTimeSteps, 0.0);

    // double distanceBetweenCompartments = granulatorLength / nCompartments;
    // for (size_t n = 1; n < nTimeSteps; n++)
    // {
    //     for (size_t s = 0; s < nFirstSolidBins; s++)
    //         for (size_t ss = 0; ss < nSecondSolidBins; ss++)
    //         {
    //             timeStep = Time[n] - Time[n - 1];
    //             double value1 = fAllCompartmentsOverTime[n][nCompartments - 1][s][ss];
    //             value1 *= (particleAverageVelocity[nCompartments - 1] / distanceBetweenCompartments) * timeStep;
    //             double value2 = value1 * externalVolumeBinsAllCompartmentsOverTime[n][nCompartments - 1][s][ss];
    //             totalStuffLeavingOverTime[n] += value2;

    //             value2 = value1 * (vs[s] + vss[ss]);
    //             totalSolidLeavingOverTime[n] += value2;

    //             value2 = value1 * (externalVolumeBinsAllCompartmentsOverTime[n][nCompartments - 1][s][ss] - vs[s] - vss[ss]);
    //             totalLiquidLeavingOverTime[n] += value2;
    //         }

    //     for (int c = 0; c < nCompartments; c++)
    //     {
    //         for (size_t s = 0; s < nFirstSolidBins; s++)
    //             for (size_t ss = 0; ss < nSecondSolidBins; ss++)
    //             {
    //                 double value1 = fAllCompartmentsOverTime[n][c][s][ss];
    //                 double value2 = value1 * externalVolumeBinsAllCompartmentsOverTime[n][nCompartments - 1][s][ss];
    //                 totalVolumeAllCompartmentsOverTime[n][c] += value2;

    //                 value2 = value1 * (vs[s] + vss[ss]);
    //                 totalSolidVolumeAllCompartmentsOverTime[n][c] += value2;

    //                 value2 = value1 * (externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss] - vs[s] - vss[ss]);
    //                 totalPoreVolumeAllCompartmentsOverTime[n][c] += value2;

    //                 value2 = value1 * liquidBinsAllCompartmentsOverTime[n][c][s][ss];
    //                 totalLiquidVolumeAllCompartmentsOverTime[n][c] += value2;

    //                 value2 = value1 * gasBinsAllCompartmentsOverTime[n][c][s][ss];
    //                 totalGasVolumeAllCompartmentsOverTime[n][c] += value2;
    //             }

        //     arrayOfDouble2D fAll = fAllCompartmentsOverTime[n][c];
        //     externalVolumeBins = externalVolumeBinsAllCompartmentsOverTime[n][c];
        //     arrayOfDouble2D diameter = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);

        //     for (int s = 0; s < nFirstSolidBins; s++)
        //         for (int ss = 0; ss < nSecondSolidBins; ss++)
        //             diameter[s][ss] = cbrt((6 / M_PI) * externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss]) * 1.0e6;

        //     vector<double> totalVolumeGrid(nSieveGrid, 0.0);
        //     for (size_t d = 0; d < nSieveGrid - 1; d++)
        //         for (int s = 0; s < nFirstSolidBins; s++)
        //             for (int ss = 0; ss < nSecondSolidBins; ss++)
        //             {
        //                 if (diameter[s][ss] < sieveGrid[d + 1] && diameter[s][ss] >= sieveGrid[d])
        //                     totalVolumeGrid[d] += fAll[s][ss] * externalVolumeBinsAllCompartmentsOverTime[n][c][s][ss];
        //             }

        //     totalVolumeGridAllCompartmentsOverTime[n][c] = totalVolumeGrid;

        //     double sum = 0.0;
        //     for (size_t d = 0; d < nSieveGrid; d++)
        //         sum += totalVolumeGrid[d];

        //     vector<double> volumeDistribution(nSieveGrid, 0.0);
        //     for (size_t d = 0; d < nSieveGrid; d++)
        //         volumeDistribution[d] = totalVolumeGrid[d] / sum;

        //     vector<double> cumulativeVolumeDistribution(nSieveGrid, 0.0);
        //     sum = 0.0;
        //     for (size_t d = 0; d < nSieveGrid; d++)
        //     {
        //         sum += volumeDistribution[d];
        //         cumulativeVolumeDistribution[d] = sum;
        //     }
        //     cumulativeVolumeDistributionAllCompartmentsOverTime[n][c] = cumulativeVolumeDistribution;

        //     double d10 = 0.1 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);
        //     double d50 = 0.5 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);
        //     double d90 = 0.9 * (sieveGrid[1] / cumulativeVolumeDistribution[0]);

        //     for (size_t d = 1; d < nSieveGrid; d++)
        //     {
        //         double value1 = (sieveGrid[d] - sieveGrid[d - 1]) / (cumulativeVolumeDistribution[d] - cumulativeVolumeDistribution[d - 1]);
        //         double value2 = sieveGrid[d - 1];
        //         if (cumulativeVolumeDistribution[d - 1] < 0.5 && cumulativeVolumeDistribution[d] >= 0.5)
        //         {
        //             double value = 0.5 - cumulativeVolumeDistribution[d - 1];
        //             d50 = value * value1 + value2;
        //         }
        //         if (cumulativeVolumeDistribution[d - 1] < 0.1 && cumulativeVolumeDistribution[d] >= 0.1)
        //         {
        //             double value = 0.1 - cumulativeVolumeDistribution[d - 1];
        //             d10 = value * value1 + value2;
        //         }
        //         if (cumulativeVolumeDistribution[d - 1] < 0.1 && cumulativeVolumeDistribution[d] >= 0.1)
        //         {
        //             double value = 0.9 - cumulativeVolumeDistribution[d - 1];
        //             d90 = value * value1 + value2;
        //         }
        //     }
        //     d10OverTime[n][c] = d10;
        //     d50OverTime[n][c] = d50;
        //     d90OverTime[n][c] = d90;
         //}
    //}
    //cout << "End computing D10, D50, D90" << endl;
    // if (mpi_id == 0)
    // {
    //     string appendFileName = string("_") + coreVal + string("_") + diaVal;
        

    //     dumpDiaCSV(Time, d10OverTime, string("d10") + appendFileName);
    //     dumpDiaCSV(Time, d50OverTime, string("d50") + appendFileName);
    //     dumpDiaCSV(Time, d90OverTime, string("d90") + appendFileName);

    //     arrayOfDouble3D dumpedLastValue = *(fAllCompartmentsOverTime.end() - 1);
    //     DUMP3DCSV(dumpedLastValue);
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    cout << "Testing... my process id = " << mpi_id << endl;
    if (mpi_id == MASTER)
    {
        string endTimeStr(timestring());
        double endTime = MPI_Wtime();
        cout << "That took " << endTime - startTime << " seconds for parallel code" << endl;
        cout << "wall clock time at beginning = " << startTimeStr << endl;
        cout << "wall clock time at end = " << endTimeStr << endl;
        cout << "Code End" << endl;
    }
    MPI_Finalize();

    //return 0;
}
