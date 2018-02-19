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
    d_ssMeshXY[bdx * bix + idx] = d_vss[bix];
    d_sAgg[bdx * bix + idx] = d_vs[idx] + d_vs[bix];
    d_ssAgg[bdx * bix + idx] = d_vss[idx] + d_vss[bix];
    d_sAggregationCheck[bdx * bix + idx] = d_sAgg[bdx * bix + idx] <= d_vs[bdx - 1] ? 1 : 0;
    d_ssAggregationCheck[bdx * bix + idx] = d_ssAgg[bdx * bix + idx] <= d_vss[bdx - 1] ? 1 : 0;
    d_sLow [bdx * bix + idx] = d_sMeshXY[bdx * bix + idx];
    d_ssLow[bdx * bix + idx] = d_ssMeshXY[bdx * bix + idx];
    __syncthreads();
    if (bix < bdx -1)
    {
        d_sHigh[bdx * bix + idx] = d_sMeshXY[bdx * (bix + 1) + idx];
        d_ssHigh[bdx * bix + idx] = d_sMeshXY[bdx * (bix + 1) + idx];
    }
    d_sHigh[bdx * (bdx -1) + idx] = 0.0;
    d_ssHigh[bdx * (bdx -1) + idx] = 0.0;
    d_sLoc[bdx * bix + idx] = floor(log(d_sAgg[bdx * bix + idx] / fsVolCoeff) / log(fsVolBase) + 1);
    d_ssLoc[bdx * bix + idx] = floor(log(d_ssAgg[bdx * bix + idx] / ssVolCoeff) / log(ssVolBase) + 1);
    d_sInd[bdx * bix + idx] = (idx <= bix) ? (bix + 1) : (idx + 1);
    d_ssInd[bdx * bix + idx] = (idx <= bix) ? (bix + 1) : (idx + 1);
    __syncthreads();
    double value = d_vs[idx] - d_vs[bix];
    double value1 = d_vss[idx] - d_vss[bix];
    d_sBreak[bdx * bix + idx] = value < 0.0 ? 0.0 : value;
    d_ssBreak[bdx * bix + idx] = value1 < 0.0 ? 0.0 : value1;
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
}

__global__ void launchCompartment(CompartmentIn *d_compartmentIn, PreviousCompartmentIn *d_prevCompInData, CompartmentOut *d_compartmentOut, CompartmentDEMIn *d_compartmentDEMIn,
                                  CompartmentVar *d_compVar, AggregationCompVar *d_aggCompVar, BreakageCompVar *d_brCompVar, double time, double timeStep, double initialTime,
                                  double *d_formationThroughAggregation, double *d_depletionThroughAggregation, double *d_formationThroughBreakage, double *d_depletionThroughBreakage,
                                  double *d_fAllCompartments, double *d_flAllCompartments, double *d_fgAllCompartments, double *d_liquidAdditionRateAllCompartments,
                                  size_t size2D, size_t size3D, size_t size4D, double *d_fIn, double initPorosity, double demTimeStep)
{
    int bix = blockIdx.x;
    int biy = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;

    int tix = threadIdx.x;
    // int tiy = threadIdx.y;

    // int idx = bix * bdx * bdy + tiy * bdx + tix;
    int ddx = bix * bdx + tix;
    __syncthreads();
    //if (tiy == 0)

    d_compartmentIn->fAll[tix] = d_fAllCompartments[tix];
    d_compartmentIn->fLiquid[tix] = d_flAllCompartments[tix];
    d_compartmentIn->fGas[tix] = d_fgAllCompartments[tix];
    d_compartmentIn->liquidAdditionRate = d_liquidAdditionRateAllCompartments[tix];

    if (bix == 0)
    {
        d_prevCompInData->fAllComingIn[tix] = d_fIn[tix];
        double value = initPorosity * timeStep;
        d_prevCompInData->fgComingIn[tix] = d_fIn[tix] * (d_compartmentIn->vs[tix % 16] + d_compartmentIn->vss[tix % 16]) * value;
    }

    else
    {
        d_prevCompInData->fAllPreviousCompartment[tix] = d_fAllCompartments[(bix - 1) * bdx + tix];
        d_prevCompInData->flPreviousCompartment[tix] = d_flAllCompartments[(bix - 1) * bdx + tix];
        d_prevCompInData->fgPreviousCompartment[tix] = d_fgAllCompartments[(bix - 1) * bdx + tix];
    }

    if (fabs(d_compartmentIn->fAll[tix] > 1e-16))
    {
        d_compartmentOut->liquidBins[tix] = d_compartmentIn->fLiquid[tix] / d_compartmentIn->fAll[tix];
        d_compartmentOut->gasBins[tix] = d_compartmentIn->fGas[tix] / d_compartmentIn->fAll[tix];
    }
    else
    {
        d_compartmentOut->liquidBins[tix] = 0.0;
        d_compartmentOut->gasBins[tix] = 0.0;
    }

    printf("d_compartmentOut->liquidBins  = %f \n", d_compartmentOut->liquidBins[tix]);
    dim3 compKernel_nblocks, compKernel_nthreads;
    compKernel_nblocks = dim3(1,1,1);
    compKernel_nthreads = dim3(size2D, size2D,1);

    if (tix == 0)
    {
        performAggCalculations<<<(1,1,1),(256,256,1)>>>(d_prevCompInData, d_compartmentIn, d_compartmentDEMIn, d_compartmentOut, d_compVar, d_aggCompVar, time, timeStep, initialTime, demTimeStep);
        cudaDeviceSynchronize();
        cudaError_t err = cudaSuccess;
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        }
        printf("comp done \n");
    }

    cudaDeviceSynchronize();
}


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
    CompartmentIn compartmentIn, *d_compartmentIn;
    PreviousCompartmentIn prevCompInData, *d_prevCompInData;
    CompartmentOut compartmentOut, *d_compartmentOut;
    CompartmentDEMIn compartmentDEMIn, *d_compartmentDEMIn;

    unsigned int nFirstSolidBins = pData->nFirstSolidBins;
    unsigned int nSecondSolidBins = pData->nSecondSolidBins;

    size_t size1D = nFirstSolidBins;
    size_t size2D = nFirstSolidBins * nSecondSolidBins;
    size_t size3D = nFirstSolidBins * nSecondSolidBins * nCompartments;
    size_t size4D = nFirstSolidBins * nFirstSolidBins * nSecondSolidBins * nSecondSolidBins;

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
    for (size_t i = 0; i < size1D; i++)
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

    vector<int> h_sCheckB(size2D, 0);
    vector<int> h_ssCheckB(size2D, 0);

    vector<int> h_sIndB(size2D, 0.0);
    vector<int> h_ssIndB(size2D, 0.0);

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
    cout << "Initialization complete" << endl;

    // copy back data required for the compartment calculations
     
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

    copy_integer_vector_fromDtoH(h_sCheckB.data(), d_sCheckB, size2D);
    copy_integer_vector_fromDtoH(h_ssCheckB.data(), d_ssCheckB, size2D);

    copy_integer_vector_fromDtoH(h_sIndB.data(), d_sIndB, size2D);
    copy_integer_vector_fromDtoH(h_ssIndB.data(), d_ssIndB, size2D);

    cudaDeviceSynchronize();

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

    compartmentDEMIn.velocityCol = colVelocity.data();

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
    compartmentDEMIn.uCriticalCol = uCritical;
    // cout << "Critical velocity for agg is " << uCritical << endl;

        int veloSize = colVelocity.size();
    for (int i = 0; i < veloSize; i++)
        sumVelo += colVelocity[i];

    unsigned int nDEMBins = pData->nDEMBins;
    double averageVelocity = sumVelo / nDEMBins;
    double stdDevVelocity = 0.0;
    double varianceVelocity = 0.0;

    for (int i = 0; i < veloSize; ++i)
        varianceVelocity += pow((colVelocity[i] - averageVelocity), 2) / 10;

    stdDevVelocity = sqrt(varianceVelocity);
    //double intVelocity = 0.0;
    vector<double> colProbablityOfVelocity(veloSize, 0.0);
    for (int i = 0; i < veloSize; i++)
    {
        colProbablityOfVelocity[i] = (1 / (colVelocity[i] * sqrt(2 * M_PI) * stdDevVelocity)) * exp(-((log(colVelocity[i]) - averageVelocity) / (2 * pow(varianceVelocity, 2))));
        // cout << "Probability at " << velocity[i] << "is " << probablityOfVelocity[i] << endl;
    }

    

    compartmentDEMIn.colProbability = colProbablityOfVelocity.data();

    // vector<double> impactFrequency = DEMImpactData;
    // for (int s = 0; s < nFirstSolidBins; s++)
    //     for (int ss = 0; ss < nSecondSolidBins; ss++)
    //         for (int i = 0; i < nDEMBins; i++)
    //         {
    //             if (fAll[s][ss] > 0.0)
    //                 impactFrequency[i] = (DEMImpactData[i] * timeStep) / demTimeStep;
    //         }

    double critStDefNum = pData->critStDefNum;
    double initPorosity = pData->initPorosity;
    // double Ubreak = (2 * critStDefNum / solDensity) * (9 / 8.0) * (pow((1 - initPorosity), 2) / pow(initPorosity, 2)) * (9 / 16.0) * (bindVisc / compartmentIn->diameter[0]);

    int size1 = velocity.size();
    double sum = 0.0;

    for (int i = 0; i < size1; i++)
        sum += velocity[i];

    double averageVelocityBr = sum / nDEMBins;
    double stdDevVelocityBr = 0.0;
    double varianceVelocityBr = 0.0;
    for (int i = 0; i < size1; ++i)
    {
        varianceVelocityBr += pow((velocity[i] - averageVelocityBr), 2) / 10;
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
        }
    }

    compartmentDEMIn.brProbability = breakageProbablityOfVelocity.data();

    DUMP2D(DEMCollisionData);
    DUMP(DEMDiameter);
    DUMP(DEMImpactData);
    DUMP(velocity);

    //Initialize DEM data for compartment
    compartmentDEMIn.DEMDiameter = DEMDiameter.data();
    compartmentDEMIn.DEMCollisionData = linearize2DVector(DEMCollisionData).data();
    compartmentDEMIn.DEMImpactData = DEMImpactData.data();

    vector<double> liquidAdditionRateAllCompartments(nCompartments, 0.0);
    double liqSolidRatio = pData->liqSolidRatio;
    double throughput = pData->throughput;
    double liqDensity = pData->liqDensity;
    double liquidAddRate = (liqSolidRatio * throughput) / (liqDensity * 3600);
    liquidAdditionRateAllCompartments[0] = liquidAddRate;
    
    vector<double> h_fAllCompartmentsOverTime(size4D, 0.0);
    vector<double> h_externalVolumeBinsAllCompartmentsOverTime(size4D, 0.0);
    vector<double> h_internalVolumeBinsAllCompartmentsOverTime(size4D, 0.0);
    vector<double> h_liquidBinsAllCompartmentsOverTime(size4D, 0.0);
    vector<double> h_gasBinsAllCompartmentsOverTime(size4D, 0.0);

    double granulatorLength = pData->granulatorLength;
    double partticleResTime = pData->partticleResTime;
    double particleAveVelo = granulatorLength /  partticleResTime;
    vector<double> particleAverageVelocity(nCompartments, particleAveVelo);


    //Initialize input data for compartment

    compartmentIn.vs = h_vs.data();
    compartmentIn.vss = h_vss.data();

    compartmentIn.diameter = diameter.data();

    compartmentIn.sMeshXY = h_sMeshXY.data();
    compartmentIn.ssMeshXY = h_ssMeshXY.data();

    compartmentIn.sAggregationCheck = h_sAggregationCheck.data();
    compartmentIn.ssAggregationCheck = h_ssAggregationCheck.data();

    compartmentIn.sLow = h_sLow.data();
    compartmentIn.sHigh = h_sHigh.data();

    compartmentIn.ssLow = h_ssLow.data();
    compartmentIn.ssHigh = h_ssHigh.data();

    compartmentIn.sInd = h_sInd.data();
    compartmentIn.ssInd = h_ssInd.data();

    compartmentIn.sCheckB = h_sCheckB.data();
    compartmentIn.ssCheckB = h_ssCheckB.data();

    compartmentIn.sIndB = h_sIndB.data();
    compartmentIn.ssIndB = h_ssIndB.data();

    compartmentIn.fAll = alloc_double_vector(size2D);
    compartmentIn.fLiquid = alloc_double_vector(size2D);
    compartmentIn.fGas = alloc_double_vector(size2D);

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

    vector<double> d10OverTime(size2D, 0.0);
    vector<double> d50OverTime(size2D, 0.0);
    vector<double> d90OverTime(size2D, 0.0);

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

    vector<double> formationThroughAggregationOverTime;
    vector<double> depletionThroughAggregationOverTime;
    vector<double> formationThroughBreakageOverTime;
    vector<double> depletionThroughBreakageOverTime;
    cout << "time" << endl;

    

    vector<double> temp(size2D, 0);
    vector<double> temp4(size4D, 0);
    prevCompInData.fAllPreviousCompartment = alloc_double_vector(size2D);
    prevCompInData.flPreviousCompartment = alloc_double_vector(size2D);
    prevCompInData.fgPreviousCompartment = alloc_double_vector(size2D);
    prevCompInData.fAllComingIn = alloc_double_vector(size2D);
    prevCompInData.fgComingIn = alloc_double_vector(size2D);


    // defining compartment varibale pointers

    CompartmentVar compVar, *d_compVar;
    AggregationCompVar aggCompVar, *d_aggCompVar;
    BreakageCompVar brCompVar, *d_brCompVar;

    // Allocating the arrays for each pointer of the compVar
    compVar.internalLiquid = alloc_double_vector(size2D);
    compVar.externalLiquid = alloc_double_vector(size2D);
    compVar.externalLiquidContent = alloc_double_vector(size2D);
    compVar.volumeBins = alloc_double_vector(size2D);
    compVar.aggregationRate = alloc_double_vector(size4D);
    compVar.breakageRate = alloc_double_vector(size4D);
    compVar.particleMovement = alloc_double_vector(size2D);
    compVar.liquidMovement = alloc_double_vector(size2D);
    compVar.gasMovement = alloc_double_vector(size2D);
    compVar.liquidBins = alloc_double_vector(size2D);
    compVar.gasBins = alloc_double_vector(size2D);

    // Allocating compVar pointers on the GPU device
    
    d_compVar->internalLiquid = device_alloc_double_vector(size2D);
    d_compVar->externalLiquid = device_alloc_double_vector(size2D);
    d_compVar->externalLiquidContent = device_alloc_double_vector(size2D);
    d_compVar->volumeBins = device_alloc_double_vector(size2D);
    d_compVar->aggregationRate = device_alloc_double_vector(size4D);
    d_compVar->breakageRate = device_alloc_double_vector(size4D);
    d_compVar->particleMovement = device_alloc_double_vector(size2D);
    d_compVar->liquidMovement = device_alloc_double_vector(size2D);
    d_compVar->gasMovement = device_alloc_double_vector(size2D);
    d_compVar->liquidBins = device_alloc_double_vector(size2D);
    d_compVar->gasBins = device_alloc_double_vector(size2D);

    // Copying data into the device from the host

    copy_double_vector_fromHtoD(d_compVar->internalLiquid, compVar.internalLiquid, size2D);
    copy_double_vector_fromHtoD(d_compVar->externalLiquid, compVar.externalLiquid, size2D);
    copy_double_vector_fromHtoD(d_compVar->externalLiquidContent, compVar.externalLiquidContent, size2D);
    copy_double_vector_fromHtoD(d_compVar->volumeBins, compVar.volumeBins, size2D);
    copy_double_vector_fromHtoD(d_compVar->aggregationRate, compVar.aggregationRate, size4D);
    copy_double_vector_fromHtoD(d_compVar->breakageRate, compVar.breakageRate, size4D);
    copy_double_vector_fromHtoD(d_compVar->particleMovement, compVar.particleMovement, size2D);
    copy_double_vector_fromHtoD(d_compVar->liquidMovement, compVar.liquidMovement, size2D);
    copy_double_vector_fromHtoD(d_compVar->gasMovement, compVar.gasMovement, size2D);
    copy_double_vector_fromHtoD(d_compVar->liquidBins, compVar.liquidBins, size2D);
    copy_double_vector_fromHtoD(d_compVar->gasBins, compVar.gasBins, size2D);

    // compartmentOut

    compartmentOut.dfAlldt = alloc_double_vector(size2D);
    compartmentOut.dfLiquiddt = alloc_double_vector(size2D);
    compartmentOut.dfGasdt = alloc_double_vector(size2D);
    compartmentOut.liquidBins = alloc_double_vector(size2D);
    compartmentOut.gasBins = alloc_double_vector(size2D);
    compartmentOut.internalVolumeBins = alloc_double_vector(size2D);
    compartmentOut.externalVolumeBins = alloc_double_vector(size2D);
    compartmentOut.aggregationKernel = alloc_double_vector(size4D);
    compartmentOut.breakageKernel = alloc_double_vector(size4D);
   
    // Allocating the compartment out variables for the device

    d_compartmentOut->dfAlldt = device_alloc_double_vector(size2D);
    d_compartmentOut->dfLiquiddt = device_alloc_double_vector(size2D);
    d_compartmentOut->dfGasdt = device_alloc_double_vector(size2D);
    d_compartmentOut->liquidBins = device_alloc_double_vector(size2D);
    d_compartmentOut->gasBins = device_alloc_double_vector(size2D);
    d_compartmentOut->internalVolumeBins = device_alloc_double_vector(size2D);
    d_compartmentOut->externalVolumeBins = device_alloc_double_vector(size2D);
    d_compartmentOut->aggregationKernel = device_alloc_double_vector(size4D);
    d_compartmentOut->breakageKernel = device_alloc_double_vector(size4D);

    copy_double_vector_fromHtoD(d_compartmentOut->dfAlldt, compartmentOut.dfAlldt, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->dfLiquiddt, compartmentOut.dfLiquiddt, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->dfGasdt, compartmentOut.dfGasdt, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->liquidBins, compartmentOut.liquidBins, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->gasBins, compartmentOut.gasBins, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->internalVolumeBins, compartmentOut.internalVolumeBins, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->externalVolumeBins, compartmentOut.externalVolumeBins, size2D);
    copy_double_vector_fromHtoD(d_compartmentOut->aggregationKernel, compartmentOut.aggregationKernel, size4D);
    copy_double_vector_fromHtoD(d_compartmentOut->breakageKernel, compartmentOut.breakageKernel, size4D);

    compartmentDEMIn.colEfficiency = alloc_double_vector(size4D);

    // Allocating the compartmentDEMIn pointers to the device and copying the data

    
    d_compartmentDEMIn->DEMDiameter = device_alloc_double_vector(size1D);
    d_compartmentDEMIn->DEMCollisionData = device_alloc_double_vector(size2D);
    d_compartmentDEMIn->DEMImpactData = device_alloc_double_vector(size1D);
    d_compartmentDEMIn->colProbability = device_alloc_double_vector(size1D);
    d_compartmentDEMIn->brProbability = device_alloc_double_vector(size1D);
    d_compartmentDEMIn->colEfficiency = device_alloc_double_vector(size4D);
    d_compartmentDEMIn->colFrequency = device_alloc_double_vector(size4D);
    d_compartmentDEMIn->velocityCol = device_alloc_double_vector(size2D);

    copy_double_vector_fromHtoD(d_compartmentDEMIn->DEMDiameter, compartmentDEMIn.DEMDiameter, size1D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->DEMCollisionData, compartmentDEMIn.DEMCollisionData, size2D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->DEMImpactData, compartmentDEMIn.DEMImpactData, size1D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->colProbability, compartmentDEMIn.colProbability, size1D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->brProbability, compartmentDEMIn.brProbability, size1D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->colEfficiency, compartmentDEMIn.colEfficiency, size4D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->colFrequency, compartmentDEMIn.colFrequency, size4D);
    copy_double_vector_fromHtoD(d_compartmentDEMIn->velocityCol, compartmentDEMIn.velocityCol, size2D);


    // Defining aggregration parameters and also allocating them on the deivce and copying the data
    double aggKernelConst = pData->aggKernelConst;
    aggCompVar.aggKernelConst = aggKernelConst;
    aggCompVar.depletionOfGasThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.depletionOfLiquidThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.birthThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.firstSolidBirthThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.secondSolidBirthThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.liquidBirthThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.gasBirthThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.firstSolidVolumeThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.secondSolidVolumeThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.birthAggLowLow = alloc_double_vector(size2D);
    aggCompVar.birthAggHighHigh = alloc_double_vector(size2D);
    aggCompVar.birthAggLowHigh = alloc_double_vector(size2D);
    aggCompVar.birthAggHighLow = alloc_double_vector(size2D);
    aggCompVar.birthAggLowLowLiq = alloc_double_vector(size2D);
    aggCompVar.birthAggHighHighLiq = alloc_double_vector(size2D);
    aggCompVar.birthAggLowHighLiq = alloc_double_vector(size2D);
    aggCompVar.birthAggHighLowLiq = alloc_double_vector(size2D);
    aggCompVar.birthAggLowLowGas = alloc_double_vector(size2D);
    aggCompVar.birthAggHighHighGas = alloc_double_vector(size2D);
    aggCompVar.birthAggLowHighGas = alloc_double_vector(size2D);
    aggCompVar.birthAggHighLowGas = alloc_double_vector(size2D);
    aggCompVar.formationThroughAggregationCA = alloc_double_vector(size2D);
    aggCompVar.formationOfLiquidThroughAggregationCA = alloc_double_vector(size2D);
    aggCompVar.formationOfGasThroughAggregationCA = alloc_double_vector(size2D);
    aggCompVar.depletionThroughAggregation = alloc_double_vector(size2D);
    aggCompVar.depletionThroughBreakage = alloc_double_vector(size2D);
    aggCompVar.depletionOfGasThroughBreakage = alloc_double_vector(size2D);
    aggCompVar.depletionOfLiquidthroughBreakage = alloc_double_vector(size2D);

    d_aggCompVar->aggKernelConst = device_alloc_double_vector(1);
    d_aggCompVar->depletionOfGasThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->depletionOfLiquidThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->birthThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->firstSolidBirthThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->secondSolidBirthThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->liquidBirthThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->gasBirthThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->firstSolidVolumeThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->secondSolidVolumeThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowLow = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighHigh = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowHigh = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighLow = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowLowLiq = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighHighLiq = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowHighLiq = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighLowLiq = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowLowGas = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighHighGas = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggLowHighGas = device_alloc_double_vector(size2D);
    d_aggCompVar->birthAggHighLowGas = device_alloc_double_vector(size2D);
    d_aggCompVar->formationThroughAggregationCA = device_alloc_double_vector(size2D);
    d_aggCompVar->formationOfLiquidThroughAggregationCA = device_alloc_double_vector(size2D);
    d_aggCompVar->formationOfGasThroughAggregationCA = device_alloc_double_vector(size2D);
    d_aggCompVar->depletionThroughAggregation = device_alloc_double_vector(size2D);
    d_aggCompVar->depletionThroughBreakage = device_alloc_double_vector(size2D);
    d_aggCompVar->depletionOfGasThroughBreakage = device_alloc_double_vector(size2D);
    d_aggCompVar->depletionOfLiquidthroughBreakage = device_alloc_double_vector(size2D);

    copy_double_vector_fromHtoD(d_aggCompVar->aggKernel, aggCompVar.a, 1);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionOfGasThroughAggregation, aggCompVar.depletionOfGasThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionOfLiquidThroughAggregation, aggCompVar.depletionOfLiquidThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthThroughAggregation, aggCompVar.birthThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->firstSolidBirthThroughAggregation, aggCompVar.firstSolidBirthThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->secondSolidBirthThroughAggregation, aggCompVar.secondSolidBirthThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->liquidBirthThroughAggregation, aggCompVar.liquidBirthThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->gasBirthThroughAggregation, aggCompVar.gasBirthThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->firstSolidVolumeThroughAggregation, aggCompVar.firstSolidVolumeThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->secondSolidVolumeThroughAggregation, aggCompVar.secondSolidVolumeThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowLow, aggCompVar.birthAggLowLow, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighHigh, aggCompVar.birthAggHighHigh, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowHigh, aggCompVar.birthAggLowHigh, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighLow, aggCompVar.birthAggHighLow, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowLowLiq, aggCompVar.birthAggLowLowLiq, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighHighLiq, aggCompVar.birthAggHighHighLiq, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowHighLiq, aggCompVar.birthAggLowHighLiq, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighLowLiq, aggCompVar.birthAggHighLowLiq, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowLowGas, aggCompVar.birthAggLowLowGas, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighHighGas, aggCompVar.birthAggHighHighGas, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggLowHighGas, aggCompVar.birthAggLowHighGas, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->birthAggHighLowGas, aggCompVar.birthAggHighLowGas, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->formationThroughAggregationCA, aggCompVar.formationThroughAggregationCA, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->formationOfLiquidThroughAggregationCA, aggCompVar.formationOfLiquidThroughAggregationCA, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->formationOfGasThroughAggregationCA, aggCompVar.formationOfGasThroughAggregationCA, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionThroughAggregation, aggCompVar.depletionThroughAggregation, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionThroughBreakage, aggCompVar.depletionThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionOfGasThroughBreakage, aggCompVar.depletionOfGasThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_aggCompVar->depletionOfLiquidthroughBreakage, aggCompVar.depletionOfLiquidthroughBreakage, size2D);


    // Allocating and copying device pointers for breakage

    brCompVar.birthThroughBreakage1 = alloc_double_vector(size2D);
    brCompVar.birthThroughBreakage2 = alloc_double_vector(size2D);
    brCompVar.firstSolidBirthThroughBreakage = alloc_double_vector(size2D);
    brCompVar.secondSolidBirthThroughBreakage = alloc_double_vector(size2D);
    brCompVar.liquidBirthThroughBreakage1 = alloc_double_vector(size2D);
    brCompVar.gasBirthThroughBreakage1 = alloc_double_vector(size2D);
    brCompVar.liquidBirthThroughBreakage2 = alloc_double_vector(size2D);
    brCompVar.gasBirthThroughBreakage2 = alloc_double_vector(size2D);
    brCompVar.firstSolidVolumeThroughBreakage = alloc_double_vector(size2D);
    brCompVar.secondSolidVolumeThroughBreakage = alloc_double_vector(size2D);
    brCompVar.fractionBreakage00 = alloc_double_vector(size2D);
    brCompVar.fractionBreakage01 = alloc_double_vector(size2D);
    brCompVar.fractionBreakage10 = alloc_double_vector(size2D);
    brCompVar.fractionBreakage11 = alloc_double_vector(size2D);
    brCompVar.formationThroughBreakageCA = alloc_double_vector(size2D);
    brCompVar.formationOfLiquidThroughBreakageCA = alloc_double_vector(size2D);
    brCompVar.formationOfGasThroughBreakageCA = alloc_double_vector(size2D);
    brCompVar.transferThroughLiquidAddition = alloc_double_vector(size2D);
    brCompVar.transferThroughConsolidation = alloc_double_vector(size2D);


    d_brCompVar->birthThroughBreakage1 = device_alloc_double_vector(size2D);
    d_brCompVar->birthThroughBreakage2 = device_alloc_double_vector(size2D);
    d_brCompVar->firstSolidBirthThroughBreakage = device_alloc_double_vector(size2D);
    d_brCompVar->secondSolidBirthThroughBreakage = device_alloc_double_vector(size2D);
    d_brCompVar->liquidBirthThroughBreakage1 = device_alloc_double_vector(size2D);
    d_brCompVar->gasBirthThroughBreakage1 = device_alloc_double_vector(size2D);
    d_brCompVar->liquidBirthThroughBreakage2 = device_alloc_double_vector(size2D);
    d_brCompVar->gasBirthThroughBreakage2 = device_alloc_double_vector(size2D);
    d_brCompVar->firstSolidVolumeThroughBreakage = device_alloc_double_vector(size2D);
    d_brCompVar->secondSolidVolumeThroughBreakage = device_alloc_double_vector(size2D);
    d_brCompVar->fractionBreakage00 = device_alloc_double_vector(size2D);
    d_brCompVar->fractionBreakage01 = device_alloc_double_vector(size2D);
    d_brCompVar->fractionBreakage10 = device_alloc_double_vector(size2D);
    d_brCompVar->fractionBreakage11 = device_alloc_double_vector(size2D);
    d_brCompVar->formationThroughBreakageCA = device_alloc_double_vector(size2D);
    d_brCompVar->formationOfLiquidThroughBreakageCA = device_alloc_double_vector(size2D);
    d_brCompVar->formationOfGasThroughBreakageCA = device_alloc_double_vector(size2D);
    d_brCompVar->transferThroughLiquidAddition = device_alloc_double_vector(size2D);
    d_brCompVar->transferThroughConsolidation = device_alloc_double_vector(size2D);

    copy_double_vector_fromHtoD(d_brCompVar->birthThroughBreakage1, brCompVar.birthThroughBreakage1, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->birthThroughBreakage2, brCompVar.birthThroughBreakage2, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->firstSolidBirthThroughBreakage, brCompVar.firstSolidBirthThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->secondSolidBirthThroughBreakage, brCompVar.secondSolidBirthThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->liquidBirthThroughBreakage1, brCompVar.liquidBirthThroughBreakage1, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->gasBirthThroughBreakage1, brCompVar.gasBirthThroughBreakage1, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->liquidBirthThroughBreakage2, brCompVar.liquidBirthThroughBreakage2, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->gasBirthThroughBreakage2, brCompVar.gasBirthThroughBreakage2, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->firstSolidVolumeThroughBreakage, brCompVar.firstSolidVolumeThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->secondSolidVolumeThroughBreakage, brCompVar.secondSolidVolumeThroughBreakage, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->fractionBreakage00, brCompVar.fractionBreakage00, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->fractionBreakage01, brCompVar.fractionBreakage01, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->fractionBreakage10, brCompVar.fractionBreakage10, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->fractionBreakage11, brCompVar.fractionBreakage11, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->formationThroughBreakageCA, brCompVar.formationThroughBreakageCA, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->formationOfLiquidThroughBreakageCA, brCompVar.formationOfLiquidThroughBreakageCA, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->formationOfGasThroughBreakageCA, brCompVar.formationOfGasThroughBreakageCA, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->transferThroughLiquidAddition, brCompVar.transferThroughLiquidAddition, size2D);
    copy_double_vector_fromHtoD(d_brCompVar->transferThroughConsolidation, brCompVar.transferThroughConsolidation, size2D);

    // // allocating memory for structures used for compartment calculations

    // err = cudaMalloc(&d_compartmentIn, sizeof(CompartmentIn));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : CompartmentIn (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_prevCompInData, sizeof(PreviousCompartmentIn));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : prevCompInData (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_compartmentDEMIn, sizeof(CompartmentDEMIn));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : compartmentDEMIn (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_compVar, sizeof(CompartmentVar));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : CompartmentVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_aggCompVar, sizeof(AggregationCompVar));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : AggregationCompVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_brCompVar, sizeof(BreakageCompVar));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : BreakageCompVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMalloc(&d_compartmentOut, sizeof(CompartmentOut));
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to  cudaMalloc : d_compartmentOut (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // // copying data to the allocated GPU

    // cudaMemcpy(d_compartmentIn, &compartmentIn, sizeof(CompartmentIn), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : CompartmentIn (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_prevCompInData, &prevCompInData, sizeof(PreviousCompartmentIn), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : PreviousCompartmentIn (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_compartmentDEMIn, &compartmentDEMIn, sizeof(CompartmentDEMIn), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : CompartmentDEMIn (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_aggCompVar, &aggCompVar, sizeof(AggregationCompVar), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : AggregationCompVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_compVar, &compVar, sizeof(CompartmentVar), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : CompartmentVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_brCompVar, &brCompVar, sizeof(BreakageCompVar), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : BreakageCompVar (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // cudaMemcpy(d_compartmentOut, &compartmentOut, sizeof(CompartmentOut), cudaMemcpyHostToDevice);
    // err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr, "Failed to cudaMemcpy : compartmentOut (error code %s)!\n", cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    vector<double> h_formationThroughAggregation(nCompartments, 0.0);
    vector<double> h_depletionThroughAggregation(nCompartments, 0.0);
    vector<double> h_formationThroughBreakage(nCompartments, 0.0);
    vector<double> h_depletionThroughBreakage(nCompartments, 0.0);

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

    CompartmentOut *h_results;
    CompartmentDEMIn *h_demr;
    h_results = (CompartmentOut *)malloc(sizeof(CompartmentOut));

    h_demr = (CompartmentDEMIn *)malloc(sizeof(CompartmentDEMIn));

    // dim3 compKernel_nblocks, compKernel_nthreads;
    // compKernel_nblocks = dim3(nCompartments,1,1);
    // compKernel_nthreads = dim3(size2D, size2D,1);
    int compKernel_nblocks = 16;
    int compKernel_nthreads = size2D * size2D;
    while (time <= finalTime)
    {
        copy_double_vector_fromHtoD(d_fAllCompartments, h_fAllCompartments.data(), size3D);
        copy_double_vector_fromHtoD(d_flAllCompartments, h_flAllCompartments.data(), size3D);
        copy_double_vector_fromHtoD(d_fgAllCompartments, h_fgAllCompartments.data(), size3D);
        
        launchCompartment<<<16,256>>>(d_compartmentIn, d_prevCompInData, d_compartmentOut, d_compartmentDEMIn, d_compVar, d_aggCompVar, d_brCompVar,
                                                    time, timeStep, stod(timeVal), d_formationThroughAggregation, d_depletionThroughAggregation,d_formationThroughBreakage, 
                                                    d_depletionThroughBreakage, d_fAllCompartments, d_flAllCompartments, d_fgAllCompartments, 
                                                    d_liquidAdditionRateAllCompartments, size2D, size3D, size4D, d_fIn, initPorosity, demTimeStep);

        cudaDeviceSynchronize();
        err = cudaSuccess;
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch launchCompartment kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        cout << "Compartment started " << endl;
        err = cudaMemcpy(h_results, d_compartmentOut, sizeof(CompartmentOut), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMemcpy : CompartmentOut fron D to H (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(h_demr, d_compartmentDEMIn, sizeof(CompartmentDEMIn), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMemcpy : CompartmentDEMIn D to Hmake (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        time = finalTime + 5.0;
    }
    // vector<double> h(size4D, 0.0);
    // for (int i = 0; i < size4D; i++)
    // {
    //     cout << "At i = " << i << "  kernel = " << h_results->aggregationKernel[i] << endl;
    // }
    cudaFree(d_vs);
    cudaFree(d_vss);
    // cudaFree(d_sMeshXY);
    // cudaFree(d_ssMeshXY);
}