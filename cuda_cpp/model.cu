#include <vector>
#include <cmath>
#include <float.h>
#include <string>
#include <iostream>
#include <stdio.h>

#include "utility.cuh"
#include "parameterData.h"
#include "liggghtsData.h"

using namespace std;

#define TWOWAYCOUPLING false
#define NTHREADS 256
#define NBLOCKS 256

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

__global__ void initialization_kernel(double *d_vs1, double *d_vss1, struct cudaPitchedPtr d_sMeshXY1, struct cudaPitchedPtr d_ssMeshXY1) 
{
    int idx = threadIdx.x;
    int bdx = blockIdx.x;
    printf("thread id = %d",idx);
    printf("block id = %d", bdx);
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
    // CompartmentIn CompartmentIn;
    // PreviousCompartmentIn prevCompInData;
    // CompartmentOut compartmentOut;

    unsigned int nFirstSolidBins = pData->nFirstSolidBins;
    unsigned int nSecondSolidBins = pData->nSecondSolidBins;

    vector<double> h_vs(nFirstSolidBins, 0.0);
    vector<double> h_vss(nSecondSolidBins, 0.0);
    
    // Bin Initiation
    double fsVolCoeff = pData->fsVolCoeff;
    double fsVolBase = pData->fsVolBase;
    for (size_t i = 0; i < nFirstSolidBins; i++)
        h_vs[i] = fsVolCoeff * pow(fsVolBase, i); // m^3

    double ssVolCoeff = pData->ssVolCoeff;
    double ssVolBase = pData->ssVolBase;
    for (size_t i = 0; i < nSecondSolidBins; i++) 
        h_vss[i] = ssVolCoeff * pow(ssVolBase, i); // m^3

    arrayOfDouble2D diameter = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    for (size_t s = 0; s < nFirstSolidBins; s++)
        for (size_t ss = 0; ss < nSecondSolidBins; ss++)
            diameter[s][ss] = cbrt((6/M_PI) * (h_vs[s] + h_vss[ss]));
    
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

    // arrayOfDouble2D d_fIn = getArrayOfDouble2D(nFirstSolidBins, nSecondSolidBins);
    double **h_fIn = alloc_double_matrix(static_cast<unsigned int>(nFirstSolidBins), static_cast<unsigned int>(nSecondSolidBins));
    for (size_t i = 0; i < particleIn.size(); i++)
        h_fIn[i][i] = particleIn[i];

    // allocation of the matrices reqd for further calculation on the host
    double **h_sMeshXY = alloc_double_matrix(static_cast<unsigned int>(nFirstSolidBins), static_cast<unsigned int>(nSecondSolidBins));
    double **h_ssMeshXY = alloc_double_matrix(static_cast<unsigned int>(nFirstSolidBins), static_cast<unsigned int>(nSecondSolidBins));

    // allocation of memory for the matrices that will be copied onto the device from the host
    double *d_vs = device_alloc_double_vector(h_vs.size());
    double *d_vss = device_alloc_double_vector(h_vss.size());
    struct cudaPitchedPtr d_sMeshXY = device_alloc_double_matrix(static_cast<unsigned int>(nFirstSolidBins), static_cast<unsigned int>(nSecondSolidBins));
    struct cudaPitchedPtr d_ssMeshXY = device_alloc_double_matrix(static_cast<unsigned int>(nFirstSolidBins), static_cast<unsigned int>(nSecondSolidBins));

    copy_double_vector_fromHtoD(d_vs, h_vs.data(), h_vs.size());
    copy_double_vector_fromHtoD(d_vss, h_vss.data(), h_vss.size());

    int nBlocks = NBLOCKS;
    int nThreads = NTHREADS;

    initialization_kernel<<<nBlocks,nThreads>>>(d_vs, d_vss, d_sMeshXY, d_ssMeshXY);
    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch initialization kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cout << "Initialization complete" << endl;

    copy_double_2Dmatrix_fromDtoH(h_sMeshXY, d_sMeshXY, h_vs.size(), h_vs.size(), 1);
    copy_double_2Dmatrix_fromDtoH(h_ssMeshXY, d_ssMeshXY, h_vss.size(), h_vss.size(), 1);

    for (size_t i = 0; i < h_vs.size(); i++)
    {
        for (size_t j = 0; j < h_vss.size(); j++)
            {
                cout << h_sMeshXY[i][j] << " ";
            }
        cout << endl;
    }
}