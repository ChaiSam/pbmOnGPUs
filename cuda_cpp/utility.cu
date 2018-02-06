#include <cmath>
#include <float.h>
#include <sstream>
#include <iomanip>
#include <dirent.h>
#include <cstring>
#include <stdio.h>
#include "utility.cuh"

using namespace std;

// make int and double vectors for the host

arrayOfInt2D getArrayOfInt2D(int n, int m, int val)
{
    arrayOfInt2D arrayInt2D(n, vector<int>(m, val));
    return arrayInt2D;
}

arrayOfInt3D getarrayOfInt3D(int n, int m, int p, int val)
{
    arrayOfInt3D arrayInt3D(n, vector<vector<int>>(m, vector<int>(p, val)));
    return arrayInt3D;
}

arrayOfInt4D getArrayOfInt4D(int n, int m, int p, int q, int val)
{
    arrayOfInt4D arrayInt4D(n, vector<vector<vector<int>>>(m, vector<vector<int>>(p, vector<int>(q, val))));
    return arrayInt4D;
}

arrayOfDouble2D getArrayOfDouble2D(int n, int m, double val)
{
    arrayOfDouble2D arrayDouble2D(n, vector<double>(m, val));
    return arrayDouble2D;
}

arrayOfDouble3D getArrayOfDouble3D(int n, int m, int p, double val)
{
    arrayOfDouble3D arrayDouble3D(n, vector<vector<double>>(m, vector<double>(p, val)));
    return arrayDouble3D;
}

arrayOfDouble4D getArrayOfDouble4D(int n, int m, int p, int q, double val)
{
    arrayOfDouble4D arrayDouble4D(n, vector<vector<vector<double>>>(m, vector<vector<double>>(p, vector<double>(q, val))));
    return arrayDouble4D;
}

// linearizing arrays

vector<double> linearize2DVector(arrayOfDouble2D array2D)
{
    // vector<double> data;
    size_t dim1 = array2D.size();
    size_t dim2 = array2D[0].size();
    vector<double> data(dim1 * dim2, 0.0);
    for (size_t d1 = 0; d1 < dim1; d1++)
        for (size_t d2 = 0; d2 < dim2; d2++)
                data[d1 * dim2 + d2] = array2D[d1][d2];

    return data;
}

vector<double> linearize3DVector(arrayOfDouble3D array3D)
{
    // vector<double> data;
    size_t dim1 = array3D.size();
    size_t dim2 = array3D[0].size();
    size_t dim3 = array3D[0][0].size();
    vector<double> data(dim1 * dim2 * dim3, 0.0);
    for (size_t d1 = 0; d1 < dim1; d1++)
        for (size_t d2 = 0; d2 < dim2; d2++)
            for (size_t d3 = 0; d3 < dim3; d3++)
                data[d1 * dim2 * dim3 + d2 * dim3 + d3] = array3D[d1][d2][d3];

    return data;
}

vector<double> linearize4DVector(arrayOfDouble4D array4D)
{
    // vector<double> data;
    size_t dim1 = array4D.size();
    size_t dim2 = array4D[0].size();
    size_t dim3 = array4D[0][0].size();
    size_t dim4 = array4D[0][0][0].size();
    vector<double> data(dim1 * dim2 * dim3 * dim4, 0.0);
    for (size_t d1 = 0; d1 < dim1; d1++)
        for (size_t d2 = 0; d2 < dim2; d2++)
            for (size_t d3 = 0; d3 < dim3; d3++)
                for (size_t d4 = 0; d4 < dim4; d4 ++)
                    data[d1 * dim2 * dim3 * dim4 + d2 * dim3 * dim4 + d3 * dim4 + d4] = array4D[d1][d2][d3][d4];

    return data;
}

// allocate pointer for variables on the host

double *alloc_double_vector(unsigned int nX)
{
    double *vector_tmp;

    if((vector_tmp = (double *) malloc((size_t) (nX * sizeof(double)))) == NULL)
    {
        fprintf(stderr, "failed to alllocate memory for the pointer. \n");
        exit(EXIT_FAILURE);
    }

    return vector_tmp;
}


// CUDA Allocation functions

// allocating int vector on the device
int *device_alloc_integer_vector(unsigned int nX)
{
    int *vector;

    if (cudaMalloc((void **) &vector, nX * sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for the CUDA integer vector.\n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s \n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return vector;
}


// allocating double vector on the device

double *device_alloc_double_vector(unsigned int nX)
{
    double *vector;

    if (cudaMalloc((void **) &vector, nX * sizeof(double)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for the CUDA double vector.\n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s \n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return vector;
}


// Clearing Memory

// Clearing memory from the host 

void free_double_vector(double *vector) 
{
    free((char *) vector);
}

// Clearing menory from the device

void free_double_vector_device(double *vector)
{
    if (cudaFree(vector) != cudaSuccess) 
    {
       fprintf(stderr, "Failed to free device memory for double vector.\n");
       cudaError_t error = cudaGetLastError();
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
       exit(EXIT_FAILURE);
    }
 }


 // Copying double vector from the host to device

void copy_double_vector_fromHtoD(double *vectorD, double *vectorH, unsigned int vectorH_size)
{
    if ((cudaMemcpy(vectorD, vectorH, vectorH_size * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to the device. \n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void copy_integer_vector_fromHtoD(int *vectorD, int *vectorH, unsigned int vectorH_size)
{
    if ((cudaMemcpy(vectorD, vectorH, vectorH_size * sizeof(int), cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to the device. \n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Copy 2D matrix from the host to device 

// Copying data from device to host

void copy_double_vector_fromDtoH(double *vectorH, double *vectorD, unsigned int vectorD_size)
{
    if ((cudaMemcpy(vectorH, vectorD, vectorD_size * sizeof(double), cudaMemcpyDeviceToHost)) != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector from device to host. \n");
         cudaError_t error = cudaGetLastError();
         fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
}

void copy_integer_vector_fromDtoH(int *vectorH, int *vectorD, unsigned int vectorD_size)
{
    if ((cudaMemcpy(vectorH, vectorD, vectorD_size * sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector from device to host. \n");
         cudaError_t error = cudaGetLastError();
         fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
}



vector<string> listFiles(string path, string ext)
{
    bool gIgnoreHidden = true;
    string dotExt = "." + ext;
    vector<string> filelist;
    DIR *dirFile = opendir(path.c_str());
    if (dirFile)
    {
        struct dirent *hFile;
        errno = 0;
        while ((hFile = readdir(dirFile)) != NULL)
        {
            if (!strcmp(hFile->d_name, "."))
                continue;
            if (!strcmp(hFile->d_name, ".."))
                continue;

            // in linux hidden files all start with '.'
            if (gIgnoreHidden && (hFile->d_name[0] == '.'))
                continue;

            // dirFile.name is the name of the file. Do whatever string comparison
            // you want here. Something like:
            if (strstr(hFile->d_name, dotExt.c_str()))
                filelist.push_back(hFile->d_name);
            //cout<< "found an " << ext << " file " << hFile->d_name << endl;
        }
        closedir(dirFile);
    }
    return filelist;
}
