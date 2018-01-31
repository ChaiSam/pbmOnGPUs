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

// allocate pointer for a 2D matrix

double **alloc_double_matrix(unsigned int nX, unsigned int nY) 
{
   long cnti;
   double **matrix;

   if((matrix = (double **) malloc((size_t) (nX * sizeof(double *)))) == NULL) 
   {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   if((matrix[0] = (double *) malloc((size_t) (nX * nY * sizeof(double)))) == NULL) 
   {
      fprintf(stderr, "Failed to allocate memory for the matrix.\n");
      exit(EXIT_FAILURE);
   }
   for(cnti = 1; cnti < nX; cnti ++)
      matrix[cnti] = matrix[cnti - 1] + nY;

   return matrix;
}


// CUDA Allocation functions

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

// Allocating a 2D matrix on the device

struct cudaPitchedPtr device_alloc_double_matrix(unsigned int nX, unsigned int nY)
{
    struct cudaPitchedPtr matrix;

    if(cudaMalloc3D(&matrix, make_cudaExtent(nY * sizeof(double), nX, 1)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate memory for the 2D matrix on the device \n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s \n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return matrix;
}

// Clearing Memory

// Clearing memory from the host 

void free_double_vector(double *vector) 
{
    free((char *) vector);
}

void free_double_matrix(double **matrix) 
{
    free((char *) matrix[0]);
    free((char *) matrix);
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

void free_double_matrix_device(struct cudaPitchedPtr matrix)
{
    if (cudaFree(matrix.ptr) != cudaSuccess)
    {
       fprintf(stderr, "Failed to free device memory for double matrix.\n");
       cudaError_t error = cudaGetLastError();
       fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
       exit(EXIT_FAILURE);
    }
 }

 // Copying double vector from the host to device

void copy_double_vector_fromHtoD(double *vectorD, double *vectorH, unsigned int vectorH_size)
{
    if ((cudaMemcpy(vectorD, vectorH, vectorH_size, cudaMemcpyHostToDevice)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to the device. \n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Copy 2D matrix from the host to device 

void copy_double_2Dmatrix_fromHtoD(struct cudaPitchedPtr d_2D, struct cudaPitchedPtr h_2D, struct cudaExtent e)
{
    struct cudaMemcpy3DParms temp_3d;
    temp_3d.extent = e;
    temp_3d.kind = cudaMemcpyHostToDevice;
    temp_3d.dstPtr = d_2D;
    temp_3d.srcPtr = h_2D;

    if ((cudaMemcpy3D(&temp_3d)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy 2D matrix from host to the device. \n");
        cudaError_t error = cudaGetLastError();
        fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Copying data from device to host

void copy_double_vector_fromDtoH(double *vectorH, double *vectorD, unsigned int vectorD_size)
{
    if ((cudaMemcpy(vectorH, vectorD, vectorD_size, cudaMemcpyDeviceToHost)) != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector from host to the device. \n");
         cudaError_t error = cudaGetLastError();
         fprintf(stderr, "CUDA error: %s.\n",cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }
}

// Copy 2D matrix from the host to device 

void copy_double_2Dmatrix_fromDtoH(struct cudaPitchedPtr h_2D, struct cudaPitchedPtr d_2D, struct cudaExtent e)
{
    struct cudaMemcpy3DParms temp_3d;
    temp_3d.extent = e;
    temp_3d.kind = cudaMemcpyDeviceToHost;
    temp_3d.srcPtr = d_2D;
    temp_3d.dstPtr = h_2D;

    if ((cudaMemcpy3D(&temp_3d)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy 2D matrix from device to the host. \n");
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
