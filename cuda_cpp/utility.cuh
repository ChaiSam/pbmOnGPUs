#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#define CSVFILEPATH "./csvDump/"
#define TXTFILEPATH "./txtDump/"

typedef std::vector<std::vector<int>> arrayOfInt2D;
typedef std::vector<std::vector<std::vector<int>>> arrayOfInt3D;
typedef std::vector<std::vector<std::vector<std::vector<int>>>> arrayOfInt4D;
typedef std::vector<std::vector<double>> arrayOfDouble2D;
typedef std::vector<std::vector<std::vector<double>>> arrayOfDouble3D;
typedef std::vector<std::vector<std::vector<std::vector<double>>>> arrayOfDouble4D;

arrayOfInt2D getArrayOfInt2D(int n, int m, int val = 0);
arrayOfInt3D getarrayOfInt3D(int n, int m, int p, int val = 0);
arrayOfInt4D getArrayOfInt4D(int n, int m, int p, int q, int val = 0);
arrayOfDouble2D getArrayOfDouble2D(int n, int m, double val = 0.0);
arrayOfDouble3D getArrayOfDouble3D(int n, int m, int p, double val = 0.0);
arrayOfDouble4D getArrayOfDouble4D(int n, int m, int p, int q, double val = 0.0);

double *alloc_double_vector(unsigned int);

int *device_alloc_integer_vector(unsigned int);
double *device_alloc_double_vector(unsigned int);

void free_double_vector(double *);
void free_double_vector_device(double **);
void free_double_matrix_device(struct cudaPitchedPtr);

void copy_double_vector_fromDtoH(double *, double *, unsigned int);
void copy_double_vector_fromHtoD(double *, double *, unsigned int);

void copy_integer_vector_fromHtoD(int *, int *, unsigned int);
void copy_integer_vector_fromDtoH(int *, int *, unsigned int);
std::vector<std::string> listFiles(std::string path, std::string ext);

#endif // UTILITY_H
