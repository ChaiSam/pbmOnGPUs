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

std::vector<double> linearize2DVector(arrayOfDouble2D);
std::vector<double> linearize3DVector(arrayOfDouble3D);
std::vector<double> linearize4DVector(arrayOfDouble4D);

int *alloc_integer_vector(unsigned int);
double *alloc_double_vector(unsigned int);

int *device_alloc_integer_vector(unsigned int);
double *device_alloc_double_vector(unsigned int);
void device_alloc_double_vector(double ** d, unsigned int);

void free_double_vector(double *);
void free_double_vector_device(double **);
void free_double_matrix_device(struct cudaPitchedPtr);

void copy_double_vector_fromDtoH(double *, double *, unsigned int);
void copy_double_vector_fromHtoD(double *, double *, unsigned int);

void copy_integer_vector_fromHtoD(int *, int *, unsigned int);
void copy_integer_vector_fromDtoH(int *, int *, unsigned int);
std::vector<std::string> listFiles(std::string path, std::string ext);

template <typename T>
void dumpData(T data, std::string varName)
{
    std::string path = TXTFILEPATH;
    std::string fileName = varName + ".txt";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }
    for (size_t i = 0; i < data.size(); i++)
        myFile << data[i] << std::endl;

    myFile.close();
}

template <typename T>
void dump2DData(T data, std::string varName)
{
    std::string path = TXTFILEPATH;
    std::string fileName = varName + ".txt";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); i++)
    {
        myFile << "i = " << i << std::endl;
        for (size_t j = 0; j < data[i].size(); j++)
            myFile << data[i][j] << '\t';
        myFile << std::endl
               << std::endl;
    }
    myFile.close();
}

template <typename T>
void dump3DData(T data, std::string varName)
{
    std::string path = TXTFILEPATH;
    std::string fileName = varName + ".txt";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }

    size_t thirdDim = data[0][0].size();
    for (size_t d1 = 0; d1 < data.size(); d1++)
    {
        myFile << varName.c_str() << "(:,:" << d1 << ")" << std::endl;
        for (size_t d2 = 0; d2 < data[d1].size(); d2++)
        {
            for (size_t d3 = 0; d3 < thirdDim ; d3++)
                myFile << data[d1][d2][d3] << '\t';
            myFile << std::endl;
        }
        myFile << std::endl
               << std::endl;
    }
    myFile.close();
}

template <typename T>
void dump4DData(T data, std::string varName)
{
    std::string path = TXTFILEPATH;
    std::string fileName = varName + ".txt";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }

    size_t thirdDim = data[0][0].size();
    size_t forthDim = data[0][0][0].size();
    for (size_t d1 = 0; d1 < data.size(); d1++)
    {
        for (size_t d2 = 0; d2 < data[d1].size(); d2++)
        {
            myFile << varName.c_str() << "(:,:" << d2 << "," << d1 << ")" << std::endl;
            for (size_t d3 = 0; d3 < thirdDim; d3++)
            {
                for (size_t d4 = 0; d4 < forthDim; d4++)
                    myFile << data[d1][d2][d3][d4] << '\t';
                myFile << std::endl;
            }
            myFile << std::endl
                   << std::endl;
        }
    }
    myFile.close();
}

//Dumping data as column

template <typename T>
void dumpCSV(T data, std::string varName)
{
    std::string path = CSVFILEPATH;
    std::string fileName = varName + ".csv";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }
    myFile << "dim_1"
           << ","
           << "value" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
    {
        if (data[i] < 1.0e-16)
            myFile << i + 1 << "," << data[i] << std::endl;
        else
            myFile << i + 1 << "," << moreSigs(data[i], 16) << std::endl;
    }
    myFile.close();
}

template <typename T>
void dump2DCSV(T data, std::string varName)
{
    std::string path = CSVFILEPATH;
    std::string fileName = varName + ".csv";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }
    myFile << "dim_1"
           << ","
           << "dim_2"
           << ","
           << "value" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
        for (size_t j = 0; j < data[i].size(); j++)
        {
            if (data[i][j] < 1.0e-16)
                myFile << i + 1 << "," << j + 1 << "," << data[i][j] << std::endl;
            else
                myFile << i + 1 << "," << j + 1 << "," << moreSigs(data[i][j], 16) << std::endl;
        }
    myFile.close();
}

template <typename T>
void dump3DCSV(T data, std::string varName)
{
    std::string path = CSVFILEPATH;
    std::string fileName = varName + ".csv";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }
    myFile << "dim_1"
           << ","
           << "dim_2"
           << ","
           << "dim_3"
           << ","
           << "value" << std::endl;
    for (size_t d1 = 0; d1 < data.size(); d1++)
        for (size_t d2 = 0; d2 < data[d1].size(); d2++)
            for (size_t d3 = 0; d3 < data[d1][d2].size(); d3++)
            {
                if (data[d1][d2][d3] < 1.0e-16)
                    myFile << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << data[d1][d2][d3] << std::endl;
                else
                    myFile << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << moreSigs(data[d1][d2][d3], 16) << std::endl;
            }
    myFile.close();
}

template <typename T>
void dump4DCSV(T data, std::string varName)
{
    std::string path = CSVFILEPATH;
    std::string fileName = varName + ".csv";
    std::ofstream myFile;
    myFile.open((path + fileName).c_str());
    if (!myFile.is_open())
    {
        std::cout << "Unable to open file to dump data" << std::endl;
        return;
    }
    myFile << "dim_1"
           << ","
           << "dim_2"
           << ","
           << "dim_3"
           << ","
           << "dim_4"
           << ","
           << "value" << std::endl;
    for (size_t d1 = 0; d1 < data.size(); d1++)
        for (size_t d2 = 0; d2 < data[d1].size(); d2++)
            for (size_t d3 = 0; d3 < data[d1][d2].size(); d3++)
                for (size_t d4 = 0; d4 < data[d1][d2][d3].size(); d4++)
                {
                    if (data[d1][d2][d3][d4] < 1.0e-16)
                        myFile << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << data[d1][d2][d3][d4] << std::endl;
                    else
                        myFile << d1 + 1 << "," << d2 + 1 << "," << d3 + 1 << "," << d4 + 1 << "," << moreSigs(data[d1][d2][d3][d4], 16) << std::endl;
                }
    myFile.close();
}

#endif // UTILITY_H
