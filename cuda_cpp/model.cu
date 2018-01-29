#include <vector>
#include <cmath>
#include <float.h>
#include <string>
#include <iostream.h>

#include "utility.h"

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

int main(int argc, char *argv[])
{
    fprint("Code begins \n");
    // Read passed arguments
    string startTimeStr;
    double startTime = 0.0;
    liggghtsData *lData = nullptr;
    


}