#ifndef PARAMETERDATA_H
#define PARAMETERDATA_H

#include "utility.h"

// Class to provide input parameters to initiating PBM
// Parameters are assigned either by reading PBM input files or values from parameters.h

class parameterData
{
    static bool instanceFlag;
    static parameterData *pData; // = nullptr;
    parameterData();

public:
    // member variables

    //Tuning parameters for kernel
    double aggKernelConst; //Aggregation Kernel Constant
    double brkKernelConst; //Breakage Kernel Constant
    double consConst; //Consolidation Constant
    double initPorosity; // Initial Porosity
    double minPorosity; //Minimum Porosity
    double granSatFactor; //Granule Stauration Factor

    //Probablistic kernel constants
    double critStDefNum; //Critical Stokes Deformation number value
    double bindVisc; //Binder viscosity
    double coefOfRest; //Coefficient of restitution
    double liqThick; //Liquid thickness
    double surfAsp; //Surface asperities

    //Material & process parameters
    double granulatorLength; //unit: meter
    unsigned int nCompartments; //Number of compartmets
    double partticleResTime; //Particle Residence Time (unit: seconds)
    double impDiameter; //Diammeter of Impeller (unit: meter)
    double impSpeed; //Speed of impeller (unit: rpm)

    //Time Parameters (unit: seconds)
    double premixTime; //Pre mixing time
    double liqAddTime; //Liquid addition time
    double postMixTime; //Post mixing time 

    //Other Process parameters
    double throughput; //unit: Kg/hr
    double solDensity; //Solid density (unit: Kg/m^3)
    double liqSolidRatio; // Liquid to Solid ratio
    double liqDensity; //Liquid density

    //Bins parameter
    //First particle
    unsigned int nFirstSolidBins;
    double fsVolCoeff; //Volume of first solid bin
    double fsVolBase;  //Base value to increase volume between two bins
    //Second particle
    unsigned int nSecondSolidBins;
    double ssVolCoeff; //Volume of second solid bin
    double ssVolBase;  //Base value to increase volume between two bins

    //DEM data
    double colEffConst; // Collision efficiency constant
    double demTimeStep;
    unsigned int nDEMBins;

    static parameterData *getInstance();
    ~parameterData();

    void readPBMInputFile (std::string pbmInFilePath);
    
    //Function to read csv file of particles, liquid or gas
    // Specify content as particles/liquid/gas
    arrayOfDouble3D readCompartmentInputFile (std::string timeStr, std::string content);
    
};


#endif // PARAMETERDATA_H