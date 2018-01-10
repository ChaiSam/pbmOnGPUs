#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "parameters.h"
#include "parameterData.h"
#define FILEPATH "./"
using namespace std;

bool parameterData::instanceFlag = false;
parameterData *parameterData::pData = nullptr;

parameterData::parameterData()
{
    cout << "LIGGGHTS data pointer created" << endl;
}

parameterData::~parameterData()
{
    cout << "LIGGGHTS data pointer destroyed" << endl;
    delete pData;
}

parameterData *parameterData::getInstance()
{
    if (!instanceFlag)
    {
        pData = new parameterData();
        instanceFlag = true;
        return pData;
    }
    else
        return pData;
}

void parameterData::readPBMInputFile (string pbmInFilePath)
{
    //For now, initialising values from parameters.h
    //Later, when format of input file is finalized, following values will be initialized by reading PBM input files

    // aggKernelConst = AGGREGATIONKERNELCONSTANT; //Aggregation Kernel Constant
    // brkKernelConst = BREAKAGEKERNELCONSTANT; //Breakage Kernel Constant
    // consConst = CONSOLIDATIONCONSTANT; //Consolidation Constant
    // initPorosity = INITIALPOROSITY; // Initial Porosity
    // minPorosity = MINIMUMPOROSITY; //Minimum Porosity
    // granSatFactor = GRANULESATURATIONFACTOR; //Granule Stauration Factor

    // //Probablistic kernel constants
    // critStDefNum = CRITICALSTOKESDEFNUMBER; //Critical Stokes Deformation number value
    // bindVisc = BINDERVISCOSITY; //Binder viscosity
    // coefOfRest = COEFFICIENTOFRESTITUTION; //Coefficient of restitution
    // liqThick = LIQUIDTHICKNESS; //Liquid thickness
    // surfAsp = SURFACEASPERITIES; //Surface asperities

    // //Material & process parameters
    // granulatorLength = GRANULATORLENGTH; //unit: meter
    // nCompartments = NUMBEROFCOMPARTMENTS; //Number of compartmets
    // partticleResTime = PARTICLERESIDENCETIME; //Particle Residence Time (unit: seconds)
    // impDiameter = IMPELLERDIAMETER; //Diammeter of Impeller (unit: meter)
    // impSpeed = IMPELLERSPEED; //Speed of impeller (unit: rpm)

    // //Time Parameters (unit: seconds)
    // premixTime = PREMIXINGTIME; //Pre mixing time
    // liqAddTime = LIQUIDADDITIONTIME; //Liquid addition time
    // postMixTime = POSTMIXINGTIME; //Post mixing time 

    // //Other Process parameters
    // throughput = THROUGHPUT; //unit: Kg/hr
    // solDensity = SOLIDDENSITY; //Solid density (unit: Kg/m^3)
    // liqSolidRatio = LIQUIDTOSOLIDRATIO; // Liquid to Solid ratio
    // liqDensity = LIQUIDDENSITY; //Liquid density

    // //Bins parameter
    // //First particle
    // nFirstSolidBins = NUMBEROFFIRSTSOLIDBINS;
    // fsVolCoeff = SCOEF; //Volume of first solid bin
    // fsVolBase = SBASE;  //Base value to increase volume between two bins
    // //Second particle
    // nSecondSolidBins = NUMBEROFSECONDSOLIDBINS;
    // ssVolCoeff = SSCOEF; //Volume of second solid bin
    // ssVolBase = SSBASE;  //Base value to increase volume between two bins

    // //DEM data
    // colEffConst = COLLISIONEFFICIENCYCONSTANT; // Collision efficiency constant
    // demTimeStep = TIMESTEPDEM;
    // nDEMBins = NUMBEROFDEMBINS;

    // Read input values from file

    ifstream pbmInputFile;
    pbmInputFile.open(pbmInFilePath.c_str(), ifstream::in);

    if (!pbmInputFile.is_open())
    {
        std::cout << "Unable to open " << "PBM_Input.in" << " file" << endl;
        return;
    }

    string line;
    string tmpStr;
    stringstream lineData;

    //Read & Ignore first line as # Input Parameters for PBM
    getline(pbmInputFile, line);

    //Read AggregationKernelConstant
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> aggKernelConst;
  
    //Read BreakageKernelConstant
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> brkKernelConst;

    //Read ConsolidationConstant
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> consConst;

    //Read InitialPorosity
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> initPorosity;

    //Read MinimumPorosity
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> minPorosity;

    //Read GranulationSaturationFactor
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> granSatFactor;

    //Read CriticalStokesDefNumber
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> critStDefNum;

    //Read BinderViscosity
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> bindVisc;

    //Read CoefficientOfRestitution
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> coefOfRest;

    //Read LiquidThickness
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> liqThick;

    //Read SurfaceAsperities
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> surfAsp;

    //Read GranulatorLength
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> granulatorLength;

    //Read NumberOfCompartments
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> nCompartments;
    cout << "Number of compaertments from in file " << nCompartments << endl;

    //Read ParticleResidenceTime
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> partticleResTime;

    //Read ImpellerDiameter
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> impDiameter;

    //Read ImpellerSpeed
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> impSpeed;

    //Read PreMixingTime
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> premixTime;

    //Read LiquidAdditionTime
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> liqAddTime;
    
    //Read PostMixingTime
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> postMixTime;

    //Read Throughput
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> throughput;

    //Read SolidDensity
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> solDensity;

    //Read LiquidSolidRatio
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> liqSolidRatio;

    //Read LiquidDensity
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> liqDensity;

    //Read NumberOfFirstSolidBins
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> nFirstSolidBins;

    //Read FirstSolidVolumeCoefficient
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> fsVolCoeff;

    //Read FirstSolidVolumeBase
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> fsVolBase;

    //Read NumberOfSecondSolidBins
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> nSecondSolidBins;

    //Read SecondSolidVolumeCoefficient
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> ssVolCoeff;

    //Read SecondSolidVolumeBase
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> ssVolBase;

    //Read CollisionEfficiencyConstant
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> colEffConst;

    //Read DEMTimeStep
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> demTimeStep;

    //Read NumberOfDEMBins
    getline(pbmInputFile, line);
    lineData = move(stringstream(line));
    lineData >> tmpStr;
    lineData >> nDEMBins;    
}

arrayOfDouble3D parameterData::readCompartmentInputFile (string timeStr, string content)
{
    arrayOfDouble3D f;

    string filePath = string("./csvDump/");
    string fileName = content + string("_") + timeStr + string(".csv");
    ifstream csvInputFile;
    csvInputFile.open((filePath + fileName).c_str(), ifstream::in);

    if (!csvInputFile.is_open())
    {
        std::cout << "No " << content << " file with " << timeStr << " time stamp " << endl;
        return f;
    }
    
    f = getArrayOfDouble3D (nCompartments, nFirstSolidBins, nSecondSolidBins);
    
    string line;
    stringstream lineData;

    getline(csvInputFile, line); // read and ignore first header line
    cout << line << endl;

    size_t compartment = 0;
    size_t firstSolid = 0;
    size_t secondSolid = 0;
    double value = 0.0;

    while(getline(csvInputFile, line))
    {
        replace( line.begin(), line.end(), ',', ' ' );
        lineData = move(stringstream(line));
        lineData >> compartment;
        lineData >> firstSolid;
        lineData >> secondSolid;
        lineData >> value;

        if(compartment > (nCompartments-1) || firstSolid > (nFirstSolidBins-1) || secondSolid > (nSecondSolidBins-1))
            {
                cout << "Dimension(s) exceed while reading csv input" << endl;
                cout << "Compartment = " << compartment << "; Should be less than " << nCompartments << endl;
                cout << "First Solid Bin = " << firstSolid << "; Should be less than " << nFirstSolidBins << endl;
                cout << "Second Solid Bin = " << secondSolid << "; Should be less than " << nSecondSolidBins << endl;
                break;
            }

        f[compartment-1][firstSolid-1][secondSolid-1] = value;        
    }
    return f;
}