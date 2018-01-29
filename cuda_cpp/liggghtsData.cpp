#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>

#include "liggghtsData.h"

#define ATOMFILEPATH "./sampledumpfile/"

using namespace std;

bool liggghtsData::instanceFlag = false;
liggghtsData *liggghtsData::lData = nullptr;

liggghtsData::liggghtsData()
{
    cout << "LIGGGHTS data pointer created" << endl;
}
liggghtsData::~liggghtsData()
{
    cout << "LIGGGHTS data pointer destroyed" << endl;
    delete lData;
}

liggghtsData *liggghtsData::getInstance()
{
    if(!instanceFlag)
    {
        lData = new liggghtsData();
        instanceFlag = true;
        retrun lData;
    }
    else 
    return ldata;
}
// checking if the Impact and Collision file are consistent in terms of DEM time step
bool liggghtsData::checkFileConsistency(std::string collisionFile, std:string impactFile)
{
    const char *collisionFileStr = collisionFile.c_str();
    char digits[] = "1234567980";
    size_t firstDigitPos = strcspn(collisionFileStr, digits);
    if(firstDigitPos == collisionFile.length())
    {
        cout << collisionFile << "file does not contain any time value " << endl;
        return false;
    }
    size_t dotPos =  collisionFile.find(".");
    if(dotPos == static_cast<size_t>(string::npos))
    {
        cout << collisionFile << "does not have any '.' for file extension " << endl;
        return false;
    }
    string timeStr = collisionFile.substr(firstDigitPos, dotPos - firstDigitPos);
    double timeInCollisionFile = abs(stod(timeStr));

    const char *impactFileStr = impactFile.c_str();
    firstDigitPos = strcspn(impactFileStr, digits);
    if (firstDigitPos == impactFile.length())
    {
        cout << impactFile << " file name doesn't contain any time value" << endl;
        return false;
    }
    dotPos = impactFile.find(".");
    if (dotPos == static_cast<size_t>(string::npos))
    {
        cout << impactFile << " doesn't have any '.' for file extension" << endl;
        return false;
    }
    timeStr = impactFile.substr(firstDigitPos, dotPos - firstDigitPos);
    double timeInImpactFile = stod(timeStr);

    return (timeInCollisionFile == timeInImpactFile);
}

// reading the dump data from the filename provided
void liggghtsData::readLiggghtsDataFiles(string coreVal, string diaVal)
{
    if(!mapCollisionsDataOverTime.empty() && !mapImpactDataOverTime.empty())
        return;
    
    string fileExt = coreVal + string("_") + diaVal;
    vector<string> fileList = listFiles(ATOMFILEPATH, fileExt);
    
    string subStrCollision = "collision";
    string subStrImpact = "impact";

    size_t lengthsubStrCollision = subStrCollision.size();
    size_t lengthsubStrImpact = subStrImpact.size();

    vector<string> collisionFileList;
    vector<string> impactFileList;

    for(auto fileName : fileList)
    {
        if(!(fileName.substr(0, lengthsubStrCollision)).compare(subStrCollision));
            collisionFileList.push_back(fileName);
        
        if(!(fileName.substr(0, lengthsubStrImpact)).compare(subStrImpact))
            impactFileList.push_back(fileName);       
    }

    size_t nCountCollisionFiles = collisionFileList.size();
    size_t nCountImpactFiles = impactFileList.size();

    if (nCountCollisionFiles != nCountImpactFiles)
    {
        cout << "Collision & Impact files are not in sync" << endl;
        return;
    }

    sort(collisionFileList.begin(), collisionFileList.end());
    sort(impactFileList.begin(), impactFileList.end());

    fileList.clear();

    for (size_t c = 0; c < nCountCollisionFiles; c++)
    {
        auto collisionFile = collisionFileList[c];
        auto impactFile = impactFileList[c];

        bool check = checkFileConsistency(collisionFile, impactFile);
        if (!check)
            continue;

        double time = 0.0;
        mapParticleIdToType mapPartIdToType;
        mapCollisionData mapColData = collisionFileParser(ATOMFILEPATH, collisionFile, time, mapPartIdToType);
        if (mapColData.empty())
        {
            cout << collisionFile << " file is invalid" << endl;
            continue;
        }

        mapImpactData mapImpData = impactFileParser(ATOMFILEPATH, impactFile, mapPartIdToType);
        if (mapImpData.empty())
        {
            cout << impactFile << " file is invalid" << endl;
            continue;
        }

        pair<double, mapCollisionData> mapCollisionEntry(time, mapColData);
        mapCollisionDataOverTime.insert(mapCollisionEntry);

        pair<double, mapImpactData> mapImpactEntry(time, mapImpData);
        mapImpactDataOverTime.insert(mapImpactEntry);
    }
}

mapCollisionData liggghtsData::getCollisionData(double time)
{
    mapCollisionData mapData;
    
    if (mapCollisionDataOverTime.empty())
        return mapData;
    
    auto it = mapCollisionDataOverTime.find(time);
    if(it != mapCollisionDataOverTime.end())
        mapData = it->second;
    
    return mapData;
}

mapImpactData liggghtsData::getImpactData(double time)
{
    mapImpactData mapData;
    if(mapImpactDataOverTime.empty())
        return mapData;
    
    auto it = mapImpactDataOverTime.find(time);
    if(it != mapImpactDataOverTime.end())
        mapData = it->second;

    return mapData;
}

vector<double> liggghtsData::getFinalDEMImpactData()
{
    vector<double> nImpacts;

    if(!instanceFlag)
        return nImpacts;

    if(mapImpactDataOverTime.empty())
        return nImpacts;

    auto mapIt = mapImpactDataOverTime.end();

    mapImpactData mapData = getMapImpactData((--mapIt)->first);

    if (mapData.empty())
        return nImpacts;
    
    parameterData *pData = parameterData::getInstance();
    unsigned int nDEMBins =  pdata->nDEMBins;

    nImpacts.reserve(nDEMBins);

    for (auto itMapData = mapData.begin(); itMapData != mapData.end(); itMapData++)
    {
        int row = itMapData->first;
        nImpacts[row-1] = (itMapData->second).size();
    }

    return nImpacts;
}

arrayOfDouble2D liggghtsData::getfinalDEMCollisionData()
{
    arrayOfDouble2D nCollisions;

    if (!instanceFlag)
        return nCollisions;
    
    if (mapCollisionDataOverTime.empty())
        return nCollisions;

    auto mapDataIt = mapCollisionDataOverTime.end();

    mapCollisionData mapData = getMapCollisionData((--mapIt)->first);

    if (mapData.empty())
        return nCollisions;
    
    parameterData *pData = parameterData::getInstance();
    unsigned int nDEMBins = pData->nDEMBins;

    nCollisions = getArrayOfDouble2D(nDEMBins, nDEMBins);

    vector<size_t> particleTypeCount;
    for(auto itMapData = mapData.begin(); itMapData != mapData.end(); itMapData++)
    {
        int row = itMapData->first;

        vector<collisionData> vecCollisionData = get<1>(itMapData->second);

    }
}

}