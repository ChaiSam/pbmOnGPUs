#ifndef LIGGGHTSDATA_H
#define LIGGGHTSDATA_H

#include <map>
#include <vector>
#include "atomFileParser.h"
#include "utility.h"

// A structure to store the DEM values from the LIGGGHTS dump files and mapped using the particle type.
class LiggghtsData
{
    static bool instanceFlag;
    static liggghtsData *lData;
    liggghtsData();
    bool checkFileConsistency(std::string collisionFile, std:: string impactFile);

    std::map<double, mapCollisionData> mapCollisionDataOverTime;
    std::map<double, mapImpactData> mapImpactDataOverTime;

    
}
