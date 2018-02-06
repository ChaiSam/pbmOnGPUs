#ifndef LIGGGHTSDATA_H
#define LIGGGHTSDATA_H

#include <map>
#include <vector>
#include "atomFileParser.h"
#include "utility.cuh"

// A structure to store the DEM values from the LIGGGHTS dump files and mapped using the particle type.
class liggghtsData
{
    static bool instanceFlag;
    static liggghtsData *lData;
    liggghtsData();
    bool checkFileConsistency(std::string collisionFile, std:: string impactFile);

    std::map<double, mapCollisionData> mapCollisionDataOverTime;
    std::map<double, mapImpactData> mapImpactDataOverTime;

    public:
      static  liggghtsData *getInstance();

      void readLiggghtsDataFiles(std::string coreVal, std::string diaVal);

      mapCollisionData getMapCollisionData(double time);
      mapImpactData getMapImpactData(double time);

      arrayOfDouble2D getFinalDEMCollisionData();
      std::vector<double> getFinalDEMImpactData();
      std::vector<double> getDEMParticleDiameters();
      std::vector<double> getFinalDEMCollisionVelocity();
      std::vector<double> getFinalDEMImpactVelocity();
      ~liggghtsData();
};

#endif //LIGGGHTSDATA_h
