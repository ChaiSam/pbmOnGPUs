#include <algorithm>
#include <cstring>
#include <cmath>

#include "liggghtsData.h"

#define ATOMFILEPATH "./sampledumpfile/"

using namespace std;

bool liggghtsData::instanceFlag = false;
liggghtsData *liggghtsData::lData = nullptr;

liggghtsData::liggghtsData()
{
    fprintf("LIGGGHTS data pointer created. \n");   
}
liggghtsData::~liggghtsData()
{
    fprintf("LIGGGHTS data pointer destroyed. \n");
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

bool liggghtsData::checkFileConsistency(std::string collisionFile, std:string impactFile)
{
    
}