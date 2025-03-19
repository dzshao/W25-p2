#ifndef DATAPOINT_HPP
#define DATAPOINT_HPP

#include <vector>

using std::vector;

struct datapoint {
    double classification;
    vector<double> features;

    datapoint(double classification = 0, vector<double> features = {}) : classification(classification), 
                                                                         features(features) {};
};

#endif