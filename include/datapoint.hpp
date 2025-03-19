#ifndef DATAPOINT_HPP
#define DATAPOINT_HPP

#include <vector>

using std::vector;

struct datapoint {
    double classification;
    // Cost to reach node + heuristic value
    vector<double> features;

    datapoint(double classification = 0, vector<double> features = {}) : classification(classification), 
                                                                         features(features) {};
};

#endif