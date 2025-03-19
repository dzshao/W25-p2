#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include "../include/datapoint.hpp"

using std::cout;
using std::vector;
using std::cin;
using std::ifstream;
using std::endl;
using std::string;
using std::stringstream;
using std::pair;

pair<double, uint32_t> search (const vector<datapoint> &, bool);
double kFoldValidation (const vector<datapoint> &, uint32_t, int, bool);

int main() {
    ifstream fin = ifstream("./data/CS170_Small_Data__106.txt");

    string currLine = "";

    vector<datapoint> allPoints;
    double currFeature = 0;
    // Read in data from text file
    while (getline(fin, currLine)) {
        stringstream sin = stringstream(currLine);
        allPoints.push_back({});

        sin >> allPoints.back().classification;
        while (sin >> currFeature) {
            allPoints.back().features.push_back(currFeature);
        }
    }

    pair<double, uint32_t> result = search(allPoints, true);
    pair<double, uint32_t> backwardResult = search(allPoints, false);

    // for (datapoint curr : allPoints) {
    //     cout << curr.classification << endl;
    //     for (double feat : curr.features) {
    //         cout << feat << " ";
    //     }
    //     cout << endl;
    // }

    return 0;
}

/* Searches for the best accuracy and the bitstream of features that yielded it
Set forwardBackward to true to use forward search, false for backward
*/
pair<double, uint32_t> search (const vector<datapoint> &data, bool forwardBackward) {
    int size = data.at(0).features.size();
    
    // Use bitfield to represent which features are included
    uint32_t includedFeatures = 0;
    uint32_t bestFeatures = 0;

    // Include all features with backwards search
    if (!forwardBackward) {
        includedFeatures = pow(2, data.at(0).features.size()) - 1;
        bestFeatures = pow(2, data.at(0).features.size()) - 1;
    }
    cout << includedFeatures << endl;
    double overallBestAccuracy = 0.0;

    for (int i = 0; i < size; ++i) {
        
        cout << "Level: " << i + 1 << endl;
        int bestFeature = 0;
        double bestAccuracy = 0.0;
        
        for (int j = 0; j < size; ++j) {
            if (forwardBackward) {
                // Skip if feature is already included
                if (includedFeatures & (0b1 << j)) {
                    continue;
                }
            } else {
                // Skip feature if already excluded
                if (includedFeatures & (0b1 << j) ^ (0b1 << j)) {
                    continue;
                }
            }
            cout << "Trying feature: " << j + 1 << endl;
            double currAccuracy = kFoldValidation(data, includedFeatures, j, forwardBackward);
            if (currAccuracy >= bestAccuracy) {
                bestFeature = j;
                bestAccuracy = currAccuracy;
            }
        }
        if (forwardBackward) { 
            includedFeatures = includedFeatures | (0b1 << bestFeature);
            cout << "Added feature: " << bestFeature + 1 << endl;
        } else {
            includedFeatures = includedFeatures ^ (0b1 << bestFeature);
            cout << "Removed feature: " << bestFeature + 1 << endl;
        }

        if (overallBestAccuracy < bestAccuracy) {
            overallBestAccuracy = bestAccuracy;
            bestFeatures = includedFeatures;
        }
        
        cout << "Current best accuracy: " << overallBestAccuracy << endl << endl;
    }
    return {overallBestAccuracy, bestFeatures};
}


double kFoldValidation (const vector<datapoint> &data, uint32_t features, int newFeature, bool add) {
    int size = data.at(0).features.size();

    // Hidden data is the point removed from the dataset
    double correct = 0.0;
    for (int hiddenData = 0; hiddenData < size; ++hiddenData) {
        double nearestNeighborClass = 0.0;
        double nearestNeighborDistance = DBL_MAX;
        for (int j = 0; j < size; ++j) {
            // Don't compare with self
            if (hiddenData == j) {
                continue;
            }
            
        }
        if (data.at(hiddenData).classification == nearestNeighborClass) {
            correct += 1;
        }
    }
    return correct / size;
}