#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <iomanip>
#include "../include/datapoint.hpp"

using std::cout;
using std::vector;
using std::cin;
using std::ifstream;
using std::endl;
using std::string;
using std::stringstream;
using std::pair;
using std::setprecision;

pair<double, uint64_t> search (const vector<datapoint> &, bool);
double kFoldValidation (const vector<datapoint> &, uint64_t, int, bool);
double calcDistance(const datapoint &, const datapoint &, uint64_t);

void printBinary(uint64_t num) {
    for (int i = 63; i >= 0; --i) {
        // Check if the bit at position i is set (1) or not (0)
        std::cout << ((num >> i) & 1);
    }
    std::cout << std::endl;
}

int main() {
    ifstream fin = ifstream("./data/CS170_Large_Data__1.txt");
    // ifstream fin = ifstream("./data/testdata.txt");
    string currLine = "";
    vector<datapoint> allPoints;
    double currFeature = 0;
    // Read in data from text file
    while (getline(fin, currLine)) {
        stringstream sin = stringstream(currLine);
        allPoints.push_back({});

        sin >> allPoints.back().classification;
        while (sin >> setprecision(15) >> currFeature) {
            allPoints.back().features.push_back(currFeature);
        }
    }
    
    // for (datapoint curr : allPoints) {
    //     cout << curr.classification << endl;
    //     for (double feat : curr.features) {
    //         cout << setprecision(15) << feat << " ";
    //     }
    //     cout << endl;
    // }
    // cout << kFoldValidation(allPoints, (uint64_t(1) << 17) | (uint64_t(1) << 33), 24, true) << endl;

    pair<double, uint64_t> result = search(allPoints, true);
    cout << "Best accuracy: " << result.first << endl << "Best features: ";
    int size = allPoints.at(0).features.size();
    for (int i = 0; i < size; ++i) {
        if (result.second & (uint64_t(1) << i)) { 
            cout << i + 1 << ", ";
        }
    }
    cout << endl;

    // pair<double, uint64_t> backwardResult = search(allPoints, false);

    return 0;
}

/* Searches for the best accuracy and the bitstream of features that yielded it
Set forwardBackward to true to use forward search, false for backward
*/
pair<double, uint64_t> search (const vector<datapoint> &data, bool forwardBackward) {
    int size = data.at(0).features.size();

    // Use bitfield to represent which features are included
    uint64_t includedFeatures = 0;
    uint64_t bestFeatures = 0;
    double overallBestAccuracy = 0.0;
    // Include all features with backwards search
    if (!forwardBackward) {
        includedFeatures = pow(2, data.at(0).features.size()) - 1;
        bestFeatures = pow(2, data.at(0).features.size()) - 1;
    }

    for (int i = 0; i < size; ++i) {
        
        cout << "Level: " << i + 1 << endl;
        int bestFeature = 0;
        double bestAccuracy = 0.0;
        
        for (int j = 0; j < size; ++j) {
            if (forwardBackward) {
                // Skip if feature is already included
                if (includedFeatures & (uint64_t(1) << j)) {
                    continue;
                }
            } else {
                // Skip feature if already excluded
                if (includedFeatures & (uint64_t(1) << j) ^ (uint64_t(1) << j)) {
                    continue;
                }
            }
            
            double currAccuracy = kFoldValidation(data, includedFeatures, j, forwardBackward);
            // cout << "Trying feature: " << j + 1 << " Accuracy: " << currAccuracy << endl;
            if (currAccuracy >= bestAccuracy) {
                bestFeature = j;
                bestAccuracy = currAccuracy;
            }
        }

        if (forwardBackward) { 
            includedFeatures = includedFeatures | (uint64_t(1) << bestFeature);
            cout << "Added feature: " << bestFeature + 1 << endl;
        } else {
            includedFeatures = includedFeatures ^ (uint64_t(1) << bestFeature);
            cout << "Removed feature: " << bestFeature + 1 << endl;
        }
        // printBinary(includedFeatures);
        if (overallBestAccuracy < bestAccuracy) {
            overallBestAccuracy = bestAccuracy;
            bestFeatures = includedFeatures;
        }
        
        cout << "Current best accuracy: " << overallBestAccuracy << endl << endl;
    }
    return {overallBestAccuracy, bestFeatures};
}


double kFoldValidation (const vector<datapoint> &data, uint64_t features, int newFeature, bool add) {
    int size = data.size();

    // Hidden data is the point removed from the dataset
    double correct = 0.0;
    uint64_t currFeatures = features;
    // Add feature if doing forward search, remove otherwise
    if (add) {
        currFeatures = features | (uint64_t(1) << newFeature);
    } else {
        currFeatures = features ^ (uint64_t(1) << newFeature);
    }

    for (int hiddenData = 0; hiddenData < size; ++hiddenData) {
        double nearestNeighborClass = 0.0;
        double nearestNeighborDistance = DBL_MAX;
        double currDistance = 0.0;
        for (int j = 0; j < size; ++j) {
            // Don't compare with self
            if (hiddenData == j) {
                continue;
            }
            // Calculate distance including/removing new feature
            currDistance = calcDistance(data.at(hiddenData), data.at(j), currFeatures);
            if (currDistance < nearestNeighborDistance) {
                nearestNeighborDistance = currDistance;
                nearestNeighborClass = data.at(j).classification;
            }
        }
        if (data.at(hiddenData).classification == nearestNeighborClass) {
            correct += 1;
        }
    }
    return correct / size;
}

double calcDistance(const datapoint &lhs, const datapoint &rhs, uint64_t features) {
    int size = lhs.features.size();
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        if (features & (uint64_t(1) << i)) {
            sum += pow(fabs(lhs.features.at(i) - rhs.features.at(i)), 2.0);
        }
    }
    return sqrt(sum);
}