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
using std::fixed;
using std::to_string;

pair<double, uint64_t> search (const vector<datapoint> &, bool);
double kFoldValidation (const vector<datapoint> &, uint64_t, int, bool);
double calcDistance(const datapoint &, const datapoint &, uint64_t);
string convertBitfieldToFeatureSet(uint64_t, int);

int main() {
    
    cout << "Searching large dataset..." << endl << endl;

    ifstream fin = ifstream("./data/CS170_Large_Data__1.txt");
    if (!fin) {
        cout << "Error opening file." << endl;
        return -1; 
    }
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

    int size = allPoints.at(0).features.size();
    pair<double, uint64_t> result = search(allPoints, true);
    cout << "Finished Search!! The best feature subset is {" << convertBitfieldToFeatureSet(result.second, size) << "}, which has an accuracy of " << fixed << setprecision(1) << result.first * 100 << "%";
    for (int i = 0; i < size; ++i) {
        if (result.second & (uint64_t(1) << i)) { 
            cout << i + 1 << ", ";
        }
    }
    cout << endl << endl;

    pair<double, uint64_t> backwardResult = search(allPoints, false);
    cout << "Finished Search!! The best feature subset is {" << convertBitfieldToFeatureSet(backwardResult.second, size) << "}, which has an accuracy of " << fixed << setprecision(1) << backwardResult.first * 100 << "%";
    cout << endl << endl;
    
    cout << "Searching small dataset..." << endl << endl;

    ifstream fin2 = ifstream("./data/CS170_Small_Data__106.txt");
    if (!fin2) {
        cout << "Error opening file." << endl;
        return -1; 
    }
    string currLine2 = "";
    vector<datapoint> allPoints2;

    double currFeature2 = 0;
    // Read in data from text file
    while (getline(fin2, currLine2)) {
        stringstream sin = stringstream(currLine2);
        allPoints2.push_back({});

        sin >> allPoints2.back().classification;
        while (sin >> currFeature2) {
            allPoints2.back().features.push_back(currFeature2);
        }
    }

    int size2 = allPoints2.at(0).features.size();
    pair<double, uint64_t> result2 = search(allPoints2, true);
    cout << "Finished Search!! The best feature subset is {" << convertBitfieldToFeatureSet(result2.second, size2) << "}, which has an accuracy of " << fixed << setprecision(1) << result2.first * 100 << "%";
    cout << endl << endl;

    pair<double, uint64_t> backwardResult2 = search(allPoints2, false);
    cout << "Finished Search!! The best feature subset is {" << convertBitfieldToFeatureSet(backwardResult2.second, size2) << "}, which has an accuracy of " << fixed << setprecision(1) << backwardResult2.first * 100 << "%";
    cout << endl << endl;

    return 0;
}

/* Searches for the best accuracy and the bitstream of features that yielded it
Set forwardBackward to true to use forward search, false for backward
*/
pair<double, uint64_t> search (const vector<datapoint> &data, bool forwardBackward) {
    int size = data.at(0).features.size();
    
    cout << "Beginning search..." << endl << endl;
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
        
        // cout << "Level: " << i + 1 << endl;
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
            // Find accuracy with including/excluding feature
            double currAccuracy = kFoldValidation(data, includedFeatures, j, forwardBackward);
            cout << "\tUsing feature(s) {" << j + 1 << "} accuracy is " << fixed << setprecision(1) << currAccuracy * 100 << "%" << endl;
            if (currAccuracy >= bestAccuracy) {
                bestFeature = j;
                bestAccuracy = currAccuracy;
            }
        }

        if (forwardBackward) { 
            includedFeatures = includedFeatures | (uint64_t(1) << bestFeature);
        } else {
            includedFeatures = includedFeatures ^ (uint64_t(1) << bestFeature);
        }
        
        cout << endl << "Feature set {" << convertBitfieldToFeatureSet(includedFeatures, size) << "} was best, accuracy is " << fixed << setprecision(1) << bestAccuracy * 100 << "%" << endl;

        if (overallBestAccuracy < bestAccuracy) {
            overallBestAccuracy = bestAccuracy;
            bestFeatures = includedFeatures;
        } else {
            cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)" << endl;
        }
        // cout << endl << "Feature set {" << convertBitfieldToFeatureSet(bestFeatures, size) << "} was best, accuracy is " << fixed << setprecision(1) << overallBestAccuracy * 100 << "%";// << endl;
        // cout << fixed << setprecision(1) << overallBestAccuracy * 100 << "%" << endl;
        cout << endl;
    }
    return {overallBestAccuracy, bestFeatures};
}


double kFoldValidation (const vector<datapoint> &data, uint64_t features, int newFeature, bool forwardBackward) {
    int size = data.size();

    // Hidden data is the point removed from the dataset
    double correct = 0.0;
    uint64_t currFeatures = features;
    // Add feature if doing forward search, remove otherwise
    if (forwardBackward) {
        currFeatures = features | (uint64_t(1) << newFeature);
    } else {
        currFeatures = features ^ (uint64_t(1) << newFeature);
    }

    // Iterate through all datapoints leaving out one each time
    for (int hiddenData = 0; hiddenData < size; ++hiddenData) {
        double nearestNeighborClass = 0.0;
        double nearestNeighborDistance = DBL_MAX;
        double currDistance = 0.0;
        // Find closest neighbor
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
    // Find distance between two points
    for (int i = 0; i < size; ++i) {
        if (features & (uint64_t(1) << i)) {
            sum += pow(fabs(lhs.features.at(i) - rhs.features.at(i)), 2.0);
        }
    }
    return sqrt(sum);
}

string convertBitfieldToFeatureSet(uint64_t bitfield, int size) {
    string result = "";
    for (int i = 0; i < size; ++i) {
        if (bitfield & (uint64_t(1) << i)) { 
            result += to_string(i + 1) + ", ";
        }
    }
    if (result.size() > 0) {
        return result.substr(0, result.size() - 2);
    }
    return "";
}