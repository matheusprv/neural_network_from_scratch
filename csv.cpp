#include "csv.hpp"

tuple<vector<vector<vector<float>>>, vector<vector<vector<float>>>> readCSV(const string& filename) {
    vector<vector<vector<float>>> X;
    vector<vector<vector<float>>> Y;

    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return {{}, {}};
    }

    string line;

    // Skip the first line (header)
    getline(file, line);

    while (getline(file, line)) {      
        vector<vector<float>> rowX(1); // Initialize with one empty vector
        vector<vector<float>> rowY(1); // Initialize with one empty vector
        
        stringstream ss(line);
        string cell;

        // Label is in the first column
        getline(ss, cell, ',');
        rowY[0].push_back(stoi(cell));

        // Remaining columns are features
        while (getline(ss, cell, ',')) {
            rowX[0].push_back(stof(cell) / 255.0);
        }

        X.push_back(transposeMatrix(rowX));
        Y.push_back(rowY);
    }

    file.close();
    return {X, Y};
}
