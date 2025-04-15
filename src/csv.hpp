#ifndef CSV_HPP
#define CSV_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "matrix_operations.hpp"
#include <tuple>
using namespace std;


tuple <vector<vector<vector<float>>>, vector<vector<vector<float>>>> readCSV(const string& filename);

#endif