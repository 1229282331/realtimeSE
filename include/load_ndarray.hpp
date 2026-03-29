#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>
#include <regex>

using namespace Eigen;

// 读取浮点矩阵文件
inline MatrixXf loadMatrixFromCSV(const std::string& filename) 
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("failed to open the file: " + filename);

    std::vector<std::vector<float>> data;
    std::string line;
    
    // 第一遍：确定矩阵维度
    int cols = 0;
    while (std::getline(file, line)) 
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        
        if(cols == 0)
            cols = static_cast<int>(row.size());
        else if(row.size() != cols) 
            throw std::runtime_error("row.size() != cols");
        data.push_back(row);
    }
    file.close();

    // 转换为 Eigen 矩阵
    MatrixXf mat(data.size(), cols);
    for (size_t i = 0; i < data.size(); ++i)
        mat.row(i) = Map<RowVectorXf>(data[i].data(), cols);
    return mat;
}

// 读取复数矩阵文件
inline MatrixXcf loadComplexMatrixFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open the file: " + filename);
    }

    std::vector<std::vector<std::complex<float>>> data;
    std::string line;
    
    // 第一遍：确定矩阵维度
    size_t cols = 0;
    while (std::getline(file, line)) {
        std::vector<std::complex<float>> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            std::smatch match;
            if (!std::regex_match(cell, match, std::regex(" \\(([+-]*[0-9]+\\.[0-9]+)([+-])([+-]*[0-9]+\\.[0-9]+)j\\)"))) {
                throw std::runtime_error("invalid complex-format: " + cell);
            }
            
            float real = std::stof(match[1]);
            float imag = (match[2] == "+") ? 
                         std::stof(match[3]) : -std::stof(match[3]);
            row.emplace_back(real, imag);
        }
        
        if (cols == 0) cols = row.size();
        else if (row.size() != cols) {
            throw std::runtime_error("row.size() != cols");
        }
        data.push_back(row);
    }
    file.close();

    // 转换为 Eigen 矩阵
    MatrixXcf mat(data.size(), cols);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}
