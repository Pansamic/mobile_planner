/**
 * @file gridmap.cpp
 * @author pansamic (pansamic@foxmail.com)
 * @brief 
 * @version 0.1
 * @date 2025-09-18
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <algorithm>
#include <mobile_planner/grid_map.h>

GridMap::GridMap(const std::vector<std::string>& names, std::size_t rows, std::size_t cols, float resolution)
{
    this->rows_ = rows;
    this->cols_ = cols;
    this->resolution_ = resolution;
    this->length_ = cols * resolution;
    this->width_ = rows * resolution;
    for ( auto& name : names )
    {
        this->maps_.emplace( name, Eigen::MatrixXf::Constant( rows, cols, std::numeric_limits<float>::quiet_NaN() ) );
    }
}

std::size_t GridMap::getRows() const
{
    return this->rows_;
}

std::size_t GridMap::getColumns() const
{
    return this->cols_;
}

float GridMap::getLength() const
{
    return this->length_;
}

float GridMap::getWidth() const
{
    return this->width_;
}

float GridMap::getResolution() const
{
    return this->resolution_;
}

void GridMap::add(const std::string name)
{
    this->maps_.emplace(name, Eigen::MatrixXf(this->rows_, this->cols_));
}

bool GridMap::resize(std::size_t rows, std::size_t cols)
{
    if ( rows == 0 || cols == 0 )
    {
        return false;
    }
    // Calculate the center of the old and new matrices
    std::size_t old_center_row = (this->rows_ - 1) / 2;
    std::size_t old_center_col = (this->cols_ - 1) / 2;
    std::size_t new_center_row = (rows - 1) / 2;
    std::size_t new_center_col = (cols - 1) / 2;

    // Calculate the top-left corner of the old matrix in the new matrix
    std::size_t new_start_row = new_center_row > old_center_row ? new_center_row - old_center_row : 0;
    std::size_t new_start_col = new_center_col > old_center_col ? new_center_col - old_center_col : 0;
    std::size_t old_start_row = old_center_row > new_center_row ? old_center_row - new_center_row : 0;
    std::size_t old_start_col = old_center_col > new_center_col ? old_center_col - new_center_col : 0;

    // Calculate the size of the overlapping region
    std::size_t min_rows = std::min(rows, this->rows_);
    std::size_t min_cols = std::min(cols, this->cols_);

    for (auto& pair : maps_)
    {
        Eigen::MatrixXf new_map = Eigen::MatrixXf::Constant(rows, cols, std::numeric_limits<float>::quiet_NaN());
        new_map.block(new_start_row, new_start_col, min_rows, min_cols) = pair.second.block(old_start_row, old_start_col, min_rows, min_cols);
        pair.second = std::move(new_map);
    }

    this->rows_ = rows;
    this->cols_ = cols;
    this->length_ = cols * this->resolution_;
    this->width_ = rows * this->resolution_;

    return true;
}