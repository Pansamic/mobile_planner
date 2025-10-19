/**
 * @file grid_map.cpp
 * @author pansamic (pansamic@foxmail.com)
 * @brief Grid map implementation for mobile planning
 * @version 0.1
 * @date 2025-09-18
 * 
 * @copyright Copyright (c) 2025
 */
#include <mobile_planner/grid_map.h>
#include <Eigen/Core>

GridMap::GridMap(std::size_t num_maps, float length_x, float length_y, float resolution)
    : rows_(static_cast<std::size_t>(length_x / resolution)),
      cols_(static_cast<std::size_t>(length_y / resolution)),
      length_x_(length_x),
      length_y_(length_y),
      resolution_(resolution)
{
    maps_.resize(num_maps, Eigen::MatrixXf::Zero(rows_, cols_));
}

std::size_t GridMap::getRows() const
{
    return rows_;
}

std::size_t GridMap::getColumns() const
{
    return cols_;
}

float GridMap::getLengthX() const
{
    return length_x_;
}

float GridMap::getLengthY() const
{
    return length_y_;
}

float GridMap::getResolution() const
{
    return resolution_;
}

void GridMap::add(const std::string name)
{
    // TODO: Implement map naming functionality
    maps_.emplace_back(Eigen::MatrixXf::Zero(rows_, cols_));
}

bool GridMap::resize(float length_x, float length_y)
{
    if (length_x <= 0 || length_y <= 0)
    {
        return false;
    }

    // Calculate new dimensions
    std::size_t new_rows = static_cast<std::size_t>(length_x / resolution_);
    std::size_t new_cols = static_cast<std::size_t>(length_y / resolution_);
    
    // Resize all maps
    for (auto& map : maps_)
    {
        map.resize(new_rows, new_cols);
    }
    
    // Update dimensions
    rows_ = new_rows;
    cols_ = new_cols;
    length_x_ = length_x;
    length_y_ = length_y;
    
    return true;
}