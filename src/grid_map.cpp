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

GridMap::GridMap(std::size_t num_maps, float length_x, float length_y, float resolution)
{
    // Keep rows and cols even numbers.
    rows_ = 2 * static_cast<std::size_t>( std::ceil ( length_x / 2 / resolution ) );
    cols_ = 2 * static_cast<std::size_t>( std::ceil ( length_y / 2 / resolution ) );
    resolution_ = resolution;
    length_x_ = rows_ * resolution;
    length_y_ = cols_ * resolution;
    
    // Initialize vector with the same number of maps as names
    maps_.resize(num_maps, Eigen::MatrixXf::Constant(rows_, cols_, std::numeric_limits<double>::quiet_NaN()));
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
    // Add a new map to the vector
    maps_.emplace_back(Eigen::MatrixXf::Constant(rows_, cols_, std::numeric_limits<double>::quiet_NaN()));
}

bool GridMap::resize(float length_x, float length_y)
{
    if ( length_x <= 0 || length_y <= 0 )
    {
        return false;
    }

    // Keep rows and cols even numbers.
    std::size_t new_rows = 2 * static_cast<std::size_t>( std::ceil ( length_x / 2 / resolution_ ) );
    std::size_t new_cols = 2 * static_cast<std::size_t>( std::ceil ( length_y / 2 / resolution_ ) );

    // Calculate the top-left corner of the old matrix in the new matrix
    std::size_t new_start_row = new_rows > rows_ ? ( ( new_rows - rows_ ) / 2 ) : 0;
    std::size_t new_start_col = new_cols > cols_ ? ( ( new_cols - cols_ ) / 2 ) : 0;
    std::size_t old_start_row = rows_ > new_rows ? ( ( rows_ - new_rows ) / 2 ) : 0;
    std::size_t old_start_col = cols_ > new_cols ? ( ( cols_ - new_cols ) / 2 ) : 0;

    // Calculate the size of the overlapping region
    std::size_t min_rows = std::min( new_rows, rows_ );
    std::size_t min_cols = std::min( new_cols, cols_ );

    for ( auto& map : maps_ )
    {
        Eigen::MatrixXf new_map = Eigen::MatrixXf::Constant(new_rows, new_cols, std::numeric_limits<double>::quiet_NaN());
        new_map.block(new_start_row, new_start_col, min_rows, min_cols) = map.block(old_start_row, old_start_col, min_rows, min_cols);
        map = std::move(new_map);
    }

    rows_ = new_rows;
    cols_ = new_cols;
    length_x_ = static_cast<float>( rows_ ) * resolution_;
    length_y_ = static_cast<float>( cols_ ) * resolution_;

    return true;
}