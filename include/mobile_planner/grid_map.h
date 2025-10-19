/**
 * @file grid_map.h
 * @author pansamic (pansamic@foxmail.com)
 * @brief Grid map base class for mobile robot navigation
 * @version 0.1
 * @date 2025-10-12
 * 
 * @copyright Copyright (c) 2025
 * 
 * This file implements the base grid map functionality for mobile robot navigation.
 */
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * @class GridMap
 * @brief Base grid map class for mobile robot navigation
 * 
 * This class implements the base grid map functionality that can be extended by
 * specialized map classes. It manages a collection of 2D grid maps with common
 * spatial properties.
 */
class GridMap
{
public:
    /**
     * @brief Delete default constructor
     */
    GridMap() = delete;
    
    /**
     * @brief Construct a new Grid Map object
     * 
     * @param num_maps Number of maps to initialize
     * @param length_x Length of map in x-direction
     * @param length_y Length of map in y-direction
     * @param resolution Resolution of the grid map in meters per cell
     */
    explicit GridMap(std::size_t num_maps, float length_x, float length_y, float resolution);
    
    /**
     * @brief Destroy the Grid Map object
     */
    virtual ~GridMap() = default;
    
    /**
     * @brief Get number of rows in the grid map
     * @return std::size_t Number of rows
     */
    std::size_t getRows() const;
    
    /**
     * @brief Get number of columns in the grid map
     * @return std::size_t Number of columns
     */
    std::size_t getColumns() const;
    
    /**
     * @brief Get length of map in x-direction
     * @return float Length in x-direction
     */
    float getLengthX() const;
    
    /**
     * @brief Get length of map in y-direction
     * @return float Length in y-direction
     */
    float getLengthY() const;
    
    /**
     * @brief Get resolution of the grid map
     * @return float Resolution in meters per cell
     */
    float getResolution() const;
    
    /**
     * @brief Add a new map with the given name
     * @param name Name of the map to add
     */
    void add(const std::string name);
    
    /**
     * @brief Resize the grid map
     * 
     * @param length_x New length in x-direction
     * @param length_y New length in y-direction
     * @return true if resized successfully
     * @return false if resize failed
     */
    bool resize(float length_x, float length_y);

protected:
    /// Collection of maps
    std::vector<Eigen::MatrixXf> maps_;

    /// Map rows, start from 1
    std::size_t rows_;
    
    /// Map columns, start from 1
    std::size_t cols_;
    
    /// Length of map in x-direction
    float length_x_;
    
    /// Length of map in y-direction
    float length_y_;
    
    /// Resolution of the grid map in meters per cell
    float resolution_;
};