/**
 * @file mobile_planner.h
 * @author pansamic (pansamic@foxmail.com)
 * @brief Mobile planner for global path planning of mobile robots
 * @version 0.1
 * @date 2025-09-17
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef __MOBILE_PLANNER_H__
#define __MOBILE_PLANNER_H__

#include <string>
#include <vector>
#include <set>
#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mobile_planner/elevation_map.h>

/**
 * @class MobilePlanner
 * @brief Mobile planner for global path planning of mobile robots
 * 
 * This class implements global path planning functionality for mobile robots
 * using elevation and traversability maps.
 */
class MobilePlanner
{
public:
    /**
     * @brief Construct a new Mobile Planner object
     * 
     * @param elevation_map Reference to the elevation map
     * @param method Planning method to use
     * @param traversability_threshold Threshold for traversability (values above this are considered non-traversable)
     */
    MobilePlanner(
        const ElevationMap& elevation_map, 
        const std::string& method, 
        float traversability_threshold
    );
    
    /**
     * @brief Destroy the Mobile Planner object
     */
    virtual ~MobilePlanner() = default;
    
    /**
     * @brief Plan a path using the specified method
     * 
     * @param start Start position transform
     * @param goal Goal position transform
     * @return std::vector<Eigen::Vector2f> Waypoints of the planned path
     */
    std::vector<Eigen::Vector2f> plan(
        const Eigen::Transform<float, 3, Eigen::Affine>& start,
        const Eigen::Transform<float, 3, Eigen::Affine>& goal
    );

    /**
     * @brief Check if a path is reachable between start and goal positions
     * 
     * @param start Start position transform
     * @param goal Goal position transform
     * @return true if reachable, false otherwise
     */
    bool checkReachability(
        const Eigen::Transform<float, 3, Eigen::Affine>& start,
        const Eigen::Transform<float, 3, Eigen::Affine>& goal
    );

private:
    /**
     * @brief A* path planning algorithm implementation
     * 
     * @param start Start position (x, y in world coordinates)
     * @param goal Goal position (x, y in world coordinates)
     * @return std::vector<Eigen::Vector2f> Waypoints of the planned path
     */
    std::vector<Eigen::Vector2f> planAStar(
        const Eigen::Vector2f& start,
        const Eigen::Vector2f& goal
    );

    /// Reference to the elevation map
    const ElevationMap& elevation_map_;
    
    /// Planning method to use
    std::string method_;
    
    /// Threshold for traversability (values above this are considered non-traversable)
    float traversability_threshold_;
};

#endif // __MOBILE_PLANNER_H__