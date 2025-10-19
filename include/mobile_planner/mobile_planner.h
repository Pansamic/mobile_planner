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
#include <Eigen/Core>
#include <Eigen/Geometry>

class MobilePlanner
{
public:
    MobilePlanner(const std::string& method, float traversability_threshold);
    virtual ~MobilePlanner() = default;
    
    /**
     * @brief Plan a path using the specified method
     * 
     * @param traversability_map The traversability map to plan on
     * @param start Start position transform
     * @param goal Goal position transform
     * @return std::vector<Eigen::Vector2f> Waypoints of the planned path
     */
    std::vector<Eigen::Vector2f> plan(
        const Eigen::MatrixXf& traversability_map,
        const Eigen::Transform<float, 3, Eigen::Affine>& start,
        const Eigen::Transform<float, 3, Eigen::Affine>& goal
    );

    /**
     * @brief Check if a path is reachable between start and goal positions
     * 
     * @param traversability_map The traversability map to check on
     * @param start Start position transform
     * @param goal Goal position transform
     * @return true if reachable, false otherwise
     */
    bool checkReachability(
        const Eigen::MatrixXf& traversability_map,
        const Eigen::Transform<float, 3, Eigen::Affine>& start,
        const Eigen::Transform<float, 3, Eigen::Affine>& goal
    );

private:
    /**
     * @brief A* path planning algorithm implementation
     * 
     * @param traversability_map The traversability map to plan on
     * @param start Start position (row, col)
     * @param goal Goal position (row, col)
     * @return std::vector<Eigen::Vector2f> Waypoints of the planned path
     */
    std::vector<Eigen::Vector2f> planAStar(
        const Eigen::MatrixXf& traversability_map,
        const Eigen::Vector2i& start,
        const Eigen::Vector2i& goal
    );

    // Planning method to use
    std::string method_;
    
    // Threshold for traversability (values above this are considered non-traversable)
    float traversability_threshold_;
};

#endif // __MOBILE_PLANNER_H__