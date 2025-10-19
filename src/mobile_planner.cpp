/**
 * @file mobile_planner.cpp
 * @author pansamic (pansamic@foxmail.com)
 * @brief Implementation of mobile planner for global path planning
 * @version 0.1
 * @date 2025-10-19
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <mobile_planner/mobile_planner.h>
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>

MobilePlanner::MobilePlanner(const ElevationMap& elevation_map, const std::string& method, float traversability_threshold)
    : elevation_map_(elevation_map), method_(method), traversability_threshold_(traversability_threshold)
{
}

std::vector<Eigen::Vector2f> MobilePlanner::plan(
    const Eigen::Transform<float, 3, Eigen::Affine>& start,
    const Eigen::Transform<float, 3, Eigen::Affine>& goal
)
{
    // Convert world coordinates to grid coordinates
    Eigen::Vector3f start_pos = start.translation();
    Eigen::Vector3f goal_pos = goal.translation();
    
    // Check coordinate validity
    if (!elevation_map_.isValidCoordinate(start_pos.x(), start_pos.y()) ||
        !elevation_map_.isValidCoordinate(goal_pos.x(), goal_pos.y())) {
        // Invalid start or goal position
        return std::vector<Eigen::Vector2f>();
    }

    // Check if start and goal positions are traversable
    float start_traversability = elevation_map_.getMapValue(ElevationMap::TRAVERSABILITY, start_pos.x(), start_pos.y());
    float goal_traversability = elevation_map_.getMapValue(ElevationMap::TRAVERSABILITY, goal_pos.x(), goal_pos.y());
    
    if (std::isnan(start_traversability) || start_traversability > traversability_threshold_) {
        // Start position is not traversable
        return std::vector<Eigen::Vector2f>();
    }
    
    if (std::isnan(goal_traversability) || goal_traversability > traversability_threshold_) {
        // Goal position is not traversable
        return std::vector<Eigen::Vector2f>();
    }
    
    // Call appropriate planning method
    if (method_ == "astar") {
        // Convert 3D vectors to 2D (x, y only)
        Eigen::Vector2f start_2d(start_pos.x(), start_pos.y());
        Eigen::Vector2f goal_2d(goal_pos.x(), goal_pos.y());
        return planAStar(start_2d, goal_2d);
    }
    
    // Default return empty path
    return std::vector<Eigen::Vector2f>();
}

bool MobilePlanner::checkReachability(
    const Eigen::Transform<float, 3, Eigen::Affine>& start,
    const Eigen::Transform<float, 3, Eigen::Affine>& goal
)
{
    // Simple check: if we can plan a path, it's reachable
    std::vector<Eigen::Vector2f> path = plan(start, goal);
    return !path.empty();
}

std::vector<Eigen::Vector2f> MobilePlanner::planAStar(
    const Eigen::Vector2f& start,
    const Eigen::Vector2f& goal
)
{
    // A* algorithm implementation
    struct Node {
        float x, y;  // World coordinates
        float g, h, f;
        bool operator>(const Node& other) const { return f > other.f; }
    };
    
    // Directions for 8-connected grid
    const float dx[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    const float dy[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const float cost[8] = {
        1.414f, 1.0f, 1.414f, 
        1.0f,           1.0f, 
        1.414f, 1.0f, 1.414f
    };
    
    // Get resolution from elevation map
    float resolution = elevation_map_.getResolution();
    
    // Create a map to track visited nodes (using a set of coordinate pairs)
    std::set<std::pair<int, int>> visited;
    
    // Priority queue for open set
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    
    // Initialize start node
    float h_start = std::hypot(goal.x() - start.x(), goal.y() - start.y());
    open_set.push({start.x(), start.y(), 0.0f, h_start, h_start});
    
    // Parent tracking for path reconstruction
    std::map<std::pair<float, float>, std::pair<float, float>> parent;
    
    // G score map
    std::map<std::pair<float, float>, float> g_score;
    g_score[std::make_pair(start.x(), start.y())] = 0.0f;
    
    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();
        
        // Check if we reached the goal (within one cell)
        if (std::hypot(current.x - goal.x(), current.y - goal.y()) <= resolution) {
            // Reconstruct path
            std::vector<Eigen::Vector2f> path;
            float x = current.x, y = current.y;
            
            std::pair<float, float> current_key = std::make_pair(x, y);
            path.push_back(Eigen::Vector2f(x, y));
            
            while (parent.find(current_key) != parent.end()) {
                auto p = parent[current_key];
                x = p.first;
                y = p.second;
                current_key = std::make_pair(x, y);
                path.push_back(Eigen::Vector2f(x, y));
            }
            
            // Reverse path to get start->goal order
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        // Skip if already visited
        std::pair<int, int> current_grid = std::make_pair(
            static_cast<int>(std::round(current.x / resolution)),
            static_cast<int>(std::round(current.y / resolution))
        );
        
        if (visited.find(current_grid) != visited.end()) {
            continue;
        }
        
        visited.insert(current_grid);
        
        // Explore neighbors
        for (int i = 0; i < 8; i++) {
            float nx = current.x + dx[i] * resolution;
            float ny = current.y + dy[i] * resolution;
            
            // Convert to grid coordinates for checking visited
            std::pair<int, int> n_grid = std::make_pair(
                static_cast<int>(std::round(nx / resolution)),
                static_cast<int>(std::round(ny / resolution))
            );
            
            // Check if not visited
            if (visited.find(n_grid) == visited.end()) {
                // Check if traversable
                float traversability = elevation_map_.getMapValue(ElevationMap::TRAVERSABILITY, nx, ny);
                
                if (!std::isnan(traversability) && traversability <= traversability_threshold_) {
                    float tentative_g = current.g + cost[i] * resolution;
                    
                    std::pair<float, float> n_key = std::make_pair(nx, ny);
                    if (g_score.find(n_key) == g_score.end() || tentative_g < g_score[n_key]) {
                        parent[n_key] = std::make_pair(current.x, current.y);
                        g_score[n_key] = tentative_g;
                        float h = std::hypot(goal.x() - nx, goal.y() - ny);
                        float f = tentative_g + h;
                        
                        open_set.push({nx, ny, tentative_g, h, f});
                    }
                }
            }
        }
    }
    
    // No path found
    return std::vector<Eigen::Vector2f>();
}