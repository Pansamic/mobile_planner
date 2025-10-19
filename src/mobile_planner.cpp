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
#include <iostream>
MobilePlanner::MobilePlanner(const std::string& method, float traversability_threshold)
    : method_(method), traversability_threshold_(traversability_threshold)
{
}

std::vector<Eigen::Vector2f> MobilePlanner::plan(
    const Eigen::MatrixXf& traversability_map,
    const Eigen::Transform<float, 3, Eigen::Affine>& start,
    const Eigen::Transform<float, 3, Eigen::Affine>& goal
)
{
    // For now, we'll use a fixed resolution assumption
    // In a more complete implementation, this would be passed as a parameter
    float resolution = 0.1f;
    int rows = traversability_map.rows();
    int cols = traversability_map.cols();
    
    // Convert world coordinates to grid coordinates
    // Assuming map is centered at origin
    Eigen::Vector3f start_pos = start.translation();
    Eigen::Vector3f goal_pos = goal.translation();
    
    Eigen::Vector2i start_grid(
        static_cast<int>(rows / 2 + start_pos.x() / resolution),
        static_cast<int>(cols / 2 + start_pos.y() / resolution)
    );
    
    Eigen::Vector2i goal_grid(
        static_cast<int>(rows / 2 + goal_pos.x() / resolution),
        static_cast<int>(cols / 2 + goal_pos.y() / resolution)
    );
    
    // Clamp to valid range
    start_grid.x() = std::max(0, std::min(rows - 1, start_grid.x()));
    start_grid.y() = std::max(0, std::min(cols - 1, start_grid.y()));
    goal_grid.x() = std::max(0, std::min(rows - 1, goal_grid.x()));
    goal_grid.y() = std::max(0, std::min(cols - 1, goal_grid.y()));
    
    // Check if start and goal positions are traversable
    if (std::isnan(traversability_map(start_grid.x(), start_grid.y())) || 
        traversability_map(start_grid.x(), start_grid.y()) > traversability_threshold_)
    {
        std::cout << "Start position is not traversable" << std::endl;
        std::cout << "Start position traversability: " << traversability_map(start_grid.x(), start_grid.y()) << std::endl;
        // Start position is not traversable
        return std::vector<Eigen::Vector2f>();
    }
    
    if (std::isnan(traversability_map(goal_grid.x(), goal_grid.y())) || 
        traversability_map(goal_grid.x(), goal_grid.y()) > traversability_threshold_)
    {
        std::cout << "Goal position is not traversable" << std::endl;
        std::cout << "Goal position traversability: " << traversability_map(goal_grid.x(), goal_grid.y()) << std::endl;
        // Goal position is not traversable
        return std::vector<Eigen::Vector2f>();
    }
    
    // Call appropriate planning method
    if (method_ == "astar")
    {
        return planAStar(traversability_map, start_grid, goal_grid);
    }
    
    // Default return empty path
    return std::vector<Eigen::Vector2f>();
}

bool MobilePlanner::checkReachability(
    const Eigen::MatrixXf& traversability_map,
    const Eigen::Transform<float, 3, Eigen::Affine>& start,
    const Eigen::Transform<float, 3, Eigen::Affine>& goal
)
{
    // Simple check: if we can plan a path, it's reachable
    std::vector<Eigen::Vector2f> path = plan(traversability_map, start, goal);
    return !path.empty();
}

std::vector<Eigen::Vector2f> MobilePlanner::planAStar(
    const Eigen::MatrixXf& traversability_map,
    const Eigen::Vector2i& start,
    const Eigen::Vector2i& goal
)
{
    int rows = traversability_map.rows();
    int cols = traversability_map.cols();
    
    // A* algorithm implementation
    struct Node {
        int x, y;
        float g, h, f;
        bool operator>(const Node& other) const { return f > other.f; }
    };
    
    // Directions for 8-connected grid
    const int dx[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    const int dy[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const float cost[8] = {
        1.414f, 1.0f, 1.414f, 
        1.0f,           1.0f, 
        1.414f, 1.0f, 1.414f
    };
    
    // Create maps for g_score, f_score and visited
    Eigen::MatrixXf g_score = Eigen::MatrixXf::Constant(rows, cols, std::numeric_limits<float>::max());
    Eigen::MatrixXf f_score = Eigen::MatrixXf::Constant(rows, cols, std::numeric_limits<float>::max());
    Eigen::MatrixXi visited = Eigen::MatrixXi::Zero(rows, cols);
    
    // Priority queue for open set
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    
    // Initialize start node
    g_score(start.x(), start.y()) = 0.0f;
    float h_start = std::hypot(goal.x() - start.x(), goal.y() - start.y());
    f_score(start.x(), start.y()) = h_start;
    
    open_set.push({start.x(), start.y(), 0.0f, h_start, h_start});
    
    // Parent tracking for path reconstruction
    std::vector<std::vector<std::pair<int, int>>> parent(rows, std::vector<std::pair<int, int>>(cols, {-1, -1}));
    
    while (!open_set.empty()) {
        Node current = open_set.top();
        open_set.pop();
        
        // Check if we reached the goal
        if (current.x == goal.x() && current.y == goal.y()) {
            // Reconstruct path
            std::vector<Eigen::Vector2f> path;
            int x = current.x, y = current.y;
            
            while (x != -1 && y != -1) {
                path.push_back(Eigen::Vector2f(x, y));
                auto p = parent[x][y];
                x = p.first;
                y = p.second;
            }
            
            // Reverse path to get start->goal order
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        // Skip if already visited
        if (visited(current.x, current.y)) {
            continue;
        }
        
        visited(current.x, current.y) = 1;
        
        // Explore neighbors
        for (int i = 0; i < 8; i++) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            
            // Check bounds
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) {
                // Check if traversable and not visited
                if (!visited(nx, ny) && !std::isnan(traversability_map(nx, ny)) && 
                    traversability_map(nx, ny) <= traversability_threshold_) {
                    
                    float tentative_g = current.g + cost[i];
                    
                    if (tentative_g < g_score(nx, ny)) {
                        parent[nx][ny] = {current.x, current.y};
                        g_score(nx, ny) = tentative_g;
                        float h = std::hypot(goal.x() - nx, goal.y() - ny);
                        f_score(nx, ny) = tentative_g + h;
                        
                        open_set.push({nx, ny, tentative_g, h, tentative_g + h});
                    }
                }
            }
        }
    }
    
    // No path found
    return std::vector<Eigen::Vector2f>();
}