/**
 * @file point_cloud_process.cpp
 * @author pansamic (pansamic@foxmail.com)
 * @brief Implementation of point cloud processing functions
 * @version 0.1
 * @date 2025-10-16
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <mobile_planner/point_cloud_process.h>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Custom hash function for std::pair<int, int>
struct PairHash
{
    std::size_t operator()( const std::pair<int, int>& p ) const
    {
        // Combine hashes of the two integers using a simple hash combining technique
        std::size_t h1 = std::hash<int>{}( p.first );
        std::size_t h2 = std::hash<int>{}( p.second );
        return h1 ^ ( h2 << 1 );
    }
};

void extractPointCloudTopSurface(
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_point_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr old_point_cloud,
    float length,
    float width,
    float resolution,
    std::size_t num_max_points_in_grid
)
{
    // Create a map to store points for each grid cell
    std::unordered_map<std::pair<int, int>, std::vector<pcl::PointXYZ>, PairHash> grid_points;
    
    // Calculate map boundaries
    float min_x = -length / 2.0f;
    float max_x = length / 2.0f;
    float min_y = -width / 2.0f;
    float max_y = width / 2.0f;
    
    // Assign points to grid cells
    for ( const auto& point : old_point_cloud->points )
    {
        // Check if point is within map boundaries
        if ( point.x < min_x || point.x > max_x || point.y < min_y || point.y > max_y )
        {
            continue;
        }
        
        // Calculate grid cell indices
        int row = static_cast<int>( ( point.y - min_y ) / resolution );
        int col = static_cast<int>( ( point.x - min_x ) / resolution );
        
        // Ensure indices are within bounds
        if ( row >= 0 && row < static_cast<int>( width / resolution ) && 
             col >= 0 && col < static_cast<int>( length / resolution ) )
        {
            grid_points[std::make_pair( row, col )].push_back( point );
        }
    }
    
    // Clear the new point cloud
    new_point_cloud->clear();
    
    // For each grid cell, keep only the highest points (up to num_max_points_in_grid)
    for ( const auto& cell : grid_points )
    {
        auto points = cell.second;
        
        // Sort points by Z-coordinate in descending order
        std::sort( points.begin(), points.end(), []( const pcl::PointXYZ& a, const pcl::PointXYZ& b )
        {
            return a.z > b.z;
        } );
        
        // Keep only the first num_max_points_in_grid points (highest ones)
        std::size_t num_points_to_keep = std::min( num_max_points_in_grid, points.size() );
        for ( std::size_t i = 0; i < num_points_to_keep; ++i )
        {
            new_point_cloud->points.push_back( points[i] );
        }
    }
    
    new_point_cloud->width = new_point_cloud->points.size();
    new_point_cloud->height = 1;
}