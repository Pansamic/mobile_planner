#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <mobile_planner/mobile_planner.h>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

int main()
{
    const std::array<std::string, 7> map_names = {"elevation", "elevation_filtered", "uncertainty", "slope", "step_height", "roughness", "traversability"};
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Create temp directory if it doesn't exist
    std::filesystem::create_directories("temp");

    // Load point cloud from file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string filename = "assets/room.ply";
    
    if (pcl::io::loadPLYFile(filename, *cloud) == -1)
    {
        std::cerr << "Failed to load point cloud from " << filename << std::endl;
        return -1;
    }
    
    std::cout << "Loaded point cloud with " << cloud->points.size() << " points" << std::endl;

    // Load parameters from config file
    YAML::Node config = YAML::LoadFile("config/config.yaml");
    
    float resolution = config["grid_map"]["resolution"].as<float>();
    std::size_t length_x = config["grid_map"]["default_length_x"].as<std::size_t>();
    std::size_t length_y = config["grid_map"]["default_length_y"].as<std::size_t>();
    std::size_t max_top_points_in_grid = config["grid_map"]["max_top_points_in_grid"].as<std::size_t>();
    std::string elevation_map_filter_type = config["elevation_map"]["elevation_map_filter_type"].as<std::string>();
    
    float max_height = config["elevation_map"]["max_height"].as<float>();
    std::string method = config["elevation_map"]["method"].as<std::string>();
    
    // Direct method parameters
    float slope_weight = config["elevation_map"]["direct"]["slope_weight"].as<float>();
    float step_height_weight = config["elevation_map"]["direct"]["step_height_weight"].as<float>();
    float roughness_weight = config["elevation_map"]["direct"]["roughness_weight"].as<float>();
    float slope_critical = config["elevation_map"]["direct"]["slope_critical"].as<float>();
    float step_height_critical = config["elevation_map"]["direct"]["step_height_critical"].as<float>();
    float roughness_critical = config["elevation_map"]["direct"]["roughness_critical"].as<float>();
    
    // Gaussian process parameters
    float l = config["elevation_map"]["gaussian_process"]["l"].as<float>();
    float sigma_f = config["elevation_map"]["gaussian_process"]["sigma_f"].as<float>();
    float sigma_n = config["elevation_map"]["gaussian_process"]["sigma_n"].as<float>();

    // Planner parameters
    std::string planner_method = config["planner"]["method"].as<std::string>();
    float traversability_threshold = config["planner"]["traversability_threshold"].as<float>();
    
    ElevationMap elevation_map(
        method,
        length_x,
        length_y,
        resolution,
        max_top_points_in_grid,
        elevation_map_filter_type,
        max_height,
        slope_weight,
        step_height_weight,
        roughness_weight,
        slope_critical,
        step_height_critical,
        roughness_critical,
        l,
        sigma_f,
        sigma_n
    );

    // Update the map with the point cloud
    auto processing_start = std::chrono::high_resolution_clock::now();
    elevation_map.update(cloud);
    auto processing_end = std::chrono::high_resolution_clock::now();

    const std::vector<Eigen::MatrixXf>& maps = elevation_map.getMaps();
    const Eigen::MatrixXf& traversability_map = maps[ElevationMap::TRAVERSABILITY];

    // Export all maps as binary files
    for ( std::size_t i = 0; i < maps.size(); i++)
    {
        const Eigen::MatrixXf& map = maps[i];
        const std::string& map_name = map_names[i];
        
        std::string filepath = "temp/binary/maps_direct_static/" + map_name + ".bin";
        
        // Create directory if it doesn't exist
        std::filesystem::create_directories("temp/binary/maps_direct_static/");
        
        std::ofstream file(filepath, std::ios::binary);
        
        if (file.is_open())
        {
            // Write matrix dimensions
            int rows = map.rows();
            int cols = map.cols();
            file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
            file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
            
            // Write matrix data
            file.write(reinterpret_cast<const char*>(map.data()), sizeof(float) * rows * cols);
            
            file.close();
            std::cout << "Exported " << map_name << " map to " << filepath << std::endl;
        }
        else
        {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        }
    }

    // Path planning
    MobilePlanner planner(elevation_map, planner_method, traversability_threshold);
    
    // Define start and goal positions (in world coordinates)
    Eigen::Transform<float, 3, Eigen::Affine> start_transform = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
    Eigen::Transform<float, 3, Eigen::Affine> goal_transform = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
    
    // Set start position to (2.5, -12, 0)
    start_transform.translation() << 2.5f, -12.0f, 0.0f;
    
    // Set goal position to (7, -9, 0)
    goal_transform.translation() << 7.0f, -9.0f, 0.0f;
    
    // Check reachability
    bool reachable = planner.checkReachability(start_transform, goal_transform);
    std::cout << "Path is " << (reachable ? "reachable" : "not reachable") << std::endl;
    
    // Plan path
    std::vector<Eigen::Vector2f> waypoints = planner.plan(start_transform, goal_transform);
    
    std::cout << "Found path with " << waypoints.size() << " waypoints" << std::endl;
    
    // Export waypoints as binary file
    std::string waypoints_filepath = "temp/binary/maps_direct_static/waypoints.bin";
    std::ofstream waypoints_file(waypoints_filepath, std::ios::binary);
    
    if (waypoints_file.is_open())
    {
        // Write number of waypoints
        int num_waypoints = waypoints.size();
        waypoints_file.write(reinterpret_cast<const char*>(&num_waypoints), sizeof(int));
        
        // Write waypoints data
        for (const auto& waypoint : waypoints)
        {
            float data[2] = {waypoint.x(), waypoint.y()};
            waypoints_file.write(reinterpret_cast<const char*>(data), sizeof(float) * 2);
        }
        
        waypoints_file.close();
        std::cout << "Exported waypoints to " << waypoints_filepath << std::endl;
    }
    else
    {
        std::cerr << "Failed to open waypoints file for writing: " << waypoints_filepath << std::endl;
    }

    // Print timing information
    auto end_time = std::chrono::high_resolution_clock::now();
    auto processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end - processing_start);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Point cloud processing time: " << processing_duration.count() << " ms" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << " ms" << std::endl;

    return 0;
}