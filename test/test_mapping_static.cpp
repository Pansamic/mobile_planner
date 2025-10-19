#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <mobile_planner/elevation_map.h>

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
            
            std::cout << "Exported " << map_name << " map (" << rows << "x" << cols << ") to " << filepath << std::endl;
        }
        else
        {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
        }
    }
    
    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();
    
    // Calculate durations
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end - processing_start);
    
    std::cout << "Total computation time: " << total_duration.count() << " ms" << std::endl;
    std::cout << "Point cloud processing time: " << processing_duration.count() << " ms" << std::endl;
    
    std::cout << "All maps exported successfully!" << std::endl;
    
    return 0;
}