/**
 * @file elevation_map.h
 * @author pansamic (pansamic@foxmail.com)
 * @brief Elevation map implementation for mobile robot navigation
 * @version 0.1
 * @date 2025-10-12
 * 
 * @copyright Copyright (c) 2025
 * 
 * This file implements the elevation mapping functionality for mobile robot navigation.
 * It includes point cloud processing, elevation estimation with uncertainty, and 
 * traversability analysis.
 */
#include <cstdint>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <mobile_planner/grid_map.h>

/**
 * @class ElevationMap
 * @brief Elevation mapping class for mobile robot navigation
 * 
 * This class implements elevation mapping functionality for mobile robots. It processes
 * point cloud data to build elevation maps with uncertainty, and computes traversability
 * metrics for navigation planning.
 */
class ElevationMap : public GridMap
{
public:
    /**
     * @brief Enum for map types
     */
    enum MapType {
        ELEVATION = 0,
        ELEVATION_FILTERED,
        UNCERTAINTY,
        SLOPE,
        STEP_HEIGHT,
        ROUGHNESS,
        TRAVERSABILITY,
    };

    /**
     * @brief Delete default constructor
     */
    ElevationMap() = delete;
    
    /**
     * @brief Construct a new Elevation Map object
     *
     * @param method Method to use for updating the elevation map, "direct" or "gaussian_process"
     * @param length_x Number of rows in the grid map
     * @param length_y Number of columns in the grid map
     * @param resolution Resolution of the grid map in meters per cell
     * @param num_max_points_in_grid Maximum number of points to keep per grid cell
     * @param elevation_map_filter_type Type of elevation map filter to use (Gaussian, boxblur, bilateral, etc.)
     * @param max_height Maximum height of point cloud (h_max)
     * @param slope_weight Weight for slope map (ω1)
     * @param step_height_weight Weight for step height map (ω2)
     * @param roughness_weight Weight for roughness map (ω3)
     * @param slope_critical Critical slope threshold (s_crit)
     * @param step_height_critical Critical step height threshold (ζ_crit)
     * @param roughness_critical Critical roughness threshold (r_crit)
     * @param l Gaussian process length scale parameter
     * @param sigma_f Gaussian process signal variance parameter
     * @param sigma_n Gaussian process noise variance parameter
     */
    explicit ElevationMap(
        const std::string& method,
        float length_x,
        float length_y,
        float resolution,
        std::size_t num_max_points_in_grid,
        const std::string& elevation_map_filter_type,
        float max_height,
        float slope_weight,
        float step_height_weight,
        float roughness_weight,
        float slope_critical,
        float step_height_critical,
        float roughness_critical,
        float l,
        float sigma_f,
        float sigma_n
    );
    
    /**
     * @brief Destroy the Elevation Map object
     */
    virtual ~ElevationMap() = default;
    
    /**
     * @brief Update elevation map with a new point cloud frame with method specified by `method_`
     * @param point_cloud Input point cloud frame aligned with odometry
     */
    void update( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud );

    /**
     * @brief Request traversability map
     * 
     * Returns the computed traversability map.
     * 
     * @return Eigen::MatrixXf The traversability map
     */
    const std::vector<Eigen::MatrixXf>& getMaps();
    
    // Getter methods for algorithm parameters
    float getSlopeWeight() const { return slope_weight_; }
    float getStepHeightWeight() const { return step_height_weight_; }
    float getRoughnessWeight() const { return roughness_weight_; }
    float getSlopeCritical() const { return slope_critical_; }
    float getStepHeightCritical() const { return step_height_critical_; }
    float getRoughnessCritical() const { return roughness_critical_; }
    float getGPLengthScale() const { return l_; }
    float getGPSignalVariance() const { return sigma_f_; }
    float getGPNoiseVariance() const { return sigma_n_; }
    
    // Setter methods for algorithm parameters
    void setSlopeWeight(float weight) { slope_weight_ = weight; }
    void setStepHeightWeight(float weight) { step_height_weight_ = weight; }
    void setRoughnessWeight(float weight) { roughness_weight_ = weight; }
    void setSlopeCritical(float critical) { slope_critical_ = critical; }
    void setStepHeightCritical(float critical) { step_height_critical_ = critical; }
    void setRoughnessCritical(float critical) { roughness_critical_ = critical; }
    void setGPLengthScale(float l) { l_ = l; }
    void setGPSignalVariance(float sigma_f) { sigma_f_ = sigma_f; }
    void setGPNoiseVariance(float sigma_n) { sigma_n_ = sigma_n; }

private:
    // Method to use for updating the elevation map, "direct" or "gaussian_process"
    std::string method_;

    std::size_t num_max_points_in_grid_;

    std::string elevation_map_filter_type_;

    // maximum height of point cloud. any point above this height will be removed.
    float max_height_;

    // Algorithm parameters for traversability map computation
    float slope_weight_;
    float step_height_weight_;
    float roughness_weight_;
    float slope_critical_;
    float step_height_critical_;
    float roughness_critical_;

    // Gaussian process regression parameters
    float l_;             // Length scale
    float sigma_f_;       // Signal variance
    float sigma_n_;       // Noise variance

    /**
     * @brief Update elevation map with a new point cloud frame using direct method
     * 
     * Processes a point cloud frame and updates the elevation map using Kalman filtering.
     * Automatically extends the map if the point cloud exceeds current boundaries.
     * Also computes slope map, step height map, roughness map and traversability map.
     * 
     * @param point_cloud Input point cloud frame aligned with odometry
     */
    void updateDirect( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud );
    
    /**
     * @brief Update elevation map using full Gaussian Process regression
     * 
     * Processes the entire point cloud using Gaussian Process regression to build
     * a complete elevation map. This is suitable for offline processing of full maps.
     * 
     * @param point_cloud Input point cloud containing the full map data
     */
    void updateFullGP( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud );
    
    /**
     * @brief Divide point cloud into grid cells
     * 
     * Partitions a point cloud into grid cells based on XY coordinates.
     * 
     * @param cell_heights Output map of grid cell indices to point vectors
     * @param point_cloud Input point cloud to partition
     */
    void dividePointCloudToGridCells(
        std::vector<std::vector<float>>& cell_heights,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud
    ) const;
    
    /**
     * @brief Extract top surface points from each grid cell
     * 
     * For each grid cell, keeps only the highest points up to the maximum limit.
     * 
     * @param cell_heights Map of grid cell indices to point vectors (modified in place)
     */
    void extractPointCloudTopSurface( std::vector<std::vector<float>>& cell_heights ) const;
    
    /**
     * @brief Remove points above maximum height limit
     * 
     * Removes points from the point cloud that are above the maximum height limit.
     * 
     * @param cell_heights Map of grid cell indices to point vectors (modified in place)
     */
    void removeOverHeightPoints( pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud ) const;

    /**
     * @brief Fuse elevation measurements using 1D Kalman filter
     * 
     * Fuses a new elevation measurement with an existing one using 1D Kalman filter.
     * 
     * @param mean Reference to existing mean (updated in place)
     * @param variance Reference to existing variance (updated in place)
     * @param new_mean Mean of new measurement
     * @param new_variance Variance of new measurement
     */
    void fuseElevationWithKalmanFilter(float& mean, float& variance, float new_mean, float new_variance) const;
    
    /**
     * @brief Check if map needs to be extended and extend if needed
     * 
     * Checks if the point cloud exceeds the current map boundaries and extends
     * the map if necessary, maintaining the origin at the center.
     * 
     * @param point_cloud Input point cloud to check
     * @return true If map was extended
     * @return false If map was not extended
     */
    bool checkAndExtendMapIfNeeded( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud );
    
    /**
     * @brief Fill NaN values with minimum value in 3x3 window
     * 
     * Replaces NaN values with the minimum valid value in a 3x3 neighborhood.
     * 
     * @param map Reference to map to process (modified in place)
     */
    void fillNaNWithMinimumInWindow( Eigen::MatrixXf& map );
    
    /**
     * @brief Apply filter to elevation map
     * 
     * Selects and applies the specified filter to elevation map.
     * 
     * @param filter_type Type of filter to apply ("boxblur", "gaussian", or "bilateral")
     */
    void filterElevation(const std::string& filter_type);
    
    /**
     * @brief Apply box blur filter to map
     * 
     * Applies a 3x3 box blur filter to the map, handling NaN values properly.
     * 
     * @param map Reference to map to filter (modified in place)
     */
    void applyBoxBlurFilter( Eigen::MatrixXf& map );
    
    /**
     * @brief Apply Gaussian filter to map
     * 
     * Applies a 5x5 Gaussian filter to the map, handling NaN values properly.
     * 
     * @param map Reference to map to filter (modified in place)
     */
    void applyGaussianFilter( Eigen::MatrixXf& map );
    
    /**
     * @brief Apply bilateral filter to map
     * 
     * Applies a 5x5 bilateral filter to the map, handling NaN values properly.
     * 
     * @param map Reference to map to filter (modified in place)
     */
    void applyBilateralFilter( Eigen::MatrixXf& map );
    
    /**
     * @brief Compute slope map from elevation data
     * 
     * Calculates the magnitude of the gradient of the elevation surface at each cell.
     * Uses finite difference method on a regular grid.
     */
    void computeSlopeMap();
    
    /**
     * @brief Compute step height map from elevation data
     * 
     * Measures the maximum vertical discontinuity within a local neighborhood.
     * Calculated as the difference between maximum and minimum elevation in a window.
     */
    void computeStepHeightMap();
    
    /**
     * @brief Compute roughness map from elevation data
     * 
     * Calculates the standard deviation of elevation within a local window.
     */
    void computeRoughnessMap();
    
    /**
     * @brief Compute traversability map from slope, step height, and roughness maps
     * 
     * Combines the three maps with weights to produce a single traversability metric.
     */
    void computeTraversabilityMap();
};