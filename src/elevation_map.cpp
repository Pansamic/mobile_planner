#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <mobile_planner/elevation_map.h>
#include <mobile_planner/point_cloud_process.h>

ElevationMap::ElevationMap(
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
)
    : GridMap(7, length_x, length_y, resolution),
      method_(method),
      num_max_points_in_grid_(num_max_points_in_grid),
      elevation_map_filter_type_(elevation_map_filter_type),
      max_height_(max_height),
      slope_weight_(slope_weight),
      step_height_weight_(step_height_weight),
      roughness_weight_(roughness_weight),
      slope_critical_(slope_critical),
      step_height_critical_(step_height_critical),
      roughness_critical_(roughness_critical),
      l_(l),
      sigma_f_(sigma_f),
      sigma_n_(sigma_n)
{
}

void ElevationMap::update(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{
    if (method_ == "direct")
    {
        updateDirect(point_cloud);
    }
    // TODO: Implement Gaussian Process method.
}

const std::vector<Eigen::MatrixXf>& ElevationMap::getMaps()
{
    return maps_;
}

float ElevationMap::getMapValue(MapType map_type, float x, float y) const
{
    auto [row, col] = getGridCellIndex(x, y);
    return maps_[map_type](row, col);
}

bool ElevationMap::isValidCoordinate(float x, float y) const
{
    return x >= -length_x_ / 2.0f && x <= length_x_ / 2.0f && 
           y >= -length_y_ / 2.0f && y <= length_y_ / 2.0f;
}

void ElevationMap::updateDirect(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{
    // Check if map needs to be extended to cover new point cloud
    checkAndExtendMapIfNeeded(point_cloud);

    removeOverHeightPoints(point_cloud);

    // Create a map to store points for each grid cell
    // Partition point cloud to grid cells
    auto cell_heights = dividePointCloudToGridCells(point_cloud);
    
    // Extract top surface points in each grid cell
    extractPointCloudTopSurface(cell_heights);
    
    // Update elevation map with new measurements using Kalman filter
    for ( auto cell : cell_heights )
    {
        if ( cell.second.empty() ) continue;
        
        std::size_t valid_amount = std::min(cell.second.size(), num_max_points_in_grid_);
        // Calculate mean and variance of points in this cell
        float sum_height = std::accumulate(cell.second.begin(), cell.second.begin() + valid_amount, 0.0f);
        float mean_height = sum_height / static_cast<float>(valid_amount);

        float sum_sq_diff = 0.0f;
        for ( float* height = cell.second.data(); height < cell.second.data() + valid_amount; height++ )
        {
            float diff = *height - mean_height;
            sum_sq_diff += diff * diff;
        }
        float var_height = sum_sq_diff / static_cast<float>(valid_amount);
        
        std::size_t row = cell.first.first;
        std::size_t col = cell.first.second;

        float& existing_mean = maps_[ELEVATION](row, col);
        float& existing_var = maps_[UNCERTAINTY](row, col);
        
        // If this is the first measurement for this cell, just use the new values
        if (std::isnan(existing_mean) || std::isnan(existing_var))
        {
            existing_mean = mean_height;
            existing_var = var_height;
        }
        else
        {
            // Fuse with existing measurement using Kalman filter
            fuseElevationWithKalmanFilter(existing_mean, existing_var, mean_height, var_height);
        }
    }
    
    // Create temporary elevation map for filtering
    maps_[ELEVATION_FILTERED] = maps_[ELEVATION];

    // Fill NaN cells with minimum elevation value in 3x3 window
    fillNaNWithMinimumInWindow(maps_[ELEVATION_FILTERED]);
    
    // Apply Gaussian filter to filtered elevation map
    filterElevation(elevation_map_filter_type_);
    
    // Compute slope map
    computeSlopeMap();
    
    // Compute step height map
    computeStepHeightMap();
    
    // Compute roughness map
    computeRoughnessMap();
    
    // Compute traversability map with default parameters
    computeTraversabilityMap();
}
[[nodiscard]]
std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<float>, ElevationMap::PairHash>
ElevationMap::dividePointCloudToGridCells(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud
) const
{
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<float>, PairHash> cell_heights;

    // Iterate through all points in the point cloud
    for (auto point = point_cloud->rbegin(); point != point_cloud->rend(); point++)
    {
        auto [row, col] = getGridCellIndex(point->x, point->y);

        // Ensure indices are within bounds
        if (row < rows_ && col < cols_)
        {
            auto cell = cell_heights.find({row, col});
            if ( cell == cell_heights.end() )
            {
                cell_heights[{row, col}] = std::vector<float>();
                cell_heights[{row, col}].reserve(256);
                cell = cell_heights.find({row, col});
            }
            // Add point to the corresponding grid cell
            cell->second.push_back(point->z);
        }
    }

    return cell_heights;
}

void ElevationMap::removeOverHeightPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) const
{
    point_cloud->erase(
        std::remove_if(point_cloud->begin(), point_cloud->end(),
            [this](const pcl::PointXYZ& point) {
                return point.z > max_height_;
            }),
        point_cloud->end()
    );
}

void ElevationMap::fuseElevationWithKalmanFilter(
    float& mean, 
    float& variance, 
    float new_mean, 
    float new_variance
) const
{
    // One-dimensional Kalman filter fusion
    // https://ieeexplore.ieee.org/document/8392399
    float fused_mean = (mean * new_variance + new_mean * variance) / (new_variance + variance);
    float fused_variance = (variance * new_variance) / (new_variance + variance);
    
    mean = fused_mean;
    variance = fused_variance;
}

void ElevationMap::filterElevation(const std::string& filter_type)
{
    if (filter_type == "boxblur")
    {
        applyBoxBlurFilter(maps_[ELEVATION_FILTERED]);
    }
    else if (filter_type == "gaussian")
    {
        applyGaussianFilter(maps_[ELEVATION_FILTERED]);
    }
    else if (filter_type == "bilateral")
    {
        applyBilateralFilter(maps_[ELEVATION_FILTERED]);
    }
}

void ElevationMap::computeSlopeMap()
{
    // Ensure slope map exists
    if (maps_.size() <= SLOPE)
    {
        maps_.resize(SLOPE + 1, Eigen::MatrixXf::Constant(rows_, cols_, std::numeric_limits<float>::quiet_NaN()));
    }
    
    // Compute gradient using central difference method
    for (int i = 1; i < static_cast<int>(rows_) - 1; i++)
    {
        for (int j = 1; j < static_cast<int>(cols_) - 1; j++)
        {
            // Skip invalid cells
            if (std::isnan(maps_[ELEVATION_FILTERED](i, j)))
            {
                maps_[SLOPE](i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            if (std::isnan(maps_[ELEVATION_FILTERED](i, j - 1)) ||
                std::isnan(maps_[ELEVATION_FILTERED](i, j + 1)) ||
                std::isnan(maps_[ELEVATION_FILTERED](i - 1, j)) ||
                std::isnan(maps_[ELEVATION_FILTERED](i + 1, j)))
            {
                maps_[SLOPE](i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            // Central difference in x-direction (dz/dx)
            float dz_dx = (maps_[ELEVATION_FILTERED](i, j + 1) - maps_[ELEVATION_FILTERED](i, j - 1)) / (2.0f * resolution_);
            
            // Central difference in y-direction (dz/dy)
            float dz_dy = (maps_[ELEVATION_FILTERED](i + 1, j) - maps_[ELEVATION_FILTERED](i - 1, j)) / (2.0f * resolution_);
            
            // Calculate magnitude of gradient
            maps_[SLOPE](i, j) = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy);
        }
    }
}

void ElevationMap::computeStepHeightMap()
{
    // Ensure step height map exists
    if (maps_.size() <= STEP_HEIGHT)
    {
        maps_.resize(STEP_HEIGHT + 1, Eigen::MatrixXf::Zero(rows_, cols_));
    }
    
    // 3x3 window size
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Compute step height in each cell
    for (int i = half_window; i < static_cast<int>(rows_) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(cols_) - half_window; j++)
        {
            // Skip invalid cells
            if (std::isnan(maps_[ELEVATION_FILTERED](i, j)))
            {
                maps_[STEP_HEIGHT](i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Find min and max elevation in 3x3 window
            float min_elevation = std::numeric_limits<float>::max();
            float max_elevation = std::numeric_limits<float>::lowest();
            
            for (int wi = -half_window; wi <= half_window; wi++)
            {
                for (int wj = -half_window; wj <= half_window; wj++)
                {
                    float value = maps_[ELEVATION_FILTERED](i + wi, j + wj);
                    if (!std::isnan(value))
                    {
                        min_elevation = std::min(min_elevation, value);
                        max_elevation = std::max(max_elevation, value);
                    }
                }
            }
            
            // Step height is difference between max and min
            maps_[STEP_HEIGHT](i, j) = max_elevation - min_elevation;
        }
    }
}

void ElevationMap::computeRoughnessMap()
{
    // Ensure roughness map exists
    if (maps_.size() <= ROUGHNESS)
    {
        maps_.resize(ROUGHNESS + 1, Eigen::MatrixXf::Zero(rows_, cols_));
    }
    
    // 3x3 window size
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Compute roughness in each cell
    for (int i = half_window; i < static_cast<int>(rows_) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(cols_) - half_window; j++)
        {
            // Skip invalid cells
            if (std::isnan(maps_[ELEVATION_FILTERED](i, j)))
            {
                maps_[ROUGHNESS](i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Calculate standard deviation of elevation in 3x3 window
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            
            for (int wi = -half_window; wi <= half_window; wi++)
            {
                for (int wj = -half_window; wj <= half_window; wj++)
                {
                    float value = maps_[ELEVATION_FILTERED](i + wi, j + wj);
                    if (!std::isnan(value))
                    {
                        sum += value;
                        sum_sq += value * value;
                        count++;
                    }
                }
            }
            
            if (count > 0)
            {
                float mean = sum / static_cast<float>(count);
                float variance = (sum_sq / static_cast<float>(count)) - (mean * mean);
                maps_[ROUGHNESS](i, j) = std::sqrt(std::max(0.0f, variance));
            }
            else
            {
                maps_[ROUGHNESS](i, j) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void ElevationMap::computeTraversabilityMap()
{
    // Ensure traversability map exists
    if (maps_.size() <= TRAVERSABILITY)
    {
        maps_.resize(TRAVERSABILITY + 1, Eigen::MatrixXf::Zero(rows_, cols_));
    }
    
    // Normalize slope, step height, and roughness maps to [0, 1]
    // Then combine with weights to produce traversability map
    
    // Compute traversability in each cell
    for (int i = 0; i < static_cast<int>(rows_); i++)
    {
        for (int j = 0; j < static_cast<int>(cols_); j++)
        {
            // Check if any input maps have NaN values
            if (std::isnan(maps_[SLOPE](i, j)) || 
                std::isnan(maps_[STEP_HEIGHT](i, j)) || 
                std::isnan(maps_[ROUGHNESS](i, j)))
            {
                maps_[TRAVERSABILITY](i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Normalize metrics to [0, 1] range using critical values
            float normalized_slope = std::min(1.0f, maps_[SLOPE](i, j) / slope_critical_);
            float normalized_step_height = std::min(1.0f, maps_[STEP_HEIGHT](i, j) / step_height_critical_);
            float normalized_roughness = std::min(1.0f, maps_[ROUGHNESS](i, j) / roughness_critical_);
            
            // Weighted sum to compute traversability
            maps_[TRAVERSABILITY](i, j) = 
                slope_weight_ * normalized_slope +
                step_height_weight_ * normalized_step_height +
                roughness_weight_ * normalized_roughness;
        }
    }
}

void ElevationMap::extractPointCloudTopSurface(
    std::unordered_map<std::pair<std::size_t, std::size_t>, std::vector<float>, PairHash>& cell_heights
) const
{
    // Lambda for insertion sort (descending)
    auto insertionSortDesc = [](std::vector<float>& vec)
    {
        for ( int i = 1; i < vec.size(); ++i )
        {
            float key = vec[i];
            int j = i - 1;

            // Move elements that are LESS than key one position ahead
            // (for descending order, we want larger elements first)
            while (j >= 0 && vec[j] < key)
            {
                vec[j + 1] = vec[j];
                --j;
            }
            vec[j + 1] = key;
        }
    };

    for ( auto cell : cell_heights )
    {
        if ( cell.second.empty() ) continue;
        // insertion sort is the fastest method if the vector is already sorted.
        // `heights` is already sorted because point cloud is sorted.
        insertionSortDesc(cell.second);
    }
}

bool ElevationMap::checkAndExtendMapIfNeeded(const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud)
{
    if (point_cloud->empty())
    {
        return false;
    }
    
    // Find min/max coordinates of existing map
    float min_map_x = -length_x_ / 2.0f;
    float max_map_x = length_x_ / 2.0f;
    float min_map_y = -length_y_ / 2.0f;
    float max_map_y = length_y_ / 2.0f;
    
    // Find min/max coordinates of point cloud
    float min_point_x = std::numeric_limits<float>::max();
    float max_point_x = std::numeric_limits<float>::lowest();
    float min_point_y = std::numeric_limits<float>::max();
    float max_point_y = std::numeric_limits<float>::lowest();
    
    for (const auto& point : point_cloud->points)
    {
        min_point_x = std::min(min_point_x, point.x);
        max_point_x = std::max(max_point_x, point.x);
        min_point_y = std::min(min_point_y, point.y);
        max_point_y = std::max(max_point_y, point.y);
    }
    
    // Check if point cloud extends beyond map boundaries
    bool extend_needed = 
        min_point_x < min_map_x || max_point_x > max_map_x ||
        min_point_y < min_map_y || max_point_y > max_map_y;
    
    if (!extend_needed)
    {
        return false;
    }
    
    // Calculate new boundaries
    float new_min_x = std::min(min_map_x, min_point_x);
    float new_max_x = std::max(max_map_x, max_point_x);
    float new_min_y = std::min(min_map_y, min_point_y);
    float new_max_y = std::max(max_map_y, max_point_y);
    
    // Calculate new geometry of map
    float new_map_length_x = 2 * std::max( std::abs( new_max_x ), std::abs( new_min_x ) );
    float new_map_length_y = 2 * std::max( std::abs( new_max_y ), std::abs( new_min_y ) );
    
    // Resize the grid map
    resize(new_map_length_x, new_map_length_y);
    
    return true;
}

void ElevationMap::fillNaNWithMinimumInWindow(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf filled_map = map;
    
    // 3x3 window size
    int window_size = 3;
    int half_window = (window_size - 1) / 2;
    
    // Fill NaN values with minimum in window
    for (int i = half_window; i < static_cast<int>(map.rows()) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(map.cols()) - half_window; j++)
        {
            // Skip valid values
            if (!std::isnan(map(i, j)))
            {
                continue;
            }
            
            // Find minimum valid value in 3x3 window
            float min_value = std::numeric_limits<float>::max();
            bool found_valid = false;
            
            for (int wi = -half_window; wi <= half_window; wi++)
            {
                for (int wj = -half_window; wj <= half_window; wj++)
                {
                    float value = map(i + wi, j + wj);
                    if (!std::isnan(value))
                    {
                        min_value = std::min(min_value, value);
                        found_valid = true;
                    }
                }
            }
            
            if (found_valid)
            {
                filled_map(i, j) = min_value;
            }
        }
    }
    
    map = filled_map;
}

void ElevationMap::applyBoxBlurFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf filtered_map = map;
    
    // 3x3 window size
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Apply box blur filter
    for (int i = half_window; i < static_cast<int>(map.rows()) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(map.cols()) - half_window; j++)
        {
            // Skip NaN values
            if (std::isnan(map(i, j)))
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Calculate average of neighbors
            float sum = 0.0f;
            int count = 0;
            
            for (int wi = -half_window; wi <= half_window; wi++)
            {
                for (int wj = -half_window; wj <= half_window; wj++)
                {
                    float value = map(i + wi, j + wj);
                    if (!std::isnan(value))
                    {
                        sum += value;
                        count++;
                    }
                }
            }
            
            if (count > 0)
            {
                filtered_map(i, j) = sum / static_cast<float>(count);
            }
            else
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map;
}

void ElevationMap::applyGaussianFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf filtered_map = map;
    
    // 5x5 window size
    int window_size = 5;
    int half_window = window_size / 2;
    
    // Gaussian kernel (5x5, sigma=1.0)
    static const float kernel[5][5] = {
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f },
        { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
        { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f }
    };
    
    // Apply Gaussian filter
    for (int i = half_window; i < static_cast<int>(map.rows()) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(map.cols()) - half_window; j++)
        {
            // Skip NaN values
            if (std::isnan(map(i, j)))
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Apply convolution
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int wi = -half_window; wi <= half_window; wi++)
            {
                for (int wj = -half_window; wj <= half_window; wj++)
                {
                    float value = map(i + wi, j + wj);
                    if (!std::isnan(value))
                    {
                        sum += value * kernel[wi + half_window][wj + half_window];
                        weight_sum += kernel[wi + half_window][wj + half_window];
                    }
                }
            }
            
            if (weight_sum > 0.0f)
            {
                filtered_map(i, j) = sum / weight_sum;
            }
            else
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map;
}

void ElevationMap::applyBilateralFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf filtered_map = map;
    
    // 5x5 window size
    int window_size = 5;
    int half_window = window_size / 2;
    
    // Bilateral filter parameters
    float spatial_sigma = 1.0f;
    float intensity_sigma = 0.1f;
    
    // Precompute spatial weights
    float spatial_weights[5][5];
    for (int k = -half_window; k <= half_window; k++)
    {
        for (int l = -half_window; l <= half_window; l++)
        {
            float distance_squared = static_cast<float>(k * k + l * l);
            spatial_weights[k + half_window][l + half_window] = 
                std::exp(-distance_squared / (2.0f * spatial_sigma * spatial_sigma));
        }
    }
    
    // Apply bilateral filter
    for (int i = half_window; i < static_cast<int>(map.rows()) - half_window; i++)
    {
        for (int j = half_window; j < static_cast<int>(map.cols()) - half_window; j++)
        {
            // Skip NaN values
            if (std::isnan(map(i, j)))
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            float center_value = map(i, j);
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = -half_window; k <= half_window; k++)
            {
                for (int l = -half_window; l <= half_window; l++)
                {
                    float neighbor_value = map(i + k, j + l);
                    
                    // Skip NaN values
                    if (!std::isnan(neighbor_value))
                    {
                        // Calculate intensity weight
                        float intensity_diff = (neighbor_value - center_value) * (neighbor_value - center_value);
                        float intensity_weight = std::exp(-intensity_diff / (2.0f * intensity_sigma * intensity_sigma));
                        
                        // Combine spatial and intensity weights
                        float weight = spatial_weights[k + half_window][l + half_window] * intensity_weight;
                        
                        weighted_sum += weight * neighbor_value;
                        weight_sum += weight;
                    }
                }
            }
            
            if (weight_sum > 0)
            {
                filtered_map(i, j) = weighted_sum / weight_sum;
            }
            else
            {
                filtered_map(i, j) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map;
}

std::tuple<std::size_t, std::size_t> ElevationMap::getGridCellIndex(float x, float y) const
{
    const float half_range_x = length_x_ / 2.0f;
    const float half_range_y = length_y_ / 2.0f;
    std::size_t row = static_cast<std::size_t>(std::floor((half_range_x - x) / resolution_));
    std::size_t col = static_cast<std::size_t>(std::floor((half_range_y - y) / resolution_));
    return std::make_tuple(row, col);
}