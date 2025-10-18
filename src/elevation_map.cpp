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
    : GridMap( 7, length_x, length_y, resolution ),
      method_( method ),
      num_max_points_in_grid_( num_max_points_in_grid ),
      elevation_map_filter_type_( elevation_map_filter_type ),
      max_height_( max_height ),
      slope_weight_( slope_weight ),
      step_height_weight_( step_height_weight ),
      roughness_weight_( roughness_weight ),
      slope_critical_( slope_critical ),
      step_height_critical_( step_height_critical ),
      roughness_critical_( roughness_critical ),
      l_( l ),
      sigma_f_( sigma_f ),
      sigma_n_( sigma_n )
{
}

void ElevationMap::update( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud )
{
    if ( method_ == "direct" )
    {
        updateDirect( point_cloud );
    }
    // TODO: Implement Gaussian Process method.
}
void ElevationMap::updateDirect( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud )
{
    // Check if map needs to be extended to cover new point cloud
    checkAndExtendMapIfNeeded( point_cloud );

    // Create a map to store points for each grid cell
    std::vector<std::vector<float>> cell_heights;
    
    removeOverHeightPoints( point_cloud );

    // Partition point cloud to grid cells
    dividePointCloudToGridCells( cell_heights, point_cloud );
    
    // Extract top surface points in each grid cell
    extractPointCloudTopSurface( cell_heights );
    
    // Update elevation map with new measurements using Kalman filter
    for ( std::size_t i = 0; i < cell_heights.size(); ++i )
    {
        const auto& heights = cell_heights[i];
        if ( heights.empty() )
        {
            continue;
        }    

        // Calculate mean and variance of points in this cell
        float sum_height = std::accumulate( heights.begin(), heights.end(), 0.0f);
        float mean_height = sum_height / heights.size();

        float sum_sq_diff = 0.0f;
        for ( const auto& height : heights )
        {
            float diff = height - mean_height;
            sum_sq_diff += diff * diff;
        }
        float var_height = sum_sq_diff / heights.size();
        
        float& existing_mean = maps_[ELEVATION].data()[i];
        float& existing_var = maps_[UNCERTAINTY].data()[i];
        
        // If this is the first measurement for this cell, just use the new values
        if ( std::isnan( existing_mean ) || std::isnan( existing_var ) )
        {
            existing_mean = mean_height;
            existing_var = var_height;
        }
        else
        {
            // Fuse with existing measurement using Kalman filter
            fuseElevationWithKalmanFilter( existing_mean, existing_var, static_cast<double>(mean_height), static_cast<double>(var_height) );
        }
    }
    
    // Create temporary elevation map for filtering
    maps_[ELEVATION_FILTERED] = maps_[ELEVATION];

    // Fill NaN cells with minimum elevation value in 3x3 window
    fillNaNWithMinimumInWindow( maps_[ELEVATION_FILTERED] );
    
    // Apply Gaussian filter to filtered elevation map
    filterElevation( elevation_map_filter_type_ );
    
    // Compute slope map
    computeSlopeMap();
    
    // Compute step height map
    computeStepHeightMap();
    
    // Compute roughness map
    computeRoughnessMap();
    
    // Compute traversability map with default parameters
    computeTraversabilityMap();
}
void ElevationMap::dividePointCloudToGridCells(
    std::vector<std::vector<float>>& cell_heights,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud
) const
{
    // Precompute map boundaries and factors
    const float min_x = -length_x_ / 2.0f;
    const float max_x = length_x_ / 2.0f;
    const float min_y = -length_y_ / 2.0f;
    const float max_y = length_y_ / 2.0f;
    const float half_range_x = length_x_ / 2.0f;
    const float half_range_y = length_y_ / 2.0f;
    
    // Reserve space for cell_heights based on grid map.
    cell_heights.resize( rows_ * cols_ );

    // Iterate through all points in the point cloud
    for ( auto point = point_cloud->rbegin(); point != point_cloud->rend(); ++point )
    {
        // Check if point is within map boundaries
        if (point->x >= min_x && point->x <= max_x && point->y >= min_y && point->y <= max_y)
        {
            // Calculate grid cell indices using multiplication instead of division
            // Fixed coordinate mapping: x-coordinate maps to row, y-coordinate maps to column
            std::size_t col = static_cast<std::size_t>( std::floor(( half_range_y - point->y ) / resolution_ ));
            std::size_t row = static_cast<std::size_t>( std::floor(( half_range_x - point->x ) / resolution_ ));

            // Ensure indices are within bounds
            if ( row < rows_ && col < cols_ )
            {
                // Add point to the corresponding grid cell
                // column-major order for Eigen::MatrixXf memory copy.
                cell_heights[col * rows_ + row].push_back( point->z );
            }
        }
    }
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

void ElevationMap::fuseElevationWithKalmanFilter( float& mean, float& variance, float new_mean, float new_variance ) const
{
    // One-dimensional Kalman filter fusion
    // https://ieeexplore.ieee.org/document/8392399
    float fused_mean = ( mean * new_variance + new_mean * variance ) / ( new_variance + variance );
    float fused_variance = ( variance * new_variance ) / ( new_variance + variance );
    
    mean = fused_mean;
    variance = fused_variance;
}

void ElevationMap::filterElevation( const std::string& filter_type )
{
    if ( filter_type == "boxblur" )
    {
        applyBoxBlurFilter( maps_[ELEVATION_FILTERED] );
    }
    else if ( filter_type == "gaussian" )
    {
        applyGaussianFilter( maps_[ELEVATION_FILTERED] );
    }
    else if ( filter_type == "bilateral" )
    {
        applyBilateralFilter( maps_[ELEVATION_FILTERED] );
    }
}

void ElevationMap::computeSlopeMap()
{
    // Use direct reference to avoid copy construction
    Eigen::MatrixXf& slope_map = maps_[SLOPE];
    const Eigen::MatrixXf& elevation = maps_[ELEVATION_FILTERED];
    
    // Initialize with zeros
    slope_map.setZero( rows_, cols_ );
    
    // Grid spacing
    float d = resolution_;
    
    // Compute slope for each cell using finite differences
    for ( int i = 1; i < static_cast<int>( rows_ ) - 1; ++i )
    {
        for ( int j = 1; j < static_cast<int>( cols_ ) - 1; ++j )
        {
            // Check if center cell is valid
            if ( std::isnan( elevation( i, j ) ) )
            {
                slope_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Check if neighbor cells are valid
            bool valid_neighbors = !std::isnan( elevation( i, j - 1 ) ) && 
                                  !std::isnan( elevation( i, j + 1 ) ) &&
                                  !std::isnan( elevation( i - 1, j ) ) &&
                                  !std::isnan( elevation( i + 1, j ) );
            
            if ( valid_neighbors )
            {
                // Partial derivatives using central difference
                float dz_dx = ( elevation( i, j + 1 ) - elevation( i, j - 1 ) ) / ( 2.0f * d );
                float dz_dy = ( elevation( i + 1, j ) - elevation( i - 1, j ) ) / ( 2.0f * d );
                
                // Slope magnitude
                slope_map( i, j ) = sqrtf( dz_dx * dz_dx + dz_dy * dz_dy );
            }
            else
            {
                slope_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void ElevationMap::computeStepHeightMap()
{ 
    // Use direct reference to avoid copy construction
    Eigen::MatrixXf& step_height_map = maps_[STEP_HEIGHT];
    const Eigen::MatrixXf& elevation = maps_[ELEVATION_FILTERED];
    
    // Initialize with zeros
    step_height_map.setZero( rows_, cols_ );
    
    // Window size (3x3)
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Compute step height for each cell
    for ( int i = half_window; i < static_cast<int>( rows_ ) - half_window; ++i )
    {
        for ( int j = half_window; j < static_cast<int>( cols_ ) - half_window; ++j )
        {
            // Check if center cell is valid
            if ( std::isnan( elevation( i, j ) ) )
            {
                step_height_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Find min and max elevation in the window, ignoring NaN values
            float min_elev = std::numeric_limits<float>::max();
            float max_elev = std::numeric_limits<float>::lowest();
            bool found_valid = false;
            
            for ( int wi = -half_window; wi <= half_window; ++wi )
            {
                for ( int wj = -half_window; wj <= half_window; ++wj )
                {
                    float elev = elevation( i + wi, j + wj );
                    if ( !std::isnan( elev ) )
                    {
                        if ( elev < min_elev )
                        {
                            min_elev = elev;
                        }
                        if ( elev > max_elev )
                        {
                            max_elev = elev;
                        }
                        found_valid = true;
                    }
                }
            }
            
            // Step height is the difference between max and min
            if ( found_valid )
            {
                step_height_map( i, j ) = max_elev - min_elev;
            }
            else
            {
                step_height_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void ElevationMap::computeRoughnessMap()
{
    // Use direct reference to avoid copy construction
    Eigen::MatrixXf& roughness_map = maps_[ROUGHNESS];
    const Eigen::MatrixXf& elevation = maps_[ELEVATION_FILTERED];
    
    // Initialize with zeros
    roughness_map.setZero( rows_, cols_ );
    
    // Window size (3x3)
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Compute roughness for each cell
    for ( int i = half_window; i < static_cast<int>( rows_ ) - half_window; ++i )
    {
        for ( int j = half_window; j < static_cast<int>( cols_ ) - half_window; ++j )
        {
            // Check if center cell is valid
            if ( std::isnan( elevation( i, j ) ) )
            {
                roughness_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Calculate mean elevation in window, ignoring NaN values
            float sum = 0.0f;
            int count = 0;
            
            for ( int wi = -half_window; wi <= half_window; ++wi )
            {
                for ( int wj = -half_window; wj <= half_window; ++wj )
                {
                    float elev = elevation( i + wi, j + wj );
                    if ( !std::isnan( elev ) )
                    {
                        sum += elev;
                        count++;
                    }
                }
            }
            
            if ( count > 0 )
            {
                float mean_elev = sum / count;
                
                // Calculate standard deviation
                float sum_sq_diff = 0.0f;
                for ( int wi = -half_window; wi <= half_window; ++wi )
                {
                    for ( int wj = -half_window; wj <= half_window; ++wj )
                    {
                        float elev = elevation( i + wi, j + wj );
                        if ( !std::isnan( elev ) )
                        {
                            float diff = elev - mean_elev;
                            sum_sq_diff += diff * diff;
                        }
                    }
                }
                
                roughness_map( i, j ) = sqrtf( sum_sq_diff / count );
            }
            else
            {
                roughness_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

void ElevationMap::computeTraversabilityMap()
{
    // Use direct reference to avoid copy construction
    Eigen::MatrixXf& traversability_map = maps_[TRAVERSABILITY];
    const Eigen::MatrixXf& slope_map = maps_[SLOPE];
    const Eigen::MatrixXf& step_height_map = maps_[STEP_HEIGHT];
    const Eigen::MatrixXf& roughness_map = maps_[ROUGHNESS];
    
    // Initialize with zeros
    traversability_map.setZero( rows_, cols_ );
    
    // Compute traversability map as weighted combination
    for ( int i = 0; i < static_cast<int>( rows_ ); ++i )
    {
        for ( int j = 0; j < static_cast<int>( cols_ ); ++j )
        {
            // Check if all input maps have valid values
            if ( std::isnan( slope_map( i, j ) ) || 
                 std::isnan( step_height_map( i, j ) ) || 
                 std::isnan( roughness_map( i, j ) ) )
            {
                traversability_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            float slope_term = slope_weight_ * slope_map( i, j ) / slope_critical_;
            float step_height_term = step_height_weight_ * step_height_map( i, j ) / step_height_critical_;
            float roughness_term = roughness_weight_ * roughness_map( i, j ) / roughness_critical_;
            
            traversability_map( i, j ) = slope_term + step_height_term + roughness_term;
        }
    }
}

void ElevationMap::extractPointCloudTopSurface( std::vector<std::vector<float>>& cell_heights ) const
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

    for ( auto& heights : cell_heights )
    {
        if ( heights.empty() )
        {
            continue;
        }

        // insertion sort is the fastest method if the vector is already sorted.
        // `heights` is already sorted because point cloud is sorted.
        insertionSortDesc(heights);

        std::size_t num_points = heights.size() > num_max_points_in_grid_ ? num_max_points_in_grid_ : heights.size();

        heights.resize(num_points);
    }
}
Eigen::MatrixXf ElevationMap::requestTraversabilityMap()
{
    // Convert from Eigen::MatrixXf to Eigen::MatrixXf for return
    Eigen::MatrixXf result = maps_[TRAVERSABILITY].cast<float>();
    return result;
}



bool ElevationMap::checkAndExtendMapIfNeeded( const pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud )
{
    if ( point_cloud->empty() )
    {
        return false;
    }
    
    // Calculate current map boundaries
    float min_map_x = -length_x_ / 2.0f;
    float max_map_x = length_x_ / 2.0f;
    float min_map_y = -length_y_ / 2.0f;
    float max_map_y = length_y_ / 2.0f;
    
    // Calculate point cloud boundaries
    float min_point_x = std::numeric_limits<float>::max();
    float max_point_x = std::numeric_limits<float>::lowest();
    float min_point_y = std::numeric_limits<float>::max();
    float max_point_y = std::numeric_limits<float>::lowest();
    
    for ( const auto& point : point_cloud->points )
    {
        if ( point.x < min_point_x ) min_point_x = point.x;
        if ( point.x > max_point_x ) max_point_x = point.x;
        if ( point.y < min_point_y ) min_point_y = point.y;
        if ( point.y > max_point_y ) max_point_y = point.y;
    }
    
    // Check if point cloud exceeds current map boundaries
    bool need_extend = false;
    if ( min_point_x < min_map_x || max_point_x > max_map_x ||
         min_point_y < min_map_y || max_point_y > max_map_y )
    {
        need_extend = true;
    }
    
    if ( !need_extend )
    {
        return false;
    }
    
    // Calculate new map dimensions to cover all points
    float new_min_x = std::min( min_map_x, min_point_x );
    float new_max_x = std::max( max_map_x, max_point_x );
    float new_min_y = std::min( min_map_y, min_point_y );
    float new_max_y = std::max( max_map_y, max_point_y );

    float new_map_length_x = 2 * std::max( std::abs( new_max_x ), std::abs( new_min_x ) );
    float new_map_length_y = 2 * std::max( std::abs( new_max_y ), std::abs( new_min_y ) );

    // Resize the grid map
    resize( new_map_length_x, new_map_length_y );
    
    return true;
}

void ElevationMap::fillNaNWithMinimumInWindow(Eigen::MatrixXf& map)
{
    const Eigen::MatrixXf& temp_map = map;
    Eigen::MatrixXf filled_map = temp_map;
    
    // 3x3 window size
    int window_size = 3;
    int half_window = window_size / 2;
    
    for ( int i = 0; i < static_cast<int>( map.rows() ); ++i )
    {
        for ( int j = 0; j < static_cast<int>( map.cols() ); ++j )
        {
            // Only process NaN values
            if ( std::isnan( map( i, j ) ) )
            {
                // Find minimum value in the 3x3 window
                float min_value = std::numeric_limits<float>::max();
                bool found_valid = false;
                
                for ( int wi = -half_window; wi <= half_window; ++wi )
                {
                    for ( int wj = -half_window; wj <= half_window; ++wj )
                    {
                        int ni = i + wi;
                        int nj = j + wj;
                        
                        // Check bounds
                        if ( ni >= 0 && ni < static_cast<int>( map.rows() ) &&
                             nj >= 0 && nj < static_cast<int>( map.cols() ) )
                        {
                            // Check if value is not NaN
                            if ( !std::isnan( temp_map( ni, nj ) ) )
                            {
                                if ( temp_map( ni, nj ) < min_value )
                                {
                                    min_value = temp_map( ni, nj );
                                    found_valid = true;
                                }
                            }
                        }
                    }
                }
                
                // If we found a valid value, use it
                if ( found_valid )
                {
                    filled_map( i, j ) = min_value;
                }
            }
        }
    }
    
    map = filled_map;
}

void ElevationMap::applyBoxBlurFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf temp_map = map;
    Eigen::MatrixXf filtered_map = temp_map;
    
    // 3x3 window size
    int window_size = 3;
    int half_window = window_size / 2;
    
    // Apply box blur filter
    for ( int i = half_window; i < static_cast<int>( map.rows() ) - half_window; ++i )
    {
        for ( int j = half_window; j < static_cast<int>( map.cols() ) - half_window; ++j )
        {
            // Skip NaN values
            if ( std::isnan( map( i, j ) ) )
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Calculate average of neighbors
            float sum = 0.0f;
            int count = 0;
            
            for ( int wi = -half_window; wi <= half_window; ++wi )
            {
                for ( int wj = -half_window; wj <= half_window; ++wj )
                {
                    float value = map( i + wi, j + wj );
                    if ( !std::isnan( value ) )
                    {
                        sum += value;
                        count++;
                    }
                }
            }
            
            if ( count > 0 )
            {
                filtered_map( i, j ) = sum / count;
            }
            else
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map.cast<float>();
}

void ElevationMap::applyGaussianFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf temp_map = map;
    Eigen::MatrixXf filtered_map = temp_map;
    
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
    for ( int i = half_window; i < static_cast<int>( map.rows() ) - half_window; ++i )
    {
        for ( int j = half_window; j < static_cast<int>( map.cols() ) - half_window; ++j )
        {
            // Skip NaN values
            if ( std::isnan( map( i, j ) ) )
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            // Apply convolution
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for ( int wi = -half_window; wi <= half_window; ++wi )
            {
                for ( int wj = -half_window; wj <= half_window; ++wj )
                {
                    float value = map( i + wi, j + wj );
                    if ( !std::isnan( value ) )
                    {
                        sum += value * kernel[wi + half_window][wj + half_window];
                        weight_sum += kernel[wi + half_window][wj + half_window];
                    }
                }
            }
            
            if ( weight_sum > 0.0f )
            {
                filtered_map( i, j ) = sum / weight_sum;
            }
            else
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map.cast<float>();
}

void ElevationMap::applyBilateralFilter(Eigen::MatrixXf& map)
{
    Eigen::MatrixXf temp_map = map;
    Eigen::MatrixXf filtered_map = temp_map;
    
    // 5x5 window size
    int window_size = 5;
    int half_window = window_size / 2;
    
    // Bilateral filter parameters
    float spatial_sigma = 1.0f;
    float intensity_sigma = 0.1f;
    
    // Precompute spatial weights
    float spatial_weights[5][5];
    for ( int k = -half_window; k <= half_window; ++k )
    {
        for ( int l = -half_window; l <= half_window; ++l )
        {
            float distance_squared = static_cast<float>( k * k + l * l );
            spatial_weights[k + half_window][l + half_window] = expf( -distance_squared / ( 2.0f * spatial_sigma * spatial_sigma ) );
        }
    }
    
    // Apply bilateral filter
    for ( int i = half_window; i < static_cast<int>( map.rows() ) - half_window; ++i )
    {
        for ( int j = half_window; j < static_cast<int>( map.cols() ) - half_window; ++j )
        {
            // Skip NaN values
            if ( std::isnan( map( i, j ) ) )
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            
            float center_value = map( i, j );
            float weighted_sum = 0.0f;
            float weight_sum = 0.0f;
            
            for ( int k = -half_window; k <= half_window; ++k )
            {
                for ( int l = -half_window; l <= half_window; ++l )
                {
                    float neighbor_value = map( i + k, j + l );
                    
                    // Skip NaN values
                    if ( !std::isnan( neighbor_value ) )
                    {
                        // Calculate intensity weight
                        float intensity_diff = ( neighbor_value - center_value ) * ( neighbor_value - center_value );
                        float intensity_weight = expf( -intensity_diff / ( 2.0f * intensity_sigma * intensity_sigma ) );
                        
                        // Combine spatial and intensity weights
                        float weight = spatial_weights[k + half_window][l + half_window] * intensity_weight;
                        
                        weighted_sum += weight * neighbor_value;
                        weight_sum += weight;
                    }
                }
            }
            
            if ( weight_sum > 0 )
            {
                filtered_map( i, j ) = weighted_sum / weight_sum;
            }
            else
            {
                filtered_map( i, j ) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    
    map = filtered_map.cast<float>();
}
