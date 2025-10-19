#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

class GridMap
{
public:
    GridMap() = delete;
    explicit GridMap(std::size_t num_maps, float length_x, float length_y, float resolution);
    virtual ~GridMap() = default;
    std::size_t getRows() const;
    std::size_t getColumns() const;
    float getLengthX() const;
    float getLengthY() const;
    float getResolution() const;
    void add(const std::string name);
    bool resize(float length_x, float length_y);

protected:
    std::vector<Eigen::MatrixXf> maps_;

    // map rows, start from 1
    std::size_t rows_;
    // map columns, start from 1
    std::size_t cols_;
    
    float length_x_;
    float length_y_;
    float resolution_;
};