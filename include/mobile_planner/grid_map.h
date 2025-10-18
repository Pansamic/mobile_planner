#include <cstdint>
#include <unordered_map>
#include <string>
#include <Eigen/Core>
#include <Eigen/Geometry>

class GridMap
{
public:
    GridMap() = delete;
    explicit GridMap(const std::vector<std::string>& names, std::size_t rows, std::size_t cols, float resolution);
    virtual ~GridMap() = default;
    std::size_t getRows() const;
    std::size_t getColumns() const;
    float getLength() const;
    float getWidth() const;
    float getResolution() const;
    void add(const std::string name);
    bool resize(std::size_t rows, std::size_t cols);
    
    // Public members
    std::unordered_map<std::string, Eigen::MatrixXf> maps_;
    std::size_t rows_;
    std::size_t cols_;
    float length_;
    float width_;
    float resolution_;
};