#pragma once
#include <tuple>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>


#include <sensor_msgs/Image.h>
//#include <visiualization_msgs/Detection2D.h>
#include <sensor_msgs/CameraInfo.h>
#include "ros/time.h"
#include "map"
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/map.hpp>
#include "unordered_map"
#include "tuple"
#include "pcl_ros/point_cloud.h"
#include "pcl_ros/publisher.h"
#include "pcl/common/centroid.h"
#include "pcl/point_types.h"
#include <pcl_conversions/pcl_conversions.h>

//namespace EFEAR_4D 

using std::cout;
using std::endl;
using std::cerr;

typedef std::tuple<int,int,int> idx_grid;

int GetX(idx_grid& idx);

int GetY(idx_grid& idx);

int GetZ(idx_grid& idx);

class voxel
{
public:

  voxel();

  voxel(const Eigen::Vector3d& p);

  void AddPoint(const Eigen::Vector3d& p);

  void ComputeMeanCov(const Eigen::Vector3d& p);

  unsigned int N_;
  Eigen::Matrix3d Cov_;
  Eigen::Vector3d mean_;
  bool has_gausian_;
  Eigen::MatrixXd pnts_;

};


class Voxelgrid
{
public:

  Voxelgrid(double resolution = 0.01):resolution_(resolution) {}

  idx_grid GetIndex(const double x, const double y, const double z);

  idx_grid GetIndex(const pcl::PointXYZI& p);

  void IdxToCenter(const idx_grid& idx, Eigen::Vector3d& center);

  Eigen::Vector3d IdxToCenterReturn(const idx_grid& idx);

  void InsertElement(const double x, const double y, const double z, const voxel& obj);

  void InsertElement(idx_grid idx, const voxel& obj);

  voxel* GetElement(const idx_grid& idx);

  std::vector<idx_grid> GetNeighbours( idx_grid& idx, int n);

  const voxel* GetElement(const double x, const double y, const double z);

  bool ElementExists(const idx_grid& idx);

  size_t Size();

  std::vector<idx_grid> GetIndecies();

  std::vector<Eigen::Vector3d> GetCenters();

  std::vector<voxel> GetCells();

  double GetGridProb();

  void SetGridProb(double prob);

  void Clear();

  double GetResolution();


  double resolution_;
  std::map<idx_grid, voxel> map_;

};


class NDTGrid
{
public:

  NDTGrid(float resolution);

  NDTGrid(const pcl::PointCloud<pcl::PointXYZI>& cld, float resolution);

private:
  Voxelgrid map_;

};


