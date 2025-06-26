#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_
#define PCL_NO_PRECOMPILE 

#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/filter.h>

#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>

#include <geometry_msgs/TwistWithCovarianceStamped.h>

#include "fast_dbscan/KDTreeVectorOfVectorsAdaptor.h"
#include "fast_dbscan/Fastdbscan.h"

#include <ceres/normal_prior.h>
#include <ceres/ceres.h>
#include <ceres/loss_function.h>

#include "efear/cloud_msgs.h"
#include "efear/save_map.h"
#include "efear/pointnormal.h"
#include "efear/n_scan_normal.h"
#include "efear/registration.h"

#define GREEN "\033[32m"

using namespace nanoflann;

using namespace std;

typedef pcl::PointXYZI PointT;
extern const int imuQueLength = 200;
extern const float scanPeriod = 0.083333;

// enum class SensorType { VELODYNE, OUSTER, LIVOX };
struct RadarPointCloudType
{
  PCL_ADD_POINT4D      // x,y,z position in [m]
  PCL_ADD_INTENSITY;
  union
    {
      struct
      {
        float doppler;
      };
      float data_c[4];
    };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 确保定义新类型点云内存与SSE对齐
} EIGEN_ALIGN16; // 强制SSE填充以正确对齐内存
POINT_CLOUD_REGISTER_POINT_STRUCT
(
    RadarPointCloudType,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, doppler, doppler)
)

struct mscCloudType
{
    float x;
    float y;
    float z;
    float power;
    float doppler;
};
POINT_CLOUD_REGISTER_POINT_STRUCT 
(mscCloudType,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, power, power)
    (float, doppler, doppler))

struct RadarEgoVelocityEstimatorIndices
{ 
  
  uint x_r          = 0;
  uint y_r          = 1;
  uint z_r          = 2;
  uint snr_db       = 3;
  uint doppler      = 4;
  uint range        = 5;
  uint azimuth      = 6;
  uint elevation    = 7;
  uint normalized_x = 8;
  uint normalized_y = 9;
  uint normalized_z = 10;
};

// 定义球面方程误差
struct SphereFittingCostFunction {
    SphereFittingCostFunction(double x_obs, double y_obs, double z_obs)
        : x_obs_(x_obs), y_obs_(y_obs), z_obs_(z_obs) {}

    template <typename T>
    bool operator()(const T* x, T* residual) const {
        residual[0] = ceres::pow((x[0] - T(x_obs_)), 2) +
                      ceres::pow((x[1] - T(y_obs_)), 2) -//+
                      //ceres::pow((x[2] - T(z_obs_)), 2) - 
                      (ceres::pow((x[0]), 2) +
                      ceres::pow((x[1]), 2) 
                      //ceres::pow((x[2]), 2)
                      );
                      // Constraint: Ensure the sphere passes through the origin
        // residual[1] = ceres::pow((x[0]), 2) +
        //               ceres::pow((x[1]), 2) +
        //               ceres::pow((x[2]), 2) - T(x[3]*x[3]);
        return true;
    }

private:
    double x_obs_;
    double y_obs_;
    double z_obs_;
    double radius_; // Radius of the sphere
};

class ParamServer
{
public:

    ros::NodeHandle nh;

    int robot_id;

    //Topics
    string pointCloudTopic;
    string imuTopic;
    string odomTopic;
    string gpsTopic;

    //Frames
    // string lidarFrame;
    string baselinkFrame;
    string odometryFrame;
    string mapFrame;

    // GPS Settings
    // bool useImuHeadingInitialization;
    // bool useGpsElevation;
    // float gpsCovThreshold;
    // float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // // Lidar Sensor Configuration
    // // SensorType sensor;
    // int N_SCAN;
    // int Horizon_SCAN;
    // int downsampleRate;
    // float lidarMinRange;
    // float lidarMaxRange;

    // IMU
    // float imuAccNoise;
    // float imuGyrNoise;
    // float imuAccBiasN;
    // float imuGyrBiasN;
    // float imuGravity;
    // float imuRPYWeight;
    vector<double> extRotV;
    // vector<double> extRPYV;
    vector<double> extTransV;
    Eigen::Matrix3d extRot;
    // Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    // Eigen::Quaterniond extQRPY;

    // LOAM
    // float edgeThreshold;
    // float surfThreshold;
    // int edgeFeatureMinValidNum;
    // int surfFeatureMinValidNum;

    // voxel filter paprams
    // float odometrySurfLeafSize;
    // float mappingCornerLeafSize;
    // float mappingSurfLeafSize ;

    // float z_tollerance; 
    // float rotation_tollerance;

    // CPU Params
    // int numberOfCores;
    // double mappingProcessInterval;

    // Surrounding map
    // float surroundingkeyframeAddingDistThreshold; 
    // float surroundingkeyframeAddingAngleThreshold; 
    // float surroundingKeyframeDensity;
    // float surroundingKeyframeSearchRadius;
    
    // Loop closure
    // bool  loopClosureEnableFlag;
    // float loopClosureFrequency;
    // int   surroundingKeyframeSize;
    // float historyKeyframeSearchRadius;
    // float historyKeyframeSearchTimeDiff;
    // int   historyKeyframeSearchNum;
    // float historyKeyframeFitnessScore;


    // ego velocity estimator 
    double min_db;
    double allowed_outlier_percentage;
    double thresh_zero_velocity;

    bool use_ransac;
    int N_ransac_points;
    float outlier_prob; // Outlier Probability, to calculate ransac_iter_
    float success_prob;
    float inlier_thresh;

    //DBSCAN
    float MinCPts; 
    float ClusterTR;
    int MinCS;
    int MaxCS;
    float FOV_degree;
    float FOV_rad;
    int Min_inlier;

    float sigma_offset_radar_x;
    float sigma_offset_radar_y;
    float sigma_offset_radar_z;
    float max_sigma_x;
    float max_sigma_y;
    float max_sigma_z;


    // preprocess
    string input_type;
    float power_threshold;
    string topic_inlier_pc2, topic_outlier_pc2;
    bool enable_dynamic_object_removal;
    double distance_near_thresh;
    double distance_far_thresh;
    double z_low_thresh;
    double z_high_thresh;
    double rcs_thresh;
    
    Eigen::Matrix4d calibration_matrix = Eigen::Matrix4d::Identity();
    std::string downsample_method;
    double downsample_resolution;
    std::string outlier_removal_method;

    // matching
    double max_egovel_cum;
    int  max_submap_frames;
    double resolution_;
    std::string reg_type;

    // keyframe_ parameters
    double keyframe_delta_trans;  // minimum distance between keyframes_
    double keyframe_delta_angle;  //
    double keyframe_delta_time;   //

    // registration validation by thresholding
    bool enable_transform_thresholding;  //
    bool enable_imu_thresholding;
    double max_acceptable_trans;  //
    double max_acceptable_angle;
    double max_diff_trans;
    double max_diff_angle;

    // bool enable_imu_fusion;
    // bool imu_debug_out;
    // double imu_fusion_ratio;

    // mapping
    double map_cloud_resolution;

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer()
    {
        nh.param<int>("efear/robot_id", robot_id, 1);

        nh.param<std::string>("efear/pointCloudTopic", pointCloudTopic, "/radar_enhanced_pcl");
        // nh.param<std::string>("efear/imuTopic", imuTopic, "/vectornav/imu");
        nh.param<std::string>("efear/odomTopic", odomTopic, "/odom");
        nh.param<std::string>("efear/gpsTopic", gpsTopic, "/ublox/fix");

        // nh.param<std::string>("efear/lidarFrame", lidarFrame, "livox");
        nh.param<std::string>("efear/baselinkFrame", baselinkFrame, "base_link");
        nh.param<std::string>("efear/odometryFrame", odometryFrame, "odom_imu"); 
        nh.param<std::string>("efear/mapFrame", mapFrame, "map");

        // nh.param<bool>("efear/useGpsElevation", useGpsElevation, false);
        // nh.param<float>("efear/gpsCovThreshold", gpsCovThreshold, 2.0);
        // nh.param<float>("efear/poseCovThreshold", poseCovThreshold, 25.0);

        nh.param<bool>("efear/savePCD", savePCD, false);
        nh.param<std::string>("efear/savePCDDirectory", savePCDDirectory, "/Downloads/LOAM/");

        // nh.param<int>("efear/downsampleRate", downsampleRate, 1);
        // nh.param<float>("efear/lidarMinRange", lidarMinRange, 1.0);
        // nh.param<float>("efear/lidarMaxRange", lidarMaxRange, 1000.0);

        // nh.param<float>("efear/imuAccNoise", imuAccNoise, 0.01);
        // nh.param<float>("efear/imuGyrNoise", imuGyrNoise, 0.001);
        // nh.param<float>("efear/imuAccBiasN", imuAccBiasN, 0.0002);
        // nh.param<float>("efear/imuGyrBiasN", imuGyrBiasN, 0.00003);
        // nh.param<float>("efear/imuGravity", imuGravity, 9.80511);
        // nh.param<float>("efear/imuRPYWeight", imuRPYWeight, 0.01);
        

        // nh.param<float>("efear/odometrySurfLeafSize", odometrySurfLeafSize, 0.2);
        // nh.param<float>("efear/mappingCornerLeafSize", mappingCornerLeafSize, 0.2);
        // nh.param<float>("efear/mappingSurfLeafSize", mappingSurfLeafSize, 0.2);

        // nh.param<int>("efear/numberOfCores", numberOfCores, 2);
        // nh.param<double>("efear/mappingProcessInterval", mappingProcessInterval, 0.15);

        // nh.param<float>("efear/surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0);
        // nh.param<float>("efear/surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2);
        // nh.param<float>("efear/surroundingKeyframeDensity", surroundingKeyframeDensity, 1.0);
        // nh.param<float>("efear/surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius, 50.0);

        // nh.param<bool>("efear/loopClosureEnableFlag", loopClosureEnableFlag, false);
        // nh.param<float>("efear/loopClosureFrequency", loopClosureFrequency, 1.0);
        // nh.param<int>("efear/surroundingKeyframeSize", surroundingKeyframeSize, 50);
        // nh.param<float>("efear/historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        // nh.param<float>("efear/historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);
        // nh.param<int>("efear/historyKeyframeSearchNum", historyKeyframeSearchNum, 25);
        // nh.param<float>("efear/historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3);



        
        // ego velocity estimator
        nh.param<double>("ego_velocity_estimator/min_db", min_db, 8.0);
        nh.param<double>("ego_velocity_estimator/allowed_outlier_percentage", allowed_outlier_percentage, 0.3);
        nh.param<double>("ego_velocity_estimator/thresh_zero_velocity", thresh_zero_velocity, 0.2);
        nh.param<bool>("ego_velocity_estimator/use_ransac", use_ransac, false);
        
        // RANSAC
        nh.param<int>("ego_velocity_estimator/N_ransac_points", N_ransac_points, 5);
        nh.param<float>("ego_velocity_estimator/outlier_prob", outlier_prob, 0.05);
        nh.param<float>("ego_velocity_estimator/success_prob", success_prob, 0.995);
        //  DBSCAN 
        nh.param<float>("ego_velocity_estimator/inlier_thresh", inlier_thresh, 0.5);
        nh.param<float>("ego_velocity_estimator/Min_core_points", MinCPts, 50);
        nh.param<float>("ego_velocity_estimator/Cluster_Tolerance", ClusterTR, 1);
        nh.param<int>("ego_velocity_estimator/Min_Cluster_size", MinCS, 300);
        nh.param<int>("ego_velocity_estimator/Max_Cluster_size", MaxCS, 2000);
        nh.param<int>("ego_velocity_estimator/Min_inlier_size", Min_inlier, 2000);


        nh.param<float>("ego_velocity_estimator/sigma_offset_radar_x", sigma_offset_radar_x, 0);
        nh.param<float>("ego_velocity_estimator/sigma_offset_radar_y", sigma_offset_radar_y, 0);
        nh.param<float>("ego_velocity_estimator/sigma_offset_radar_z", sigma_offset_radar_z, 0);
        nh.param<float>("ego_velocity_estimator/max_sigma_x", max_sigma_x, 0.2);
        nh.param<float>("ego_velocity_estimator/max_sigma_y", max_sigma_y, 0.2);
        nh.param<float>("ego_velocity_estimator/max_sigma_z", max_sigma_z, 0.2);
        nh.param<float>("ego_velocity_estimator/FOV", FOV_degree, 113);
        FOV_rad = FOV_degree * M_PI / 180; 

        nh.param<vector<double>>("efear/extrinsicRot", extRotV, vector<double>());
        // nh.param<vector<double>>("efear/extrinsicRPY", extRPYV, vector<double>());
        nh.param<vector<double>>("efear/extrinsicTrans", extTransV, vector<double>());
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        // extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        // extQRPY = Eigen::Quaterniond(extRPY);
        nh.param<string>("efear/input_type", input_type, "PointCloud");
        nh.param<float>("efear/power_threshold", power_threshold, 0);
        nh.param<std::string>("efear/topic_outlier_pc2", topic_outlier_pc2, "/eagle_data/outlier_pc2");
        nh.param<std::string>("efear/topic_inlier_pc2", topic_inlier_pc2, "/eagle_data/inlier_pc2");
        nh.param<bool>("efear/enable_dynamic_object_removal", enable_dynamic_object_removal, false);
        nh.param<double>("efear/distance_near_thresh", distance_near_thresh, 1.0);
        nh.param<double>("efear/distance_far_thresh", distance_far_thresh, 100.0);
        nh.param<double>("efear/z_low_thresh", z_low_thresh, -5.0);
        nh.param<double>("efear/z_high_thresh", z_high_thresh, 10.0);
        nh.param<std::string>("efear/downsample_method", downsample_method, "VOXELGRID");
        nh.param<double>("efear/downsample_resolution", downsample_resolution, 0.1);
        nh.param<std::string>("efear/outlier_removal_method", outlier_removal_method, "STATISTICAL");


        // matching 
        nh.param<double>("efear/max_egovel_cum", max_egovel_cum, 1.0);
        nh.param<int>("efear/max_submap_frames", max_submap_frames, 3.0);
        nh.param<std::string>("efear/registration_type", reg_type, "P2L");
        nh.param<double>("efear/resolution", resolution_, 1);
        // The minimum tranlational distance and rotation angle between keyframes_.
        // If this value is zero, frames are always compared with the previous frame
        nh.param<double>("efear/keyframe_delta_trans", keyframe_delta_trans, 0.25);
        nh.param<double>("efear/keyframe_delta_angle", keyframe_delta_angle, 0.15);

        // Registration validation by thresholding
        nh.param<bool>("efear/enable_transform_thresholding", enable_transform_thresholding, false);
        nh.param<bool>("efear/enable_imu_thresholding", enable_imu_thresholding, false);
        nh.param<double>("efear/max_acceptable_trans", max_acceptable_trans, 2.0);
        nh.param<double>("efear/max_acmax_submap_framesceptable_angle", max_acceptable_angle, 1.0);
        nh.param<double>("mefear/ax_diff_trans", max_diff_trans, 2.0);
        nh.param<double>("efear/max_diff_angle", max_diff_angle, 1.0);
        nh.param<double>("efear/max_egovel_cum", max_egovel_cum,1.0);

        nh.param<double>("efear/map_cloud_resolution", map_cloud_resolution, 0.05);
        nh.param<float>("efear/globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 1e3);
        nh.param<float>("efear/globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 10.0);
        nh.param<float>("efear/globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 1.0);  
        

        usleep(100);
    }

/*    sensor_msgs::Imu imuConverter(const sensor_msgs::Imu& imu_in)
    {
        sensor_msgs::Imu imu_out = imu_in;
        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();
        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();
        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            ROS_ERROR("Invalid quaternion, please use a 9-axis IMU!");
            // ros::shutdown();
        }

        return imu_out;
    }*/
};

double restrict_rad(double rad){
  double out;
  if (rad < -M_PI/2)
    out = rad + M_PI;
  else if (rad > M_PI/2)
    out = rad - M_PI;
  else out = rad;
  return out;
}

Eigen::Vector3d R2ypr(const Eigen::Matrix3d& R) {
  Eigen::Vector3d n = R.col(0);
  Eigen::Vector3d o = R.col(1);
  Eigen::Vector3d a = R.col(2);

  Eigen::Vector3d ypr(3);
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr;
}


/**
 * @brief convert Eigen::Matrix to geometry_msgs::TransformStamped
 * @param stamp            timestamp
 * @param pose             Eigen::Matrix to be converted
 * @param frame_id         tf frame_id
 * @param child_frame_id   tf child frame_id
 * @return converted TransformStamped
 */
static geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix4d& pose, const std::string& frame_id, const std::string& child_frame_id) {
  Eigen::Quaterniond quat(pose.block<3, 3>(0, 0));
  quat.normalize();
  geometry_msgs::Quaternion odom_quat;
  odom_quat.w = quat.w();
  odom_quat.x = quat.x();
  odom_quat.y = quat.y();
  odom_quat.z = quat.z();

  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = stamp;
  odom_trans.header.frame_id = frame_id;
  odom_trans.child_frame_id = child_frame_id;

  odom_trans.transform.translation.x = pose(0, 3);
  odom_trans.transform.translation.y = pose(1, 3);
  odom_trans.transform.translation.z = pose(2, 3);
  odom_trans.transform.rotation = odom_quat;

  return odom_trans;
}

static nav_msgs::Odometry matrix2odom(const ros::Time& stamp, const Eigen::Matrix4d& pose, const std::string& frame_id, const std::string& child_frame_id){
  Eigen::Quaterniond quat(pose.block<3, 3>(0, 0));
  quat.normalize();
  geometry_msgs::Quaternion odom_quat;
  odom_quat.w = quat.w();
  odom_quat.x = quat.x();
  odom_quat.y = quat.y();
  odom_quat.z = quat.z();
  
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = frame_id;
  odom.child_frame_id = child_frame_id;

  odom.pose.pose.position.x = pose(0, 3);
  odom.pose.pose.position.y = pose(1, 3);
  odom.pose.pose.position.z = pose(2, 3);
  odom.pose.pose.orientation = odom_quat;
  
  return odom;
}

static Eigen::Isometry3d pose2isometry(const geometry_msgs::Pose& pose) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  mat.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z).toRotationMatrix();
  return mat;
}

static Eigen::Isometry3d transform2isometry(const geometry_msgs::Transform& transf) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.translation() = Eigen::Vector3d(transf.translation.x, transf.translation.y, transf.translation.z);
  mat.linear() = Eigen::Quaterniond(transf.rotation.w, transf.rotation.x, transf.rotation.y, transf.rotation.z).toRotationMatrix();
  return mat;
}

static Eigen::Isometry3d tf2isometry(const tf::StampedTransform& trans) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.translation() = Eigen::Vector3d(trans.getOrigin().x(), trans.getOrigin().y(), trans.getOrigin().z());
  mat.linear() = Eigen::Quaterniond(trans.getRotation().w(), trans.getRotation().x(), trans.getRotation().y(), trans.getRotation().z()).toRotationMatrix();
  return mat;
}

static Eigen::Isometry3d quaternion2isometry(const Eigen::Quaterniond& orient) {
  Eigen::Isometry3d mat = Eigen::Isometry3d::Identity();
  mat.linear() = Eigen::Quaterniond(orient.w(), orient.x(), orient.y(), orient.z()).toRotationMatrix();
  return mat;
}

static geometry_msgs::Pose isometry2pose(const Eigen::Isometry3d& mat) {
  Eigen::Quaterniond quat(mat.linear());
  Eigen::Vector3d trans = mat.translation();

  geometry_msgs::Pose pose;
  pose.position.x = trans.x();
  pose.position.y = trans.y();
  pose.position.z = trans.z();
  pose.orientation.w = quat.w();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();

  return pose;
}

static nav_msgs::Odometry isometry2odom(const ros::Time& stamp, const Eigen::Isometry3d& mat, const std::string& frame_id, const std::string& child_frame_id) {
  Eigen::Quaterniond quat(mat.linear());
  Eigen::Vector3d trans = mat.translation();
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = frame_id;
  odom.child_frame_id = child_frame_id;
  odom.pose.pose.position.x = trans.x();
  odom.pose.pose.position.y = trans.y();
  odom.pose.pose.position.z = trans.z();
  odom.pose.pose.orientation.w = quat.w();
  odom.pose.pose.orientation.x = quat.x();
  odom.pose.pose.orientation.y = quat.y();
  odom.pose.pose.orientation.z = quat.z();

  return odom;
}

static Eigen::Isometry3d odom2isometry(const nav_msgs::OdometryConstPtr& odom_msg) {
  const auto& orientation = odom_msg->pose.pose.orientation;
  const auto& position = odom_msg->pose.pose.position;

  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}

float pointDistance(PointT p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointT p1, PointT p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}


template<typename T>
sensor_msgs::PointCloud2 publishCloud(const ros::Publisher& thisPub, const T& thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub.getNumSubscribers() != 0)
        thisPub.publish(tempCloud);
    return tempCloud;
}

pcl::PointCloud<PointT>::Ptr convertToXYZI(const pcl::PointCloud<RadarPointCloudType>::Ptr& input)
{
    auto output = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    output->reserve(input->size());

    for (const auto& pt : *input)
    {
        pcl::PointXYZI xyzi;
        xyzi.x = pt.x;
        xyzi.y = pt.y;
        xyzi.z = pt.z;
        xyzi.intensity = pt.intensity;  // 或者其他映射方式
        output->push_back(xyzi);
    }

    output->width = output->size();
    output->height = 1;
    output->is_dense = false;
    return output;
}

#endif
