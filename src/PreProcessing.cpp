#include "utility.h"
#include "dbscan/DBSCAN_simple.h"
#include "dbscan/DBSCAN_precomp.h"
#include "dbscan/DBSCAN_kdtree.h"
#include "tictoc.h"
#include <random>
#include <algorithm>
#include <ros/ros.h>
class Preprocess: public ParamServer
{

private:

  ros::Subscriber points_sub;  
  ros::Publisher points_pub;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;

  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Filter<PointT>::Ptr outlier_removal_filter;

  ros::Publisher pub_twist, pub_inlier_pc2, pub_outlier_pc2, pc2_raw_pub;

  std::vector<Eigen::VectorXi> num_at_dist_vec;
  const RadarEgoVelocityEstimatorIndices idx_;

  typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, PointCloudAdaptor<PointT>>,
    PointCloudAdaptor<PointT>, 3> KDTree;

public: 

  Preprocess(): idx_()
  {

    if (input_type == "PointCloud") {
    points_sub = nh.subscribe(pointCloudTopic, 64, &Preprocess::cloud_callback, this);
    // ROS_INFO("PointCloud subscribe!");
    } else if (input_type == "PointCloud2") {
        points_sub = nh.subscribe(pointCloudTopic, 64, &Preprocess::cloud2_callback, this);
        // ROS_INFO("PointCloud2 subscribe!");
    } else {
        ROS_ERROR("Unsupported point cloud type");
        // 处理错误
    }

    pc2_raw_pub = nh.advertise<sensor_msgs::PointCloud2>("/eagle_data/pc2_raw",1);
    pub_inlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_inlier_pc2, 5);
    pub_outlier_pc2 = nh.advertise<sensor_msgs::PointCloud2>(topic_outlier_pc2, 5);

    points_pub = nh.advertise<efear::cloud_msgs>("/filtered_points", 32);
    initializationValue();
  }
  void initializationValue()
  {
    calibration_matrix.block<3,3>(0,0) = extRot;
    calibration_matrix.topRightCorner(3, 1) = extTrans;
    cout << calibration_matrix << endl;

    if(downsample_method == "VOXELGRID") {
      // std::cout << "downsample: VOXELGRID " << downsample_resolution << std::endl;
      auto voxelgrid = new pcl::VoxelGrid<PointT>();
      voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter.reset(voxelgrid);
    } else if(downsample_method == "APPROX_VOXELGRID") {
      // std::cout << "downsample: APPROX_VOXELGRID " << downsample_resolution << std::endl;
      pcl::ApproximateVoxelGrid<PointT>::Ptr approx_voxelgrid(new pcl::ApproximateVoxelGrid<PointT>());
      approx_voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
      downsample_filter = approx_voxelgrid;
    } else {
      if(downsample_method != "NONE") {
        std::cerr << "warning: unknown downsampling type (" << downsample_method << ")" << std::endl;
        std::cerr << "       : use passthrough filter" << std::endl;
      }
      // std::cout << "downsample: NONE" << std::endl;
    }
     
    if(outlier_removal_method == "STATISTICAL") {
      int mean_k = 20;
      double stddev_mul_thresh  = 1.0;
      // std::cout << "outlier_removal: STATISTICAL " << mean_k << " - " << stddev_mul_thresh << std::endl;

      pcl::StatisticalOutlierRemoval<PointT>::Ptr sor(new pcl::StatisticalOutlierRemoval<PointT>());
      sor->setMeanK(mean_k);
      sor->setStddevMulThresh(stddev_mul_thresh);
      outlier_removal_filter = sor;
    } 
    else if(outlier_removal_method == "RADIUS") 
    {
      double radius = 0.8;
      int min_neighbors = 2;
      // std::cout << "outlier_removal: RADIUS " << radius << " - " << min_neighbors << std::endl;

      pcl::RadiusOutlierRemoval<PointT>::Ptr rad(new pcl::RadiusOutlierRemoval<PointT>());
      rad->setRadiusSearch(radius);
      rad->setMinNeighborsInRadius(min_neighbors);
      outlier_removal_filter = rad;
    } 
  }

  void cloud_callback(const sensor_msgs::PointCloud::ConstPtr&  eagle_msg) 
  {
    
    std::vector<Eigen::Vector4d> raw_points;
    std::vector<float> dopplers;
    std::vector<float> powers;

    for (size_t i = 0; i < eagle_msg->points.size(); ++i)
    {
        const auto& pt = eagle_msg->points[i];

        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;

        raw_points.emplace_back(pt.x, pt.y, pt.z, 1.0);  
        dopplers.push_back(eagle_msg->channels[0].values[i]);
        powers.push_back(eagle_msg->channels[2].values[i]);
    }
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw(new pcl::PointCloud<RadarPointCloudType>);
    pcl::PointCloud<PointT>::Ptr radarcloud_xyzi(new pcl::PointCloud<PointT>);
    processRadarCloud(raw_points, dopplers, powers, radarcloud_raw, radarcloud_xyzi, eagle_msg->header.stamp, baselinkFrame);
    
    //********** Ego Velocity Estimation **********
    preprocessing(radarcloud_raw, radarcloud_xyzi, eagle_msg->header.stamp, baselinkFrame);
  }
  void processRadarCloud(
    const std::vector<Eigen::Vector4d>& raw_points, 
    const std::vector<float>& dopplers,
    const std::vector<float>& powers,
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw, 
    pcl::PointCloud<PointT>::Ptr radarcloud_xyzi,
    const ros::Time& stamp,
    const std::string& frame_id)
  {
    
    radarcloud_xyzi->header.frame_id = frame_id;
    radarcloud_xyzi->header.stamp = stamp.toSec() * 1e6;

    radarcloud_raw->header.frame_id = frame_id;
    radarcloud_raw->header.stamp = stamp.toSec() * 1e6;

    for (size_t i = 0; i < raw_points.size(); ++i)
    {
        if (powers[i] <= power_threshold) continue;

        const auto& pt = raw_points[i];
        if (!std::isfinite(pt.x()) || !std::isfinite(pt.y()) || !std::isfinite(pt.z())) continue;

        Eigen::Vector4d dst_pt = calibration_matrix * pt;

        RadarPointCloudType p_raw;
        p_raw.x = dst_pt.x();
        p_raw.y = dst_pt.y();
        p_raw.z = dst_pt.z();
        p_raw.intensity = powers[i];
        p_raw.doppler = dopplers[i];

        PointT p_xyzi;
        p_xyzi.x = dst_pt.x();
        p_xyzi.y = dst_pt.y();
        p_xyzi.z = dst_pt.z();
        p_xyzi.intensity = powers[i];

        radarcloud_raw->push_back(p_raw);
        radarcloud_xyzi->push_back(p_xyzi);
    }

    publishCloud(pc2_raw_pub, radarcloud_raw, stamp, frame_id);

}

  void cloud2_callback(const sensor_msgs::PointCloud2ConstPtr&  eagle_msg) { // const pcl::PointCloud<PointT>& src_cloud_r
    
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw( new pcl::PointCloud<RadarPointCloudType> );
    pcl::PointCloud<PointT>::Ptr radarcloud_xyzi( new pcl::PointCloud<PointT> );
    // ROS_INFO_STREAM(GREEN << "begin");
    pcl::PointCloud<mscCloudType> scan_mmwave;
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*eagle_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, scan_mmwave);

    std::vector<Eigen::Vector4d> raw_points;
    std::vector<float> dopplers;
    std::vector<float> powers;

    for (const auto& p : scan_mmwave)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;

        raw_points.emplace_back(p.x, p.y, p.z, 1.0);
        dopplers.push_back(p.doppler);
        powers.push_back(p.power);
    }
    processRadarCloud(raw_points, dopplers, powers, radarcloud_raw, radarcloud_xyzi, 
    eagle_msg->header.stamp, baselinkFrame);

    //********** Ego Velocity Estimation **********
    // ROS_INFO_STREAM(GREEN << "begin ego estimation: ");
    preprocessing(radarcloud_raw, radarcloud_xyzi, eagle_msg->header.stamp, baselinkFrame);
  } 

  void preprocessing(const pcl::PointCloud<RadarPointCloudType>::ConstPtr radarcloud_raw, 
    const pcl::PointCloud<PointT>::ConstPtr radarcloud_xyzi, 
    const ros::Time& stamp,
    const std::string& frame_id)
  {
    Eigen::Vector3d v_r, sigma_v_r;
    sensor_msgs::PointCloud2 inlier_radar_msg, outlier_radar_msg;
    clock_t start_ms = clock();
    geometry_msgs::TwistWithCovarianceStamped twist;

    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_inlier( new pcl::PointCloud<RadarPointCloudType> );
    pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_outlier( new pcl::PointCloud<RadarPointCloudType> );
    if (estimate(radarcloud_raw, v_r, sigma_v_r, radarcloud_inlier, radarcloud_outlier))
    {
        clock_t end_ms = clock();
      
        twist.header.stamp         = stamp;
        twist.twist.twist.linear.x = v_r.x();
        twist.twist.twist.linear.y = v_r.y();
        twist.twist.twist.linear.z = v_r.z();

        twist.twist.covariance.at(0)  = std::pow(sigma_v_r.x(), 2);
        twist.twist.covariance.at(7)  = std::pow(sigma_v_r.y(), 2);
        twist.twist.covariance.at(14) = std::pow(sigma_v_r.z(), 2);

        publishCloud(pub_inlier_pc2, radarcloud_inlier, stamp, frame_id);
        publishCloud(pub_outlier_pc2, radarcloud_outlier, stamp, frame_id);

    }


    // pcl::fromROSMsg (inlier_radar_msg, *radarcloud_inlier);

    pcl::PointCloud<PointT>::ConstPtr src_cloud;
    if (enable_dynamic_object_removal)
    {
      src_cloud = convertToXYZI(radarcloud_inlier);
    }
    else
      src_cloud = radarcloud_xyzi;

    
    if(src_cloud->empty()) {
      return;
    }

    // // if baselinkFrame is defined, transform the input cloud to the frame
    // if(!baselinkFrame.empty()) {
    //   if(!tf_listener.canTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0))) {
    //     std::cerr << "failed to find transform between " << baselinkFrame << " and " << src_cloud->header.frame_id << std::endl;
    //   }

    //   tf::StampedTransform transform;
    //   tf_listener.waitForTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), ros::Duration(2.0));
    //   tf_listener.lookupTransform(baselinkFrame, src_cloud->header.frame_id, ros::Time(0), transform);

    //   pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
    //   pcl_ros::transformPointCloud(*src_cloud, *transformed, transform);
    //   transformed->header.frame_id = baselinkFrame;
    //   transformed->header.stamp = src_cloud->header.stamp;
    //   src_cloud = transformed;
    // }

    pcl::PointCloud<PointT>::ConstPtr filtered = distance_filter(src_cloud);
      if(src_cloud->size()>2000)
    {
      filtered = downsample(filtered);
      filtered = outlier_removal(filtered);
    }

    efear::cloud_msgs msg;

    msg.header.stamp = stamp;
    msg.header.frame_id = baselinkFrame;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*src_cloud, cloud_msg);
    cloud_msg.header = msg.header;
    msg.cloud = cloud_msg;

    
    msg.twist = twist;
    points_pub.publish(msg);
  }
  
  bool estimate(const pcl::PointCloud<RadarPointCloudType>::ConstPtr radar_scan,
                                         Eigen::Vector3d& v_r,
                                         Eigen::Vector3d& sigma_v_r,
                                         pcl::PointCloud<RadarPointCloudType>::Ptr radar_scan_inlier,
                                         pcl::PointCloud<RadarPointCloudType>::Ptr radar_scan_outlier)
  {
    
    bool success = false;

    std::vector<Eigen::Matrix<double, 11, 1>> valid_targets;
    std::vector<Eigen::Matrix<double, 11, 1>> all_targets;
    std::vector<std::pair<double, Eigen::Matrix<double, 11, 1>>> intensity_valid_points;

    for (uint i = 0; i < radar_scan->size(); ++i)
    {
      const auto& target = radar_scan->at(i);
      const double r = Eigen::Vector3d(target.x, target.y, target.z).norm();

      double azimuth = std::atan2(target.y, target.x);
      double elevation = std::atan2(std::sqrt(target.x * target.x + target.y * target.y), target.z) - M_PI_2;

      Eigen::Matrix<double, 11, 1> v_pt;
      v_pt << target.x, target.y, target.z, target.intensity, -target.doppler,
              r, azimuth, elevation, target.x / r, target.y / r, target.z / r;

      if (target.intensity > min_db)
      {
        intensity_valid_points.emplace_back(target.intensity, v_pt);  // 暂存强度和点
      }

        all_targets.emplace_back(v_pt);
    }

    // Sort by intensity in descending order
    std::sort(intensity_valid_points.begin(), intensity_valid_points.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Too much points will be too slow.
    size_t N = std::min<size_t>(MaxCS, intensity_valid_points.size());
    for (size_t i = 0; i < N; ++i)
    {
      valid_targets.emplace_back(intensity_valid_points[i].second);
    }

    if (valid_targets.size() > 2)
    {
      // check for zero velocity
      std::vector<double> v_dopplers;
      for (const auto& v_pt : valid_targets) v_dopplers.emplace_back(std::fabs(v_pt[4])); // idx_doppler
      const size_t n = v_dopplers.size() * (1.0 - allowed_outlier_percentage);
      std::nth_element(v_dopplers.begin(), v_dopplers.begin() + n, v_dopplers.end());
      const auto median = v_dopplers[n];

      Eigen::MatrixXd valid_radar_data(valid_targets.size(), 4);  // rx, ry, rz, v
      Eigen::MatrixXd valid_radar_velocity(valid_targets.size(), 3);
      Eigen::MatrixXd all_radar_data(all_targets.size(), 4);
      uint idx = 0, idx_o = 0;

      for (const auto& v_pt : valid_targets)
      {
          valid_radar_data.row(idx) = Eigen::Vector4d(v_pt[idx_.normalized_x], v_pt[idx_.normalized_y], 
          v_pt[idx_.normalized_z], v_pt[idx_.doppler]);
          double vx = v_pt[idx_.normalized_x] * v_pt[idx_.doppler];
          double vy = v_pt[idx_.normalized_y] * v_pt[idx_.doppler];
          double vz =  v_pt[idx_.normalized_z] * v_pt[idx_.doppler];
          valid_radar_velocity.row(idx) = Eigen::Vector3d(vx, vy, vz);  // vx, vy, vz
          idx++;
      }

      for (const auto& v_pt : all_targets)
      {
          all_radar_data.row(idx_o++) = Eigen::Vector4d(v_pt[idx_.normalized_x], v_pt[idx_.normalized_y], 
          v_pt[idx_.normalized_z], v_pt[idx_.doppler]);
      }
         
      if (median < thresh_zero_velocity)
      {
        // ROS_INFO_STREAM_THROTTLE(0.5, kPrefix << "Zero velocity detected!");
        v_r = Eigen::Vector3d(0, 0, 0);
        
        sigma_v_r = ((valid_radar_velocity.rowwise() - valid_radar_velocity.colwise().mean()).array().square().colwise().sum() / 
                                 (valid_radar_velocity.rows() - 1)).sqrt();

        for (const auto& item : all_targets)
          if (std::fabs(item[4]) < thresh_zero_velocity) 
            radar_scan_inlier->push_back(toRadarPointCloudType(item, idx_));
        
        success = true;
      }
      else
      {
        // LSQ velocity estimation
        
        std::vector<uint> inlier_idx_best;
        std::vector<uint> outlier_idx_best;

        if (!estimateRadarVelocity(valid_radar_data, all_radar_data, v_r, sigma_v_r, inlier_idx_best, outlier_idx_best, median, 
                                  use_ransac ? "ransac" : "dbscan"))
        {
          ROS_WARN_STREAM("Velocity estimation failed, fallback to RANSAC.");
          estimateRadarVelocity(valid_radar_data, all_radar_data, v_r, sigma_v_r, inlier_idx_best, outlier_idx_best, median, "ransac");
        }

        for (const auto& idx : inlier_idx_best)
          radar_scan_inlier->push_back(toRadarPointCloudType(all_targets.at(idx), idx_));

        for (const auto& idx : outlier_idx_best)
          radar_scan_outlier->push_back(toRadarPointCloudType(all_targets.at(idx), idx_));

        success = true;

      }
    }
    else ROS_INFO("To small valid_targets (< 2) in radar_scan (%ld)", radar_scan->size());
    radar_scan_inlier->height = 1;
    radar_scan_inlier->width  = radar_scan_inlier->size();

    // pcl::toROSMsg(*radar_scan_inlier, inlier_radar_msg);  

    radar_scan_outlier->height = 1;
    radar_scan_outlier->width  = radar_scan_outlier->size();

    // pcl::toROSMsg(*radar_scan_outlier, outlier_radar_msg);


    // if(success)
    //   ROS_INFO_STREAM(GREEN << "Ego Velocity estimation Successful! speed:" <<v_r);
    // else
    //   ROS_INFO_STREAM(GREEN << "Ego Velocity estimation Failed");

    return success; 
  } 

  bool estimateRadarVelocity(
      const Eigen::MatrixXd& radar_data, const Eigen::MatrixXd& all_radar_data,
      Eigen::Vector3d& v_r, Eigen::Vector3d& sigma_v_r,
      std::vector<uint>& inlier_idx, std::vector<uint>& outlier_idx, double median, 
      const std::string& method)
  {
    if (method == "ransac")
    {
      return solve3DFullRansac(radar_data, all_radar_data, v_r, sigma_v_r, inlier_idx, outlier_idx);
    }
    else if (method == "dbscan")
    {
      Eigen::MatrixXd dbscan_data;
      bool success = dbscan(radar_data, dbscan_data, median);
      if (!success) return false;

      if (!solvebyNonlinearOptimization(dbscan_data, v_r)) return false;

      classifyInliers(all_radar_data.leftCols(3), all_radar_data.col(3), v_r, inlier_thresh, inlier_idx, outlier_idx);
      compute_sigma(all_radar_data.leftCols(3), all_radar_data.col(3), v_r, sigma_v_r);

      float outlier_ratio = float(outlier_idx.size()) / (inlier_idx.size() + outlier_idx.size());
      if (outlier_ratio > 0.5 && inlier_idx.size() < Min_inlier)
      {
        ROS_WARN_STREAM("Too many outliers in DBSCAN, merging into inliers. outlier = " << outlier_idx.size() <<" inlier =" << inlier_idx.size()
      <<" ratio = " << outlier_ratio);
        inlier_idx.insert(inlier_idx.end(), outlier_idx.begin(), outlier_idx.end());
        outlier_idx.clear();
      }

      return true;
    }
    else
    {
      ROS_ERROR_STREAM("Unknown method: " << method);
      return false;
    }
  }

  bool solve3DFullRansac(const Eigen::MatrixXd& radar_data,
                        const Eigen::MatrixXd& all_radar_data,
                                                    Eigen::Vector3d& v_r,
                                                    Eigen::Vector3d& sigma_v_r,
                                                    std::vector<uint>& inlier_idx_best,
                                                    std::vector<uint>& outlier_idx_best)
  {
    Eigen::MatrixXd H_all(radar_data.rows(), 3);
    H_all.col(0)       = radar_data.col(0);
    H_all.col(1)       = radar_data.col(1);
    H_all.col(2)       = radar_data.col(2);
    const Eigen::VectorXd y_all = radar_data.col(3);

    std::vector<uint> idx(radar_data.rows());
    for (uint k = 0; k < radar_data.rows(); ++k) idx[k] = k;

    std::random_device rd;
    std::mt19937 g(rd());

    uint ransac_iter_ = uint((std::log(1.0 - success_prob)) /
                          std::log(1.0 - std::pow(1.0 - outlier_prob, N_ransac_points)));
    // ROS_INFO_STREAM("Number of Ransac iterations: " << ransac_iter_);

    if (radar_data.rows() >= N_ransac_points)
    {
      for (uint k = 0; k < ransac_iter_; ++k)
      {
        std::shuffle(idx.begin(), idx.end(), g);
        Eigen::MatrixXd radar_data_iter;
        radar_data_iter.resize(N_ransac_points, 4);

        for (uint i = 0; i < N_ransac_points; ++i) radar_data_iter.row(i) = radar_data.row(idx.at(i));
        bool rtn = solve3DFull(radar_data_iter, v_r, sigma_v_r, false);
        if (rtn == false) ROS_INFO("Failure at solve3DFullRansac() 1");
        if (rtn)
        {
          std::vector<uint> inlier_idx;
          std::vector<uint> outlier_idx;
          v_r[2]=0;
          classifyInliers(all_radar_data.leftCols(3), all_radar_data.col(3), v_r, inlier_thresh, inlier_idx, outlier_idx);
          // ROS_INFO("Inlier number: %ld, Outlier number: %ld, outlier Ratio: %f", 
                    // inlier_idx.size(), outlier_idx.size(), float(outlier_idx.size())/(inlier_idx.size()+outlier_idx.size()));
          // if too small number of inlier detected, the error is too high, so regard outlier as inlier
          if ( float(outlier_idx.size())/(inlier_idx.size()+outlier_idx.size()) > 0.05 )
          {
            inlier_idx.insert(inlier_idx.end(), outlier_idx.begin(), outlier_idx.end());
            outlier_idx.clear();
            // outlier_idx.swap(std::vector<uint>());
          }

          // ROS_INFO("Inlier number: %ld, Outlier number: %ld, outlier Ratio: %f", 
                    // inlier_idx.size(), outlier_idx.size(), float(outlier_idx.size())/(inlier_idx.size()+outlier_idx.size()));

          if (inlier_idx.size() > inlier_idx_best.size())
          {
            inlier_idx_best = inlier_idx;
          }
          if (outlier_idx.size() > outlier_idx_best.size())
          {
            outlier_idx_best = outlier_idx;
          }
        }
      }
    }
    else{ROS_INFO("Warning: radar_data.rows() < config_.N_ransac_points");}

    if (!inlier_idx_best.empty())
    {
      Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
      for (uint i = 0; i < inlier_idx_best.size(); ++i) radar_data_inlier.row(i) = all_radar_data.row(inlier_idx_best.at(i));
      
      bool rtn = solve3DFull(radar_data_inlier, v_r, sigma_v_r, true);
      if (rtn == false) ROS_INFO("Failure at solve3DFullRansac() 2");
      return rtn;
    }
    ROS_INFO("Failure at solve3DFullRansac() 3");
    return false;
  }

  bool solve3DFull(const Eigen::MatrixXd& radar_data,
                                              Eigen::Vector3d& v_r,
                                              Eigen::Vector3d& sigma_v_r,
                                              bool estimate_sigma)
  {
    Eigen::MatrixXd H(radar_data.rows(), 3);
    H.col(0)         = radar_data.col(0);
    H.col(1)         = radar_data.col(1);
    H.col(2)         = radar_data.col(2);
    const Eigen::MatrixXd HTH = H.transpose() * H;

    const Eigen::VectorXd y = radar_data.col(3);
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(HTH);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

    // cond > 1000, error occurs
    if (1)//std::fabs(cond) < 1.0e3
    {
      if (1) // use_cholesky_instead_of_bdcsvd
      {
        v_r = (HTH).ldlt().solve(H.transpose() * y);
      }
      else
        v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

      if (estimate_sigma)
      {
        compute_sigma(H, y, v_r, sigma_v_r);
      }
      else
      {
        return true;
      }
    }
    //ROS_INFO("cond too large, cond = %f", cond);

    return true;//return false;
  }

  bool dbscan(const Eigen::MatrixXd& radar_data, Eigen::MatrixXd& radar_inlier_data, double median)
  {  
      pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
      const int N = radar_data.rows();

      for(int i = 0; i < radar_data.rows(); i++)
      {
        PointT point;
        point.x = radar_data(i,0) * radar_data(i,3);
        point.y = radar_data(i,1) * radar_data(i,3);
        point.z = 0; // radar_data(i,2)*radar_data(i,3); due to fov limitation, we set to zero
        point.intensity = i;
        if (point.x <= thresh_zero_velocity && !robot_id) continue;  // The car cannot move backward or horizontally, it must have forward speed except hand draft
        
        cloud->push_back(point);

      }
      // cout <<cloud->points.size() << " points in cloud" << std::endl << median << " median doppler" << std::endl;
      
      // Two types DBSCAN for choice
      // TicToc t1;
      
      // std::vector<pcl::PointIndices> cluster_indices1;
      // pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
      // tree->setInputCloud(cloud);

      // DBSCANKdtreeCluster<PointT> ec;
      // ec.setCorePointMinPts(50);
      // ec.setClusterTolerance(1);
      // ec.setMinClusterSize(300);
      // ec.setMaxClusterSize(5000);
      // ec.setSearchMethod(tree);
      // ec.setInputCloud(cloud);
      // ec.extract(cluster_indices1);
      // std::cout << "category: " << cluster_indices1.size() << std::endl;
      // t1.toc("dbscan time: ");

      // TicToc t2;
      std::vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>> points = cloud->points;
      std::vector<pcl::PointIndices> cluster_indices;

      FastDBSCAN<pcl::PointXYZI> dbscan;
      dbscan.setInputCloud(points);
      dbscan.setCorePointMinPts(MinCPts);
      dbscan.setClusterTolerance(ClusterTR);
      dbscan.setMinClusterSize(MinCS);
      dbscan.setMaxClusterSize(MaxCS);
      dbscan.extract(cluster_indices);

      // std::cout << "cluster count: " << cluster_indices.size() << std::endl;
      // t2.toc("dbscan time: ");

      if (!cluster_indices.size())
        return false;

      double best_angular_range = -1.0;
      int best_cluster = -1;
      std::vector<Eigen::Vector4d> best_cluster_points;

      for (size_t j = 0; j < cluster_indices.size(); ++j) {
          const auto& indices = cluster_indices[j].indices;
          std::vector<double> angles;
          std::vector<Eigen::Vector4d> current_points;
          current_points.reserve(indices.size());

          for (int idx : indices) {
              const auto& pt = cloud->points[idx];
              float x = pt.x, y = pt.y;
              double angle = std::atan2(y, x);  // [-pi, pi]
              angles.push_back(angle);
              current_points.emplace_back(x, y, 0.0f, pt.intensity);
          }

          // Sort and compute angular coverage
          std::sort(angles.begin(), angles.end());
          double min_angle = angles.front();
          double max_angle = angles.back();
          double angular_range = max_angle - min_angle;

          // -π ~ π
          if (angular_range > M_PI) {
              // If cross -pi ~ pi, calculate the supplementary angle in reverse
              angular_range = 2 * M_PI - angular_range;
          }

          angular_range = std::abs(angular_range);  

          // std::cout << "cluster " << j << ", size: " << indices.size()
          //           << ", angular range (rad): " << angular_range
          //           << ", degree: " << angular_range * 180.0 / M_PI 
          //           << ", max angles " << max_angle * 180.0 / M_PI 
          //           << ", min angles " << min_angle * 180.0 / M_PI
          //           << std::endl;

          //Due to the limitation of the field of view, it should be closer to the arc of FOV
          
          if (std::abs(angular_range-FOV_rad) < std::abs(best_angular_range- FOV_rad)) { 
              best_angular_range = angular_range;
              best_cluster = j;
              best_cluster_points.swap(current_points);
          }
        }

      radar_inlier_data.resize(best_cluster_points.size(), 4);
      for (size_t i = 0; i < best_cluster_points.size(); ++i) {
          radar_inlier_data.row(i) = best_cluster_points[i];
      }

      // std::cout << "Selected cluster with angular coverage: " 
      //           << best_angular_range * 180.0 / M_PI << " degrees" << std::endl;

      return true;
  }

  bool solvebyNonlinearOptimization(const Eigen::MatrixXd radar_data, Eigen::Vector3d& v_r)
  {

      ceres::Problem problem;
      std::vector<Eigen::Vector3d> parameter;
      Eigen::Vector2d center;

      for (int i=0; i<radar_data.rows(); i++) {
        Eigen::Vector3d point;
        point[0] = radar_data(i,0);
        point[1] = radar_data(i,1);
        point[2] = radar_data(i,2);

        ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<SphereFittingCostFunction, 1, 3>(
            new SphereFittingCostFunction(point[0], point[1], point[2])
        );
        
        problem.AddResidualBlock(costFunction, nullptr, center.data());
      }
      //problem.AddResidualBlock(costFunction, nullptr, parameter);

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      // options.minimizer_progress_to_stdout = true;
      options.max_num_iterations = 500;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      v_r[0] = center[0] * 2;
      v_r[1] = center[1] * 2;
      v_r[2] = 0;//center[2] * 2; 
      return summary.IsSolutionUsable();
  }

  void classifyInliers(const Eigen::MatrixXd& H_all,
                      const Eigen::VectorXd& y_all,
                      const Eigen::Vector3d& v_r,
                      double inlier_thresh,
                      std::vector<uint>& inlier_idx,
                      std::vector<uint>& outlier_idx)
  {
      Eigen::VectorXd err = (y_all - H_all * v_r).array().abs();
      for (int i = 0; i < err.size(); ++i) {
          if (err(i) < inlier_thresh)
              inlier_idx.push_back(i);
          else
              outlier_idx.push_back(i);
      }
      
  }
  void compute_sigma(const Eigen::MatrixXd& H,
                      const Eigen::VectorXd& y,
                      const Eigen::Vector3d& v_r,
                      Eigen::Vector3d& sigma_v_r)
  {
        const Eigen::MatrixXd HTH = H.transpose() * H;               
        const Eigen::VectorXd e = H * v_r - y;
        const Eigen::MatrixXd C = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
        sigma_v_r      = Eigen::Vector3d(C(0, 0), C(1, 1), C(2, 2));
        sigma_v_r      = sigma_v_r.array();
        if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.)
        {
          sigma_v_r = sigma_v_r.array().sqrt();
          sigma_v_r += Eigen::Vector3d(sigma_offset_radar_x, sigma_offset_radar_y, sigma_offset_radar_z);
        }            
  }

  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());

    filtered->reserve(cloud->size());
    std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const PointT& p) {
      double d = p.getVector3fMap().norm();
      double z = p.z;
      return d > distance_near_thresh && d < distance_far_thresh && z < z_high_thresh && z > z_low_thresh;
    });

    filtered->width = filtered->size();
    filtered->height = 1;
    filtered->is_dense = false;

    filtered->header = cloud->header;

    return filtered;
  }
 pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      // Remove NaN/Inf points
      pcl::PointCloud<PointT>::Ptr cloudout(new pcl::PointCloud<PointT>());
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloud, *cloudout, indices);
      
      return cloudout;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  pcl::PointCloud<PointT>::ConstPtr outlier_removal(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!outlier_removal_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    outlier_removal_filter->setInputCloud(cloud);
    outlier_removal_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }


  static RadarPointCloudType toRadarPointCloudType(const Eigen::Matrix<double, 11, 1>& item, const RadarEgoVelocityEstimatorIndices& idx)
  {
    RadarPointCloudType point;
    point.x             = item[idx.x_r];
    point.y             = item[idx.y_r];
    point.z             = item[idx.z_r];
    point.doppler       = -item[idx.doppler];
    point.intensity     = item[idx.snr_db];
    return point;
  }


};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "efear");

    Preprocess P;
    
    ROS_INFO("\033[1;32m---->\033[0m PreProcess Started.");
    
    ros::spin();
    return 0;
}
