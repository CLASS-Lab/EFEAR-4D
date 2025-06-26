#include "utility.h"


using namespace EFEAR_4D;

typedef std::vector<std::pair< Eigen::Isometry3d, MapNormalPtr> > PoseScanVector;
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float qx;
    float qy;
    float qz;
    float qw;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, qx, qx) (float, qy, qy) (float, qz, qz)
                                   (float, qw, qw))

typedef PointXYZIRPYT  PointTypePose;

class KeyframeUpdater {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief constructor
   * @param delta_trans
   * @param delta_angle
   */
  KeyframeUpdater(double delta_trans, double delta_angle) : is_first(true), prev_keypose(Eigen::Isometry3d::Identity()) {
    keyframe_delta_trans = delta_trans;
    keyframe_delta_angle = delta_angle;
    
    // keyframe_delta_time = pnh.param<double>("keyframe_delta_time", 2.0);
    // keyframe_min_size = pnh.param<int>("keyframe_min_size", 1000);

    accum_distance = 0.0;
  }

  /**
   * @brief decide if a new frame should be registered to the graph
   * @param pose  pose of the frame
   * @return  if true, the frame should be registered
   */
  bool decide(const Eigen::Isometry3d& pose, const ros::Time& stamp) {
    // first frame is always registered to the graph
    if(is_first) {
      is_first = false;
      prev_keypose = pose;
      prev_keytime = stamp.toSec();
      return true;
    }
    
    // calculate the delta transformation from the previous keyframe
    Eigen::Isometry3d delta = prev_keypose.inverse() * pose;
    double dx = delta.translation().norm();
    double da = Eigen::AngleAxisd(delta.linear()).angle();
    
    // too close to the previous frame
    if((dx < keyframe_delta_trans && da < keyframe_delta_angle)) {
      return false;
    }

    accum_distance += dx;
    prev_keypose = pose;
    prev_keytime = stamp.toSec();
    return true;
  }

  /**
   * @brief the last keyframe's accumulated distance from the first keyframe
   * @return accumulated distance
   */
  double get_accum_distance() const {
    return accum_distance;
  }

private:
  // parameters
  double keyframe_delta_trans;  //
  double keyframe_delta_angle;  //
  std::size_t keyframe_min_size;  //

  bool is_first;
  double accum_distance;
  Eigen::Isometry3d prev_keypose;
  double prev_keytime;
};

class Scan_Matching : public ParamServer {

private:
  // ROS topics
  ros::Subscriber points_sub;

  double timeLaserOdometry = 0;

  // Submap
  // ros::Publisher submap_pub;
  ros::Publisher odom_pub;
  ros::Publisher trans_pub;
  ros::Publisher feature_pub;

  ros::Publisher keyframe_pub;
  // ros::Publisher read_until_pub;

  ros::Publisher pubLaserCloudSurround;
  ros::ServiceServer srvSaveMap;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster map2odom_broadcaster; // map => odom_frame
  //tf::TransformBroadcaster RealtimePose;

  // std::string points_topic;
  Eigen::Matrix4d last_radar_delta = Eigen::Matrix4d::Identity();


  Eigen::Matrix4d egovel_cum = Eigen::Matrix4d::Identity();
  bool use_ego_vel;

  Eigen::Isometry3d prev_frame_trans;                // last frame relative pose 
  Eigen::Isometry3d keyframe_pose;               // latest keyframe pose
  ros::Time keyframe_stamp;                    // keyframe_ time
  pcl::PointCloud<PointT>::ConstPtr keyframe_cloud;  // keyframe_ point cloud

  PoseScanVector submaps;
  Eigen::Matrix4d guess;
  std::unique_ptr<KeyframeUpdater> keyframe_updater;
  nav_msgs::Path globalPath;
  ros::Publisher pubPath;

  pcl::PointCloud<PointT>::Ptr cloudKeyPoses3D; // pose for map
  vector<pcl::PointCloud<PointT>::Ptr> CloudKeyFrames;
  std::mutex mtx;
  pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // pose for map
  ros::Time timeInfoStamp;
  pcl::VoxelGrid<PointT> downSizeFilter;


public:


  Scan_Matching() {
    points_sub = nh.subscribe<efear::cloud_msgs>("/filtered_points", 64, &Scan_Matching::cloud_callback, this, ros::TransportHints().tcpNoDelay());
    srvSaveMap  = nh.advertiseService("efear/save_map", &Scan_Matching::saveMapService, this);



    //******** Publishers **********
    // read_until_pub = nh.advertise<std_msgs::Header>("/scan_matching_odometry/read_until", 32);

    // Odometry of Radar scan-matching_
    odom_pub = nh.advertise<nav_msgs::Odometry>(odomTopic, 32);
    // Transformation of Radar scan-matching_
    trans_pub = nh.advertise<geometry_msgs::TransformStamped>("/scan_matching_odometry/transform", 32);
    feature_pub = nh.advertise<visualization_msgs::MarkerArray>("/scan_matching/feature_visualize",100);

    //  submap_pub = nh.advertise<sensor_msgs::PointCloud2>("/scan_matching/submap", 2);
    keyframe_pub = nh.advertise<sensor_msgs::PointCloud2>("/scan_matching/keyframe", 2);

    pubPath  = nh.advertise<nav_msgs::Path>("/mapping/path", 1);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/mapping/map_global", 1);
    
    // *********ptr*******  // 
  
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointT>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    keyframe_updater.reset(new KeyframeUpdater(keyframe_delta_trans, keyframe_delta_angle));
    downSizeFilter.setLeafSize(map_cloud_resolution, map_cloud_resolution, map_cloud_resolution);
  
  }

  void cloud_callback(const efear::cloud_msgsConstPtr& msg_in)
  {
    
    // ROS_INFO_STREAM(GREEN <<"Registration begin! size: "<< msg_in->cloud.width);
    timeInfoStamp = msg_in->header.stamp;
    timeLaserOdometry = msg_in->header.stamp.toSec();
    double this_cloud_time = msg_in->header.stamp.toSec();
    static double last_cloud_time = this_cloud_time;

    double dt = this_cloud_time - last_cloud_time;

    const geometry_msgs::TwistWithCovarianceStamped twist = msg_in->twist;
    double egovel_cum_x = twist.twist.twist.linear.x * dt;
    double egovel_cum_y = twist.twist.twist.linear.y * dt;
    double egovel_cum_z = twist.twist.twist.linear.z * dt;
    // If too large, set 0
    if (pow(egovel_cum_x,2)+pow(egovel_cum_y,2)+pow(egovel_cum_z,2) > pow(max_egovel_cum, 2));
    else egovel_cum.block<3, 1>(0, 3) = Eigen::Vector3d(egovel_cum_x, egovel_cum_y, egovel_cum_z);
    
    last_cloud_time = this_cloud_time;


    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(msg_in->cloud, *cloud);

    Eigen::Isometry3d pose = matching(msg_in->header.stamp, cloud);
    
    // geometry_msgs::TwistWithCovariance twist = twistMsg->twist;
    // publish map to odom frame
    publish_odometry(msg_in->header.stamp, mapFrame, odometryFrame, pose.matrix(), twist.twist);
    publishCloud(keyframe_pub, keyframe_cloud, msg_in->header.stamp, odometryFrame); // publish keyframe cloud

    // ROS_INFO_STREAM(GREEN << "odometry finished! ");
    // In offline estimation, point clouds will be supplied until the published time
    // std_msgs::HeaderPtr read_until(new std_msgs::Header());
    // read_until->frame_id = points_topic;
    // read_until->stamp = msg_in->header.stamp + ros::Duration(1, 0);
    // read_until_pub.publish(read_until);

    // read_until->frame_id = "/filtered_points";
    // read_until_pub.publish(read_until);
  }

  Eigen::Isometry3d matching(const ros::Time& stamp, const pcl::PointCloud<PointT>::ConstPtr& cloud) 
  {
    pcl::PointCloud<PointT>::Ptr src (new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cloud, *src); 
    if(!keyframe_cloud) {

      prev_frame_trans = Eigen::Isometry3d::Identity();
      keyframe_pose = Eigen::Isometry3d::Identity();
      keyframe_cloud = cloud; // last keyframe

      MapNormalPtr scan = MapNormalPtr(new MapPointNormal(src,    resolution_));

      AddToReference(submaps,scan,keyframe_pose,max_submap_frames);
      double accum_d = 0;

      return Eigen::Isometry3d::Identity();
    }
 
    auto filtered = cloud;
 
    Eigen::Isometry3d transform_initial;

    std::string msf_source;
    Eigen::Isometry3d msf_delta = Eigen::Isometry3d::Identity();
    
    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
 
    Eigen::Isometry3d odom_now;

    // **********  Matching  **********

    guess = prev_frame_trans * egovel_cum * msf_delta.matrix();  // last frame * delta

    Eigen::Isometry3d trans_s2m;
    MapNormalPtr scan = MapNormalPtr 
    ( new MapPointNormal(src,    resolution_));
  
  
    // if(src->size()>600 || scan->GetCells().size()>100)

    compute_Initial_Transformation(src, transform_initial, guess);
    // else GICP(cloud,transform_initial,guess);
    
    odom_now = keyframe_pose * transform_initial;  
    trans_s2m = transform_initial; // relative to the newst keyframe
    // cout << "odom: " << odom_now.matrix()<<endl;
    

    // Add abnormal judgment, that is, if the difference between the two frames matching point cloud 
    // transition matrix is too large, it will be discarded
    bool thresholded = false;
    bool too_large_trans = false;

    Eigen::Matrix4d radar_delta;

    radar_delta = (prev_frame_trans.inverse() * trans_s2m).matrix();

    double dx_rd = radar_delta.block<3, 1>(0, 3).norm();

    Eigen::AngleAxisd rotation_vector;
    rotation_vector.fromRotationMatrix(radar_delta.block<3, 3>(0, 0));
    double da_rd = rotation_vector.angle();
    Eigen::Matrix3d rot_rd = radar_delta.block<3, 3>(0, 0).cast<double>();

    too_large_trans = dx_rd > max_acceptable_trans || da_rd > max_acceptable_angle;
    double da, dx, delta_rot_imu = 0;
    Eigen::Matrix3d matrix_rot; Eigen::Vector3d delta_trans_egovel;

    
    // TODO:  Use IMU orientation to determine whether the matching result is good or not
    if (enable_imu_thresholding) ;
      
    else {
      //cout << "debug!!  " << dx_rd << "[m] " << da_rd << "[degree]" << "frame (" << stamp << ")"<< endl;
      if (too_large_trans) {
        cout << "Too large transform!!  " << dx_rd << "[m] " << da_rd << "[degree]"<<
          " Ignore this frame (" << stamp << ")" << endl;
        thresholded = true;

        // prev_frame_trans = trans_s2m;
        prev_frame_trans = guess;
        odom_now = keyframe_pose * prev_frame_trans; //* radar_delta;

      }
    }
    last_radar_delta = radar_delta;

    // prev_time = stamp;
    if (!thresholded) {

      prev_frame_trans = trans_s2m;
    }
    
    //********** Decided whether to accept the frame as a key frame or not **********
    if(!too_large_trans)
    {
      if((keyframe_updater->decide(Eigen::Isometry3d(odom_now), stamp)))     
      {
        
        keyframe_stamp = stamp;

        double accum_d = keyframe_updater->get_accum_distance();

        keyframe_pose = odom_now;
        // keyframe_pose = odom_now;

        prev_frame_trans.setIdentity();
        AddToReference(submaps, scan, odom_now, max_submap_frames);
        // cout <<cloudKeyPoses3D->size()<<endl;
        
        //save key poses
        PointT thisPose3D;
        thisPose3D.x = odom_now.translation().x();
        thisPose3D.y = odom_now.translation().y();
        thisPose3D.z = odom_now.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
        // cout << "cloud key pose 3d " << endl;

        // save key frame cloud
        CloudKeyFrames.push_back(src);

        PointTypePose thisPose6D;
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        Eigen::Quaterniond quat(odom_now.rotation());
        thisPose6D.qw = quat.w();
        thisPose6D.qx = quat.x();
        thisPose6D.qy = quat.y();
        thisPose6D.qz = quat.z();
        cloudKeyPoses6D->push_back(thisPose6D);
        // cout << "cloud key pose 6d " << endl;
        
      }
    }

    // cout<<"estimate: "<< odom_now.matrix() <<endl;

    return odom_now;
  }
  void AddToReference(PoseScanVector& reference, MapNormalPtr cloud,  const Eigen::Isometry3d& T, size_t submap_scan_size){
  reference.push_back( std::make_pair(T, cloud) );
  if(reference.size() > submap_scan_size){
    reference.erase(reference.begin());
  }
  }
  void compute_Initial_Transformation(pcl::PointCloud<PointT>::Ptr &src,
                    Eigen::Isometry3d &transform, 
                    Eigen::Matrix4d &guess) 
  {
    Eigen::MatrixXd cov_Identity = Eigen::Matrix<double,6,6>::Identity();
    std::vector<Eigen::Matrix<double,6,6>> reg_cov;
    cost_metric cost = Str2Cost(reg_type);
    n_scan_normal_reg reg(cost);

    MapNormalPtr Nsrc = MapNormalPtr(new MapPointNormal(src,    resolution_));
    MapPointNormal::PublishMap(feature_pub, Nsrc, odometryFrame);
    
    std::vector<MapNormalPtr> scans;
    std::vector<Eigen::Isometry3d> Tscans;
    const Eigen::Isometry3d Tinit(guess);
    
    FormatScans(submaps, Nsrc, Tinit, reg_cov,scans,Tscans);
    bool success =  reg.Register(scans, Tscans, reg_cov,0);
    if(success)
    {
      transform = Tscans.back().cast<double>(); // keyframe pose frame
      // cout <<"transform: "<< transform.matrix() <<endl;
    }
    else
    transform = guess;
  }
  void FormatScans(const PoseScanVector& reference,
                                          const MapNormalPtr& Pcurrent,
                                          const Eigen::Isometry3d& Tcurrent,
                                          std::vector<Matrix6d>& cov_vek,
                                          std::vector<MapNormalPtr>& scans_vek,
                                          std::vector<Eigen::Isometry3d>& T_vek
                                          ){

    Eigen::Isometry3d initial_rel_pose = reference[reference.size()-1].first;// the newest one 
    for (int i=0;i<reference.size();i++) {
      cov_vek.push_back(Eigen::Matrix<double, 6, 6>::Identity());
      scans_vek.push_back(reference[i].second);
      T_vek.push_back(initial_rel_pose.inverse() * reference[i].first);
    }
    cov_vek.push_back(Eigen::Matrix<double, 6, 6>::Identity());
    scans_vek.push_back(Pcurrent);
    T_vek.push_back(Tcurrent);

  }
  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const std::string& father_frame_id, 
  const std::string& child_frame_id, const Eigen::Matrix4d& pose_in, 
  const geometry_msgs::TwistWithCovariance twist_in) {
    // publish transform stamped for IMU integration
    geometry_msgs::TransformStamped odom_trans = matrix2transform(stamp, pose_in, father_frame_id, child_frame_id); //"map" 
    trans_pub.publish(odom_trans);

    // broadcast the transform over TF
    map2odom_broadcaster.sendTransform(odom_trans);

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = father_frame_id;   // frame: /odom
    odom.child_frame_id = child_frame_id;

    odom.pose.pose.position.x = pose_in(0, 3);
    odom.pose.pose.position.y = pose_in(1, 3);
    odom.pose.pose.position.z = pose_in(2, 3);
    odom.pose.pose.orientation = odom_trans.transform.rotation;
    odom.twist = twist_in;

    odom_pub.publish(odom);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = stamp;
    pose_stamped.header.frame_id = father_frame_id;
    pose_stamped.pose.position.x = pose_in(0, 3);
    pose_stamped.pose.position.y = pose_in(1, 3);
    pose_stamped.pose.position.z = pose_in(2, 3);
    // tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation = odom_trans.transform.rotation;
    // pose_stamped.pose.orientation.y = q.y();
    // pose_stamped.pose.orientation.z = q.z();
    // pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
    globalPath.header.stamp = stamp;
    globalPath.header.frame_id = father_frame_id;
    pubPath.publish(globalPath);
  }

 
  void publishGlobalMap()
  {
    // if (pubLaserCloudSurround.getNumSubscribers() == 0)
    //         return;

    if (cloudKeyPoses3D->points.empty() == true)
        return;

    pcl::KdTreeFLANN<PointT>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointT>());;
    pcl::PointCloud<PointT>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointT>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
        globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointT> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for(auto& pt : globalMapKeyPosesDS->points)
    {
        kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
        pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
            continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(CloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointT> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeInfoStamp, mapFrame);
  }

  pcl::PointCloud<PointT>::Ptr transformPointCloud(pcl::PointCloud<PointT>::Ptr cloudIn, PointTypePose* transformIn)
  {
    pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    // 从四元数构造旋转矩阵
    Eigen::Quaternionf rotation(transformIn->qw, transformIn->qx, transformIn->qy, transformIn->qz);
    rotation.normalize(); // 确保四元数规范化
    
    // 构建完整变换矩阵
    Eigen::Affine3f transCur = Eigen::Affine3f::Identity();
    transCur.rotate(rotation);
    transCur.translation() << transformIn->x, transformIn->y, transformIn->z;
    
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
  }
  
  void visualizeGlobalMapThread()
  {
    ros::Rate rate(0.2);
    while (ros::ok()){
        rate.sleep();
        publishGlobalMap();
    }

    if (savePCD == false)
        return;

    efear::save_mapRequest  req;
    efear::save_mapResponse res;

    if(!saveMapService(req, res)){
        cout << "Fail to save map" << endl;
    }
  }
    bool saveMapService(efear::save_mapRequest& req, efear::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // create directory and remove old files;
      int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // save key frame transformations
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // extract global point cloud map
      // pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      // pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      // pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointT>::Ptr globalMapCloudDS(new pcl::PointCloud<PointT>());
      pcl::PointCloud<PointT>::Ptr globalMapCloud(new pcl::PointCloud<PointT>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalMapCloud += *transformPointCloud(CloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // down-sample and save corner cloud
        downSizeFilter.setInputCloud(globalMapCloud);
        downSizeFilter.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilter.filter(*globalMapCloudDS);
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // down-sample and save surf cloud
        // downSizeFilterSurf.setInputCloud(globalSurfCloud);
        // downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        // downSizeFilterSurf.filter(*globalSurfCloudDS);
        // pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      // else
      // {
      //   // save corner cloud
      //   pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
      //   // save surf cloud
      //   // pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      // }

      // // save global point cloud map
      // *globalMapCloud += *globalCornerCloud;
      // *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilter.setLeafSize(map_cloud_resolution, map_cloud_resolution, map_cloud_resolution);
      // downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "efear");

    Scan_Matching S;
    
    ROS_INFO("\033[1;32m---->\033[0m Scan Matching Started.");
    std::thread visualizeMapThread(&Scan_Matching::visualizeGlobalMapThread, &S);
    
    ros::spin();
    
    visualizeMapThread.join();
    return 0;
}