// LAST UPDATE: 2024.01.25
//
// AUTHOR: Neset Unver Akmandor (NUA)
//
// E-MAIL: akmandor.n@northeastern.edu
//
// DESCRIPTION: TODO...
// 
// TODO:
//

// --CUSTOM LIBRARIES--
#include "mobiman_simulation/scan_utility.h"

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility::ScanUtility(NodeHandle& nh)
{
  tflistener_ = new tf::TransformListener;

  pkg_dir_ = ros::package::getPath("mobiman_simulation") + "/";

  // Publishers
  pub_pc2_msg_scan_ = nh.advertise<sensor_msgs::PointCloud2>("pc2_scan", 10);
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility::ScanUtility(NodeHandle& nh, string data_path)
{
  tflistener_ = new tf::TransformListener;

  pkg_dir_ = ros::package::getPath("mobiman_simulation") + "/";

  readPointcloud2Data(data_path);

  // Publishers
  pub_pc2_msg_scan_ = nh.advertise<sensor_msgs::PointCloud2>("pc2_scan", 10);
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility::ScanUtility(NodeHandle& nh,
                         string obj_name,
                         string data_dir,
                         string world_frame_name,
                         vector<string> pc2_msg_name_vec_,
                         double scan_bbx_x_min,
                         double scan_bbx_x_max,
                         double scan_bbx_y_min,
                         double scan_bbx_y_max,
                         double scan_bbx_z_min,
                         double scan_bbx_z_max,
                         double oct_resolution)
{
  tflistener_ = new tf::TransformListener;

  pkg_dir_ = ros::package::getPath("mobiman_simulation") + "/";
  obj_name_ = obj_name;
  data_dir_ = data_dir;
  world_frame_name_ = world_frame_name;

  for (size_t i = 0; i < pc2_msg_name_vec_.size(); i++)
  {
    switch (i)
    {
      case 0:
        pc2_msg_name_sensor1_ = pc2_msg_name_vec_[i];
        break;
      case 1:
        pc2_msg_name_sensor2_ = pc2_msg_name_vec_[i];
        break;
      case 2:
        pc2_msg_name_sensor3_ = pc2_msg_name_vec_[i];
        break;
      case 3:
        pc2_msg_name_sensor4_ = pc2_msg_name_vec_[i];
        break;
      default:
        pc2_msg_name_sensor1_ = pc2_msg_name_vec_[i];
        break;
    }
  }

  scan_bbx_x_min_ = scan_bbx_x_min;
  scan_bbx_x_max_ = scan_bbx_x_max;
  scan_bbx_y_min_ = scan_bbx_y_min;
  scan_bbx_y_max_ = scan_bbx_y_max;
  scan_bbx_z_min_ = scan_bbx_z_min;
  scan_bbx_z_max_ = scan_bbx_z_max;

  oct_resolution_ = oct_resolution;
  oct_ = std::make_shared<octomap::ColorOcTree>(oct_resolution_);

  point3d scan_bbx_mini(scan_bbx_x_min_, scan_bbx_y_min_, scan_bbx_z_min_);
  point3d scan_bbx_maxi(scan_bbx_x_max_, scan_bbx_y_max_, scan_bbx_z_max_);
  oct_->setBBXMin(scan_bbx_mini);
  oct_->setBBXMax(scan_bbx_maxi);

  // Subscribers
  //sub_pc2_sensor1_ = nh.subscribe(pc2_msg_name_sensor1_, 1000, &ScanUtility::pc2CallbackSensor1, this);
  //sub_pc2_sensor2_ = nh.subscribe(pc2_msg_name_sensor2_, 1000, &ScanUtility::pc2CallbackSensor2, this);
  //sub_pc2_sensor3_ = nh.subscribe(pc2_msg_name_sensor3_, 1000, &ScanUtility::pc2CallbackSensor3, this);
  //sub_pc2_sensor4_ = nh.subscribe(pc2_msg_name_sensor4_, 1000, &ScanUtility::pc2CallbackSensor4, this);

  // Publishers
  pub_oct_msg_ = nh.advertise<octomap_msgs::Octomap>("octomap_scan", 10);
  pub_pc2_msg_scan_ = nh.advertise<sensor_msgs::PointCloud2>("pc2_scan", 10);
  //pub_debug_array_visu_ = nh.advertise<visualization_msgs::MarkerArray>("scan_debug_array", 10);
  //pub_debug_visu_ = nh.advertise<visualization_msgs::Marker>("scan_debug", 10);
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility::~ScanUtility()
{
  //std::std::cout << "[ScanUtility::~ScanUtility] Calling Destructor for ScanUtility..." << std::std::endl;
  delete[] tflistener_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility::ScanUtility(const ScanUtility& su)
{
  tflistener_ = su.tflistener_;
  
  obj_name_ = su.obj_name_;
  pkg_dir_ = su.pkg_dir_;
  data_dir_ = su.data_dir_;
  data_path_ = su.data_path_;
  world_frame_name_ = su.world_frame_name_;

  pc2_msg_name_sensor1_ =  su.pc2_msg_name_sensor1_;
  pc2_msg_name_sensor2_ =  su.pc2_msg_name_sensor2_;
  pc2_msg_name_sensor3_ =  su.pc2_msg_name_sensor3_;
  pc2_msg_name_sensor4_ =  su.pc2_msg_name_sensor4_;

  scan_bbx_x_min_ = su.scan_bbx_x_min_;
  scan_bbx_x_max_ = su.scan_bbx_x_max_;
  scan_bbx_y_min_ = su.scan_bbx_y_min_;
  scan_bbx_y_max_ = su.scan_bbx_y_max_;
  scan_bbx_z_min_ = su.scan_bbx_z_min_;
  scan_bbx_z_max_ = su.scan_bbx_z_max_;
  
  pose_sensor1_ = su.pose_sensor1_;
  pose_sensor2_ = su.pose_sensor2_;
  pose_sensor3_ = su.pose_sensor3_;
  pose_sensor4_ = su.pose_sensor4_;

  pc2_msg_sensor1_ = su.pc2_msg_sensor1_;
  pc2_msg_sensor2_ = su.pc2_msg_sensor2_;
  pc2_msg_sensor3_ = su.pc2_msg_sensor3_;
  pc2_msg_sensor4_ = su.pc2_msg_sensor4_;

  oct_pc_sensor1_ = su.oct_pc_sensor1_;
  oct_pc_sensor2_ = su.oct_pc_sensor2_;
  oct_pc_sensor3_ = su.oct_pc_sensor3_;
  oct_pc_sensor4_ = su.oct_pc_sensor4_;

  oct_resolution_ = su.oct_resolution_;
  oct_ = su.oct_;
  oct_msg_ = su.oct_msg_;

  pcl_pc_scan_ = su.pcl_pc_scan_;
  pc2_msg_scan_ = su.pc2_msg_scan_;

  obj_dim_ = su.obj_dim_;
  obj_bbx_min_ = su.obj_bbx_min_;
  obj_bbx_max_ = su.obj_bbx_max_;

  //debug_array_visu_ = su.debug_array_visu_;
  //debug_visu_ = su.debug_visu_;
    
  sub_pc2_sensor1_ = su.sub_pc2_sensor1_;
  sub_pc2_sensor2_ = su.sub_pc2_sensor2_;
  sub_pc2_sensor3_ = su.sub_pc2_sensor3_;
  sub_pc2_sensor4_ = su.sub_pc2_sensor4_;

  pub_pc2_msg_scan_ = su.pub_pc2_msg_scan_;
  pub_oct_msg_ = su.pub_oct_msg_;
  //pub_debug_array_visu_ = su.pub_debug_array_visu_;
  //pub_debug_visu_ = su.pub_debug_visu_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
ScanUtility& ScanUtility::operator=(const ScanUtility& su) 
{
  tflistener_ = su.tflistener_;
  
  obj_name_ = su.obj_name_;
  pkg_dir_ = su.pkg_dir_;
  data_dir_ = su.data_dir_;
  data_path_ = su.data_path_;
  world_frame_name_ = su.world_frame_name_;

  pc2_msg_name_sensor1_ =  su.pc2_msg_name_sensor1_;
  pc2_msg_name_sensor2_ =  su.pc2_msg_name_sensor2_;
  pc2_msg_name_sensor3_ =  su.pc2_msg_name_sensor3_;
  pc2_msg_name_sensor4_ =  su.pc2_msg_name_sensor4_;

  scan_bbx_x_min_ = su.scan_bbx_x_min_;
  scan_bbx_x_max_ = su.scan_bbx_x_max_;
  scan_bbx_y_min_ = su.scan_bbx_y_min_;
  scan_bbx_y_max_ = su.scan_bbx_y_max_;
  scan_bbx_z_min_ = su.scan_bbx_z_min_;
  scan_bbx_z_max_ = su.scan_bbx_z_max_;

  pose_sensor1_ = su.pose_sensor1_;
  pose_sensor2_ = su.pose_sensor2_;
  pose_sensor3_ = su.pose_sensor3_;
  pose_sensor4_ = su.pose_sensor4_;

  pc2_msg_sensor1_ = su.pc2_msg_sensor1_;
  pc2_msg_sensor2_ = su.pc2_msg_sensor2_;
  pc2_msg_sensor3_ = su.pc2_msg_sensor3_;
  pc2_msg_sensor4_ = su.pc2_msg_sensor4_;

  oct_pc_sensor1_ = su.oct_pc_sensor1_;
  oct_pc_sensor2_ = su.oct_pc_sensor2_;
  oct_pc_sensor3_ = su.oct_pc_sensor3_;
  oct_pc_sensor4_ = su.oct_pc_sensor4_;

  oct_resolution_ = su.oct_resolution_;
  oct_ = su.oct_;
  oct_msg_ = su.oct_msg_;

  pcl_pc_scan_ = su.pcl_pc_scan_;
  pc2_msg_scan_ = su.pc2_msg_scan_;

  obj_dim_ = su.obj_dim_;
  obj_bbx_min_ = su.obj_bbx_min_;
  obj_bbx_max_ = su.obj_bbx_max_;

  //debug_array_visu_ = su.debug_array_visu_;
  //debug_visu_ = su.debug_visu_;
    
  sub_pc2_sensor1_ = su.sub_pc2_sensor1_;
  sub_pc2_sensor2_ = su.sub_pc2_sensor2_;
  sub_pc2_sensor3_ = su.sub_pc2_sensor3_;
  sub_pc2_sensor4_ = su.sub_pc2_sensor4_;

  pub_pc2_msg_scan_ = su.pub_pc2_msg_scan_;
  pub_oct_msg_ = su.pub_oct_msg_;
  //pub_debug_array_visu_ = su.pub_debug_array_visu_;
  //pub_debug_visu_ = su.pub_debug_visu_;

  return *this;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
sensor_msgs::PointCloud2 ScanUtility::getPC2MsgScan()
{
  return pc2_msg_scan_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
geometry_msgs::Point ScanUtility::getObjBbxMin()
{
  return obj_bbx_min_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
geometry_msgs::Point ScanUtility::getObjBbxMax()
{
  return obj_bbx_max_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
geometry_msgs::Point ScanUtility::getObjDim()
{
  return obj_dim_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::getPointcloud2wrtWorld(const sensor_msgs::PointCloud2& msg_in, 
                                         sensor_msgs::PointCloud2& msg_out)
{
  //std::cout << "[ScanUtility::getPointcloud2wrtWorld] START" << std::endl;
  tf::StampedTransform transform_sensor_wrt_world;
  try
  {
    tflistener_->lookupTransform(world_frame_name_, msg_in.header.frame_id, ros::Time(0), transform_sensor_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("[ScanUtility::getPointcloud2wrtWorld] Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, transform_sensor_wrt_world, msg_in, msg_out);

  //std::cout << "[ScanUtility::getPointcloud2wrtWorld] END" << std::endl;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::getSensorPoseAndTransformPointcloud2(const sensor_msgs::PointCloud2& msg_in, 
                                                       geometry_msgs::Pose& sensor_pose, 
                                                       sensor_msgs::PointCloud2& msg_out)
{
  //std::cout << "[ScanUtility::getSensorPoseAndTransformPointcloud2] START" << std::endl;
  tf::StampedTransform transform_sensor_wrt_world;
  try
  {
    tflistener_->lookupTransform(world_frame_name_, msg_in.header.frame_id, ros::Time(0), transform_sensor_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("[ScanUtility::getSensorPoseAndTransformPointcloud2] Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED SENSOR POSE
  sensor_pose.position.x = transform_sensor_wrt_world.getOrigin().x();
  sensor_pose.position.y = transform_sensor_wrt_world.getOrigin().y();
  sensor_pose.position.z = transform_sensor_wrt_world.getOrigin().z();
  tf::quaternionTFToMsg(transform_sensor_wrt_world.getRotation(), sensor_pose.orientation);

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, transform_sensor_wrt_world, msg_in, msg_out);

  //std::cout << "[ScanUtility::getSensorPoseAndTransformPointcloud2] END" << std::endl;
}

/*
void ScanUtility::pc2CallbackSensor1(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  tf::StampedTransform measured_transform_sensor1_wrt_world;

  //std::cout << "[ScanUtility::pc2CallbackSensor1] Incoming data..." << std::endl;
  try
  {
    //tflistener_ -> waitForTransform(world_frame_name_, msg -> header.frame_id, ros::Time::now(), ros::Duration(1.0));
    tflistener_ -> lookupTransform(world_frame_name_, msg -> header.frame_id, ros::Time(0), measured_transform_sensor1_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("ScanUtility::pc2CallbackSensor1 -> Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED SENSOR POSE
  pose_sensor1_.position.x = measured_transform_sensor1_wrt_world.getOrigin().x();
  pose_sensor1_.position.y = measured_transform_sensor1_wrt_world.getOrigin().y();
  pose_sensor1_.position.z = measured_transform_sensor1_wrt_world.getOrigin().z();
  tf::quaternionTFToMsg(measured_transform_sensor1_wrt_world.getRotation(), pose_sensor1_.orientation);

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, measured_transform_sensor1_wrt_world, *msg, measured_pc2_msg_sensor1_);
}

void ScanUtility::pc2CallbackSensor2(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  tf::StampedTransform measured_transform_sensor2_wrt_world;

  //std::cout << "[ScanUtility::pc2CallbackSensor2] Incoming data..." << std::endl;
  try
  {
    //tflistener_ -> waitForTransform(world_frame_name_, msg -> header.frame_id, ros::Time::now(), ros::Duration(1.0));
    tflistener_ -> lookupTransform(world_frame_name_, msg -> header.frame_id, ros::Time(0), measured_transform_sensor2_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("ScanUtility::pc2CallbackSensor2 -> Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED SENSOR POSE
  pose_sensor2_.position.x = measured_transform_sensor2_wrt_world.getOrigin().x();
  pose_sensor2_.position.y = measured_transform_sensor2_wrt_world.getOrigin().y();
  pose_sensor2_.position.z = measured_transform_sensor2_wrt_world.getOrigin().z();
  tf::quaternionTFToMsg(measured_transform_sensor2_wrt_world.getRotation(), pose_sensor2_.orientation);

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, measured_transform_sensor2_wrt_world, *msg, measured_pc2_msg_sensor2_);
}

void ScanUtility::pc2CallbackSensor3(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  tf::StampedTransform measured_transform_sensor3_wrt_world;

  //std::cout << "[ScanUtility::pc2CallbackSensor3] Incoming data..." << std::endl;
  try
  {
    //tflistener_ -> waitForTransform(world_frame_name_, msg -> header.frame_id, ros::Time::now(), ros::Duration(1.0));
    tflistener_ -> lookupTransform(world_frame_name_, msg -> header.frame_id, ros::Time(0), measured_transform_sensor3_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("ScanUtility::pc2CallbackSensor3 -> Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED SENSOR POSE
  pose_sensor3_.position.x = measured_transform_sensor3_wrt_world.getOrigin().x();
  pose_sensor3_.position.y = measured_transform_sensor3_wrt_world.getOrigin().y();
  pose_sensor3_.position.z = measured_transform_sensor3_wrt_world.getOrigin().z();
  tf::quaternionTFToMsg(measured_transform_sensor3_wrt_world.getRotation(), pose_sensor3_.orientation);

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, measured_transform_sensor3_wrt_world, *msg, measured_pc2_msg_sensor3_);
}

void ScanUtility::pc2CallbackSensor4(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  tf::StampedTransform measured_transform_sensor4_wrt_world;

  //std::cout << "[ScanUtility::pc2CallbackSensor4] Incoming data..." << std::endl;
  try
  {
    //tflistener_ -> waitForTransform(world_frame_name_, msg -> header.frame_id, ros::Time::now(), ros::Duration(1.0));
    tflistener_ -> lookupTransform(world_frame_name_, msg -> header.frame_id, ros::Time(0), measured_transform_sensor4_wrt_world);
  }
  catch (tf::TransformException ex)
  {
    ROS_INFO("ScanUtility::pc2CallbackSensor4 -> Couldn't get transform!");
    ROS_ERROR("%s", ex.what());
  }

  // SET MEASURED SENSOR POSE
  pose_sensor4_.position.x = measured_transform_sensor4_wrt_world.getOrigin().x();
  pose_sensor4_.position.y = measured_transform_sensor4_wrt_world.getOrigin().y();
  pose_sensor4_.position.z = measured_transform_sensor4_wrt_world.getOrigin().z();
  tf::quaternionTFToMsg(measured_transform_sensor4_wrt_world.getRotation(), pose_sensor4_.orientation);

  // SET MEASURED PC2
  pcl_ros::transformPointCloud(world_frame_name_, measured_transform_sensor4_wrt_world, *msg, measured_pc2_msg_sensor4_);
}
*/

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::getScanPointcloud2(string data_path, sensor_msgs::PointCloud2& pc2_msg)
{
  readPointcloud2Data(data_path);
  pc2_msg = pc2_msg_scan_;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::octomapToPclPointcloud()
{
  pcl_pc_scan_.clear();
  for (octomap::ColorOcTree::iterator it = oct_ -> begin(); it != oct_ -> end(); ++it)
  {
    if (oct_->isNodeOccupied(*it))
    {
      if (!std::isnan(it.getX()) && !std::isnan(it.getY()) && !std::isnan(it.getZ()))
      {
        pcl_pc_scan_.push_back(pcl::PointXYZ(it.getX(), it.getY(), it.getZ()));
      }
    }
  }

  /*
  for (octomap::ColorOcTree::leaf_iterator it = oct_ -> begin_leafs(); it != oct_ -> end_leafs(); ++it)
  {
    if (oct_->isNodeOccupied(*it))
    {
      if (!std::isnan(it.getX()) && !std::isnan(it.getY()) && !std::isnan(it.getZ()))
      {
        pcl_pc_scan_.push_back(pcl::PointXYZ(it.getX(), it.getY(), it.getZ()));
      }
    }
  }
  */
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::PclPointcloudToVec(vector<double>& pcl_pc_scan_x, 
                                     vector<double>& pcl_pc_scan_y, 
                                     vector<double>& pcl_pc_scan_z)
{
  pcl::PointCloud<pcl::PointXYZ> pcl_pc_scan = pcl_pc_scan_;
  for (auto c : pcl_pc_scan)
  {
    pcl_pc_scan_x.push_back(c.x);
    pcl_pc_scan_y.push_back(c.y);
    pcl_pc_scan_z.push_back(c.z);
  }

  if (printOutFlag_)
  {
    std::cout << "[ScanUtility::PclPointcloudToVec] pcl_pc_scan_ size: " << pcl_pc_scan_.size() << std::endl;
    std::cout << "[ScanUtility::PclPointcloudToVec] pcl_pc_scan_x size: " << pcl_pc_scan_x.size() << std::endl;
    std::cout << "[ScanUtility::PclPointcloudToVec] pcl_pc_scan_y size: " << pcl_pc_scan_y.size() << std::endl;
    std::cout << "[ScanUtility::PclPointcloudToVec] pcl_pc_scan_z size: " << pcl_pc_scan_z.size() << std::endl;
  }
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::vecToPclPointcloud(vector<double>& pcl_pc_scan_x, 
                                     vector<double>& pcl_pc_scan_y, 
                                     vector<double>& pcl_pc_scan_z)
{
  pcl_pc_scan_.clear();
  for (size_t i = 0; i < pcl_pc_scan_x.size(); i++)
  {
    pcl_pc_scan_.push_back(pcl::PointXYZ(pcl_pc_scan_x[i], pcl_pc_scan_y[i], pcl_pc_scan_z[i]));
  }

  if (printOutFlag_)
  {
    std::cout << "[ScanUtility::vecToPclPointcloud] pcl_pc_scan_ size: " << pcl_pc_scan_.size() << std::endl;
    std::cout << "[ScanUtility::vecToPclPointcloud] pcl_pc_scan_x size: " << pcl_pc_scan_x.size() << std::endl;
    std::cout << "[ScanUtility::vecToPclPointcloud] pcl_pc_scan_y size: " << pcl_pc_scan_y.size() << std::endl;
    std::cout << "[ScanUtility::vecToPclPointcloud] pcl_pc_scan_z size: " << pcl_pc_scan_z.size() << std::endl;
  } 
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::updateObjBbxDim(vector<double>& pcl_pc_scan_x, 
                                  vector<double>& pcl_pc_scan_y, 
                                  vector<double>& pcl_pc_scan_z)
{
  obj_bbx_min_.x = *min_element(pcl_pc_scan_x.begin(), pcl_pc_scan_x.end());
  obj_bbx_min_.y = *min_element(pcl_pc_scan_y.begin(), pcl_pc_scan_y.end());
  obj_bbx_min_.z = *min_element(pcl_pc_scan_z.begin(), pcl_pc_scan_z.end());

  obj_bbx_max_.x = *max_element(pcl_pc_scan_x.begin(), pcl_pc_scan_x.end());
  obj_bbx_max_.y = *max_element(pcl_pc_scan_y.begin(), pcl_pc_scan_y.end());
  obj_bbx_max_.z = *max_element(pcl_pc_scan_z.begin(), pcl_pc_scan_z.end());

  obj_dim_.x = obj_bbx_max_.x - obj_bbx_min_.x;
  obj_dim_.y = obj_bbx_max_.y - obj_bbx_min_.y;
  obj_dim_.z = obj_bbx_max_.z - obj_bbx_min_.z;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::fillOctMsgFromOct()
{
  oct_msg_.data.clear();
  oct_msg_.header.frame_id = world_frame_name_;
  oct_msg_.binary = false;
  oct_msg_.id = "object_scan";
  oct_msg_.resolution = oct_resolution_;
  octomap_msgs::fullMapToMsg(*oct_, oct_msg_);
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::publishOctMsg()
{
  oct_msg_.header.frame_id = world_frame_name_;
  //oct_msg.header.seq++;
  //oct_msg.header.stamp = ros::Time(0);
  oct_msg_.header.stamp = ros::Time::now();
  pub_oct_msg_.publish(oct_msg_);
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::publishPC2Msg()
{
  pc2_msg_scan_.header.frame_id = world_frame_name_;
  pc2_msg_scan_.header.seq++;
  //pc2_msg.header.stamp = ros::Time(0);
  pc2_msg_scan_.header.stamp = ros::Time::now();
  pub_pc2_msg_scan_.publish(pc2_msg_scan_);
}

/*
void ScanUtility::mainCallback(const ros::TimerEvent& e)
{
  //std::cout << "[ScanUtility::mainCallback] START" << std::endl;

  pc2_msg_sensor1_ = measured_pc2_msg_sensor1_;
  pc2_msg_sensor2_ = measured_pc2_msg_sensor2_;
  pc2_msg_sensor3_ = measured_pc2_msg_sensor3_;
  pc2_msg_sensor4_ = measured_pc2_msg_sensor4_;

  pointCloud2ToOctomap(pc2_msg_sensor1_, oct_pc_sensor1_);
  pointCloud2ToOctomap(pc2_msg_sensor2_, oct_pc_sensor2_);
  pointCloud2ToOctomap(pc2_msg_sensor3_, oct_pc_sensor3_);
  pointCloud2ToOctomap(pc2_msg_sensor4_, oct_pc_sensor4_);

  point3d sensor1_origin(pose_sensor1_.position.x, pose_sensor1_.position.y, pose_sensor1_.position.z);
  point3d sensor2_origin(pose_sensor2_.position.x, pose_sensor2_.position.y, pose_sensor2_.position.z);
  point3d sensor3_origin(pose_sensor3_.position.x, pose_sensor3_.position.y, pose_sensor3_.position.z);
  point3d sensor4_origin(pose_sensor4_.position.x, pose_sensor4_.position.y, pose_sensor4_.position.z);

  oct_ -> insertPointCloud(oct_pc_sensor1_, sensor1_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor2_, sensor2_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor3_, sensor3_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor4_, sensor4_origin, 10, false, true);

  octomapToPclPointcloud();

  //std::cout << "[ScanUtility::mainCallback] pcl_pc_scan_ size: " << pcl_pc_scan_.size() << std::endl;

  pcl::toROSMsg(pcl_pc_scan_, pc2_msg_scan_);

  fillOctMsgFromOct();
  
  publishOctMsg();
  publishPC2Msg();

  //std::cout << "[ScanUtility::mainCallback] END" << std::endl << std::endl;
}
*/

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::waitAndCheckForPointCloud2Message(string msg_name, double duration_time, sensor_msgs::PointCloud2& pc_msg)
{
  boost::shared_ptr<sensor_msgs::PointCloud2 const> sharedPtr;

  sharedPtr = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(msg_name, ros::Duration(duration_time));
  
  if (sharedPtr == NULL)
  {
    ROS_INFO("[ScanUtility::waitAndCheckForPointCloud2Message] No point clound messages received"); 
  }
  else
  {
    pc_msg = *sharedPtr;
  }
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::scanner()
{
  //std::cout << "[ScanUtility::scanner] START" << std::endl;

  waitAndCheckForPointCloud2Message(pc2_msg_name_sensor1_, 10, measured_pc2_msg_sensor1_);
  waitAndCheckForPointCloud2Message(pc2_msg_name_sensor2_, 10, measured_pc2_msg_sensor2_);
  waitAndCheckForPointCloud2Message(pc2_msg_name_sensor3_, 10, measured_pc2_msg_sensor3_);
  waitAndCheckForPointCloud2Message(pc2_msg_name_sensor4_, 10, measured_pc2_msg_sensor4_);

  getSensorPoseAndTransformPointcloud2(measured_pc2_msg_sensor1_, pose_sensor1_, pc2_msg_sensor1_);
  getSensorPoseAndTransformPointcloud2(measured_pc2_msg_sensor2_, pose_sensor2_, pc2_msg_sensor2_);
  getSensorPoseAndTransformPointcloud2(measured_pc2_msg_sensor3_, pose_sensor3_, pc2_msg_sensor3_);
  getSensorPoseAndTransformPointcloud2(measured_pc2_msg_sensor4_, pose_sensor4_, pc2_msg_sensor4_);

  pointCloud2ToOctomap(pc2_msg_sensor1_, oct_pc_sensor1_);
  pointCloud2ToOctomap(pc2_msg_sensor2_, oct_pc_sensor2_);
  pointCloud2ToOctomap(pc2_msg_sensor3_, oct_pc_sensor3_);
  pointCloud2ToOctomap(pc2_msg_sensor4_, oct_pc_sensor4_);

  point3d sensor1_origin(pose_sensor1_.position.x, pose_sensor1_.position.y, pose_sensor1_.position.z);
  point3d sensor2_origin(pose_sensor2_.position.x, pose_sensor2_.position.y, pose_sensor2_.position.z);
  point3d sensor3_origin(pose_sensor3_.position.x, pose_sensor3_.position.y, pose_sensor3_.position.z);
  point3d sensor4_origin(pose_sensor4_.position.x, pose_sensor4_.position.y, pose_sensor4_.position.z);

  oct_ -> insertPointCloud(oct_pc_sensor1_, sensor1_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor2_, sensor2_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor3_, sensor3_origin, 10, false, true);
  oct_ -> insertPointCloud(oct_pc_sensor4_, sensor4_origin, 10, false, true);

  octomapToPclPointcloud();

  //std::cout << "[ScanUtility::scanner] pcl_pc_scan_ size: " << pcl_pc_scan_.size() << std::endl;

  pcl::toROSMsg(pcl_pc_scan_, pc2_msg_scan_);

  fillOctMsgFromOct();
  
  publishOctMsg();
  publishPC2Msg();

  //std::cout << "[ScanUtility::scanner] END" << std::endl << std::endl;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::writePointcloud2Data()
{
  //std::cout << "[ScanUtility::writePointcloud2Data] START" << std::endl;

  json j;
  boost::filesystem::create_directories(pkg_dir_ + data_dir_);
  data_path_ = pkg_dir_ + data_dir_ + obj_name_ + ".json";

  vector<double> pcl_pc_scan_x, pcl_pc_scan_y, pcl_pc_scan_z;
  PclPointcloudToVec(pcl_pc_scan_x, pcl_pc_scan_y, pcl_pc_scan_z);
  pcl::PointCloud<pcl::PointXYZ> main = pcl_pc_scan_;

  /// NUA NOTE: DO WE NEED THIS? 
  //pcl::io::savePCDFileASCII("/home/mobiman/mobiman/test.pcd", main);
  
  updateObjBbxDim(pcl_pc_scan_x, pcl_pc_scan_y, pcl_pc_scan_z);

  j["obj_name"] = obj_name_;
  j["data_path"] = data_path_;
  j["world_frame_name"] = world_frame_name_;
  j["oct_resolution"] = oct_resolution_;
  j["obj_bbx"]["min"]["x"] = obj_bbx_min_.x;
  j["obj_bbx"]["min"]["y"] = obj_bbx_min_.y;
  j["obj_bbx"]["min"]["z"] = obj_bbx_min_.z;
  j["obj_bbx"]["max"]["x"] = obj_bbx_max_.x;
  j["obj_bbx"]["max"]["y"] = obj_bbx_max_.y;
  j["obj_bbx"]["max"]["z"] = obj_bbx_max_.z;
  j["obj_dim"]["x"] = obj_dim_.x;
  j["obj_dim"]["y"] = obj_dim_.y;
  j["obj_dim"]["z"] = obj_dim_.z;

  // Pointcloud2 data
  j["pc2"]["x"] = pcl_pc_scan_x;
  j["pc2"]["y"] = pcl_pc_scan_y;
  j["pc2"]["z"] = pcl_pc_scan_z;

  /*
  // Sensor1 pose
  j["obj"]["pos"]["x"] = pose_sensor1_.position.x;
  j["obj"]["pos"]["y"] = pose_sensor1_.position.y;
  j["obj"]["pos"]["z"] = pose_sensor1_.position.z;
  j["obj"]["quat"]["x"] = pose_sensor1_.orientation.x;
  j["obj"]["quat"]["y"] = pose_sensor1_.orientation.y;
  j["obj"]["quat"]["z"] = pose_sensor1_.orientation.z;
  j["obj"]["quat"]["w"] = pose_sensor1_.orientation.w;
  */

  std::ofstream o(data_path_);
  o << std::setw(4) << j << std::endl;

  //std::cout << "[ScanUtility::writePointcloud2Data] END" << std::endl << std::endl;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::readPointcloud2Data(string& data_path)
{
  //std::cout << "[ScanUtility::readPointcloud2Data] START" << std::endl;
  //std::cout << "[ScanUtility::readPointcloud2Data] pkg_dir_: " << pkg_dir_ << std::endl;
  //std::cout << "[ScanUtility::readPointcloud2Data] data_path: " << data_path << std::endl;

  std::ifstream f(pkg_dir_ + data_path);
  json data = json::parse(f);
  
  obj_name_ = data["obj_name"];
  data_path_ = data["data_path"];
  world_frame_name_ = data["world_frame_name"];
  oct_resolution_ = data["oct_resolution"];

  vector<double> pcl_pc_scan_x, pcl_pc_scan_y, pcl_pc_scan_z;
  boost::property_tree::ptree pt;
  boost::property_tree::read_json(pkg_dir_ + data_path, pt);
  BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("pc2").get_child("x"))
  {
    pcl_pc_scan_x.push_back(stod(v.second.data()));
  }
  BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("pc2").get_child("y"))
  {
    pcl_pc_scan_y.push_back(stod(v.second.data()));
  }
  BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt.get_child("pc2").get_child("z"))
  {
    pcl_pc_scan_z.push_back(stod(v.second.data()));
  }
  vecToPclPointcloud(pcl_pc_scan_x, pcl_pc_scan_y, pcl_pc_scan_z);

  if (printOutFlag_)
  {
    std::cout << "[ScanUtility::readPointcloud2Data] obj_name_: " << obj_name_ << std::endl;
    std::cout << "[ScanUtility::readPointcloud2Data] data_path_: " << data_path_ << std::endl;
    std::cout << "[ScanUtility::readPointcloud2Data] world_frame_name_: " << world_frame_name_ << std::endl;
    std::cout << "[ScanUtility::readPointcloud2Data] oct_resolution_: " << oct_resolution_ << std::endl;
    std::cout << "[ScanUtility::scanner] pcl_pc_scan_ size: " << pcl_pc_scan_.size() << std::endl << std::endl;
  }
  
  pcl::toROSMsg(pcl_pc_scan_, pc2_msg_scan_);

  //std::cout << "[ScanUtility::readPointcloud2Data] END" << std::endl << std::endl;
}

//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------
void ScanUtility::readObjBbxDim(string& data_path)
{
  //std::cout << "[ScanUtility::readObjBbxDim] START" << std::endl;

  std::ifstream f(pkg_dir_ + data_path);
  json data = json::parse(f);
  
  obj_bbx_min_.x = data["obj_bbx"]["min"]["x"];
  obj_bbx_min_.y = data["obj_bbx"]["min"]["y"];
  obj_bbx_min_.z = data["obj_bbx"]["min"]["z"];

  obj_bbx_max_.x = data["obj_bbx"]["max"]["x"];
  obj_bbx_max_.y = data["obj_bbx"]["max"]["y"];
  obj_bbx_max_.z = data["obj_bbx"]["max"]["z"];

  obj_dim_.x = data["obj_dim"]["x"];
  obj_dim_.y = data["obj_dim"]["y"];
  obj_dim_.z = data["obj_dim"]["z"];

  if (printOutFlag_)
  {
    std::cout << "[ScanUtility::readObjBbxDim] pkg_dir_: " << pkg_dir_ << std::endl;
    std::cout << "[ScanUtility::readObjBbxDim] data_path: " << data_path << std::endl;
    std::cout << "[ScanUtility::readObjBbxDim] obj_bbx_min_: " << std::endl;
    print(obj_bbx_min_);
    std::cout << "[ScanUtility::readObjBbxDim] obj_bbx_max_: " << std::endl;
    print(obj_bbx_max_);
    std::cout << "[ScanUtility::readObjBbxDim] obj_dim_: " << std::endl;
    print(obj_dim_);
  }

  //std::cout << "[ScanUtility::readObjBbxDim] END" << std::endl;
}
