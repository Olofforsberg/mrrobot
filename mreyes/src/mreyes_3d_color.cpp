#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <highgui.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types_conversion.h>
#include <visualization_msgs/Marker.h>
#include <ras_msgs/RAS_Evidence.h>
#include <geometry_msgs/TransformStamped.h>
#include <unistd.h>

static void callback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_ori(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZHSV>::Ptr scene_hsv(new pcl::PointCloud<pcl::PointXYZHSV>());

  // save point cloud to .pcd
  pcl::PointCloud <pcl::PointXYZRGB> cloud_rgb;
  pcl::fromROSMsg(*cloud, cloud_rgb);
  std::vector<int> indices_rgb;
  pcl::removeNaNFromPointCloud(cloud_rgb, cloud_rgb, indices_rgb);
  pcl::io::savePCDFileASCII("test_rgb.pcd", cloud_rgb);
  
  // rotate point cloud
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
  float theta = 2.05; // The angle of rotation in radians 
  transform_1 (2,2) = cos (theta);
  transform_1 (2,1) = -sin(theta);
  transform_1 (1,2) = sin (theta);
  transform_1 (1,1) = cos (theta);
  transform_1 (2,3) = 0.125;
  transform_1 (1,3) = -0.015;
  transform_1 (0,3) = -0.015;
 
  pcl::io::loadPCDFile("test_rgb.pcd", *scene_ori);
  pcl::transformPointCloud (*scene_ori, *scene_ori, transform_1);

  pcl::PointCloudXYZRGBtoXYZHSV(*scene_ori, *scene_hsv);

  pcl::PassThrough<pcl::PointXYZHSV> pass;
  pass.setInputCloud(scene_hsv);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits(0.008,0.5);
  pass.filter(*scene_hsv);
  
  pass.setInputCloud(scene_hsv);
  pass.setFilterFieldName ("h");
  pass.setFilterLimits(0,6);
  pass.filter(*scene_hsv);
  
  pass.setInputCloud(scene_hsv);
  pass.setFilterFieldName ("s");
  pass.setFilterLimits(0,255);
  pass.filter(*scene_hsv);

  pass.setInputCloud(scene_hsv);
  pass.setFilterFieldName ("v");
  pass.setFilterLimits(0,255);
  pass.filter(*scene_hsv);

  std::cout << scene_hsv->size() << std::endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mreyes_hough");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, callback);
  ros::spin();
}
