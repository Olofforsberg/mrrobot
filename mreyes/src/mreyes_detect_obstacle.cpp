#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <visualization_msgs/Marker.h>
#include <ras_msgs/RAS_Evidence.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <unistd.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <pcl/io/pcd_io.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
//#include <point_cloud2_iterator.h>
#include <pcl/octree/octree.h>
#include <algorithm> // for std::find
#include <iterator> // for std::begin, std::end
#include <utility> 
#include <pcl/filters/voxel_grid.h>

using namespace std;
namespace enc = sensor_msgs::image_encodings;

tf2_ros::Buffer *tfb;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;




static ros::Publisher pub;

static float min_z, max_z;


void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*msg, *cloud);
  geometry_msgs::TransformStamped t_in;
  try {
    t_in = tfb->lookupTransform("map", msg->header.frame_id, msg->header.stamp, ros::Duration(0.0));
  } catch(const tf2::ExtrapolationException& e) {
    std::cerr << "could not transform" << std::endl;
    return;
  } catch(const tf2::LookupException& e) {
    std::cerr << "could not transform" << std::endl;
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  Eigen::Transform<float,3,Eigen::Affine> t = Eigen::Translation3f(t_in.transform.translation.x,
                                                                   t_in.transform.translation.y,
                                                                   t_in.transform.translation.z)
  *Eigen::Quaternion<float>(t_in.transform.rotation.w,
                            t_in.transform.rotation.x,
                            t_in.transform.rotation.y,
                            t_in.transform.rotation.z);
  pcl::transformPointCloud(*cloud, *transformed_cloud, t);
  transformed_cloud->header.frame_id = "map";
  
  //change to PCL to cut of floor
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZ>());

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*transformed_cloud, *transformed_cloud, indices);

  //pcl::toPCLPointCloud2(transformed_cloud,cloud2)
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(transformed_cloud);
  sor.setFilterFieldName ("z");
  sor.setFilterLimits (-0.02, 0.08);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);

  for (size_t i = 0; i < cloud_filtered->points.size (); ++i)
    {	if (cloud_filtered->points[i].z>0.02f)
		{
    		cloud_filtered->points[i].z = 0.02;
         	}
	else
		{
		cloud_filtered->points[i].z = 0;
		}
    }
  pub.publish(cloud_filtered);
  
}

int main(int argc, char** argv){

  ros::init(argc, argv, "mreyes_detect_obstacle");
  ros::NodeHandle nh;
  nh.param("min_z", min_z, 0.02f);
  nh.param("max_z", max_z, 1.00f);
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  tfb = &tfBuffer;
  ros::Subscriber pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points", 1, pc_callback);
  pub = nh.advertise<PointCloud>("camera/input_to_map", 1);
  ros::spin();
  return 0;
}
