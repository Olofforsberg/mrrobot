#include "ros/ros.h"
#include "tf/transform_listener.h"
#include "sensor_msgs/PointCloud.h"
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include "laser_geometry/laser_geometry.h"

class LaserScanToPointCloud{

public:

  ros::NodeHandle n_;
  laser_geometry::LaserProjection projector_;
  tf::TransformListener listener_;
  message_filters::Subscriber<sensor_msgs::LaserScan> laser_sub_;
  tf::MessageFilter<sensor_msgs::LaserScan> laser_notifier_;
  ros::Publisher scan_pub_;

  sensor_msgs::PointCloud cloud, cloud_old, cloud_new;
  int count;
  LaserScanToPointCloud(ros::NodeHandle n) : 
    n_(n),
    laser_sub_(n_, "scan", 10),
    laser_notifier_(laser_sub_,listener_, "odom", 10)
  {
    laser_notifier_.registerCallback(
      boost::bind(&LaserScanToPointCloud::scanCallback, this, _1));
    laser_notifier_.setTolerance(ros::Duration(0.01));
    scan_pub_ = n_.advertise<sensor_msgs::PointCloud>("/my_cloud",10000);
  }

  void scanCallback (const sensor_msgs::LaserScan::ConstPtr& scan_in)
  {
	
    try
    {
        projector_.transformLaserScanToPointCloud(
          "odom",*scan_in, cloud,listener_);
	
	
    }
    catch (tf::TransformException& e)
    {
        std::cout << e.what();
        return;
    }
    
    // Do something with cloud.
    
	int N=2;
    if(count >N)
    {
	cloud_new.points.resize(cloud.points.size()+cloud_old.points.size());
	std::copy(cloud.points.begin(),cloud.points.end(),cloud_new.points.begin()+cloud_old.points.size());	
	//ROS_INFO_STREAM("cloud:"<<cloud.points.size()<<"cloud_old: " << cloud_old.points.size()<<"cloud_new:" << cloud_new.points.size());
	cloud.points.resize(cloud_new.points.size());
	std::copy(cloud_new.points.begin(),cloud_new.points.end(),cloud.points.begin());
	scan_pub_.publish(cloud);
	cloud_old.points.resize(cloud_new.points.size());
	std::copy(cloud_new.points.begin(),cloud_new.points.end(),cloud_old.points.begin());
	count=0;

	if(cloud_old.points.size()>10000)
		cloud_old.points.resize(0);
}	
else
count++;
	
  }
};

int main(int argc, char** argv)
{
  
   ros::init(argc, argv, "my_scan_to_cloud");
  ros::NodeHandle n;
  LaserScanToPointCloud lstopc(n);
  ros::spin();


  
  return 0;
}
