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
#include <std_srvs/SetBool.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <sensor_msgs/PointCloud2.h>

using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

bool service_call = false;
volatile bool have_object;
bool arm_status = false;
volatile bool check_status = false;
/*
static const char WINDOW[] = "Image window";
//0-red:    0-8,    0-255,  0-255;
//1-blue:   72-112, 77-255, 82-255;
//2-green:  44-66,  108-235, 62-165;

int iLowH[6] = {0,85,37,101,20,9};
int iHighH[6] = {6,101,80,179,23,14};

int iLowS[6] = {94,140,80,42,140,224}; 
int iHighS[6] = {255,255,216,138,255,255};

int iLowV[6] = {0,47,30,0,119,137};
int iHighV[6] = {203,200,172,179,255,253};

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
public:
  ImageConverter()
    : it_(nh_)
  {
    image_pub_ = it_.advertise("out", 1);
    image_sub_ = it_.subscribe("/camera/rgb/image_color", 1, &ImageConverter::imageCb, this);

//    cv::namedWindow(WINDOW);
  }

  ~ImageConverter()
  {
  //  cv::destroyWindow(WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    if(arm_status) //shape detection does not work
    {
      cout << "Start checking..." << endl;
      cv_bridge::CvImagePtr cv_ptr;
      try
	{
	  cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
	}
      catch (cv_bridge::Exception& e)
	{
	  ROS_ERROR("cv_bridge exception: %s", e.what());
	  return;
	}
    

      cv::Mat &imageBGR = cv_ptr->image;
    
      Mat imgHSV;
      cvtColor(imageBGR, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
      for(int k=0;k<6;k++)
	{
	  Mat imgThresholded,imgThresholded_compare1,imgThresholded_compare2;
	  inRange(imgHSV, Scalar(iLowH[k], iLowS[k], iLowV[k]), Scalar(iHighH[k], iHighS[k], iHighV[k]), imgThresholded); //Threshold the image
	  std::vector<std::vector<cv::Point> > contours,contours_app;
	  std::vector<cv::Vec4i> hierarchy,hierarchy_compare1,hierarchy_compare2;
      
	  //finding all contours in the image

	  cv::findContours(imgThresholded, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	  contours_app = contours;
	  // calculate center of contour 
	  vector<Moments> mu(contours_app.size());
	  for(int i=0; i<contours_app.size();i++)
	    { mu[i]= moments(contours_app[i],false);}
	  vector<Point2f> mc(contours_app.size());
	  for(int i=0; i<contours_app.size();i++)
	    { mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);}

       
	  for (int i = 0; i < contours.size(); i++) {      
	    const std::vector<cv::Point> &c = contours_app[i];
	    double area = cv::contourArea(c);
            if (area > 15000 && mc[i].y <  280)
             {
                std::cout << "You picked up a object! Area:" << area << " Y_cor: " << mc[i].y << std::endl;
	        
                 have_object = true;
             }
            else if(area > 8000 && mc[i].y > 280)
              {
                std::cout << "You did not pick up a object! Area:" << area << " Y_cor: " << mc[i].y << std::endl;
              }
            else if(area > 8000 && area < 15000 && mc[i].y < 280)
              {
                std::cout << "You did not pick up a object! Area:" << area << " Y_cor: " << mc[i].y << std::endl;
              }
          }
        }
        check_status = true;
        // cv::waitKey(2);   
    }
    arm_status = false;
  }
};
*/
bool check(std_srvs::SetBool::Request  &req,
           std_srvs::SetBool::Response &res)
{
    // start detection
    arm_status = true;
    while(!check_status) { }
    res.success = have_object;
    res.message = "Checking Finished";
    check_status = false;
    return true;
}

void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
    if(arm_status)
    {
        pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_rgb(new pcl::PointCloud <pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud, *cloud_rgb);
        std::vector<int> indices_rgb;
        pcl::removeNaNFromPointCloud(*cloud_rgb, *cloud_rgb, indices_rgb);
        pcl::PointCloud <pcl::PointXYZ>::Ptr scene(new pcl::PointCloud <pcl::PointXYZ>());
        
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_rgb);
        pass.setFilterFieldName ("z");
        pass.setFilterLimits(0.0,0.12);
        pass.filter(*scene);
        
        cout << "Point cloud size: " << scene->points.size() << endl;
        if(scene->points.size() > 5000)
        {
           // cout << "Point cloud size: " << scene->points.size() << endl;
            have_object = true;
            check_status = true;
        } else {
            have_object = false;
            check_status = true;
        }
        arm_status = false;
    }
    // arm_status = false;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "color_dect");
  // ImageConverter ic;
  ros::NodeHandle nh;
  ros::Subscriber pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points",1,pc_callback);
  ros::ServiceServer service = nh.advertiseService("check_object_status", check);
  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();
  // ros::spin();
  return 0;
}


