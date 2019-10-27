#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <highgui.h>
using namespace cv;
using namespace std;
namespace enc = sensor_msgs::image_encodings;

static const char WINDOW[] = "Image window";

//0-red:    0-7,    0-255,  0-255;
//1-blue:   91-112, 77-255, 82-255;
//2-green:  28-90,  50-220, 40,255;

int iLowH = 0;
int iHighH = 7;

int iLowS = 100; 
int iHighS = 255;

int iLowV = 0;
int iHighV = 255;

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

    cv::namedWindow(WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
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
 
    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"


    // Create trackbars in "Control" window
    
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);
    

    
    Mat imgHSV,imgHSV_compare1;
    cvtColor(imageBGR, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    Mat imgThresholded,imgThresholded_compare1;
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
       
      
    std::vector<std::vector<cv::Point> > contours,contours_app;
    std::vector<cv::Vec4i> hierarchy;
      
    //finding all contours in the image


    cv::findContours(imgThresholded, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // calculate center of contour 
    vector<Moments> mu(contours.size());
    for(int i=0; i<contours.size();i++)
    { mu[i]= moments(contours[i],false);}
    vector<Point2f> mc(contours.size());
    for(int i=0; i<contours.size();i++)
    { mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);}
  
       
    contours_app.resize(contours.size());
    for(size_t s= 0; s < contours.size();s++)
      approxPolyDP(Mat(contours[s]), contours_app[s], 3, true);
    
    for (int i = 0; i < contours.size(); i++) {
      const std::vector<cv::Point> &c = contours_app[i];
      double area = cv::contourArea(c);
      if (area > 3000) {
	cv::drawContours(imageBGR, contours_app, i, cv::Scalar(0, 0, 255), 3, 8, hierarchy);  	 
        cout << "Position: " << mc[i] << endl;
	cout << "Distance: " << endl;
	cout << "x: " << -12.3 + 0.0364 * mc[i].x - 0.0037 * mc[i].y << endl;
	cout << "y: " << -0.0629 * mc[i].y + 41.2325 << endl;
      }	
    }
 
  imshow(WINDOW,imageBGR);
  cv::waitKey(2);
    
  image_pub_.publish(cv_ptr->toImageMsg());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "k_means_seg");
  ImageConverter ic;
  ros::spin();
  return 0;
}


