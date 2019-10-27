#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace enc = sensor_msgs::image_encodings;

static const char WINDOW[] = "Image window";
static cv::Scalar mu(92.41914641,  67.51901308,  40.76365032);
static cv::Mat sigma2 = (cv::Mat_<float>(3, 3) << 
          361.82843715,   397.22744036,   459.17613117,
          397.22744036,   480.29406139,   604.84159703,
          459.17613117,   604.84159703,  1019.99118737 );
static cv::Mat sigma2_inv;
static cv::Mat x;
static float theta = 0.0;
static cv::Scalar r_min(0.5);


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
    cv::Mat samples(imageBGR.rows * imageBGR.cols, 5, CV_32F);
    std::vector<cv::Mat> bgr;
    cv::split(imageBGR,bgr);
    for(int i = 0; i < imageBGR.rows * imageBGR.cols; i++)
    {     
      samples.at<float>(i,0) = i/imageBGR.cols/imageBGR.cols;
      samples.at<float>(i,1) = i%imageBGR.cols/imageBGR.cols;
      samples.at<float>(i,2) = bgr[0].data[i] / 255.0;
      samples.at<float>(i,3) = bgr[1].data[i] / 255.0;
      samples.at<float>(i,4) = bgr[2].data[i] / 255.0;

    }
    int clusterCount = 20;
    cv::Mat labels;
    int attempts = 2;
    cv::Mat centers;
    cv::kmeans(samples, clusterCount, labels, cv::TermCriteria(CV_TERMCRIT_ITER,10,1), attempts, cv::KMEANS_PP_CENTERS, centers);
    //replace color of its centers' color
    for(int i = 0; i < imageBGR.rows * imageBGR.cols; i++)
    {
      bgr[0].data[i] = centers.at<float>(labels.at<int>(0,i),2) * 255;
      bgr[1].data[i] = centers.at<float>(labels.at<int>(0,i),3) * 255;
      bgr[2].data[i] = centers.at<float>(labels.at<int>(0,i),4) * 255;
    }
    cv::Mat reshaped(imageBGR.rows, imageBGR.cols, CV_8U);
    cv::merge(bgr,reshaped);
    
    cv::Mat canny_out;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Canny(reshaped, canny_out, 100, 100*3, 3);
    cv::findContours(canny_out,contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    for(int i = 0; i < contours.size(); i++)
    {
      const std::vector<cv::Point> &c = contours[i];
      double area = cv::contourArea(c);
      if( area > 500 && area > 1500)
      {
	cv::drawContours(reshaped, contours, i, cv::Scalar(0, 255, 0), 3, 8, hierarchy);
      }
    }
    
    imshow(WINDOW,reshaped);

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


