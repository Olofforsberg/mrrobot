#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static cv::Scalar mu(92.41914641,  67.51901308,  40.76365032);
static cv::Mat sigma2 = (cv::Mat_<float>(3, 3) << 
          361.82843715,   397.22744036,   459.17613117,
          397.22744036,   480.29406139,   604.84159703,
          459.17613117,   604.84159703,  1019.99118737 );
static cv::Mat sigma2_inv;
static cv::Mat x;
static float theta = 0.0;
static cv::Scalar r_min(0.5);

static void rgb_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat &im = cv_ptr->image;
    cv::Mat out;
    cv::cvtColor(im, out, cv::COLOR_BGR2XYZ);
    out.convertTo(out, CV_32FC3);
    out -= mu;
    cv::transform(out, out, sigma2_inv);
    cv::pow(out, 2, out);
    cv::Mat r;
    cv::transform(out, r, cv::Matx13f(1, 1, 1));
    cv::pow(r, 0.5, r);

    if (x.rows < r.rows || x.cols < r.cols) {
        x = r;
    } else {
        x *= theta;
        x += r;
    }

    cv::Mat inrange;
    cv::inRange(x, -r_min, r_min, inrange);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(inrange, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        const std::vector<cv::Point> &c = contours[i];
        double area = cv::contourArea(c);
        if (area > 1000) {
            cv::drawContours(im, contours, i, cv::Scalar(0, 255, 0), 3, 8, hierarchy);
        }
    }

    cv::imshow("Output", im);
    cv::waitKey(3);
}

static void depth_callback(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    cv::imshow("Output", cv_ptr->image);
    cv::waitKey(3);
}

int main(int argc, char** argv)
{
    cv::invert(sigma2, sigma2_inv);
    ros::init(argc, argv, "mreyes_2d_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    //image_transport::Subscriber depth_sub = it.subscribe("/camera/depth/image", 1, depth_callback);
    image_transport::Subscriber rgb_sub = it.subscribe("/camera/rgb/image_color", 1, rgb_callback);
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    ros::spin();
}
