#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <pcl/visualization/cloud_viewer.h>

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;

static cv::Scalar mu(92.41914641,  67.51901308,  40.76365032);
static cv::Mat sigma2 = (cv::Mat_<float>(3, 3) << 
          361.82843715,   397.22744036,   459.17613117,
          397.22744036,   480.29406139,   604.84159703,
          459.17613117,   604.84159703,  1019.99118737 );
static cv::Mat sigma2_inv;
static cv::Scalar r_min(0.5);

//static pcl::visualization::CloudViewer clv("Simple Cloud Viewer");

static ros::Publisher marker_pub;

static void callback(const PointCloud::ConstPtr& cloud)
{
    //cv::Mat_<cv::Vec3b> im(cloud->height, cloud->width, cv::Vec3b(0, 0, 0));
    cv::Mat pts(cloud->size(), 1, CV_32FC3);
    for (int i = 0; i < cloud->size(); i++) {
        const PointType &pt = cloud->points[i];
        pts.at<cv::Vec3f>(i, 0) = cv::Vec3f(pt.r, pt.g, pt.b);
        //std::cerr << "pt.r " << pt.getRGBVector3i() << "\n";
        //im((int)pt.x*100, (int)pt.y*100) = cv::Vec3b((int)100*pt.z, (int)100*pt.z, (int)100*pt.z);
    }
    cv::cvtColor(pts, pts, cv::COLOR_RGB2XYZ);
    pts.convertTo(pts, CV_32FC3);
    pts -= mu;
    cv::transform(pts, pts, sigma2_inv);
    cv::pow(pts, 2, pts);
    cv::Mat r;
    cv::transform(pts, r, cv::Matx13f(1, 1, 1));
    cv::pow(r, 0.5, r);

    cv::Mat inrange;
    cv::inRange(r, -r_min, r_min, inrange);

    float sumx = 0, sumy = 0, sumz = 0;
    int num = 0;

    //PointCloud::Ptr inliers(new PointCloud(cloud->width, cloud->height));
    //inliers->is_dense = false;

    for (int i = 0; i < inrange.rows; i++) {
        if (inrange.at<uint8_t>(i, 0)) {
            const PointType &pt = cloud->points[i];
            if (isnan(pt.x)) {
                continue;
            }
            //inliers->push_back(pt);
            sumx += pt.x;
            sumy += pt.y;
            sumz += pt.z;
            num += 1;
        }
    }

    visualization_msgs::Marker marker;
    marker.header.frame_id = "camera";
    marker.header.stamp = ros::Time::now();
    marker.ns = "camera";
    marker.id = 1;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = sumx/num;
    marker.pose.position.y = sumy/num;
    marker.pose.position.z = sumz/num;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.05;
    marker.scale.y = 0.05;
    marker.scale.z = 0.05;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration(0.2);

    std::cerr << num << " points, mid " << sumx/num << ", " << sumy/num << ", " << sumz/num << "\n";
    marker_pub.publish(marker);

    //clv.showCloud(inliers);
    //iv.addRGBImage<PointType>(inliers);

    //cv::imshow("Output", im);
    //cv::waitKey(1);
}

int main(int argc, char** argv)
{
    cv::invert(sigma2, sigma2_inv);
    ros::init(argc, argv, "mreyes_node");
    ros::NodeHandle nh;
    marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    ros::Subscriber sub = nh.subscribe<PointCloud>("/camera/depth_registered/points", 5, callback);
    //cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    ros::spin();
/*
    while (1) {
        ros::spinOnce();
    }
*/
}
