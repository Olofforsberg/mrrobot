#include <iostream>
#include <cstdlib>
#include <libgen.h>
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
#include <std_msgs/Float32.h>
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
#include <visualization_msgs/Marker.h>
#include <ras_msgs/RAS_Evidence.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <unistd.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <Eigen/Dense>
#include <mreyes/Vision_drive.h>

using namespace cv;
using namespace std;
using namespace Eigen;
namespace enc = sensor_msgs::image_encodings;

tf2_ros::Buffer *tfb;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

static ros::Publisher order;
static ros::Publisher color;
std_msgs::Bool shape_status;
bool color_detection_status = false;
std_msgs::String obj_color;

string object_ID;
bool shape_detection_status = true;
int color_nr = 0;
bool arm_status = false;
int color_area_index = -1;
bool service_call = true;

static const char WINDOW[] = "Image window";
//0-red:    0-8,    0-255,   0-255;
//1-blue:   72-112, 77-255,  82-255;
//2-green:  44-66,  108-235, 62-165;
//3-purple: 101-179 0-255,   0-255;
//4-yellow: 20-22,  169-255, 153-255;
//5-orange: 9-14,   224-255, 137-253

int iLowH[6] = {0,85,37,101,16,9};
int iHighH[6] = {6,101,80,179,23,14};

int iLowS[6] = {94,140,120,42,170,174}; 
int iHighS[6] = {255,255,235,138,255,255};

int iLowV[6] = {0,47,30,0,150,137};
int iHighV[6] = {203,200,172,179,255,253};

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (true);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.004f); // 0.004
float scene_ss_ (0.004f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

bool found_obj = false;
bool search_order = false;
float marker_x, marker_y, marker_z;
sensor_msgs::Image image_rec;
string shape[8];
int mode_nr = 1; //1:exploring 2:pick up object
const int camera_x_center = 350, camera_y_center = 310;
const string color_prefixes[] = { "Red ", "Blue ", "Green ", "Purple ", "Yellow ", "Orange " };
Matrix<double, 2, 4> camera_floor_projection;
geometry_msgs::Twist motor_msg;

// initialize oject position matrix
struct det {
public:
    float x, y, z;
    int color;
    string desc() {
        stringstream s;
        s << x << ", " << y << " (";
        if (0 <= color && color < 6) {
            s << color_prefixes[color];
        } else {
            s << color;
        }
        s << ")";
        return s.str();
    }
};
det objects[128];
int num_objects = 50;
int position_index = 0;
// Estimate of currently-being-detected object's pose from camera
geometry_msgs::PointStamped obj_pos;

// initialize publisher
static ros::Publisher marker_pub;
static ros::Publisher evidence_pub;
static ros::Publisher sound_pub;
static ros::Publisher motor_pub;
static ros::Publisher move_to_pub;
static ros::Publisher target_point_pub;

// model pointcloud initialize
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model_rgb (new pcl::PointCloud<pcl::PointXYZRGBA> ());
pcl::PointCloud<PointType>::Ptr model_hollow (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_triangle (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_cube (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_cross (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_ball (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_cylinder (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_star (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_cube_2 (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_hollow_2 (new pcl::PointCloud<PointType> ());

pcl::PointCloud<NormalType>::Ptr model_normals_hollow (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_triangle (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_cube (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_cross (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_ball (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_cylinder (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_star (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_cube_2 (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr model_normals_hollow_2 (new pcl::PointCloud<NormalType> ());

pcl::PointCloud<PointType>::Ptr model_keypoints_hollow (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_triangle (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_cube (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_cross (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_ball (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_cylinder (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_star (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_cube_2 (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints_hollow_2 (new pcl::PointCloud<PointType> ());

pcl::PointCloud<DescriptorType>::Ptr model_descriptors_hollow (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_triangle (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_cube (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_cross (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_ball (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_cylinder (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_star (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_cube_2 (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors_hollow_2 (new pcl::PointCloud<DescriptorType> ());


bool vision_investigate_service(mreyes::Vision_drive::Request  &req,
                                mreyes::Vision_drive::Response &res)
{
    // start investigation
    ROS_INFO("service invoked, driving to colored thing (color=%d)", color_area_index);

    if (arm_status) {
        ROS_WARN("arm_status is set; aborting service call");
        return false;
    }

    service_call = false;

    ros::Rate rate(10);
    while (!color_detection_status && color_area_index >= 0) {
        motor_pub.publish(motor_msg);
        rate.sleep();
    }

    if (color_area_index < 0) {
        ROS_WARN("color area index < 0");
        return false;
    }

    ROS_INFO("commence shape detection");
    service_call = true;
    // start shape detection
    ros::Rate rate1(10);
    while(service_call){
        rate1.sleep();
    }
    service_call = false;
    ROS_INFO("constructing service response");
    res.object_x = objects[position_index - 1].x;
    res.object_y = objects[position_index - 1].y; 
    res.object_z = objects[position_index - 1].z; 
    res.the_object_id = object_ID;
    if(object_ID.compare("An object") == 0) {
        return false;
    }
    return true;
}


void arm_callback(const std_msgs::Bool::ConstPtr& arm_msg)
{
    arm_status = arm_msg->data;
}


#include <geometry_msgs/PoseArray.h>

static geometry_msgs::PointStamped point_from_pixel(ros::Time t, double mx, double my) {
    // transform the contour center position to base position
    geometry_msgs::PointStamped p_ori;
    p_ori.header.frame_id = "base";
    p_ori.header.stamp = t;
    Vector4d v;
    Vector2d w;
    v << mx, my, mx*my, 1;
    w = camera_floor_projection*v;
    // pose_ori.transform.translation.x = 0.23;
    //p.position.x = (-0.0629 * my + 41.2325) / 100;
    //p.position.y = -(-12.3 + 0.0364 * mx - 0.0037 * my) / 100;
    p_ori.point.x = 10e-2 + w(1);
    p_ori.point.y = 2e-2 + -w(0);
    p_ori.point.z = 0.0;
    // transform base pose to map pose
    return tfb->transform(p_ori, "map", ros::Duration(100e-3));
}

void rgb_callback(const sensor_msgs::ImageConstPtr& msg)
{
    if (color_detection_status || arm_status) {
        return;
    }

    color_nr = 0;
    image_rec.header = msg->header;
    image_rec.height = msg->height;
    image_rec.width = msg->width;
    image_rec.encoding = msg->encoding;
    image_rec.is_bigendian = msg->is_bigendian;
    image_rec.step = msg->step;
    image_rec.data = msg->data;
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat &imageBGR = cv_ptr->image;
    Mat imgHSV;
    cvtColor(imageBGR, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    // color detection
    float color_area_max = 0.0;
    //if (color_area_index < 0) {
    if (color_area_index < 0) {

        for(int k=0;k<6;k++) {
            Mat imgThresholded,imgThresholded_compare1,imgThresholded_compare2;
            inRange(imgHSV, Scalar(iLowH[k], iLowS[k], iLowV[k]), Scalar(iHighH[k], iHighS[k], iHighV[k]), imgThresholded); //Threshold the image
            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;
        
            cv::findContours(imgThresholded, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
            // calculate center of contour 
            vector<Moments> mu(contours.size());
            vector<Point2f> mc(contours.size());
            for(int i=0; i<contours.size();i++) { mu[i]= moments(contours[i],false);}
            for(int i=0; i<contours.size();i++) { mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);}

            for (int i = 0; i < contours.size(); i++) {
                const std::vector<cv::Point> &c = contours[i];
                double area = cv::contourArea(c);

                if (area < 3000) 
                    continue;

                try {
                    obj_pos = point_from_pixel(msg->header.stamp, mc[i].x, mc[i].y);
                } catch (tf2::ExtrapolationException &e) {
                    ROS_ERROR("tf transform failed: %s", e.what());
                    return;
                }

                // check this POI has been detected or not
                int min_j = -1;
                float min_distance = 20;
                for(int j = 0; j < num_objects && objects[j].x != 0; j++)
                {
                    float distance = sqrt(pow(obj_pos.point.x - objects[j].x,2)
                                        + pow(obj_pos.point.y - objects[j].y,2));
                    //cout << "Distance to object #" << j << " " << objects[j].desc() << ": " << distance << endl;
                    if(min_distance > distance) {
                        min_j = j;
                        min_distance = distance; 
                    }
                }

                if (min_j >= 0) {
                    //cout << "min_distance: " << min_distance
                    //     << " to object " << min_j
                    //     << " " << objects[min_j].desc() << endl;
                    if ((min_distance > 0.1) || (objects[min_j].color != k))
                    {
                        if(area > color_area_max)
                        {
                            color_area_max = area;
                            color_area_index = k;
                        }       
                    }
                } else {
                    if(area > color_area_max) {
                        color_area_max = area;
                        color_area_index = k;
                    }
                }
            }
        }
    
    }

    if (color_area_index >= 0) {
        //cout << "index: " << color_area_index << endl;
        Mat imgThresholded,imgThresholded_compare1,imgThresholded_compare2;
        inRange(imgHSV, Scalar(iLowH[color_area_index], iLowS[color_area_index], iLowV[color_area_index]), Scalar(iHighH[color_area_index], iHighS[color_area_index], iHighV[color_area_index]), imgThresholded); //Threshold the image
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
    
        cv::findContours(imgThresholded, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // calculate center of contour 
        vector<Moments> mu(contours.size());
        vector<Point2f> mc(contours.size());
        for(int i=0; i<contours.size();i++) { mu[i]= moments(contours[i],false);}
        for(int i=0; i<contours.size();i++) { mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);}

        double max_area = 0;
        int max_i = -1;
        for (int i = 0; i < contours.size(); i++) {      
            const std::vector<cv::Point> &c = contours[i];
            double area = cv::contourArea(c);
            if(area > max_area) {
                max_area = area;
                max_i = i;
            }
        }

        // sometimes no contour is found; only publish new estimate if found
        if (max_i >= 0) {
            try {
                obj_pos = point_from_pixel(msg->header.stamp, mc[max_i].x, mc[max_i].y);
            } catch (tf2::ExtrapolationException &e) {
                ROS_ERROR("tf transform failed: %s", e.what());
                return;
            }
            target_point_pub.publish(obj_pos);
        } else {
            color_area_index = -1;
            motor_msg.linear.x = 0.0;
            motor_msg.angular.z = 0.0;
            ROS_WARN("target disappeared");
            return;
        }

        assert(0 <= color_area_index && color_area_index < 6);
        if(!service_call) {
            if (max_area > 10000 && mc[max_i].y > camera_y_center && abs(mc[max_i].x - camera_x_center) < 50) {
                // robot should stop here
                color_detection_status = true;
                shape_detection_status = false;
                color_nr = color_area_index+1;
                object_ID = color_prefixes[color_area_index];
                motor_msg.linear.x = 0.0;
                motor_msg.angular.z = 0.0;
                cout << "Object color: " << object_ID << endl;
                cout << "Object position: " << mc[max_i] << endl << endl;
                cout << "wait for shape detection..." << endl;
                usleep(1e6);
            } else if (max_area > 3000) {
                // drive towards thing
                //cout << "Driving towards color " << color_prefixes[color_area_index] << endl
                //     << "Area: " << max_area << endl
                //     << "Object position: " << mc[max_i].x << ", " << mc[max_i].y << endl;
                move_to_pub.publish(obj_pos);
                if(abs(mc[max_i].x - camera_x_center) > 30 && max_area < 23000) {
                    motor_msg.linear.x = 0.02;
                    motor_msg.angular.z = -0.1 * (mc[max_i].x - camera_x_center) / abs(mc[max_i].x - camera_x_center);
                } else if(abs(mc[max_i].x - camera_x_center) < 30) {
                    motor_msg.linear.x = 0.06;
                    motor_msg.angular.z = - 0.005 * (mc[max_i].x - camera_x_center);
                } 

                else if(abs(mc[max_i].x - camera_x_center) > 30 && max_area > 23000) {
                    motor_msg.linear.x = -0.03;
                    motor_msg.angular.z = 0.0; //-0.1 * (mc[max_i].x - camera_x_center) / abs(mc[max_i].x - camera_x_center);
                }
                //motor_pub.publish(motor_msg);
            }
        }
        if (max_area < 3000)  {
            // contour is too small; stop driving to it
            color_area_index = -1;
            motor_msg.linear.x = 0.0;
            motor_msg.angular.z = 0.0;
        }
    }
}

void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
  if(!shape_detection_status && service_call)
  {
  //std::cout << "Start shape detection..." << std::endl;
  // save point cloud to .pcd
  pcl::PointCloud <pcl::PointXYZ>::Ptr cloud_rgb(new pcl::PointCloud <pcl::PointXYZ>());
  pcl::fromROSMsg(*cloud, *cloud_rgb);
  std::vector<int> indices_rgb;
  pcl::removeNaNFromPointCloud(*cloud_rgb, *cloud_rgb, indices_rgb);
  //pcl::io::savePCDFileASCII("test_rgb.pcd", *cloud_rgb);


  // start matching
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  // pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model_rgb (new pcl::PointCloud<pcl::PointXYZRGBA> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_ori (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  
  // rotate point cloud
  Matrix4f transform_1 = Matrix4f::Identity();
  float theta = 2.1; // The angle of rotation in radians 
  transform_1 (2,2) = cos (theta);
  transform_1 (2,1) = -sin(theta);
  transform_1 (1,2) = sin (theta);
  transform_1 (1,1) = cos (theta);
  transform_1 (2,3) = 0.125;
  transform_1 (1,3) = 0;
  transform_1 (0,3) = 0.015;
 
  int corr_value[] = {0, 0, 0, 0, 0};
  float pos_x[] = {0, 0, 0};
  float pos_y[] = {0, 0, 0};
  float pos_z[] = {0, 0, 0};

  int model_nr = 0;
  if(color_nr == 1) //red
    {
      model_nr = 5;
      shape[0] = "hollow cube";
      shape[1] = "ball";
      shape[2] = "cube";
      shape[3] = "hollow cube";
      shape[4] = "cube";
    }
  if(color_nr == 2) //blue
    {
      model_nr = 3;
      shape[0] = "cube";
      shape[1] = "triangle";
      shape[2] = "cube";
    }
  if(color_nr == 3) //green
    {
      model_nr = 3;
      shape[0] = "cube";
      shape[1] = "cylinder";
      shape[2] = "cube";
    }
  if(color_nr == 4) //purple
    {
      model_nr = 2;
      shape[0] = "cross";
      shape[1] = "star";
    }
  if(color_nr == 5) //yellow
    {
      model_nr = 3;
      shape[0] = "cube";
      shape[1] = "ball";
      shape[2] = "cube";
    }
  if(color_nr == 6) //orange
    {
      model_nr = 1;
      shape[0] = "Patric";
    }
  // searching shape
  for(int shape_nr = 0; shape_nr < model_nr; shape_nr++)
  {
    if(color_nr == 1)
      {
        if(shape_nr == 0)
          {
            model = model_hollow;
            model_descriptors = model_descriptors_hollow;
            model_normals = model_normals_hollow;
            model_keypoints = model_keypoints_hollow;
          }
        if(shape_nr == 1)
          {
            model = model_ball;
            model_descriptors = model_descriptors_ball;
            model_normals = model_normals_ball;
            model_keypoints = model_keypoints_ball;
          }
        if(shape_nr == 2)
          {
            model = model_cube;
            model_descriptors = model_descriptors_cube;      
            model_normals = model_normals_cube;
            model_keypoints = model_keypoints_cube;
          }
        if(shape_nr == 3)
          {
            model = model_hollow_2;
            model_descriptors = model_descriptors_hollow_2;      
            model_normals = model_normals_hollow_2;
            model_keypoints = model_keypoints_hollow_2;
          }
        if(shape_nr == 4)
          {
            model = model_cube_2;
            model_descriptors = model_descriptors_cube_2;      
            model_normals = model_normals_cube_2;
            model_keypoints = model_keypoints_cube_2;
          }
      }
    if(color_nr == 2)
      {
        if(shape_nr == 0)
          {
            model = model_cube;
            model_descriptors = model_descriptors_cube;
            model_normals = model_normals_cube;
            model_keypoints = model_keypoints_cube;
          }
        if(shape_nr == 1)
          {
            model = model_triangle;
            model_descriptors = model_descriptors_triangle;
            model_normals = model_normals_triangle;
            model_keypoints = model_keypoints_triangle;
          }     
        if(shape_nr == 2)
          {
            model = model_cube_2;
            model_descriptors = model_descriptors_cube_2;      
            model_normals = model_normals_cube_2;
            model_keypoints = model_keypoints_cube_2;
          }
      }
    if(color_nr == 3)
      {
        if(shape_nr == 0)
          {
            model = model_cube;
            model_descriptors = model_descriptors_cube;
            model_normals = model_normals_cube;
            model_keypoints = model_keypoints_cube;
          }
        if(shape_nr == 1)
          {
            model = model_cylinder;
            model_descriptors = model_descriptors_cylinder;
            model_normals = model_normals_cylinder;
            model_keypoints = model_keypoints_cylinder;
          }     
        if(shape_nr == 2)
          {
            model = model_cube_2;
            model_descriptors = model_descriptors_cube_2;      
            model_normals = model_normals_cube_2;
            model_keypoints = model_keypoints_cube_2;
          }
      }
    if(color_nr == 4)
      {
        if(shape_nr == 0)
          {
            model = model_cross;
            model_descriptors = model_descriptors_cross;
            model_normals = model_normals_cross;
            model_keypoints = model_keypoints_cross;
          }
        if(shape_nr == 1)
          {
            model = model_star;
            model_descriptors = model_descriptors_star;
            model_normals = model_normals_star;
            model_keypoints = model_keypoints_star;
          }     
      }
    if(color_nr == 5)
      {
        if(shape_nr == 0)
          {
            model = model_cube;
            model_descriptors = model_descriptors_cube;
            model_normals = model_normals_cube;
            model_keypoints = model_keypoints_cube;
          }
        if(shape_nr == 1)
          {
            model = model_ball;
            model_descriptors = model_descriptors_ball;
            model_normals = model_normals_ball;
            model_keypoints = model_keypoints_ball;
          }     
        if(shape_nr == 2)
          {
            model = model_cube_2;
            model_descriptors = model_descriptors_cube_2;      
            model_normals = model_normals_cube_2;
            model_keypoints = model_keypoints_cube_2;
          }
      }
    if(color_nr == 6)
      {
        if(shape_nr == 0)
          {
            model = model_star;
            model_descriptors = model_descriptors_star;
            model_normals = model_normals_star;
            model_keypoints = model_keypoints_star;
          }     
      }

    //pcl::io::loadPCDFile("test_rgb.pcd", *scene_ori);
    pcl::transformPointCloud (*cloud_rgb, *scene_ori, transform_1);
    //pcl::transformPointCloud (*scene_ori, *scene_ori, transform_1);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(scene_ori);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits(0.01,0.05);
    pass.filter(*scene_ori);
    
    pass.setInputCloud(scene_ori);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits(-0.1,0.1);
    pass.filter(*scene_ori);
    
    pass.setInputCloud(scene_ori);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits(0.05,0.2);
    pass.filter(*scene);

    //  Compute Normals
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (scene);
    norm_est.compute (*scene_normals);
    // norm_est.setKSearch (10);
    // norm_est.setInputCloud (model);
    // norm_est.compute (*model_normals);

    //  Downsample Clouds to Extract keypoints
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (scene_ss_);
    pcl::PointCloud<int> keypointIndices2;
    uniform_sampling.compute(keypointIndices2);
    pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints); 
    /*
    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (model_ss_);
    pcl::PointCloud<int> keypointIndices1;
    uniform_sampling.compute(keypointIndices1);
    pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints); 
    */
    //  Compute Descriptor for keypoints
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad_);
    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);
    /*
    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);
    */
    //  Find Model-Scene Correspondences with KdTree
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud (model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
      {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);
        if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
          {
            continue;
          }
        int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
          {
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back (corr);
          }
      }
    // std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

    //  Actual Clustering
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;


    //  Compute (Keypoints) Reference Frames only for Hough
    pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
    pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

    pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
    rf_est.setFindHoles (true);
    rf_est.setRadiusSearch (rf_rad_);

    rf_est.setInputCloud (model_keypoints);
    rf_est.setInputNormals (model_normals);
    rf_est.setSearchSurface (model);
    rf_est.compute (*model_rf);

    rf_est.setInputCloud (scene_keypoints);
    rf_est.setInputNormals (scene_normals);
    rf_est.setSearchSurface (scene);
    rf_est.compute (*scene_rf);

    //  Clustering
    pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);

    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);

    clusterer.recognize (rototranslations, clustered_corrs);


    //  Output results
    int obj_nr = 0;
    for (size_t i = 0; i < rototranslations.size (); ++i)
      {
        if(clustered_corrs[i].size () > corr_value[shape_nr])
        {
          corr_value[shape_nr] = clustered_corrs[i].size ();
          std::cout << corr_value[shape_nr] << std::endl;
          float curr_x, curr_y, curr_z;
          curr_x = 0;
          curr_y = 0;
          curr_z = 0;
          for(size_t j = 0; j < clustered_corrs[i].size (); j++)
          {
            curr_x += scene_keypoints->at(clustered_corrs[i][j].index_match).x;
            curr_y += scene_keypoints->at(clustered_corrs[i][j].index_match).y;
            curr_z += scene_keypoints->at(clustered_corrs[i][j].index_match).z;
          }
          pos_x[shape_nr] = curr_x / clustered_corrs[i].size();
          pos_y[shape_nr] = curr_y / clustered_corrs[i].size();
          pos_z[shape_nr] = curr_z / clustered_corrs[i].size();
        }

        if( clustered_corrs[i].size() > 90) {found_obj = true;}
      }
    if(found_obj) 
    {
      std::stringstream ss;
      std::string s;
      ss << shape[shape_nr];
      ss >> s;
      object_ID = object_ID + s;
      std::cout << "object_id: " << object_ID << std::endl;
      std::cout << "Object position:" << " x." << pos_x[shape_nr] << " y." << pos_y[shape_nr] << " z." << pos_z[shape_nr] << std::endl; 
      Eigen::Vector4f p(pos_x[shape_nr], pos_y[shape_nr], pos_z[shape_nr], 1);
      Eigen::Vector4f pCamera(transform_1.colPivHouseholderQr().solve(p));
      marker_x = pCamera(0);
      marker_y = pCamera(1) ;
      marker_z = pCamera(2);
      std::cout << "Object position camera:" << " x." << marker_x << " y." << marker_y << " z." << marker_z << "\n" << std::endl; 
      // publish sound
      std_msgs::String sound;
      sound.data = "Wow, I see a " + object_ID;
      sound_pub.publish(sound);
      shape_detection_status = true;
      color_detection_status = false;
      break;
    }
  }
  
 
  if(found_obj == false)
    {
      if(corr_value[0] < 15 && corr_value[1] < 15 && corr_value[2] < 15 && corr_value[3] < 15 && corr_value[4] < 15)
        {
          std::cout << "No shape matches!!" << std::endl;
          object_ID = "An object";
          shape_detection_status = true;
          color_detection_status = false;
          found_obj = false;
          objects[position_index++] = { obj_pos.point.x,
                                        obj_pos.point.y,
                                        obj_pos.point.z,
                                        color_nr - 1 };
          color_area_index = -1;
          std_msgs::String sound;
          sound.data = "Wow, I see a " + object_ID;
          sound_pub.publish(sound);
        }
      else
        {
          int index, max_value;
          index = 0;
          max_value = 0;
          for(int i = 0; i < model_nr; i++)
            {
              if(corr_value[i] > max_value)
                {
                  max_value = corr_value[i];
                  index = i; 
                }
            }
          std::stringstream ss;
          std::string s;
          ss << shape[index];
          ss >> s;
          object_ID = object_ID + s;
          std::cout << "object_id: " << object_ID << std::endl;
          std::cout << "Object position:" << " x." << pos_x[index] << " y." << pos_y[index] << " z." << pos_z[index] << "\n" << std::endl; 
          Eigen::Vector4f p(pos_x[index], pos_y[index], pos_z[index], 1);
          Eigen::Vector4f pCamera(transform_1.colPivHouseholderQr().solve(p));
          marker_x = pCamera(0);
          marker_y = pCamera(1) ;
          marker_z = pCamera(2);
          std::cout << "Object position camera:" << " x." << marker_x << " y." << marker_y << " z." << marker_z << "\n" << std::endl; 
          found_obj = true;
          shape_detection_status = true;
          color_detection_status = false;
          color_area_index = -1;
          // publish sound
          std_msgs::String sound;
          sound.data = "Wow, I see a " + object_ID;
          sound_pub.publish(sound);
        }
    }
    service_call = false;
  }


  if(found_obj)
    {
      // transform object position to map frame
      geometry_msgs::TransformStamped object_point_ori;
      object_point_ori.header.frame_id = cloud->header.frame_id;
      object_point_ori.header.stamp = cloud->header.stamp;
      object_point_ori.child_frame_id = "map";
      object_point_ori.transform.translation.x = marker_x;
      object_point_ori.transform.translation.y = marker_y;
      object_point_ori.transform.translation.z = marker_z;
      object_point_ori.transform.rotation.x = 0;
      object_point_ori.transform.rotation.y = 0;
      object_point_ori.transform.rotation.z = 0;
      object_point_ori.transform.rotation.w = 0;
      geometry_msgs::TransformStamped object_point;
      try {
          object_point = tfb->transform(object_point_ori, "map", ros::Duration(0.5));
      } catch (tf2::ExtrapolationException &e) {
          ROS_ERROR("tf transform failed: %s", e.what());
          return;
      }
      // publish evidence
      ras_msgs::RAS_Evidence evidence;
      evidence.stamp = ros::Time::now();
      evidence.group_number = 2;
      evidence.image_evidence = image_rec;
      evidence.object_id = object_ID;
      //geometry_msgs::TransformStamped location;
      //location.header.frame_id = "map";
      //location.transform.translation.x = object_point.point.x;
      //location.transform.translation.y = object_point.point.y;
      //location.transform.translation.z = object_point.point.z;
      evidence.object_location = object_point;
      evidence_pub.publish(evidence);
      std::cout << "publishing evidence..." << std::endl;
        
      // save object position to position matrix
      objects[position_index++] = { object_point.transform.translation.x,
                                    object_point.transform.translation.y,
                                    object_point.transform.translation.z,
                                    color_nr - 1 };
      assert(position_index < num_objects);

      // publish marker
      int marker_nr;
      visualization_msgs::Marker marker;
      marker.header.frame_id = cloud->header.frame_id; // "map";
      marker.header.stamp = ros::Time::now();
      marker.ns = object_ID;
      marker.id = position_index - 1;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;
      /*
      marker.pose.position.x = object_point.transform.translation.x;
      marker.pose.position.y = object_point.transform.translation.y;
      marker.pose.position.z = object_point.transform.translation.z;
      */
      marker.pose.position.x = marker_x;
      marker.pose.position.y = marker_y;
      marker.pose.position.z = marker_z;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = 0.01;
      marker.scale.y = 0.01;
      marker.scale.z = 0.01;

      // Set the color -- be sure to set alpha to something non-zero!
      if(color_nr == 1)
        {
            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0f;
        }
      else if(color_nr == 2)
        {
            marker.color.r = 0.0f;
            marker.color.g = 0.0f;
            marker.color.b = 1.0f;
        }
      else if(color_nr == 3) // green
        {
            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
        }
      else if(color_nr == 4) // purple
        {   
            marker.color.r = 0.5f;
            marker.color.g = 0.0f;
            marker.color.b = 0.5f;
        }
      else if(color_nr == 5) // yellow
        {
            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
        }
      else if(color_nr == 6) // orange
        {
            marker.color.r = 1.0f;
            marker.color.g = 0.6f;
            marker.color.b = 0.0f;
        }
      else
        {
            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 1.0f;
        }
      marker.color.a = 1.0;

      marker_pub.publish(marker);
      // ros::spin();
      found_obj = false;
      arm_status = true;
    }
    if(mode_nr == 1)
    {
        arm_status = false;
    }
}



int main(int argc, char** argv)
{
  cout << "pointcloud initializing..." << endl;
  assert(0 == chdir(dirname(argv[0])));
  assert(0 == chdir("../../../src/mrrobot"));
  assert(!pcl::io::loadPCDFile("test_rgb_hollow_cube.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_hollow);
  assert(!pcl::io::loadPCDFile("test_rgb_triangle.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_triangle);
  assert(!pcl::io::loadPCDFile("test_rgb_cube.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_cube);
  assert(!pcl::io::loadPCDFile("test_rgb_cross.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_cross);
  assert(!pcl::io::loadPCDFile("test_rgb_ball.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_ball);
  assert(!pcl::io::loadPCDFile("test_rgb_cylinder.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_cylinder);
  assert(!pcl::io::loadPCDFile("test_rgb_star.pcd", *model_rgb));
  pcl::copyPointCloud(*model_rgb, *model_star);

  camera_floor_projection << 8.59375000e-04, 3.43750000e-04, -1.07421875e-06, -2.75000000e-01,
                             1.29887837e-17, -9.79166667e-04, 0.00000000e+00,  4.70000000e-01;


  pcl::io::loadPCDFile("test_xyz_cube.pcd", *model_cube_2);
  pcl::io::loadPCDFile("test_xyz_hollow.pcd", *model_hollow_2);

  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_hollow);
  norm_est.compute (*model_normals_hollow);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_triangle);
  norm_est.compute (*model_normals_triangle);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_cube);
  norm_est.compute (*model_normals_cube);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_cross);
  norm_est.compute (*model_normals_cross);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_ball);
  norm_est.compute (*model_normals_ball);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_cylinder);
  norm_est.compute (*model_normals_cylinder);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_star);
  norm_est.compute (*model_normals_star);
  
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_cube_2);
  norm_est.compute (*model_normals_cube_2);
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model_hollow_2);
  norm_est.compute (*model_normals_hollow_2);

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model_hollow);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_hollow;
  uniform_sampling.compute(keypointIndices_hollow);
  pcl::copyPointCloud(*model_hollow, keypointIndices_hollow.points, *model_keypoints_hollow); 
  uniform_sampling.setInputCloud (model_triangle);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_triangle;
  uniform_sampling.compute(keypointIndices_triangle);
  pcl::copyPointCloud(*model_triangle, keypointIndices_triangle.points, *model_keypoints_triangle); 
  uniform_sampling.setInputCloud (model_cube);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_cube;
  uniform_sampling.compute(keypointIndices_cube);
  pcl::copyPointCloud(*model_cube, keypointIndices_cube.points, *model_keypoints_cube); 
  uniform_sampling.setInputCloud (model_cross);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_cross;
  uniform_sampling.compute(keypointIndices_cross);
  pcl::copyPointCloud(*model_cross, keypointIndices_cross.points, *model_keypoints_cross); 
  uniform_sampling.setInputCloud (model_ball);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_ball;
  uniform_sampling.compute(keypointIndices_ball);
  pcl::copyPointCloud(*model_ball, keypointIndices_ball.points, *model_keypoints_ball);
  uniform_sampling.setInputCloud (model_cylinder);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_cylinder;
  uniform_sampling.compute(keypointIndices_cylinder);
  pcl::copyPointCloud(*model_cylinder, keypointIndices_cylinder.points, *model_keypoints_cylinder);
  uniform_sampling.setInputCloud (model_star);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_star;
  uniform_sampling.compute(keypointIndices_star);
  pcl::copyPointCloud(*model_star, keypointIndices_star.points, *model_keypoints_star);
  
  uniform_sampling.setInputCloud (model_cube_2);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_cube_2;
  uniform_sampling.compute(keypointIndices_cube_2);
  pcl::copyPointCloud(*model_cube_2, keypointIndices_cube_2.points, *model_keypoints_cube_2);
  uniform_sampling.setInputCloud (model_hollow_2);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> keypointIndices_hollow_2;
  uniform_sampling.compute(keypointIndices_hollow_2);
  pcl::copyPointCloud(*model_hollow_2, keypointIndices_hollow_2.points, *model_keypoints_hollow_2);

  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);
  descr_est.setInputCloud (model_keypoints_hollow);
  descr_est.setInputNormals (model_normals_hollow);
  descr_est.setSearchSurface (model_hollow);
  descr_est.compute (*model_descriptors_hollow);
 
  descr_est.setInputCloud (model_keypoints_triangle);
  descr_est.setInputNormals (model_normals_triangle);
  descr_est.setSearchSurface (model_triangle);
  descr_est.compute (*model_descriptors_triangle);
  
  descr_est.setInputCloud (model_keypoints_cube);
  descr_est.setInputNormals (model_normals_cube);
  descr_est.setSearchSurface (model_cube);
  descr_est.compute (*model_descriptors_cube);

  descr_est.setInputCloud (model_keypoints_cross);
  descr_est.setInputNormals (model_normals_cross);
  descr_est.setSearchSurface (model_cross);
  descr_est.compute (*model_descriptors_cross);

  descr_est.setInputCloud (model_keypoints_ball);
  descr_est.setInputNormals (model_normals_ball);
  descr_est.setSearchSurface (model_ball);
  descr_est.compute (*model_descriptors_ball);

  descr_est.setInputCloud (model_keypoints_cylinder);
  descr_est.setInputNormals (model_normals_cylinder);
  descr_est.setSearchSurface (model_cylinder);
  descr_est.compute (*model_descriptors_cylinder);

  descr_est.setInputCloud (model_keypoints_star);
  descr_est.setInputNormals (model_normals_star);
  descr_est.setSearchSurface (model_star);
  descr_est.compute (*model_descriptors_star);

  descr_est.setInputCloud (model_keypoints_cube_2);
  descr_est.setInputNormals (model_normals_cube_2);
  descr_est.setSearchSurface (model_cube_2);
  descr_est.compute (*model_descriptors_cube_2);

  descr_est.setInputCloud (model_keypoints_hollow_2);
  descr_est.setInputNormals (model_normals_hollow_2);
  descr_est.setSearchSurface (model_hollow_2);
  descr_est.compute (*model_descriptors_hollow_2);
  
  cout << "pointclout initialized..." << endl;

  ros::init(argc, argv, "mreyes_final");
  ros::NodeHandle nh;
  tf2_ros::Buffer tfBuffer(ros::Duration(60.0));
  tf2_ros::TransformListener tfListener(tfBuffer);
  tfb = &tfBuffer;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber rgb_sub = it.subscribe("/camera/rgb/image_color", 1, rgb_callback);
  ros::Subscriber pointcloud = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points", 1, pc_callback);
  ros::Subscriber arm = nh.subscribe<std_msgs::Bool>("/arm_to_camera", 100, arm_callback);
  evidence_pub = nh.advertise<ras_msgs::RAS_Evidence>("/evidence", 1000);
  marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1000);
  sound_pub = nh.advertise<std_msgs::String>("/espeak/string",100);
  motor_pub = nh.advertise<geometry_msgs::Twist>("/motor_controller/twist",100);
  move_to_pub = nh.advertise<geometry_msgs::PointStamped>("/brain/vision_target", 10);
  target_point_pub = nh.advertise<geometry_msgs::PointStamped>("/vis/point", 10);
 
  ros::ServiceServer vision_drive = nh.advertiseService("vision_investigate", vision_investigate_service); 

  //ros::spin();
  ros::MultiThreadedSpinner spinner(2);
  spinner.spin();
  return 0;
}
