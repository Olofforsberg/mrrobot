#include <ros/ros.h>
#include <iostream>
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
//#include <pcl/filters/uniform_sampling.h>
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

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (true);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.004f);
float scene_ss_ (0.004f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);

const char *shape[] = {"Hollow Cube", "Triangle", "Cube"};
const int model_nr = 3;
bool found_obj = false;
bool search_order = false;
float marker_x, marker_y, marker_z;
std::string object_ID;
sensor_msgs::Image image_rec;

static ros::Publisher marker_pub;
static ros::Publisher detection_status;
static ros::Publisher evidence_pub;
static ros::Publisher sound_pub;

void imagecallback(const sensor_msgs::Image::ConstPtr & msg_image)
{
  image_rec.header = msg_image->header;
  image_rec.height = msg_image->height;
  image_rec.width = msg_image->width;
  image_rec.encoding = msg_image->encoding;
  image_rec.is_bigendian = msg_image->is_bigendian;
  image_rec.step = msg_image->step;
  image_rec.data = msg_image->data;
}

void colorcallback(const std_msgs::String::ConstPtr& msg_color)
{
  object_ID = msg_color->data;
}

void ordercallback(const std_msgs::Bool::ConstPtr& msg_order)
{ 
  search_order = msg_order->data;
}

static void callback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
  if(search_order)
  {
  std::cout << "Searching object shape..." << std::endl;
  // save point cloud to .pcd
  pcl::PointCloud <pcl::PointXYZ> cloud_rgb;
  pcl::fromROSMsg(*cloud, cloud_rgb);
  std::vector<int> indices_rgb;
  pcl::removeNaNFromPointCloud(cloud_rgb, cloud_rgb, indices_rgb);
  pcl::io::savePCDFileASCII("test_rgb.pcd", cloud_rgb);


  // start matching
  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model_rgb (new pcl::PointCloud<pcl::PointXYZRGBA> ());
  pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_ori (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
  pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
  
  // rotate point cloud

  float theta = 2.05; // The angle of rotation in radians 
  transform_1 (2,2) = cos (theta);
  transform_1 (2,1) = -sin(theta);
  transform_1 (1,2) = sin (theta);
  transform_1 (1,1) = cos (theta);
  transform_1 (2,3) = 0.125;
  transform_1 (1,3) = -0.015;
  transform_1 (0,3) = -0.015;
 
  int corr_value[] = {0, 0, 0};
  float pos_x[] = {0, 0, 0};
  float pos_y[] = {0, 0, 0};
  float pos_z[] = {0, 0, 0};
  // searching shape
  for(int shape_nr = 0; shape_nr < model_nr; shape_nr++)
  {
    // load  pointcloud
    if(shape_nr == 0) {pcl::io::loadPCDFile("test_rgb_hollow_cube.pcd", *model_rgb);}
    if(shape_nr == 1) {pcl::io::loadPCDFile("test_rgb_triangle.pcd", *model_rgb);}
    if(shape_nr == 2) {pcl::io::loadPCDFile("test_rgb_cube.pcd", *model_rgb);}
    pcl::copyPointCloud(*model_rgb, *model);
    pcl::io::loadPCDFile("test_rgb.pcd", *scene_ori);
    pcl::transformPointCloud (*scene_ori, *scene_ori, transform_1);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(scene_ori);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits(0.008,0.5);
    pass.filter(*scene);

    //  Compute Normals
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (scene);
    norm_est.compute (*scene_normals);
    norm_est.setKSearch (10);
    norm_est.setInputCloud (model);
    norm_est.compute (*model_normals);

    //  Downsample Clouds to Extract keypoints
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (scene_ss_);
    pcl::PointCloud<int> keypointIndices2;
    uniform_sampling.compute(keypointIndices2);
    pcl::copyPointCloud(*scene, keypointIndices2.points, *scene_keypoints); 
    // std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

    uniform_sampling.setInputCloud (model);
    uniform_sampling.setRadiusSearch (model_ss_);
    pcl::PointCloud<int> keypointIndices1;
    uniform_sampling.compute(keypointIndices1);
    pcl::copyPointCloud(*model, keypointIndices1.points, *model_keypoints); 
    // std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

    //  Compute Descriptor for keypoints
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad_);
    descr_est.setInputCloud (scene_keypoints);
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);

    descr_est.setInputCloud (model_keypoints);
    descr_est.setInputNormals (model_normals);
    descr_est.setSearchSurface (model);
    descr_est.compute (*model_descriptors);

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

    if (use_hough_)
      {
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
      }

    //  Output results
    int obj_nr = 0;
    // std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    for (size_t i = 0; i < rototranslations.size (); ++i)
      {
	/*
	std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
	std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;
	*/
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
	

	if( clustered_corrs[i].size() > 90) {obj_nr = obj_nr + 1;}

	/*
	// Print the rotation matrix and translation vector
	Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
	Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

	printf ("\n");
	printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
	printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
	printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
	printf ("\n");
	printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
	*/
      }
    if(obj_nr != 0) 
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
      found_obj = true;
      // publish sound
      std_msgs::String sound;
      sound.data = "Wow, I see a " + object_ID;
      sound_pub.publish(sound);
      break;
    }

    /*
    
    //  Visualization
    pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    viewer.addPointCloud (scene, "scene_cloud");

    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    if (show_correspondences_ || show_keypoints_)
      {
	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
	pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
	viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
      }

    if (show_keypoints_)
      {
	pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
	viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
	viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
      }

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
    pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
    pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

    std::stringstream ss_cloud;
    ss_cloud << "instance" << i;

    pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
    viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

    if (show_correspondences_)
      {
	for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
	  {
	    std::stringstream ss_line;
	    ss_line << "correspondence_line" << i << "_" << j;
	    PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
	    PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

	    //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
	    viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
	  }
      }
    }

    while (!viewer.wasStopped ())
    {
      viewer.spinOnce ();
    }   
    */
  }
  
 
  if(found_obj == false)
    {
      if(corr_value[0] < 20 && corr_value[1] < 20 && corr_value[2] < 20)
	{
	  std::cout << "No shape matches!!" << std::endl;
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
	  // publish sound
	  std_msgs::String sound;
	  sound.data = "Wow, I see a " + object_ID;
	  sound_pub.publish(sound);
	}
    }

  search_order = false;
  }


  if(found_obj)
    {
      // publish evidence
      ras_msgs::RAS_Evidence evidence;
      evidence.stamp = ros::Time::now();
      evidence.group_number = 2;
      evidence.image_evidence = image_rec;
      evidence.object_id = object_ID;
      geometry_msgs::TransformStamped location;
      location.child_frame_id = cloud->header.frame_id;
      location.transform.translation.x = marker_x;
      location.transform.translation.y = marker_y;
      location.transform.translation.z = marker_z;
      evidence.object_location = location;
      evidence_pub.publish(evidence);
      std::cout << "publishing evidence..." << std::endl;

      // publish marker
      visualization_msgs::Marker marker;
      marker.header.frame_id = cloud->header.frame_id;
      marker.header.stamp = ros::Time::now();
      marker.ns = "object_marker";
      marker.id = 1;
      marker.type = visualization_msgs::Marker::SPHERE;
      marker.action = visualization_msgs::Marker::ADD;

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
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 1.0;

      marker_pub.publish(marker);

      found_obj = false;
    }
  // told color_detc to start
  std_msgs::Bool shape_status;
  shape_status.data = false;
  detection_status.publish(shape_status);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mreyes_hough");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth/points", 1, callback);
  ros::Subscriber order = nh.subscribe("order", 100, ordercallback);
  ros::Subscriber color = nh.subscribe("object_color",100, colorcallback);
  ros::Subscriber image = nh.subscribe("/camera/rgb/image_color", 1, imagecallback);
  marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);
  detection_status = nh.advertise<std_msgs::Bool>("shape_detect_status",100);
  evidence_pub = nh.advertise<ras_msgs::RAS_Evidence>("/evidence", 1000);
  sound_pub = nh.advertise<std_msgs::String>("/espeak/string",100);
  ros::spin();
}
