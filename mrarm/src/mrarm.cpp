#include "ros/ros.h"
#include "std_msgs/UInt8.h"
#include "uarm/Coords.h"

#include <sstream>
#include <iostream>
#include "geometry_msgs/PointStamped.h"

static geometry_msgs::PointStamped New_coords1;
static geometry_msgs::PointStamped alwaysgohere;

static int New_messege1=0;

static geometry_msgs::PointStamped New_coords2;
static geometry_msgs::PointStamped putdown;
static int New_messege2=0;

static geometry_msgs::PointStamped moveside;
static std_msgs::UInt8 suck;
static std_msgs::UInt8 realise;



//uarm::Coords New_coords3;
static int New_messege3=0;

void chatterCallback1(const geometry_msgs::PointStamped coords)
{
    std::cerr << "received new coords\n";
    New_coords1 = coords;
    New_messege1 = 1;
}

void chatterCallback2(const geometry_msgs::PointStamped coords)
{
    New_coords2 = coords;
    New_messege2 = 1;

}
void chatterCallback3(const std_msgs::UInt8 putdown)
{
    if(putdown.data==1){
    New_messege3 = 1;
    }
}

int main(int argc, char **argv)
{
    alwaysgohere.header.frame_id = "arm_coords_origin";
    alwaysgohere.point.x=0.10; alwaysgohere.point.y=-0.10; alwaysgohere.point.z = 0.15;
    putdown.header.frame_id = "arm_coords_origin";
    putdown.point = alwaysgohere.point;
    putdown.point.z=-0.1;
    moveside.header.frame_id = "arm_coords_origin";
    moveside.point = alwaysgohere.point;
    moveside.point.x = 0.0; moveside.point.y=-0.18; moveside.point.z=0.0;
    suck.data = 1;
    realise.data=0;
    ros::init(argc, argv, "mrarm");


    //subscriber
    ros::NodeHandle n;
    ros::Subscriber sub1 = n.subscribe("arm_grab", 1000, chatterCallback1);
    ros::Subscriber sub2 = n.subscribe("arm_move", 1000, chatterCallback2);
    ros::Subscriber sub3 = n.subscribe("arm_putdown", 1000, chatterCallback3);

    //publisher
    ros::Publisher pub1 = n.advertise<geometry_msgs::PointStamped>("/move_to", 1000);
    ros::Publisher pub2 = n.advertise<std_msgs::UInt8>("/pump_control", 1000);
    ros::Rate loop_rate(100);

    while (ros::ok()) {

        if(New_messege1==1){
            New_messege1 = 0;
            std::cerr << "reset to default position\n";
            pub1.publish(alwaysgohere);
            sleep(1);
            std::cerr << "go to new position\n" << New_coords1;
            pub1.publish(New_coords1);
            sleep(1);
            std::cerr << "start pump\n";
            pub2.publish(suck);
            sleep(1);
            std::cerr << "reset to default position\n";
            pub1.publish(alwaysgohere);
		//pub2.publish(realise);
        }

        if(New_messege2==1){
            New_messege2=0;
            pub1.publish(alwaysgohere);
            sleep(1);
            pub1.publish(New_coords2);
            sleep(1);
            pub2.publish(suck);
            sleep(1);
            pub1.publish(alwaysgohere);
            sleep(1);
            pub1.publish(moveside);
            sleep(2);
            pub2.publish(realise);
            pub1.publish(alwaysgohere);
        }

        if(New_messege3==1){
            New_messege3 =0;
            pub1.publish(alwaysgohere);
            sleep(1);
            pub1.publish(putdown);
            sleep(1);
            pub2.publish(realise);
            sleep(1);
            pub1.publish(alwaysgohere);
            }

    loop_rate.sleep();
    ros::spinOnce();
    }
  return 0;
}
