#include <cmath>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <phidgets/motor_encoder.h>
#include <tf/transform_broadcaster.h>

// wheel base and radius
static float b = 21.5e-2;
static float r = 7.3e-2/2.0;

// control gain
static float alpha_left = 0.1;
static float alpha_right = 0.1;

static float ticks_per_revolution = 900.0; // hehe 3591.84;
static float control_frequency = 125.0;

static double estimated_w_left = 0.0, estimated_w_right = 0.0;
static double desired_w_left = 0.0, desired_w_right = 0.0;

static ros::Time t_last;

// pid param
static double Kp_left = 5.0;
static double Kp_right = 5.0;
static double Ki_left = 0.5;
static double Ki_right = 0.5;
static double Kd_left = 0.1;
static double Kd_right = 0.1;
static double error_left_2 = 0.0;
static double error_left_1 = 0.0;
static double error_left_0 = 0.0;
static double error_right_2 = 0.0;
static double error_right_1 = 0.0;
static double error_right_0 = 0.0;

// NOTE: Setting dt = control_frequency, should really be the arrival times of
// motor encoder messages.

#define ENCODER_TO_ANGULAR(x) ((double)((x)*2.0*M_PI*control_frequency)/ticks_per_revolution)

void encoderCallbackLeft(const phidgets::motor_encoder::ConstPtr& msg)
{
    estimated_w_left = ENCODER_TO_ANGULAR(msg->count_change);
}

void encoderCallbackRight(const phidgets::motor_encoder::ConstPtr& msg)
{
    estimated_w_right = ENCODER_TO_ANGULAR(msg->count_change);
}

void twistCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
    //desired_w=(v-(b/2)*w)/r
    t_last = ros::Time::now();
    desired_w_left  = (float)((msg->linear.x-(b/2)*msg->angular.z)/r);
    desired_w_right = -(float)((msg->linear.x+(b/2)*msg->angular.z)/r);
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "motor_controller");
    ros::NodeHandle n;
    ros::Rate loopRate(control_frequency);
    ros::Publisher pub_pwm_left = n.advertise<std_msgs::Float32>("/left_motor/cmd_vel", 1000);
    ros::Publisher pub_pwm_right = n.advertise<std_msgs::Float32>("/right_motor/cmd_vel", 1000);
    ros::Subscriber sub_encoder_left = n.subscribe("/left_motor/encoder",  1000, encoderCallbackLeft);
    ros::Subscriber sub_encoder_right = n.subscribe("/right_motor/encoder", 1000, encoderCallbackRight);
    ros::Subscriber sub_twist = n.subscribe("/motor_controller/twist", 1000, twistCallback);

    ros::Publisher set_speed_left = n.advertise<std_msgs::Float32>("left_speed_set",1000);
    ros::Publisher set_speed_right = n.advertise<std_msgs::Float32>("right_speed_set",1000);
    ros::Publisher estimate_left = n.advertise<std_msgs::Float32>("left_speed",1000);
    ros::Publisher estimate_right = n.advertise<std_msgs::Float32>("right_speed",1000);

    ros::Publisher pub_odom = n.advertise<nav_msgs::Odometry>("/odom/encoder", 1000);

    std_msgs::Float32 move_left, move_right;
    float &pwm_left(move_left.data);
    float &pwm_right(move_right.data);
        
    std_msgs::Float32 speed_left, speed_right, set_left, set_right;
    nav_msgs::Odometry odom_msg;

    sensor_msgs::JointState joint_state;
    joint_state.name.resize(1);
    joint_state.position.resize(1);
    tf::TransformBroadcaster broadcaster;
    geometry_msgs::TransformStamped odom_trans;
    geometry_msgs::TransformStamped map_to_odom;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base";
    map_to_odom.header.frame_id = "map";
    map_to_odom.child_frame_id = "odom";
    double theta = 0.0;

    while (ros::ok()) {

        if ((ros::Time::now() - t_last).toSec() > 0.2 && (desired_w_left != 0.0 || desired_w_right != 0.0)) {
            desired_w_left = desired_w_right = 0.0;
            std::cerr << "idle, shutting down engines" << std::endl;
        }

        if ((desired_w_left  == 0.0 && abs(estimated_w_left)  < 0.05) &&
            (desired_w_right == 0.0 && abs(estimated_w_right) < 0.05)) {
            pwm_left = pwm_right = 0.0;
        }

        //pwm = pwm + alpha*(desired_w - estimated_w)
        /*
        pwm_left  = std::max(-100.0, std::min(+100.0, pwm_left  + alpha_left*(desired_w_left - estimated_w_left)));
        pwm_right = std::max(-100.0, std::min(+100.0, pwm_right + alpha_right*(desired_w_right - estimated_w_right)));
        */
        //pid u[k]=u[k-1]+Kp*(e[k]-e[k-1])+Ki*e[k]+Kd*(e[k]-2*e[k-1]+e[k-2])
        error_left_2 = error_left_1;
        error_left_1 = error_left_0;
        error_left_0 = desired_w_left - estimated_w_left;
        error_right_2 = error_right_1;
        error_right_1 = error_right_0;
        error_right_0 = desired_w_right - estimated_w_right;
        pwm_left = std::max(-100.0, std::min(+100.0, pwm_left  + Kp_left*(error_left_0 - error_left_1) + Ki_left * error_left_0 + Kd_left * (error_left_0 - 2 * error_left_1 + error_left_2)));
        pwm_right = std::max(-100.0, std::min(+100.0, pwm_right  + Kp_right*(error_right_0 - error_right_1) + Ki_right * error_right_0 + Kd_right * (error_right_0 - 2 * error_right_1 + error_right_2)));
        //ROS_INFO_STREAM("PWM left:  " << pwm_left << " (desired " << desired_w_left << " estimated " << estimated_w_left << ")");
        //ROS_INFO_STREAM("PWM right:  " << pwm_right << " (desired " << desired_w_right << " estimated " << estimated_w_right << ")");
        pub_pwm_left.publish(move_left);
        pub_pwm_right.publish(move_right);

        set_left.data = desired_w_left;
        set_right.data = desired_w_right;
        set_speed_left.publish(set_left);
        set_speed_right.publish(set_right);
        speed_left.data = estimated_w_left;
        speed_right.data = estimated_w_right;
        estimate_left.publish(speed_left);
        estimate_right.publish(speed_right);

        double f = control_frequency;
        double D = (r/2.0)*((-estimated_w_right) + estimated_w_left);
        double dtheta = (r/b)*((-estimated_w_right) - estimated_w_left);

        //publish odom info for mrKalman
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.pose.pose.position.x += D*std::cos(theta)/f;
        odom_msg.pose.pose.position.y += D*std::sin(theta)/f;
        odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromYaw(theta += dtheta/f);
        odom_msg.twist.twist.linear.x = D;
        odom_msg.twist.twist.angular.z = dtheta;
        pub_odom.publish(odom_msg);

        ros::spinOnce();
        loopRate.sleep();
    }

    ROS_INFO_STREAM("!ros::ok()");

    return 0;
}
