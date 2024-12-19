#!/usr/bin/env python
import rospy
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

# ROS Topics
input_topic = '/safe_cmd_vel'
output_topic = '/cmd_vel'
laser_topic = '/scan'

# Global Variables
LEFT_THRESHOLD = 0.5
RIGHT_THRESHOLD = 0.5
g_last_scan = None

def get_front_readings(scan_data):
    """Get minimum distances for front-left and front-right sectors (45° each)"""
    total_angles = len(scan_data.ranges)
    angles_per_degree = total_angles / 360.0
    
    # Calculate indices for 45° sectors on each side
    left_end = int(45 * angles_per_degree)  # 0° to 45°
    right_start = int(315 * angles_per_degree)  # 315° to 360°
    
    # Get minimum distances for each sector
    min_left = min(scan_data.ranges[0:left_end])
    min_right = min(scan_data.ranges[right_start:])
    
    return min_left, min_right

def calculate_obstacle_avoidance(scan_data, input_twist):
    min_left, min_right = get_front_readings(scan_data)
    rospy.loginfo(f"Front distances - Left(0-45°): {min_left:.2f}m, Right(315-360°): {min_right:.2f}m")
    
    output_twist = Twist()
    output_twist.linear.x = input_twist.linear.x
    output_twist.angular.z = input_twist.angular.z

    if input_twist.linear.x > 0:  # Only avoid obstacles when moving forward
        if min_left < LEFT_THRESHOLD and min_right < RIGHT_THRESHOLD:
            rospy.loginfo("OBSTACLE: Both sides blocked - Stopping")
            output_twist.linear.x = 0.0
            output_twist.angular.z = 0.0
        elif min_left < LEFT_THRESHOLD:
            rospy.loginfo(f"OBSTACLE: Left side at {min_left:.2f}m - Turning right")
            output_twist.angular.z = -0.2
            output_twist.linear.x = input_twist.linear.x * 0.5
        elif min_right < RIGHT_THRESHOLD:
            rospy.loginfo(f"OBSTACLE: Right side at {min_right:.2f}m - Turning left")
            output_twist.angular.z = 0.2
            output_twist.linear.x = input_twist.linear.x * 0.5
    
    return output_twist

def callback_cmd_vel(input_twist):
    global g_last_scan
    if g_last_scan is not None:
        output_twist = calculate_obstacle_avoidance(g_last_scan, input_twist)
        pub.publish(output_twist)
    else:
        rospy.logwarn("No laser scan data received yet")

def callback_laser(scan_data):
    global g_last_scan
    g_last_scan = scan_data

def callback_shutdown():
    rospy.loginfo("Shutting down OANode")
    # Publish stop command
    stop_msg = Twist()
    stop_msg.linear.x = 0.0
    stop_msg.angular.z = 0.0
    pub.publish(stop_msg)
    rospy.sleep(1)

if __name__ == '__main__':
    try:
        rospy.init_node('oa_node', anonymous=True)
        rospy.loginfo("Starting OANode...")
        
        # Publishers and Subscribers
        pub = rospy.Publisher(output_topic, Twist, queue_size=10)
        rospy.Subscriber(input_topic, Twist, callback_cmd_vel)
        rospy.Subscriber(laser_topic, LaserScan, callback_laser)
        g_last_scan = None
        
        # Register shutdown hook
        rospy.on_shutdown(callback_shutdown)
        
        rospy.loginfo("=== OANode initialized ===")
        rospy.loginfo(f"Monitoring front 45° sectors for obstacles")
        rospy.loginfo(f"Thresholds - Left: {LEFT_THRESHOLD}m, Right: {RIGHT_THRESHOLD}m")
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception! Node shutting down.")
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")