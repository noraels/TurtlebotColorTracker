#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError

# ROS Topics
outputTopic = '/safe_cmd_vel'
imageTopic = '/camera/rgb/image_raw'
odomTopic = '/odom'

# Global Variables
gCurrentImage = None
gBridge = CvBridge()
gImageStarted = False
gRobotPosition = {'x': 0.0, 'y': 0.0}
task_start_time = 0.0
FULL_SPIN_DURATION = 12.56  # Approx time for a 360° spin at 0.5 rad/s


def preprocess_mask(image, target_color):
    mask = cv2.inRange(image, target_color[0], target_color[1])
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    MIN_AREA = 500
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_AREA]

    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    return cleaned_mask


def check_straight_edges(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    if contour_area < 500:
        return False

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    angle = rect[2]
    if angle < -45:
        angle += 90
    if abs(angle) > 20:
        return False

    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0
    if solidity < 0.85:
        return False

    vertical_edges = 0
    for i in range(4):
        pt1, pt2 = box[i], box[(i + 1) % 4]
        dx = abs(pt2[0] - pt1[0])
        dy = abs(pt2[1] - pt1[1])
        if dy > dx * 2:
            vertical_edges += 1

    return vertical_edges >= 2


def callbackImage(img):
    global gCurrentImage, gBridge, gImageStarted
    try:
        gCurrentImage = gBridge.imgmsg_to_cv2(img, "bgr8")
        gImageStarted = True
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")


def callbackOdom(msg):
    global gRobotPosition
    gRobotPosition['x'] = msg.pose.pose.position.x
    gRobotPosition['y'] = msg.pose.pose.position.y


def save_image_with_text(targetIndex, cx, area, robot_position, elapsed_time):
    global gCurrentImage

    annotated_image = gCurrentImage.copy()
    text_1 = f"Target: {targetIndex}"
    text_2 = f"Position: ({robot_position['x']:.2f}, {robot_position['y']:.2f})"
    text_3 = f"Elapsed Time: {elapsed_time:.2f}s"
    
    cv2.putText(annotated_image, text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (19, 69, 139), 2)
    cv2.putText(annotated_image, text_2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (19, 69, 139), 2)
    cv2.putText(annotated_image, text_3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (19, 69, 139), 2)

    filename = f"goal{targetIndex}.jpg"
    cv2.imwrite(filename, annotated_image)
    rospy.loginfo(f"Saved image: {filename}")


def trackNode(colorTargets):
    global gCurrentImage, task_start_time

    rospy.init_node('ctrackNode', anonymous=True)
    rospy.Subscriber(imageTopic, Image, callbackImage)
    rospy.Subscriber(odomTopic, Odometry, callbackOdom)
    vel_pub = rospy.Publisher(outputTopic, Twist, queue_size=10)

    cv2.namedWindow('Turtlebot Camera', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Target', cv2.WINDOW_AUTOSIZE)

    rospy.sleep(2)
    while not gImageStarted:
        rospy.sleep(1)

    rate = rospy.Rate(10)
    task_start_time = rospy.get_time()

    for targetIndex, targetColor in enumerate(colorTargets):
        rospy.loginfo(f"=== Starting search for target {targetIndex} ===")
        targetFound = False
        spin_start_time = rospy.get_time()
        target_in_sight = False  # New flag to track if we currently see the target

        while not rospy.is_shutdown() and not targetFound:
            if gCurrentImage is None:
                continue

            cv2.imshow('Turtlebot Camera', cv2.resize(gCurrentImage, (320, 240)))
            h, w = gCurrentImage.shape[:2]
            targetMask = preprocess_mask(gCurrentImage, targetColor)
            cv2.imshow('Target', cv2.resize(targetMask, (320, 240)))

            msg = Twist()
            msg.angular.z = 0.5
            msg.linear.x = 0.0

            m = cv2.moments(targetMask)
            target_in_sight = False  # Reset at start of loop
            
            if m['m00'] > 0:
                cx = m['m10'] / m['m00']
                delx = w / 2 - cx
                dist = m['m00'] / (h * w)

                if dist >= 0.05 or check_straight_edges(targetMask):
                    target_in_sight = True  # We have a valid target in sight
                    if dist > 15:
                        if abs(delx) < 30:
                            msg.angular.z = 0
                            if dist > 60:
                                msg.linear.x = -0.2
                                rospy.loginfo("TARGET: Too close, backing up")
                            elif dist < 40:
                                msg.linear.x = 0.2
                                rospy.loginfo("TARGET: Moving closer")
                            else:
                                elapsed_time = rospy.get_time() - task_start_time
                                save_image_with_text(targetIndex, int(cx), dist, gRobotPosition, elapsed_time)
                                rospy.loginfo(f"SUCCESS: Target {targetIndex} reached and image saved")
                                targetFound = True
                                continue
                        else:
                            msg.angular.z = 0.003 * delx
                            msg.linear.x = 0.1
                            rospy.loginfo(f"TARGET: Centering with offset {delx:.2f}")
                    else:
                        turn_scale = abs(delx) / (w / 2)
                        msg.angular.z = 0.002 * delx
                        msg.linear.x = 0.5 * (1 - turn_scale)
                        rospy.loginfo(f"TARGET: Direct approach - Distance: {dist:.2f}")
                else:
                    rospy.loginfo("TARGET: Detected but too small/invalid")

            # Only check spin timer if we're not currently tracking a valid target
            if not target_in_sight:
                rospy.loginfo("SEARCHING: No valid target in sight")
                elapsed_time = rospy.get_time() - spin_start_time
                if elapsed_time >= FULL_SPIN_DURATION:
                    rospy.loginfo("SEARCHING: Full rotation completed - Moving to new position")
                    msg.angular.z = 0.5
                    msg.linear.x = 0.3
                    vel_pub.publish(msg)
                    rospy.sleep(2)
                    spin_start_time = rospy.get_time()

            vel_pub.publish(msg)
            cv2.waitKey(1)
            rate.sleep()

    rospy.loginfo("=== All targets processed successfully ===")
    cv2.destroyAllWindows()


def callback_shutdown():
    rospy.loginfo("Shutting down CTrackNode")
    stop_msg = Twist()
    stop_msg.linear.x = 0.0
    stop_msg.angular.z = 0.0
    temp_vel_pub = rospy.Publisher(outputTopic, Twist, queue_size=10)
    rospy.sleep(1)
    temp_vel_pub.publish(stop_msg)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        rospy.on_shutdown(callback_shutdown)

        colorTargets = [
            [(0, 30, 75), (5, 50, 89)],
            [(240, 240, 240), (255, 255, 255)],
            [(0, 80, 80), (10, 110, 110)],
            [(0, 0, 0), (20, 20, 20)]
        ]

        trackNode(colorTargets)
    except rospy.ROSInterruptException:
        pass
