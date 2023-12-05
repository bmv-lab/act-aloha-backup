#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def right_image_callback(msg):
    try:
        bridge = CvBridge()
        # Convert the ROS Image message to a CV2 image
        right_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Process the right_image as needed (e.g., display or save)
        cv2.imshow("Right Camera Feed", right_image)
        cv2.waitKey(1)  # You may need this line to update the display
        
    except Exception as e:
        rospy.logerr(e)

def main():
    rospy.init_node('recorder', anonymous=True)

    # Create subscribers for the left and right camera feeds
    right_subscriber = rospy.Subscriber('/usb_cam_high/image_raw', Image, right_image_callback)

    rospy.spin()  # Keep the script running

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
