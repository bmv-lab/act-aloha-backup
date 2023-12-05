#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Initialize the ROS node
    rospy.init_node('realsense_feed', anonymous=True)

    # Create a publisher for the RGB image
    image_pub = rospy.Publisher('/usb_cam_wrist/image_raw', Image, queue_size=10)

    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Start the RealSense pipeline
    pipeline.start(config)

    # Create a CvBridge instance for image conversion
    bridge = CvBridge()
    rate = rospy.Rate(60)
    try:
        while not rospy.is_shutdown():
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if color_frame:
                # Convert the RealSense color frame to an OpenCV image
                color_image = np.asanyarray(color_frame.get_data())
                cv2_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Create a ROS Image message from the OpenCV image
                ros_image = bridge.cv2_to_imgmsg(cv2_image, encoding="rgb8")

                # Publish the ROS Image message
                image_pub.publish(ros_image)

                rate.sleep()
        pipeline.stop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
