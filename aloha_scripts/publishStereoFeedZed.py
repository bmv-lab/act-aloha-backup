#!/usr/bin/env python3
import sys
import pyzed.sl as sl
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Change the resolution as needed
    init_params.camera_fps = 60  # Change the frame rate as needed
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth calculation
    init_params.camera_disable_self_calib = True

    # Initialize the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error initializing the camera: {err}")
        zed.close()
        return

    # Create a runtime parameters object
    runtime_params = sl.RuntimeParameters()

    rospy.init_node('zed_feed', anonymous=True)

    # Create publishers for the left and right camera feeds
    # left_feed_publisher = rospy.Publisher('/stereo/left/image_raw', Image, queue_size=10)
    right_feed_publisher = rospy.Publisher('/usb_cam_low/image_raw', Image, queue_size=10)
    # Create a CvBridge instance for image conversion
    bridge = CvBridge()
    rate = rospy.Rate(60)
    try:
        while not rospy.is_shutdown():
            # Grab a new frame from the camera
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # left_image = sl.Mat()
                right_image = sl.Mat()
                # zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                # left_frame = cv2.resize(left_image.get_data(), (640,480))
                right_frame = cv2.resize(right_image.get_data(), (640,480))

                # left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
                right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)

                # Convert the frames to ROS image messages
                # left_msg = bridge.cv2_to_imgmsg(left_frame, encoding="rgb8")
                right_msg = bridge.cv2_to_imgmsg(right_frame, encoding="rgb8")

                # Publish the left and right camera feeds
                # left_feed_publisher.publish(left_msg)
                right_feed_publisher.publish(right_msg)
                # print("image published")
                rate.sleep()


    except KeyboardInterrupt:
        zed.close()

    # Close the camera
    

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
