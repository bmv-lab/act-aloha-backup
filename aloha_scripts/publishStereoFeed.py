import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import subprocess

gpu_frame = cv2.cuda_GpuMat()
gpu_result = cv2.cuda_GpuMat()
vidChannel = int(subprocess.check_output("ls /dev/video*", shell=True).decode("utf-8").splitlines()[0][-1])

def publish_camera_feed():
    # Initialize ROS node
    rospy.init_node('camera_feed_publisher', anonymous=True)

    # Create publishers for the left and right camera feeds
    left_feed_publisher = rospy.Publisher('/stereo/left/image_raw', Image, queue_size=10)
    right_feed_publisher = rospy.Publisher('/stereo/right/image_raw', Image, queue_size=10)

    # Create a CvBridge instance for image conversion
    bridge = CvBridge()

    # Initialize the left and right camera captures
    cap = cv2.VideoCapture(vidChannel)  
    # Set the camera frame size (optional)
    # left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Loop to continuously capture and publish camera frames
    rate = rospy.Rate(1)  # Adjust the rate as needed
    while not rospy.is_shutdown():
        # start_time = rospy.Time.now()
        ret, frame = cap.read()
    
        # Split the frame into left and right images
        height, width, _ = frame.shape
        split_width = width // 2
        left_frame = frame[:, :split_width, :]
        right_frame = frame[:, split_width:, :]
        
        # cv2.imwrite("saved.jpg", left_frame)
        
        # Resize the frames if needed
        left_frame = cv2.resize(left_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        right_frame = cv2.resize(right_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # Convert the frames to ROS image messages
        left_msg = bridge.cv2_to_imgmsg(left_frame, encoding="bgr8")
        right_msg = bridge.cv2_to_imgmsg(right_frame, encoding="bgr8")

        # Publish the left and right camera feeds
        left_feed_publisher.publish(left_msg)
        right_feed_publisher.publish(right_msg)

        print(left_frame.shape)

        # end_time = rospy.Time.now()

        # # Calculate the time difference
        # time_diff = end_time - start_time

        # # Print the time lag in milliseconds
        # print("Time Lag (ms):", round(time_diff.to_sec() * 1000,2), end="\r")
        rate.sleep()

    # Release the camera captures
    cap.release()


if __name__ == '__main__':
    try:
        publish_camera_feed()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        exit()