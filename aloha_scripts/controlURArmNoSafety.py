import socket
import time
import rtde_receive
import rtde_control
import numpy as np
import struct
from datetime import datetime
import csv
import keyboard
# import winsound
from visual_kinematics.RobotSerial import *
from math import pi

beepFrequency = 1000  # Frequency of the beep sound in Hertz
beepDuration = 200  # Duration of the beep sound in milliseconds

# initialize object for robot serial and DH params
dh_params = np.array(
        [
            [0.1625, 0.0, 0.5 * pi, 0.0],
            [0.0, -0.425, 0, 0.0],
            [0.0, -0.3922, 0, 0.0],
            [0.1333, 0.0, 0.5 * pi, 0.0],
            [0.0997, 0.0, -0.5 * pi, 0.0],
            [0.0996, 0.0, 0.0, 0.0],
        ]
    )
replica = RobotSerial(dh_params)

# Receiver (server) address and port
receiver_address = ('192.168.137.12', 5000)

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the receiver address
server_socket.bind(receiver_address)

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.137.32")
rtde_c = rtde_control.RTDEControlInterface("192.168.137.32")
# Listen for incoming connections
server_socket.listen(1)

print("Waiting for a connection...")

# Accept a connection
client_socket, client_address = server_socket.accept()
print("Connected by", client_address)


def getReplicaTCPPose(jointAngles):
    global replica
    forward = replica.forward(jointAngles)
    xyz = forward.t_3_1.reshape([3,])
    rpy = forward.euler_3
    replicaPose = np.concatenate((xyz,rpy)) 
    return replicaPose , xyz

def checkCollision(robotPose, ylim = 0.661, zlim = 0.20): # limits in meters
    x,y,z,rl,pt,yw = robotPose
    if (y > ylim) or (z < zlim):
        return True
    else: 
        return False

def writeData2File(dataArray):
    filename_prefix = "ur5e_pose_data"
    current_date = datetime.now().strftime('%m-%d_%H-%M')
    filename = f"{filename_prefix}_{current_date}.csv"
    heading = ['Sr.no.', 'X', ' Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Timestamp']
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(heading)
        for row in dataArray:
            csv_writer.writerow(row)

def moveArmToPose(jointAngles, asnx = True):
        # coord = np.array(jointAngles)
        # Swap x and z values in orientation data
        # coord[3], coord[5] = coord[5], coord[3]

        # print("Moving to pose: " + str(jointAngles))

        # Move to the inferenced pose
        rtde_c.moveJ(jointAngles, 3.14, 1.5, asnx) #increase to V= 3.14/sec to reduce Robot Teleoperation lag in synchronous mode
        # self.rtde_c.stopL(0.5)
        return True

def moveArmToJPose(jointAngles, asnx = True):
    velocity = 0.5
    acceleration = 0.5
    dt = 1.0/500  # 2ms
    lookahead_time = 0.1
    gain = 300
    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(jointAngles, velocity, acceleration, dt, lookahead_time, gain)
    rtde_c.waitPeriod(t_start)

def moveArmToHomePose():
    coord = [-0.13300146849405034, -0.4923589921772682, 0.4894790515662251, -2.218162605515039, 2.214123626635495, -0.009187629843795802]
    
    # coord[3:] = self.ypr_2_rot_vec(coord[3:])

    # Swap x and z values in orientation data
    # coord[3], coord[5] = coord[5], coord[3]

    print("Moving to home pose: " + str(coord))

    # Move to the inferenced pose
    rtde_c.moveL(coord, 1.500, 0.05, False)  

def main():
    # Receive and print the message
    moveArmToHomePose()
    # Receive and print the message
    itr=1
    readings=[]

    start_time = time.time()
    time_threshold = 0.2  # Set the time threshold in seconds
    Flag= False
    while (True):
        try:
            
            moveArm = True
            data = client_socket.recv(24)
            received_data = list(struct.unpack('6f', data))
            if Flag == False:
                Flag= moveArmToPose(received_data, asnx = False)
            else:
                moveArmToJPose(received_data, asnx = True)

            
        except KeyboardInterrupt:
            print("bye bye ")
            rtde_c.stopScript()
            if len(readings)>0:
                writeData2File(readings)
            client_socket.close()
            server_socket.close()
            break
        
        except rtde_r.RTDEException as e:
            print("RTDE error: {}".format(e))
            rtde_c.stopScript()
            if len(readings)>0:
                writeData2File(readings)
            client_socket.close()
            server_socket.close()
            break
        # time.sleep(0.001)
if __name__ == "__main__":
    main()
    