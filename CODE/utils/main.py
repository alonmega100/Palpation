import panda_py
import time
from tools import *
ROBOT_IP = "172.16.0.2"
def main():
    robot = panda_py.Panda(ROBOT_IP)


    while True:
        current_pose_matrix =robot.get_pose()

        # Print the pose matrix

        # Extract position (example)
        position = current_pose_matrix[:3, 3]

        rot_mat = current_pose_matrix[:3, :3]
        yaw_pitch_roll = rot_mat_to_euler_zyx(rot_mat, True)

        print("############")
        print(position)
        print(" ## \n ## ")
        print(yaw_pitch_roll)
        print("############")

        time.sleep(1)


if __name__ == '__main__':
    main()

