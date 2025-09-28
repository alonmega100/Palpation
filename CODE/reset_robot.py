####   RESET robot to start position  ######

import panda_py as pp

robot = pp.Panda("172.16.0.2")
robot.move_to_start()