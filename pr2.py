# -*- coding: utf-8 -*-

from Robot import Robot
import random
import numpy as np

robot = Robot()
robot.set_size(2)
robot.set_error(0.3)
robot.generate_map()
robot.print_map()
map = robot.get_map()
valid_states = map.size - np.count_nonzero(map)
for x in range(valid_states):
    print(robot.state_to_coordinates(x))
#robot.make_b_mat()
#robot.print_b_mat()
