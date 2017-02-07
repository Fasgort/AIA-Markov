#-*-coding:utf-8-*-

from Hmm import Hmm

import numpy as np
import random

class Robot(Hmm):

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def get_obstacle_rate(self):
        return self.obstacle_rate

    def set_obstacle_rate(self, obstacle_rate):
        self.obstacle_rate = obstacle_rate

    def get_map(self):
        return self.map_mat
        
    def generate_map(self):
        self.map_mat = np.zeros((size, size))
        for x in range(0, self.size):
            for y in range(0, self.size):
                self.map_mat[x][y] = random.sample([0,0,0,0,1], 1)
                
    def print_map(self):
        for x in range(0, self.size):
            print(self.map_mat[x])

    def get_error(self):
        return self.error

    def set_error(self, error):
        self.error = error

    def coordinates_to_state(self, point):
        return str(point[0]) + ',' + (point[1])

    def make_a_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, valid_states)
        a_mat = np.zeros((shape[0], shape[1]))
        for state1 in range(valid_states):
            for state2 in range(valid_states):
                return # not implemented
        self.a_mat = a_mat
