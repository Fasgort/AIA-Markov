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
        self.map_mat = np.zeros((self.size, self.size),dtype=int)
        for x in range(0, self.size):
            for y in range(0, self.size):
                self.map_mat[x][y] = random.sample([0,0,0,0,1], 1)[0]
                
    def print_map(self):
        for x in range(0, self.size):
            print(self.map_mat[x])

    def get_error(self):
        return self.error

    def set_error(self, error):
        self.error = error

    def state_to_coordinates(self, state):
        # Tiene errores
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        if state > valid_states:
            raise Exception
        for x in range(self.size):
            if (self.map_mat[x].size - np.count_nonzero(self.map_mat[x])) != 0:
                state = state - (self.map_mat[x].size - np.count_nonzero(self.map_mat[x]))
                if state <= 0:
                    state = state + (self.map_mat[x].size - np.count_nonzero(self.map_mat[x]))
                    y = -1
                    while True:
                        y += 1
                        state = state - 1 + self.map_mat[x][y]
                        print("y: " + str(y) + "; state: " + str(state))
                        if state != 0:
                            break
                        return (x,y)

    def make_a_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, valid_states)
        a_mat = np.zeros((shape[0], shape[1]))
        for state1 in range(valid_states):
            for state2 in range(valid_states):
                return # not implemented
        self.a_mat = a_mat
        
    def make_b_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, 16) # From 1111(0000), to 2222(1111), NSWE
        b_mat = np.zeros((shape[0], shape[1]))
        for state in range(valid_states):
            print(state)
            coords_state = self.state_to_coordinates(state)
            print(coords_state)
            obstacles = 1111
            if coords_state[0]-1 < 0 or self.map_mat[coords_state[0]-1][coords_state[1]] == 1:
                obstacles += 1000
            if coords_state[0]+1 >= self.size or self.map_mat[coords_state[0]+1][coords_state[1]] == 1:
                obstacles += 100
            if coords_state[1]-1 < 0 or self.map_mat[coords_state[0]][coords_state[1]-1] == 1:
                obstacles += 10
            if coords_state[1]+1 >= self.size or self.map_mat[coords_state[0]][coords_state[1]+1] == 1:
                obstacles += 1
            obstacles = str(obstacles)
            for observation in range(16):
                probability = 1
                for n in range(2):
                    for s in range(2):
                        for w in range(2):
                            for e in range(2):
                                obstacle_check = n*1000 + s*100 + w*10 + e + 1111
                                obstacle_check = str(obstacle_check)
                                for c in range(len(obstacle_check)):
                                    if obstacle_check[c] == obstacles[c]:
                                        probability *= 1 - self.get_error()
                                    else:
                                        probability *= self.get_error()
                b_mat[state][observation] = probability
        self.b_mat = b_mat
        
    def print_b_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        np.set_printoptions(threshold=np.inf)
        for x in range(valid_states):
            print(self.b_mat[x])
