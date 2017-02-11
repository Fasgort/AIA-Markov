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
        # Not implemented
        return self.obstacle_rate

    def set_obstacle_rate(self, obstacle_rate):
        # Not implemented
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
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        if state > valid_states:
            raise Exception
        state_countdown = state
        for x in range(self.map_mat.shape[0]):
            for y in range(self.map_mat.shape[1]):
                if self.map_mat[x][y] == 0:
                    if state_countdown < 1:
                        return x, y
                    else:
                        state_countdown -= 1


    def make_a_mat(self):
        ''' Calculate the state transition probability matrix
        '''
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, valid_states)
        a_mat = np.zeros((shape[0], shape[1]))
        for state1 in range(valid_states):
            for state2 in range(valid_states):
                a_mat[state1][state2] = self._get_estate_transition_probability(state1,state2)
        self.a_mat = a_mat

    def make_pi_v(self):
        ''' Calculate the initial state probability vector
        '''
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        pi_v = np.zeros((valid_states, 1))
        pi_v += 1/valid_states
        self.pi_v=pi_v

    def make_b_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, 16) # From 1111(0000), to 2222(1111), NSWE
        b_mat = np.zeros((shape[0], shape[1]))
        for state in range(valid_states):
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
            observation = 0
            for n in range(2):
                for s in range(2):
                    for w in range(2):
                        for e in range(2):
                            obstacle_check = n*1000 + s*100 + w*10 + e + 1111
                            obstacle_check = str(obstacle_check)
                            probability = 1.0
                            for c in range(len(obstacle_check)):
                                if obstacle_check[c] == obstacles[c]:
                                    probability *= (1 - self.get_error())
                                else:
                                    probability *= self.get_error()
                            b_mat[state][observation] = probability
                            observation += 1
        self.b_mat = b_mat
        
    def print_b_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        np.set_printoptions(threshold=np.inf)
        for x in range(valid_states):
            print(self.b_mat[x])

    def _get_estate_transition_probability(self, state, prev_state):
        ''' Calculate transition probability between states.
        Needs map matrix with paths (self.map_mat)
        Args:
            state (int) Target state identifier
            prev_state (int) Start state identifier
        Returns:
            (float) Probability of transition between start to target states
        '''
        state_pos = self.state_to_coordinates(state)
        prev_state_pos = self.state_to_coordinates(prev_state)

        valid_adjacents = 4
        transition_found = False

        if state_pos != prev_state_pos and (state_pos[0] == prev_state_pos[0] or state_pos[1] == prev_state_pos[1]):
            #N
            if prev_state_pos[0] <= 0 or self.map_mat[prev_state_pos[0] - 1 ,prev_state_pos[1]] == 1:
                valid_adjacents -= 1
            elif state_pos[0] == prev_state_pos[0] - 1:
                transition_found = True
            #S
            if prev_state_pos[0] >= self.map_mat.shape[0] - 1 or self.map_mat[prev_state_pos[0] + 1 ,prev_state_pos[1]] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[0] == prev_state_pos[0] + 1:
                transition_found = True
            #W
            if prev_state_pos[1] <= 0 or self.map_mat[prev_state_pos[0] ,prev_state_pos[1] - 1] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[1] == prev_state_pos[1] - 1:
                transition_found = True
            #E
            if prev_state_pos[1] >= self.map_mat.shape[1] - 1 or self.map_mat[prev_state_pos[0] ,prev_state_pos[1] + 1] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[1] == prev_state_pos[1] + 1:
                transition_found = True
            if transition_found == True:
                return 1/valid_adjacents
        return 0

