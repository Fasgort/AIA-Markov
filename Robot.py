# -*-coding:utf-8-*-

import random

import numpy as np
from PIL import Image

from Hmm import Hmm


class Robot(Hmm):
    _OBSERVATIONS_COUNT = 16

    def __init__(self, map_size=15, obstacle_rate=0.3, observation_error=0.05):
        self.size = map_size
        self.obstacle_rate = obstacle_rate
        self.error = observation_error
        self.pi_v = None
        self.a_mat = None
        self.b_mat = None
        self.observations = self._make_robot_observations()

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def get_obstacle_rate(self):
        return self.obstacle_rate

    def set_obstacle_rate(self, obstacle_rate):
        self.obstacle_rate = obstacle_rate

    def get_error(self):
        return self.error

    def set_error(self, error):
        self.error = error

    def set_observations(self, observations):
        pass

    def get_map(self):
        return self.map_mat

    def generate_map(self):
        """ Build map

        :return: None
        .. warning:: Set obstacle rate before
        """
        self.map_mat = np.zeros((self.size, self.size), dtype=int)
        for x in range(0, self.size):
            for y in range(0, self.size):
                if random.random() <= self.obstacle_rate:
                    self.map_mat[x][y] = 1
                else:
                    self.map_mat[x][y] = 0

    def print_map(self):
        for x in range(0, self.size):
            print(self.map_mat[x])

    def get_valid_states(self):
        """ Returns valid states count

        :return: current amount of system state
        .. warning:: Generate map before using it
        """
        return self.map_mat.size - np.count_nonzero(self.map_mat)

    def coordinates_to_state(self, position):
        res = 0
        for x in range(self.map_mat.shape[0]):
            for y in range(self.map_mat.shape[1]):
                if self.map_mat[x][y] == 0:
                    if x == position[x] and y == position[y]:
                        return res
                    else:
                        res += 1

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
        """ Calculate the state transition probability matrix

        .. warning:: Generate map before using it
        """
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, valid_states)
        a_mat = np.zeros((shape[0], shape[1]))
        for state2 in range(valid_states):
            for state1 in range(valid_states):
                a_mat[state1][state2] = self._get_state_transition_probability(state1, state2)
            # The correction done below is to eliminate rounding error
            a_mat[valid_states - 1][state2] -= (a_mat[:, state2].sum() - 1.0)
            if a_mat[:, state2].sum() != 1.0:
                raise Exception(
                    "Unable to generate A matrix: from_state:{} state_transition={} accumulated_probability={}".format(
                        state2, a_mat[:, state2], a_mat[:, state2].sum()))
        self.a_mat = a_mat

    def make_pi_v(self):
        """ Calculate the initial state probability vector

        .. warning:: Generate map before using it
        """
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        pi_v = np.zeros((valid_states, 1))
        pi_v += 1.0 / valid_states
        # The correction done below is to eliminate rounding error
        for atepmt in range(5):
            if pi_v.sum() != 1.0:
                pi_v[valid_states - 1] -= (pi_v.sum() - 1.0)
            if pi_v.sum() == 1.0:
                break
        if pi_v.sum() != 1.0:
            raise Exception("Unable to generate Pi vector: pi_v={} accumulated probability={}".format(pi_v, pi_v.sum()))
        self.pi_v = pi_v

    def make_b_mat(self):
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (valid_states, 16)  # From 1111(0000), to 2222(1111), NSWE
        b_mat = np.zeros((shape[0], shape[1]))
        for state in range(valid_states):
            coords_state = self.state_to_coordinates(state)
            obstacles = 1111
            if coords_state[0] - 1 < 0 or self.map_mat[coords_state[0] - 1][coords_state[1]] == 1:
                obstacles += 1000
            if coords_state[0] + 1 >= self.size or self.map_mat[coords_state[0] + 1][coords_state[1]] == 1:
                obstacles += 100
            if coords_state[1] - 1 < 0 or self.map_mat[coords_state[0]][coords_state[1] - 1] == 1:
                obstacles += 10
            if coords_state[1] + 1 >= self.size or self.map_mat[coords_state[0]][coords_state[1] + 1] == 1:
                obstacles += 1
            obstacles = str(obstacles)
            observation = 0
            for n in range(2):
                for s in range(2):
                    for w in range(2):
                        for e in range(2):
                            obstacle_check = n * 1000 + s * 100 + w * 10 + e + 1111
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

    def _get_state_transition_probability(self, state, prev_state):
        """ Calculate transition probability between states.
        Needs map matrix with paths (self.map_mat)
        Args:
            state (int) Target state identifier
            prev_state (int) Start state identifier
        Returns:
            (float) Probability of transition between start to target states
        """
        state_pos = self.state_to_coordinates(state)
        prev_state_pos = self.state_to_coordinates(prev_state)

        valid_adjacents = 4
        transition_found = False

        if state_pos != prev_state_pos and (state_pos[0] == prev_state_pos[0] or state_pos[1] == prev_state_pos[1]):
            # N
            if prev_state_pos[0] <= 0 or self.map_mat[prev_state_pos[0] - 1, prev_state_pos[1]] == 1:
                valid_adjacents -= 1
            elif state_pos[0] == prev_state_pos[0] - 1:
                transition_found = True
            # S
            if prev_state_pos[0] >= self.map_mat.shape[0] - 1 or self.map_mat[
                        prev_state_pos[0] + 1, prev_state_pos[1]] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[0] == prev_state_pos[0] + 1:
                transition_found = True
            # W
            if prev_state_pos[1] <= 0 or self.map_mat[prev_state_pos[0], prev_state_pos[1] - 1] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[1] == prev_state_pos[1] - 1:
                transition_found = True
            # E
            if prev_state_pos[1] >= self.map_mat.shape[1] - 1 or self.map_mat[
                prev_state_pos[0], prev_state_pos[1] + 1] == 1:
                valid_adjacents -= 1
            elif not transition_found and state_pos[1] == prev_state_pos[1] + 1:
                transition_found = True
            if transition_found:
                return 1 / valid_adjacents
        return 0

    def _make_robot_observations(self):
        obs_list = []
        for o in ['{0:04b}'.format(o) for o in range(16)]:
            obs = ''
            if o[0] == '1': obs = 'N'
            if o[1] == '1': obs += 'S'
            if o[2] == '1': obs += 'W'
            if o[3] == '1': obs += 'E'
            obs_list.append(obs)
        return obs_list

    def make_map_image(self):
        image_size = 500
        unit_space_dim = int(image_size / self.get_size())
        res_image_array = np.empty((self.map_mat.shape[0] * unit_space_dim, self.map_mat.shape[1] * unit_space_dim, 3),
                                   dtype=np.uint8)
        for i in range(self.map_mat.shape[0]):
            space_i = i * unit_space_dim
            for j in range(self.map_mat.shape[1]):
                space_j = j * unit_space_dim
                for row in range(unit_space_dim):
                    value_color = 255 if self.map_mat[i, j] == 0 else 0
                    res_image_array[space_i + row, space_j:space_j + unit_space_dim, :] = value_color
        self.map_image = Image.fromarray(res_image_array)

    def save_map_image(self, output_path):
        if not hasattr(self, 'map_image') or self.map_image:
            self.make_map_image()
        return self.map_image.save(output_path, 'PNG')

    def display_map(self):
        if not hasattr(self, 'map_image') or self.map_image:
            self.make_map_image()
        self.map_image.show()
