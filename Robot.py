# -*-coding:utf-8-*-

import random

import PIL
import numpy as np
from PIL import Image

from Hmm import Hmm


class Robot(Hmm):
    _OBSERVATIONS_COUNT = 16

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

    def get_error(self):
        return self.error

    def set_error(self, error):
        self.error = error

    def get_valid_states(self):
        """ Returns valid states count

        :return: current amount of system state
        .. warning:: Generate map before using it
        """
        return self.map_mat.size - np.count_nonzero(self.map_mat)

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
        if pi_v.sum() != 1.0:
            pi_v[valid_states - 1] -= (pi_v.sum() - 1.0)
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

    def generate_sample(self, size):
        """ Generates a sequence of states and its observations for the present system

        :param size: Length of string of states
        :return: Tuple with string of states and observations

        .. warning:: size must be greater than 0
            Make A, B and Pi before using it
        """
        if size is None or size < 1:
            raise ValueError("size must be greater than 0")
        valid_states = self.pi_v.shape[0]
        sample_s = np.empty(size, int)
        sample_o = np.empty(size, int)
        np.put(sample_s, 0, np.random.choice(
            valid_states,
            1,
            p=np.transpose(self.pi_v[:, 0])))
        np.put(sample_o, 0, np.random.choice(
            self._OBSERVATIONS_COUNT,
            1,
            p=np.transpose(self.b_mat[sample_s[0]])))
        for i in range(1, size):
            np.put(sample_s, i, np.random.choice(
                valid_states,
                1,
                p=np.transpose(self.a_mat[:, sample_s[i - 1]])))
            np.put(sample_o, i, np.random.choice(
                self._OBSERVATIONS_COUNT,
                1,
                p=np.transpose(self.b_mat[sample_s[i]])))
        return sample_s, sample_o

    # observations must give values between 0 (no obstacles) and 15 (NWSE obstacles)
    def forward(self, observations):
        # Exceptions for senseless arguments
        if observations is None:
            raise Exception

        # Generate matrix if they aren't built yet
        if self.a_mat is None:
            self.make_a_mat
        if self.pi_v is None:
            self.make_pi_v
        if self.b_mat is None:
            self.b_mat

        # Initialization
        time = len(observations)
        valid_states = self.map_mat.size - np.count_nonzero(self.map_mat)
        shape = (time, valid_states)
        forward_mat = np.zeros((shape[0], shape[1]))

        # Step 1
        for s in range(valid_states):
            forward_mat[0][s] = self.b_mat[s][observations[0]] * self.pi_v[s]

        # Next steps
        if time >= 1:
            for t in range(1, time):
                for s in range(valid_states):
                    accumulated = 0.0
                    for ss in range(valid_states):
                        accumulated += (self.a_mat[s][ss] * forward_mat[t - 1][ss])
                    forward_mat[t][s] = self.b_mat[s][observations[t - 1]] * accumulated
                               
        # Returning the most probable state
        state = max(forward_mat[time-1])

        return state

    def viterbi(self, observations):
        """ Implements Viterbi algorithm

        Returns the most probable sequence of system states from a system observation sequence

        :param observations: system observations sequence
        :return: system state estimated sequence
        """
        if (observations < 0).all() or (observations >= self._OBSERVATIONS_COUNT).all():
            raise ValueError("Given observation sequence contains invalid observation: {}".format(observations))
        time = observations.size - 1
        nu, pr = self._viterbi_recursion(observations, time)
        s_seq = np.array([np.argmax(nu)])
        for t in range(time, 0, -1):
            s_seq = np.insert(s_seq, 0, pr[t, s_seq[0]])
        return s_seq

    def _viterbi_recursion(self, observations, time):
        valid_states = self.b_mat.shape[0]
        nu = np.empty(valid_states, dtype=float)

        if time == 0:
            pr = np.empty((observations.size, valid_states))
            for s in range(valid_states):
                nu[s] = self.b_mat[s, observations[time]] * self.pi_v[s]
                pr[time, s] = -1
        else:
            prev_nu, pr = self._viterbi_recursion(observations, time - 1)
            for j in range(valid_states):
                tran_prob = np.empty(valid_states)
                for i in range(valid_states):
                    tran_prob[i] = self.a_mat[i, j] * prev_nu[i]
                nu[j] = self.b_mat[j, observations[time]] * np.amax(tran_prob)
                pr[time, j] = np.argmax(tran_prob)
        return nu, pr

    def forward_error(self, state, estimated_state):
        """ Calculates error in forward estimated state

        :param state: original system state
        :param estimated_state: estimated state
        :return: manhattan distance between original and estimated state
        """
        if state < 0 or state > self.get_valid_states():
            raise ValueError(
                "given state is not valid, must be in [0,{}]: state={}".format(self.get_valid_states(), state))
        if estimated_state < 0 or estimated_state > self.get_valid_states():
            raise ValueError(
                "given estimated state is not valid, must be in [0,{}]: estimated_state={}".format(
                    self.get_valid_states(), estimated_state))
        s_coordinates = self.state_to_coordinates(state)
        estimated_s_coordinates = self.state_to_coordinates(estimated_state)
        return abs(s_coordinates[0] - estimated_s_coordinates[0]) + abs(s_coordinates[1] - estimated_s_coordinates[1])

    @staticmethod
    def viterbi_error(state_sequence, estimated_state_sequence):
        """ Calculates error in viterbi estimated sequence

        :param state_sequence: original system state sequence
        :param estimated_state_sequence: estimated state sequence
        :return: matching rate between original and estimated state sequences
        """
        if state_sequence.size <= 0:
            return 1
        elif state_sequence.size != estimated_state_sequence.size:
            return 1
        matches = np.sum(state_sequence == estimated_state_sequence)
        return (state_sequence.size - matches) / state_sequence.size

    def make_map_image(self):
        image_size = 500
        unit_space_dim = int(image_size/self.get_size())
        res_image_array = np.empty((self.map_mat.shape[0] * unit_space_dim, self.map_mat.shape[1] * unit_space_dim, 3), dtype=np.uint8)
        for i in range(self.map_mat.shape[0]):
            space_i = i * unit_space_dim
            for j in range(self.map_mat.shape[1]):
                space_j = j * unit_space_dim
                for row in range(unit_space_dim):
                    value_color = 255 if self.map_mat[i, j] == 0 else 0
                    res_image_array[space_i + row, space_j:space_j + unit_space_dim, :] = value_color
        self.map_image = Image.fromarray(res_image_array)

    def display_map(self):
        if not hasattr(self, 'map_image') or self.map_image:
            self.make_map_image()
        self.map_image.show()
