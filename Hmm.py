# -*-coding:utf-8-*-

import numpy as np

''' Hidden Markov model
'''


class Hmm:
    def __init__(self, observations, pi_vector, a_matrix, b_matrix):
        self.a_mat = a_matrix
        self.b_mat = b_matrix
        self.pi_v = pi_vector
        self.observations = observations

    def get_observations(self):
        return self.observations

    def set_observations(self, observations):
        self.observations = observations

    def get_a_mat(self):
        return self.a_mat

    def set_a_mat(self, a_mat):
        self.a_mat = a_mat

    def get_pi_v(self):
        return self.pi_v

    def set_pi_v(self, pi_v):
        self.pi_v = pi_v

    def get_b_mat(self):
        return self.b_mat

    def set_b_mat(self, b_mat):
        self.b_mat = b_mat

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
                    forward_mat[t][s] = self.b_mat[s][observations[t]] * accumulated

        # Returning the most probable state
        state = np.argmax(forward_mat[time - 1])

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
