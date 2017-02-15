import unittest
import numpy as np
from Robot import Robot


class MyTestCase(unittest.TestCase):
    _OBSERVATION_COUNT = 16

    def test_size(self):
        size_x = 5
        r = Robot()
        r.set_size(size_x)
        self.assertEqual(r.get_size(), size_x)

    def test_obstacle_rate(self):
        obstacle_rate = 0.25
        r = Robot()
        r.set_obstacle_rate(obstacle_rate)
        self.assertEqual(r.get_obstacle_rate(), obstacle_rate)

    def test_error(self):
        observation_error = 0.05
        r = Robot()
        r.set_error(observation_error)
        self.assertEqual(r.get_error(), observation_error)

    def test_generate_map(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        map_mat = r.get_map()
        self.assertEqual(map_mat.shape, (size_x, size_x))
        spaces_count = np.count_nonzero(map_mat)
        self.assertGreater(spaces_count, 0)

    def test_generate_map_obstacle_rate(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        map_mat = r.get_map()
        self.assertEqual(map_mat.shape, (size_x, size_x))
        spaces_count = np.count_nonzero(map_mat)
        self.assertGreater(spaces_count, 0)

    def test_state_to_coordinates(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        map_mat = r.get_map()
        current_state = 0
        for i in range(size_x):
            for j in range(size_x):
                if map_mat[i][j] == 0:
                    self.assertEqual(r.state_to_coordinates(current_state), (i, j))
                    current_state += 1

    def test_make_a_mat(self):
        size_x = 3
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_a_mat()
        map_mat = r.get_map()
        states_count = map_mat.size - np.count_nonzero(map_mat)
        a_mat = r.get_a_mat()
        self.assertEqual(a_mat.shape, (states_count, states_count))
        for i in range(size_x):
            if a_mat[i].sum() == 0:
                self.assertEqual(a_mat[:i].sum(), 0,
                                 "A matrix, state {} is unreachable BUT it can reach others".format(i))
            else:
                self.assertAlmostEqual(a_mat[:, i].sum(), 1, 7,
                                       "A matrix, state {} total transition probability: {} != 1".format(i, a_mat[
                                           i].sum()))

    def test_make_pi_v(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_pi_v()
        map_mat = r.get_map()
        states_count = map_mat.size - np.count_nonzero(map_mat)
        pi_v = r.get_pi_v()
        self.assertEqual(pi_v.shape, (states_count, 1))
        self.assertAlmostEqual(pi_v.sum(), 1, 7, "Pi vector, probability amount: {} != 1".format(pi_v.sum()))

    def test_make_b_mat(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_error(0.05)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_b_mat()
        map_mat = r.get_map()
        states_count = map_mat.size - np.count_nonzero(map_mat)
        b_mat = r.get_b_mat()
        self.assertEqual(b_mat.shape, (states_count, self._OBSERVATION_COUNT))
        self.assertEqual(len(b_mat[np.where(b_mat < 0)]), 0, "B matrix contains probabilities < 0")
        self.assertEqual(len(b_mat[np.where(b_mat > 1)]), 0, "B matrix contains probabilities > 1")

    def test_estate_transition_probability(self):
        size_x = 3
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.map_mat = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(r._get_state_transition_probability(0, 0), 0)
        self.assertEqual(r._get_state_transition_probability(1, 0), 1)
        self.assertEqual(r._get_state_transition_probability(2, 0), 0)
        self.assertEqual(r._get_state_transition_probability(3, 0), 0)
        self.assertEqual(r._get_state_transition_probability(4, 0), 0)
        self.assertEqual(r._get_state_transition_probability(5, 0), 0)
        self.assertEqual(r._get_state_transition_probability(6, 0), 0)
        self.assertEqual(r._get_state_transition_probability(0, 1), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(1, 1), 0)
        self.assertEqual(r._get_state_transition_probability(2, 1), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(3, 1), 0)
        self.assertEqual(r._get_state_transition_probability(4, 1), 0)
        self.assertEqual(r._get_state_transition_probability(5, 1), 0)
        self.assertEqual(r._get_state_transition_probability(6, 1), 0)
        self.assertEqual(r._get_state_transition_probability(0, 2), 0)
        self.assertEqual(r._get_state_transition_probability(1, 2), 1 / 3)
        self.assertEqual(r._get_state_transition_probability(3, 2), 1 / 3)
        self.assertEqual(r._get_state_transition_probability(4, 2), 0)
        self.assertEqual(r._get_state_transition_probability(5, 2), 1 / 3)
        self.assertEqual(r._get_state_transition_probability(6, 2), 0)
        self.assertEqual(r._get_state_transition_probability(0, 3), 0)
        self.assertEqual(r._get_state_transition_probability(1, 3), 0)
        self.assertEqual(r._get_state_transition_probability(2, 3), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(6, 3), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(0, 4), 0)
        self.assertEqual(r._get_state_transition_probability(1, 4), 0)
        self.assertEqual(r._get_state_transition_probability(2, 4), 0)
        self.assertEqual(r._get_state_transition_probability(3, 4), 0)
        self.assertEqual(r._get_state_transition_probability(4, 4), 0)
        self.assertEqual(r._get_state_transition_probability(5, 4), 1)
        self.assertEqual(r._get_state_transition_probability(6, 4), 0)
        self.assertEqual(r._get_state_transition_probability(0, 5), 0)
        self.assertEqual(r._get_state_transition_probability(1, 5), 0)
        self.assertEqual(r._get_state_transition_probability(2, 5), 1 / 3)
        self.assertEqual(r._get_state_transition_probability(3, 5), 0)
        self.assertEqual(r._get_state_transition_probability(4, 5), 1 / 3)
        self.assertEqual(r._get_state_transition_probability(5, 5), 0)
        self.assertEqual(r._get_state_transition_probability(0, 6), 0)
        self.assertEqual(r._get_state_transition_probability(1, 6), 0)
        self.assertEqual(r._get_state_transition_probability(2, 6), 0)
        self.assertEqual(r._get_state_transition_probability(3, 6), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(4, 6), 0)
        self.assertEqual(r._get_state_transition_probability(5, 6), 1 / 2)
        self.assertEqual(r._get_state_transition_probability(6, 6), 0)

    def test_generate_sample(self):
        sample_size = 10
        obstacle_rate = 0.25
        size_x = 5
        r = Robot()
        r.set_size(size_x)
        r.set_error(0.05)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        sam_s, sam_o = r.generate_sample(sample_size)
        self.assertEqual(sam_s.size, sample_size)
        self.assertEqual(sam_o.size, sample_size)
        states_count = r.pi_v.size
        self.assertTrue((0 <= sam_s).all() and (sam_s < states_count).all(),
                        "States sample contains unknown states: {}".format(sam_s))
        self.assertTrue((0 <= sam_o).all() and (sam_o < self._OBSERVATION_COUNT).all(),
                        "Observations sample contains unknown observations: {}".format(sam_o))
        for i in range(sample_size - 1):
            state_transition_probability = r._get_state_transition_probability(sam_s[i + 1], sam_s[i])
            self.assertTrue(0 < state_transition_probability <= 1)

    def test_generate_sample_variable_sample_size(self):
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_error(0.05)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        for sample_size in range(1, 10):
            sam_s, sam_o = r.generate_sample(sample_size)
            self.assertEqual(sam_s.size, sample_size)
            self.assertEqual(sam_o.size, sample_size)
            states_count = r.pi_v.size
            self.assertTrue((0 <= sam_s).all() and (sam_s < states_count).all(),
                            "States sample contains unknown states: {}".format(sam_s))
            self.assertTrue((0 <= sam_o).all() and (sam_o < self._OBSERVATION_COUNT).all(),
                            "Observations sample contains unknown observations: {}".format(sam_o))
            for i in range(sample_size - 1):
                state_transition_probability = r._get_state_transition_probability(sam_s[i + 1], sam_s[i])
                self.assertTrue(0 < state_transition_probability <= 1)

    def test_viterbi(self):
        sample_size = 3
        size_x = 5
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_error(0.05)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        sam_s, sam_o = r.generate_sample(sample_size)
        viterbi_s_seq = r.viterbi(sam_o)
        self.assertEqual(viterbi_s_seq.size, sample_size)
        states_count = r.pi_v.size
        self.assertTrue((0 <= viterbi_s_seq).all() and (viterbi_s_seq < states_count).all(),
                        "States sequence contains unknown states: {}".format(sam_s))
        for i in range(sample_size - 1):
            state_transition_probability = r._get_state_transition_probability(viterbi_s_seq[i + 1], viterbi_s_seq[i])
            self.assertTrue(0 < state_transition_probability <= 1)

    def test_viterbi_variable_sample_size(self):
        size_x = 10
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_error(0.05)
        r.set_obstacle_rate(obstacle_rate)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        for sample_size in range(1, 10):
            sam_s, sam_o = r.generate_sample(sample_size)
            viterbi_s_seq = r.viterbi(sam_o)
            self.assertEqual(viterbi_s_seq.size, sample_size)
            states_count = r.pi_v.size
            self.assertTrue((0 <= viterbi_s_seq).all() and (viterbi_s_seq < states_count).all(),
                            "States sequence contains unknown states: {}".format(sam_s))
            for i in range(sample_size - 1):
                state_transition_probability = r._get_state_transition_probability(viterbi_s_seq[i + 1],
                                                                                   viterbi_s_seq[i])
                self.assertTrue(0 < state_transition_probability <= 1)

    def test_forward_error(self):
        sample_size = 10
        obstacle_rate = 0.25
        size_x = 10
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.set_error(0.05)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        sam_s, sam_o = r.generate_sample(sample_size)
        forward_err = r.forward_error(sam_s[0], sam_s[3])
        self.assertTrue(forward_err >= 0)

    def test_viterbi_error(self):
        sample_size = 10
        size_x = 10
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.set_error(0.05)
        r.generate_map()
        r.make_a_mat()
        r.make_pi_v()
        r.make_b_mat()
        sam_s, sam_o = r.generate_sample(sample_size)
        viterbi_s_seq = r.viterbi(sam_o)
        viterbi_err = r.viterbi_error(sam_s, viterbi_s_seq)
        self.assertTrue(0.0 <= viterbi_err <= 1.0,
                        "Viterbi estimated sequence error must be in [0,1]: viterbi_err={}".format(viterbi_err))

    def test_make_map_image(self):
        size_x = 20
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.set_error(0.05)
        r.generate_map()
        r.make_map_image()
        self.assertTrue(hasattr(r, 'map_image'))

    def test_display_map(self):
        size_x = 20
        obstacle_rate = 0.25
        r = Robot()
        r.set_size(size_x)
        r.set_obstacle_rate(obstacle_rate)
        r.set_error(0.05)
        r.generate_map()
        r.make_map_image()
        r.display_map()


if __name__ == '__main__':
    unittest.main()
