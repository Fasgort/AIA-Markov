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
        obstacle_rate = 0.05
        r = Robot()
        r.set_obstacle_rate(obstacle_rate)
        self.assertEqual(r.get_obstacle_rate(), obstacle_rate)

    def test_error(self):
        obstacle_rate = 0.05
        r = Robot()
        r.set_error(obstacle_rate)
        self.assertEqual(r.get_error(), obstacle_rate)

    def test_generate_map(self):
        size_x = 5
        r = Robot()
        r.set_size(size_x)
        r.generate_map()
        map_mat = r.get_map()
        self.assertEqual(map_mat.shape, (size_x, size_x))
        spaces_count = np.count_nonzero(map_mat)
        self.assertGreater(spaces_count, 0)

    def test_state_to_coordinates(self):
        size_x = 5
        r = Robot()
        r.set_size(size_x)
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
        r = Robot()
        r.set_size(size_x)
        r.generate_map()
        r.make_a_mat()
        map_mat = r.get_map()
        r.print_map()
        print(r.a_mat)
        states_count = map_mat.size - np.count_nonzero(map_mat)
        a_mat = r.get_a_mat()
        self.assertEqual(a_mat.shape, (states_count, states_count))
        for i in range(size_x):
            if a_mat[i].sum() == 0:
                self.assertEqual(a_mat[:i].sum(), 0,
                                 "A matrix, state {} is unreachable BUT it can reach others".format(i))
            else:
                self.assertAlmostEqual(a_mat[:,i].sum(), 1, 7,
                                       "A matrix, state {} total transition probability: {} != 1".format(i, a_mat[
                                           i].sum()))

    def test_make_pi_v(self):
        size_x = 5
        r = Robot()
        r.set_size(size_x)
        r.generate_map()
        r.make_pi_v()
        map_mat = r.get_map()
        states_count = map_mat.size - np.count_nonzero(map_mat)
        pi_v = r.get_pi_v()
        self.assertEqual(pi_v.shape, (states_count, 1))
        self.assertAlmostEqual(pi_v.sum(), 1, 7, "Pi vector, probability amount: {} != 1".format(pi_v.sum()))

    def test_make_b_mat(self):
        size_x = 5
        r = Robot()
        r.set_size(size_x)
        r.generate_map()
        r.make_b_mat()
        map_mat = r.get_map()
        states_count = map_mat.size - np.count_nonzero(map_mat)
        b_mat = r.get_b_mat()
        self.assertEqual(b_mat.shape, (self._OBSERVATION_COUNT, states_count))
        self.assertGreater(len(b_mat[np.where(b_mat < 0)]), 0, "B matrix contains probabilities < 0".format(i))
        self.assertGreater(len(b_mat[np.where(b_mat > 1)]), 0, "B matrix contains probabilities > 1".format(i))

    def test_estate_transition_probability(self):
        size_x = 3
        r = Robot()
        r.set_size(size_x)
        r.map_mat = np.array([[0,0,1],[1,0,0],[0,0,0]])
        self.assertEqual(r._get_estate_transition_probability(0,0), 0)
        self.assertEqual(r._get_estate_transition_probability(1,0), 1)
        self.assertEqual(r._get_estate_transition_probability(2,0), 0)
        self.assertEqual(r._get_estate_transition_probability(3,0), 0)
        self.assertEqual(r._get_estate_transition_probability(4,0), 0)
        self.assertEqual(r._get_estate_transition_probability(5,0), 0)
        self.assertEqual(r._get_estate_transition_probability(6,0), 0)
        self.assertEqual(r._get_estate_transition_probability(0,1), 1/2)
        self.assertEqual(r._get_estate_transition_probability(1,1), 0)
        self.assertEqual(r._get_estate_transition_probability(2,1), 1/2)
        self.assertEqual(r._get_estate_transition_probability(3,1), 0)
        self.assertEqual(r._get_estate_transition_probability(4,1), 0)
        self.assertEqual(r._get_estate_transition_probability(5,1), 0)
        self.assertEqual(r._get_estate_transition_probability(6,1), 0)
        self.assertEqual(r._get_estate_transition_probability(0,2), 0)
        self.assertEqual(r._get_estate_transition_probability(1,2), 1/3)
        self.assertEqual(r._get_estate_transition_probability(3,2), 1/3)
        self.assertEqual(r._get_estate_transition_probability(4,2), 0)
        self.assertEqual(r._get_estate_transition_probability(5,2), 1/3)
        self.assertEqual(r._get_estate_transition_probability(6,2), 0)
        self.assertEqual(r._get_estate_transition_probability(0,3), 0)
        self.assertEqual(r._get_estate_transition_probability(1,3), 0)
        self.assertEqual(r._get_estate_transition_probability(2,3), 1/2)
        self.assertEqual(r._get_estate_transition_probability(6,3), 1/2)
        self.assertEqual(r._get_estate_transition_probability(0,4), 0)
        self.assertEqual(r._get_estate_transition_probability(1,4), 0)
        self.assertEqual(r._get_estate_transition_probability(2,4), 0)
        self.assertEqual(r._get_estate_transition_probability(3,4), 0)
        self.assertEqual(r._get_estate_transition_probability(4,4), 0)
        self.assertEqual(r._get_estate_transition_probability(5,4), 1)
        self.assertEqual(r._get_estate_transition_probability(6,4), 0)
        self.assertEqual(r._get_estate_transition_probability(0,5), 0)
        self.assertEqual(r._get_estate_transition_probability(1,5), 0)
        self.assertEqual(r._get_estate_transition_probability(2,5), 1/3)
        self.assertEqual(r._get_estate_transition_probability(3,5), 0)
        self.assertEqual(r._get_estate_transition_probability(4,5), 1/3)
        self.assertEqual(r._get_estate_transition_probability(5,5), 0)
        self.assertEqual(r._get_estate_transition_probability(0,6), 0)
        self.assertEqual(r._get_estate_transition_probability(1,6), 0)
        self.assertEqual(r._get_estate_transition_probability(2,6), 0)
        self.assertEqual(r._get_estate_transition_probability(3,6), 1/2)
        self.assertEqual(r._get_estate_transition_probability(4,6), 0)
        self.assertEqual(r._get_estate_transition_probability(5,6), 1/2)
        self.assertEqual(r._get_estate_transition_probability(6,6), 0)

if __name__ == '__main__':
    unittest.main()
