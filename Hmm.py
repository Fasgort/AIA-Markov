#-*-coding:utf-8-*-

import numpy as np

class Hmm:
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

	def make_a_mat(self):
		raise NotImplementedError

	def make_pi_v(self):
		raise NotImplementedError
	
	def make_b_mat(self):
		raise NotImplementedError
