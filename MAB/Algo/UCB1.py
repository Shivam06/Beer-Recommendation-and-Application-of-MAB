import numpy as np 
import math

class UCB1:	
	def __init__(self, counts, values, arms):
		self.counts = counts
		self.values = values
		self.n_arms = len(arms)

	def initialize(self, n_arms):
		self.counts = [0.0 for n in range(n_arms)]
		self.values = [0.0 for n in range(n_arms)]
		self.n_arms = n_arms

	def select_arm(self):
		for arm_idx in range(self.n_arms):
			if self.counts[arm_idx] == 0:
				return arm_idx

		ucb_values = [0.0 for i in range(self.n_arms)]
		total_counts = sum(self.counts)

		for arm_idx in range(self.n_arms):
			bonus = math.sqrt(2*math.log(total_counts))/ float(self.counts[arm_idx])
			ucb_values = values[arm_idx] + bonus
			
		return np.argmax(np.array(ucb_values))

	def update(self, arm, reward):
		self.counts[arm] += 1
		n = self.counts[arm]
		self.values[arm] = float((n-1)*self.values[arm] + reward)/n
