import numpy as np 
import random
import math


def categorical_draw(probs):
	z = random.random()
	cum_prob = 0.0

	for i in range(len(probs)):
		cum_prob += probs[i]
		if cum_prob > z:
			return i

class Softmax:
	def __init__(self, counts, values, temperature):
		self.temperature = temperature
		self.counts = counts
		self.values = values
		self.num_arms = len(counts)

	def initialize(self, n_arms):
		self.counts = [0.0 for a in range(n_arms)]
		self.values = [0.0 for a in range(n_arms)]
		self.num_arms = n_arms

	def select_arm(self):
		z = sum([math.exp(v/self.temperature) for v in self.values])
		probs = [math.exp(v/self.temperature)/z for v in self.values]
		return categorical_draw(probs)

	def update(self, arm, reward):
		self.counts[arm] += 1
		n = self.counts[arm]
		self.values[arm] = float(self.values[arm]*(n-1) + reward)/n


class AnnealingSoftmax:
	def __init__(self, counts, values, arms):
		self.counts = counts
		self.values = values
		self.num_arms = len(arms)

	def initialize(self, n_arms):
		self.counts = [0.0 for a in range(len(arms))]
		self.values = [0.0 for a in range(len(arms))]
		self.num_arms = n_arms

	def select_arm(self):
		t = sum(self.counts)
		temperature = 1/math.log(t + 0.0000001)
		z = sum([math.exp(v/temperature) for v in self.values])
		probs = [math.exp(v/temperature)/z for v in self.values]
		return categorical_draw(probs)

	def update(self, arm, reward):
		self.counts[arm] += 1
		n = self.counts[arm]
		self.values[arm] = float(self.values[arm]*(n-1) + reward)/n

if __name__ == "__main__":
	arms = [0.1, 0.1, 0.1, 0.9, 0.1]
	random.shuffle(arms)
	algo = Softmax([], [], 1, arms)
	algo.initialize(5)
	print algo.select_arm()
