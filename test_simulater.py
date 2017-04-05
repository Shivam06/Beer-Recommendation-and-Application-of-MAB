import numpy as np 
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms.BernauliArm import BernauliArm

def test(algo, arms, num_sims, horizons):
	# We want convergence with probability of best arms, Avg Reward per trial, Avg Cumulative Reward per trial.
	arms_chosen = np.array((num_sims, horizons))

if __name__ == "__main__":
	p_values = [0.1,0.1,0.1,0.9,0.1]
	np.random.shuffle(p_values)
	arms = map(lambda (x): BernauliArm(x), p_values)
	algo = EpsilonGreedy()
