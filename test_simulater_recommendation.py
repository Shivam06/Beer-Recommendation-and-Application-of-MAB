import numpy as np
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms import KNNarm, CossineArm, JacArm
import cPickle as pickle

def test(algo, arms, num_sims, horizons):
	arm_chosen = np.zeros((num_sims, horizons))
	reward_obtained = np.zeros((num_sims, horizons))
	cum_reward = np.zeros((num_sims, horizons))

	for i in range(num_sims):
		algo = algo
		algo.initialize(len(arms))
		sum = 0
		for j in range(horizons):
			arm_idx = algo.select_arm()
			arm_chosen = arms[arm_idx]
			reward = arm_chosen.dra

if __name__ == "__main__":
	arms = [KNNarm.KNNarm(), CossineArm.CossineArm(), JacArm.JacArm()]
	algo = EpsilonGreedy([], [], 0.1)
