import numpy as np
from MAB.Algo.UCB1 import UCB1
from MAB.Arms.BernauliArm import BernauliArm
import cPickle as pickle
import os

filedir = os.path.dirname(os.path.realpath('__file__'))+'/Data/'

def test(algo, arms, num_sims, horizons, best_arm):
	# arm_idx, reward, cum_reward
	arm_chosen = np.zeros((num_sims, horizons))
	reward_obtained = np.zeros((num_sims, horizons))
	cum_reward = np.zeros((num_sims, horizons))

	for i in range(num_sims):
		algo.initialize(len(arms))
		sum = 0
		for j in range(horizons):
			arm_idx = algo.select_arm()
			arm_chosen[i,j] = arm_idx
			reward = arms[arm_idx].draw()
			algo.update(arm_idx, reward)
			reward_obtained[i,j] = reward
			sum += reward
			cum_reward[i,j] = sum

	return [arm_chosen, reward_obtained, cum_reward, best_arm]


if __name__ == "__main__":
	p_values = [0.1, 0.1, 0.1, 0.9, 0.1]
	np.random.shuffle(p_values)
	print p_values
	best_arm = np.argmax(np.array(p_values))
	print best_arm
	arms = map(lambda x: BernauliArm(x), p_values)

	with open(filedir+'convergence_ucb1.pickle', "wb") as handle:
		algo = UCB1([], [])
		ans = test(algo, arms, 5000, 250, best_arm)
		pickle.dump(ans, handle)