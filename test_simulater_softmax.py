import numpy as np 
from MAB.Algo.Softmax import Softmax
from MAB.Arms.BernauliArm import BernauliArm
import cPickle as pickle

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
	best_arm = np.argmax(np.array(p_values))
	arms = map(lambda x: BernauliArm(x), p_values)	

	with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\convergence_softmax.pickle", "wb") as handle:
		for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
			algo = Softmax([], [], t)
			ans = test(algo, arms, 5000, 250, best_arm)
			pickle.dump(ans, handle)