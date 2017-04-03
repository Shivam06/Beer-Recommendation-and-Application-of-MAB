import numpy as np 
import cPickle as pickle 
from CossineArm import CossineArm
from JacArm import JacArm
from KNNarm import KNNarm
with open(r"C:\Users\SHIVAM MAHAJAN\Desktop\Github\Recommendation-System\Data\beer_profile_hash.pickle", "rb") as handle:
	hash_beers = pickle.load(handle)
	hash_profiles = pickle.load(handle)
	hash_beers_inv = pickle.load(handle)
	hash_profiles_inv = pickle.load(handle)

class EpsilonGreedy:
	def __init__(self, counts, values, e):
		self.epsilon = e
		self.counts = counts
		self.values = values	
		self.n_arms = len(counts)

	def initialize(self, arms):
		self.n_arms = arms
		self.counts = [0 for i in range(arms)]
		self.values = [0.0 for i in range(arms)]

	def select_arm(self):                        # return index of arm chosen
 		if np.random.random() > self.epsilon:
			return np.argmax(self.values)
		else:
			return np.random.randint(self.n_arms)

	def update(self, chosen_idx, val):
		self.counts[chosen_idx] += 1
		n = self.counts[chosen_idx]
		self.values[chosen_idx] = self.values[chosen_idx]*float(n-1)/float(n) + float(val)/float(n)
		return


def simulator():
	algo = EpsilonGreedy([], [], 0.2)
	algo.initialize(3)
	arms = [CossineArm(), JacArm(), KNNarm()]
	ans = 1
	beer_input = raw_input("Chose your favourite beer!")

	while (True):
		arm_idx = algo.select_arm()
		display_arm(arm_idx)
		arr = arms[arm_idx].recommend(beer_input, 10)
		for i in range(len(arr)):
			print str(i+1) + ") " + arr[i]
		response = raw_input("Would you like to have any of these ? Y/N.")
		if response == 'Y' or response == 'y':
			input_rank = raw_input("Enter the corresponding number")
			beer_input = hash_beers_inv[int(input_rank)]
			score = arms[arm_idx].draw(int(input_rank))
		else:
			score = 0
			beer_input = raw_input("Chose your favourite beer!")
		algo.update(arm_idx, score)
		print algo.values
		print algo.counts	

def display_arm(idx):
	if idx == 0:
		print "Cossine"
	elif idx == 1:
		print "Jaccardian"
	elif idx == 2:
		print "KNN"



if __name__ == "__main__":
	simulator()


