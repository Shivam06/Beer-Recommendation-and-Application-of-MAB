import numpy as np 
import cPickle as pickle
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms.CossineArm import CossineArm
from MAB.Arms.JacArm import JacArm
from MAB.Arms.KNNarm import KNNarm

with open(r"C:\Users\SHIVAM MAHAJAN\Desktop\Github\Recommendation-System\Data\beer_profile_hash.pickle", "rb") as handle:
	hash_beers = pickle.load(handle)
	hash_profiles = pickle.load(handle)
	hash_beers_inv = pickle.load(handle)
	hash_profiles_inv = pickle.load(handle)
	
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


