import numpy as np
import cPickle as pickle
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms.CossineArm import CossineArm
from MAB.Arms.JacArm import JacArm
from MAB.Arms.KNNarm import KNNarm
import os


with open(filedir+'beer_profile_hash.pickle', "rb") as handle:
	hash_beers = pickle.load(handle)
	hash_profiles = pickle.load(handle)
	hash_beers_inv = pickle.load(handle)
	hash_profiles_inv = pickle.load(handle)

def simulator():
	favs = []
	algo = EpsilonGreedy([], [], 0.2)
	algo.initialize(3)
	arms = [CossineArm(), JacArm(), KNNarm()]
	ans = 1
	count = 1
	beer_input = raw_input("Chose your favourite beer!")
	favs.append(beer_input)

	while (True):
		arm_idx = algo.select_arm()
		print "Iteration number " + str(count)
		count += 1
		display_arm(arm_idx)
		arr = arms[arm_idx].recommend(favs, 10)
		print "Recommendations for you are"
		for i in range(len(arr)):
			print str(i+1) + ") " + arr[i]
		response = raw_input("Would you like to have any of these ? Y/N.")
		if response == 'Y' or response == 'y':
			input_rank = raw_input("Enter the corresponding number")
			beer_input = arr[int(input_rank)-1]
			favs.append(beer_input)
			score = arms[arm_idx].draw(int(input_rank))
		else:
			score = 0
			beer_input = raw_input("Chose your favourite beer!")
			favs.append(beer_input)
		print "Your Profile :" + str(favs)
		algo.update(arm_idx, score)
		print "Scores of different arms is" + str(algo.values)
		print "Counts of differnt arms is " + str(algo.counts)
		print "\n"

def display_arm(idx):
	if idx == 0:
		print "Arm Chosen is Cossine"
	elif idx == 1:
		print "Arm Chosen is Jaccardian"
	elif idx == 2:
		print "Arm Chosen is KNN"

if __name__ == "__main__":
	simulator()


