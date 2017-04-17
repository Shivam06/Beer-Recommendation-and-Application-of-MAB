import numpy as np 
import pandas as pd
import cPickle as pickle

with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\beer_profile_hash.pickle", "rb") as handle:
	hash_beers = pickle.load(handle)
	hash_profiles = pickle.load(handle)
	hash_beers_inv = pickle.load(handle)
	hash_profiles_inv = pickle.load(handle)

class JacArm:
	def __init__(self):
		with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\jac_arm.pickle", 'rb') as handle:
			self.sim = np.array(pickle.load(handle))

	def recommend(self, input_beer, k = 10):
		self.k = k
		beer_no = hash_beers[input_beer]
		arr = self.sim[beer_no, :]
		top_k_beers = [hash_beers_inv[a] for a in np.argsort(-arr)[1:k+1]]
		#top_k_beers_sno = [a for a in np.argsort(-arr)[1:k+1]]
		return top_k_beers

	def draw(self, input):
		return self.k - input + 1  # Index 0 has rank 1.


if __name__ == "__main__":
	arm = JacArm()
	input_beer = raw_input("Enter a beer you like!")
	arr = arm.recommend(input_beer, 10)
	for i in range(len(arr)):
		print str(i+1) + ") " + arr[i]
	print str(arm.k + 1) + ") " + "None"
	input_rank = raw_input("Enter the Beer you would like to have ?")
	print "Score obtained is " + str(arm.draw(int(input_rank)))