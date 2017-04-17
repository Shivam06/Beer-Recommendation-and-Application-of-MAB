from flask import Flask, render_template, request, url_for
import numpy as np 
import cPickle as pickle
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms.CossineArm import CossineArm
from MAB.Arms.JacArm import JacArm
from MAB.Arms.KNNarm import KNNarm

with open(r"E:\SHIVAM MAHAJAN\Desktop\Github\Beer-Recommendation-and-Application-of-MAB\Data\beer_profile_hash.pickle", "rb") as handle:
	hash_beers = pickle.load(handle)
	hash_profiles = pickle.load(handle)
	hash_beers_inv = pickle.load(handle)
	hash_profiles_inv = pickle.load(handle)

def display_arm(idx):
	if idx == 0:
		return "Jaccardian"
	elif idx == 1:
		return "Cossine"	
	elif idx == 2:
		return "KNN"

def find_rank(arr, input):
	for i in range(len(arr)):
		if arr[i] == input:
			return i+1
	return 0

algo = EpsilonGreedy([], [], 0.2)
algo.initialize(3)
arms = [JacArm(), CossineArm(), KNNarm()]
arr = []
chosen_arm = 0	

app = Flask(__name__)
wsgi_app = app.wsgi_app

@app.route('/', methods = ['GET', 'POST'])
def index():	
	if request.method == 'GET':
		global flag
		global chosen_arm
		global arr
		flag = 0
		css_url = url_for('static', filename='main.css')
		return render_template('index.html', css_url = css_url)

	elif request.method == 'POST':
		beer_input = request.form['beer']
		if flag == 0:
			chosen_arm = algo.select_arm()
			try:
				arr = arms[chosen_arm].recommend(beer_input, 10)
			except:
				return "No such Beer Available."
			flag = 1
		else:
			rank = find_rank(arr, beer_input) # Define the function
			if rank:
				score = arms[chosen_arm].draw(rank)
			else:
				score = 0
			algo.update(chosen_arm, score)
			chosen_arm = algo.select_arm()
			try:
				arr = arms[chosen_arm].recommend(beer_input, 10)
			except:
				return "No such Beer Available."
		return render_template('index.html', rec_beers = arr, arm = display_arm(chosen_arm), values = algo.values)

		

if __name__ == "__main__":
	app.run()	