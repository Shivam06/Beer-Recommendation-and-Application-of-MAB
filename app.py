from flask import Flask, render_template, request, url_for
import numpy as np
import cPickle as pickle
from MAB.Algo.EpsilonGreedy import EpsilonGreedy
from MAB.Arms.CossineArm import CossineArm
from MAB.Arms.JacArm import JacArm
from MAB.Arms.KNNarm import KNNarm

with open('Data/beer_profile_hash.pickle', "rb") as handle:
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
favs = []
chosen_arm = 0


class person:
	def __init__(self):
		self.arr = []
		self.favs = []
		self.flag = 0
shivam = person()
rakshit = person()
rahul = person()

my_dict = {}
my_dict["shivam"] = shivam
my_dict["rakshit"] = rakshit
my_dict["rahul"] = rahul

app = Flask(__name__)
wsgi_app = app.wsgi_app
# After logging in. After registering I will add the user in my_dict
@app.route('/user/<username>', methods = ['GET', 'POST'])
def recommend(username):
	if request.method == 'GET':
		user = my_dict[username]
		if user.flag == 0:
			global chosen_arm
			css_url = url_for('static', filename='main.css')
			return render_template('index.html', user = username, css_url = css_url)
		else:
			css_url = url_for('static', filename='combined.css')
			return render_template('index.html', user = username, rec_beers = user.arr,
					values = algo.values, favs = user.favs, css_url = css_url, arms = arms, arm = arms[chosen_arm].name)

	elif request.method == 'POST':
		user = my_dict[username]
		css_url = url_for('static', filename='combined.css')
		beer_input = request.form['beer']
		if beer_input not in user.favs:
			user.favs.append(beer_input)
		if user.flag == 0:
			chosen_arm = algo.select_arm()
			try:
				user.arr = arms[chosen_arm].recommend(user.favs, 10)
			except:
				return "No such Beer Available."
			user.flag = 1
		else:
			rank = find_rank(user.arr, beer_input) # Define the function
			if rank:
				score = arms[chosen_arm].draw(rank)
			else:
				score = 0
			algo.update(chosen_arm, score)
			arms[chosen_arm].count = algo.counts[chosen_arm]
			arms[chosen_arm].value = algo.values[chosen_arm]
			chosen_arm = algo.select_arm()
			try:
				user.arr = arms[chosen_arm].recommend(user.favs, 10)
			except:
				return "No such Beer Available."
		return render_template('index.html', user = username, rec_beers = user.arr, arms = arms, arm = arms[chosen_arm].name ,values = algo.values, favs = user.favs, css_url = css_url)



if __name__ == "__main__":
	app.run()