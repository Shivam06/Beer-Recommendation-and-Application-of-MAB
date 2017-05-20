# from flask import Flask, render_template, request, url_for
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



from flask import Flask, request, render_template, url_for, Response, json, jsonify
app = Flask(__name__)
class UserList():
    def __init__(self):
        self.users = {}


userlist = UserList()

class person:
	def __init__(self):
		self.arr = []
		self.favs = []
		self.flag = 0



@app.route('/user/<username>', methods=['GET', 'POST'])
def index(username):
    if request.method == 'POST':
        user = userlist.users[username]
        css_url = url_for('static', filename='combined.css')
        beers = json.loads(request.data)
        beer = beers['beers'][0]
        # import pdb
        # pdb.set_trace()
        if beer not in user.favs:
            user.favs.append(beer)
        if user.flag == 0:
            chosen_arm = algo.select_arm()
            try:
                user.arr = arms[chosen_arm].recommend(user.favs, 10)
            except:
                return "No such Beer Available."
            user.flag = 1
        else:
            rank = find_rank(user.arr, beer) # Define the function
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
        a= {}
            # for i in range(10):
            #     x.append({'beer':'beer1'})
        beerlist = user.arr.tolist()
        beers = []
        for i in beerlist:
            beers.append({'beer':i})
        a['result']=beers
        # import pdb
        # pdb.set_trace()
        return jsonify(a)
    else:
        if username not in userlist.users.keys():
            userlist.users[username] = person()
        user = userlist.users[username]
        css_url = url_for('static', filename='css/main.css')
        jquery_url = url_for('static', filename='js/jquery-1.10.2.min.js')
        beers_url = url_for('static', filename='js/beers.js')
        highlight_url = url_for('static', filename='js/code.js')
        js_url = url_for('static', filename='js/main.js')
        if user.flag == 0:
            return render_template('index.html', css_url=css_url,
                               jquery_url=jquery_url, beers_url=beers_url,
                               js_url=js_url, highlight_url=highlight_url)
        else:
            return render_template('index.html', css_url=css_url,
                               jquery_url=jquery_url, beers_url=beers_url,
                               js_url=js_url, highlight_url=highlight_url)

if __name__ == '__main__':
    app.run(debug=True)