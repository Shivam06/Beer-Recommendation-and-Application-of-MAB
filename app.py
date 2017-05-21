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
		if arr[i] == str(input):
			return i+1
	return 0

algo = EpsilonGreedy([], [], 0.2)
algo.initialize(3)
arms = [JacArm(), CossineArm(), KNNarm()]
arms_name = [ 'Jaccardian','Cossine', 'KNN'  ]
arr = []
favs = []



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
        self.prev_recommendations = 0

class Data:
    def __init__(self):
        self.chosen_arm = 0

data = Data()

@app.route('/user/<username>/train', methods=['GET', 'POST'])
def train_user(username):
    if request.method == 'POST':
        user = userlist.users[username]
        css_url = url_for('static', filename='combined.css')
        beers = json.loads(request.data)
        beer = ''
        # beer = beers['beers'][0]
        # import pdb
        # pdb.set_trace()
        #
        for i in beers['beers']:
            if i not in user.favs:
                user.favs.append(i)
                beer = i

        data.chosen_arm = algo.select_arm()

        if user.flag == 0:
            try:
                user.arr = arms[data.chosen_arm].recommend(user.favs, 10)
            except:
                return "No such Beer Available."
            user.flag = 1

        else:
            # chosen_arm = 0
            rank = find_rank(user.arr, beer) # Define the function
            if rank:
                score = arms[data.chosen_arm].draw(rank)
            else:
                score = 0
            algo.update(data.chosen_arm, score)
            arms[data.chosen_arm].count = algo.counts[data.chosen_arm]
            arms[data.chosen_arm].value = algo.values[data.chosen_arm]
            data.chosen_arm = algo.select_arm()
            try:
                user.arr = arms[data.chosen_arm].recommend(user.favs, 10)
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
        user.prev_recommendations = a
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
        train_url = request.url
        user_url = request.url.replace('/train','')
        engine=arms_name[data.chosen_arm]
        # if user.flag == 0:
        return render_template('index.html', css_url=css_url,
            jquery_url=jquery_url, beers_url=beers_url,
            js_url=js_url, highlight_url=highlight_url, train_url=train_url, user_url=user_url,name=username,engine=engine)
        # else:
        #     return render_template('index.html', css_url=css_url,
        #                        jquery_url=jquery_url, beers_url=beers_url,
        #                        js_url=js_url, highlight_url=highlight_url, arms=arms, prev=user.prev_recommendations)

@app.route('/user/<username>', methods=['GET', 'POST'])
def user_data(username):
    if username not in userlist.users.keys():
        userlist.users[username] = person()
    user = userlist.users[username]
    css_url = url_for('static', filename='css/main.css')
    jquery_url = url_for('static', filename='js/jquery-1.10.2.min.js')
    beers_url = url_for('static', filename='js/beers.js')
    highlight_url = url_for('static', filename='js/code.js')
    js_url = url_for('static', filename='js/main.js')
    img_url = url_for('static', filename='img/beer.svg')
    train_url = request.url+'/train'
    user_url = request.url
    engine=arms_name[data.chosen_arm]
        # if user.flag == 0:
        #     return render_template('index.html', css_url=css_url,
        #                        jquery_url=jquery_url, beers_url=beers_url,
        #                        js_url=js_url, highlight_url=highlight_url)
        # else:
    return render_template('user.html', css_url=css_url,
        jquery_url=jquery_url, beers_url=beers_url,
        highlight_url=highlight_url, arms=arms, engine=engine ,prev=user.prev_recommendations, favs=user.favs, img_url = img_url, train_url=train_url, user_url=user_url,name=username)

@app.route('/user/<username>/prev', methods=['POST'])
def user_prev(username):
    user = userlist.users[username]
    return jsonify(user.prev_recommendations)

if __name__ == '__main__':
    app.run(debug=True)