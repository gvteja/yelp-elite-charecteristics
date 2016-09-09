import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import networkx as nx
import pickle
import random
import csv 
import sys
import plotly.plotly as py
from plotly.graph_objs import *
import json
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score


dir_path = '/Users/bobby/Documents/Notes/DS/A3/yelp_dataset_challenge_academic_dataset'
cd $dir_path

def convert_json_to_df(filename):
    data = []
    with open(filename) as json_data:
        for line in json_data:
            data.append(json.loads(line))

    df = pd.DataFrame.from_dict({i:data[i] for i in range(len(data))}, orient='index')
    return df

bdf = convert_json_to_df('yelp_academic_dataset_business.json')
cdf = convert_json_to_df('yelp_academic_dataset_checkin.json')
udf = convert_json_to_df('yelp_academic_dataset_user.json')
tdf = convert_json_to_df('yelp_academic_dataset_tip.json')
rdf = convert_json_to_df('yelp_academic_dataset_review.json')

# calc elite dist per yr
elite = {yr:0 for yr in range(2000, 2016)}
new_elites = {yr:0 for yr in range(2000, 2016)}
drop_elites = {yr:0 for yr in range(2000, 2016)}
for i, user in udf.iterrows():
    if len(user.elite) > 0:
        new_elites[user.elite[0]] += 1
        pyr = user.elite[0] - 1
        for yr in user.elite:
            elite[yr] += 1
            if yr != pyr + 1:
                drop_elites[pyr + 1] += 1
            pyr = yr
        if yr != 2015:
            drop_elites[yr + 1] += 1

# calc join dist
join = {yr:{} for yr in range(2000,2016)}
for i, user in udf.iterrows():
    yr = int(user.yelpling_since[:4])
    m = int(user.yelpling_since[5:])
    join[yr][m] = join[yr][m] + 1 if m in join[yr] else 1

# join dist by year
join_yr = {}
for yr, m_dist in join.iteritems():
    join_yr[yr]  = sum(m_dist.values())

# build frn network
friends = {}
for i, user in udf.iterrows():
    if len(user.friends) == 0:
        continue
    if user.user_id not in friends:
        friends[user.user_id] = []
    frn_list = friends[user.user_id]
    for frn in user.friends:
        frn_list.append(frn)

g = nx.Graph()
for node in friends.keys():
    g.add_node(node)

for node, neighbours in friends.iteritems():
    for neighbour in neighbours:
        g.add_edge(node, neighbour)

# build full frn nw
g = nx.Graph()

for i, user in udf.iterrows():
    for frn in user.friends:
        g.add_edge(user.user_id, frn)



# calc compliments
compliments = {}
for i, user in udf.iterrows():
    for x, v in user.compliments.iteritems():
        if x not in compliments:
            compliments[x] = v
        else:
            compliments[x] += v

# calc neighbourhoods dist
nbhood = {}
for i, b in bdf.iterrows():
    for nb in b.neighborhoods:
        if nb in nbhood:
            nbhood[nb] += 1
        else:
            nbhood[nb] = 0

# find all att:
att = {}
for i, b in bdf.iterrows():
    for a in b.attributes:
        # note here 'a' could be a dict
        if a in att:
            att[a] += 1
        else:
            att[a] = 0

# find category dist
cat = {}
for i, b in bdf.iterrows():
    for c in b.categories:
        if c in cat:
            cat[c] += 1
        else:
            cat[c] = 1

catl = []
for c, v in cat.iteritems():
    catl.append((v,c))
catl.sort(reverse=True)

# agg check in info by hr and day
chk_day = {}
chk_hr = {}
for i, c in cdf.iterrows():
    for x, v in c.checkin_info.iteritems():
        hr, day = x.split('-')
        hr, day = int(hr), int(day)
        chk_day[day] = chk_day[day] + v if day in chk_day else v
        chk_hr[hr] = chk_hr[hr] + v if hr in chk_hr else v

# read and build g from fiends file
g = nx.read_graphml('friends.graphml')
partitions = community.best_partition(g)

# pickle obj
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# save and read partition
save_obj(partitions, 'partitions')
partitions = load_obj('partitions')

# build cluster set
num_clusters = max(partitions.values()) + 1
clusters = {i:set() for i in range(num_clusters)}
for n, c in partitions.iteritems():
    clusters[c].add(n)

cluster_size = {x:len(clusters[x]) for x in range(num_clusters)}

l = [(s, c) for c, s in cluster_size.iteritems()]
l.sort(reverse=True)

# build cluster graph for biggest components
cluster_graph = nx.Graph()
for i in range(11):
    cluster_graph.add_node(i)


for i in range(11):
    for j in range(i + 1, 11):
        cluster_j = clusters[j]
        for node in clusters[i]:
            if node in cluster_j:
                cluster_graph.add_edge(i, j)
                break

# draw show graph
nx.draw(cluster_graph)
plt.show()

# data tranformation for classification
features_o = ['yelping_since', 'votes', 'fans', 'average_stars', 'review_count', 'friends', 'elite']
features_td = ['yelping_since', 'funny_votes', 'useful_votes', 'cool_votes', 'fans', 'average_stars', 'review_count', 'friends']

data = udf[features_o]
data['elite'] = data.elite.map(lambda x: len(x) > 0)
data['friends'] = data.friends.map(lambda x: len(x))
data['yelping_since'] = data.yelping_since.map(lambda x: int(x[:4]) - 2003)
data['funny_votes'] = data.votes.map(lambda x: x['funny'])
data['useful_votes'] = data.votes.map(lambda x: x['useful'])
data['cool_votes'] = data.votes.map(lambda x: x['cool'])
data.drop('votes', inplace=True, axis=1)

# training and classification

# sample data
sample_data = data.sample(frac=0.3)

random.seed()
data_train, data_test, labels_train, labels_test = train_test_split(
data[features_td], data.elite, test_size=0.33, random_state=int(random.random() * 10000) )

nb_model = GaussianNB()
nb_model.fit(data_train, labels_train)

expected = labels_test
predicted = nb_model.predict(data_test)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# labels = sample_data.elite
# sample_data.drop('elite', inplace=True, axis=1)

# cross-validation
nb_model = GaussianNB()
cross_val_score(nb_model, data_train, labels_train, scoring='f1')

# manual cross validate
k = 10
data_chunks = np.array_split(data, k)
for chunk in data_chunks:
    random.seed()
    data_train, data_test, labels_train, labels_test = train_test_split(
    chunk[features_td], chunk.elite, test_size=0.33, random_state=int(random.random() * 10000) )

    nb_model = GaussianNB()
    nb_model.fit(data_train, labels_train)

    expected = labels_test
    predicted = nb_model.predict(data_test)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))


###########################################################################################

s=0
x = 0
for i in range(20):
    v,c = catl[i]
    x += v

x = clusters[14]
for i in range(0, 11):
    for node in clusters[i]:
        if node in x:
            cluster_graph.add_edge(i, 14)
            break

cluster_sc = {c:{s:0 for s in bdf.state.unique()} for c in range(11)}
for i, r in rdf.iterrows():
    try:
        c = partitions[r.user_id]
    except:
        continue
    if not (-1 < c < 11):
        continue
    b = bdf.loc[r.business_id]
    cluster_sc[c][b.state] += 1



svm = SVC()
svm.fit(data, labels) 
print(svm.predict([[-0.8, -1]]))


from sklearn.cluster import DBSCAN
DBSCAN(min_samples=1).fit_predict(mat)

###########################################################################################



# plotting

# elite per time
x=range(2005,2016)
y1=[elite[yr] for yr in x]
y2=[new_elites[yr] for yr in x]
y3=[drop_elites[yr] for yr in x]

plot_data = [ Scatter(x=x, y=y1, name='No of elites in that year'), Scatter(x=x, y=y2, name='No of NEW elites in that year'), Scatter(x=x, y=y3, name='No of elites who dropped in that year')]
layout = Layout(
    title='Number of yelp elites over the years',
    xaxis=XAxis( title='Year', showline=False),
    yaxis=YAxis( title='Number of yelp elites', showline=True)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='elite')

# user growth over year
x=range(2004,2015)
y1 = [join_yr[yr] for yr in x]
y2 = []
s = 0
for yr in x:
    s += join_yr[yr]
    y2.append(s)

plot_data = [ Scatter(x=x, y=y1, name='No of users in that year'), Scatter(x=x, y=y2, name='Cumulative no of users') ]
layout = Layout(
    title='Number of yelp users over the years',
    xaxis=XAxis( title='Year', showline=False),
    yaxis=YAxis( title='Number of users', showline=True)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='users')

# checkins by hour
plot_data = [ Scatter(x=range(24), y=[chk_hr[x] for x in range(24)]) ]
layout = Layout(
    title='Number of hourly checkins',
    xaxis=XAxis( title='Hour in 24-hr format', showline=True),
    yaxis=YAxis( title='Number of checkins', showline=True)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='checkin_hr')

# checkins by day
plot_data = [ Scatter(x=['Sun', 'Mon', 'Tue','Wed', 'Thu',' Fri', 'Sat'], y=[chk_day[x] for x in range(7)]) ]
layout = Layout(
    title='Checkins across days in the week ',
    xaxis=XAxis( title='Day of the week', showline=True),
    yaxis=YAxis( title='Number of checkins', showline=True)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='checkin_day')

# plotting compliments
plot_data = [ Scatter(x=[k for k in compliments], y=[compliments[k] for k in compliments]) ]
layout = Layout(
    title='Compliments',
    xaxis=XAxis( title='Compliment', showline=True),
    yaxis=YAxis( title='Number of compliments', showline=True)
)
fig = Figure(data=plot_data, layout=layout)
plot_url = py.plot(fig, filename='compliments')

# plotting box plot for review stars in bdf
trace = Box(
    y = bdf.stars.values
)

data = [trace]
plot_url = py.plot(data, filename='bdf_rating')

# plot businesses by state
sc = bdf.state.value_counts()
x = sc.keys()
y = [c for s, c in sc.iteritems()]

fig = {
    'data': [{'labels': x,
              'values': y,
              'type': 'pie'}],
    'layout': {'title': 'Distribution of Businesses by State'}
}
plot_url = py.plot(fig, validate=False, filename='businesses_state')