from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import networkx as nx
import networkit as nk


def make_ER_Graph(N,k,SEED):

	G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

	G_nk = nk.nxadapter.nx2nk(G_nx)

	return G_nk


def get_degree_seq(G):

	degree = nk.centrality.DegreeCentrality(G)

	degree.run()

	degree_sequence = degree.ranking()

	degree_sequence.sort()

	degree_sequence_final = list(map(lambda x : x[1], degree_sequence))

	return degree_sequence_final


def get_between_seq(G):

	between = nk.centrality.DynBetweenness(G)

	between.run()

	between_sequence = between.ranking()

	between_sequence.sort()

	between_sequence_final = list(map(lambda x : x[1], between_sequence))

	return between_sequence_final


def normalize(l):

	total_sum = sum(l)

	final_list = list(map(lambda x : x / total_sum, l))

	return final_list





N=int(sys.argv[1]) 

k=int(sys.argv[2])

SEED=int(sys.argv[3])

G = make_ER_Graph(N,k,SEED)

all_nodes = list(G.nodes())

all_degree_list = get_degree_seq(G)

all_between_list = get_between_seq(G)

normalize_degree = normalize(all_degree_list)

normalize_between = normalize(all_between_list)
















X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, \
                 n_redundant=0,random_state=0, shuffle=False)


clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

clf.fit(X, y)  

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', \
            max_depth=2, max_features='auto', max_leaf_nodes=None, \
            min_impurity_decrease=0.0, min_impurity_split=None, \
            min_samples_leaf=1, min_samples_split=2, \
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None, \
            oob_score=False, random_state=0, verbose=0, warm_start=False)

print(clf.feature_importances_)

print(clf.predict([[0, 0, 0, 0]]))