import pickle
from diagram_utils import bottleneck_distance_matrix

pickle_filename = 'diagrams.pkl'

with open(pickle_filename, 'rb') as h:
    all_diagrams = pickle.load(h)

#########
dgms = all_diagrams
labels = list(dgms.keys())
label1 = labels[0]
label2 = labels[1]
########

D, labels = bottleneck_distance_matrix(all_diagrams)

