import pickle
import glob
from ripser import ripser

from embeddings import coordinate_embedding
from local_file_locations import *

pickle_filename = 'diagrams.pkl'

pdb_files = glob.glob(pdb_dir+'*.pdb')

all_diagrams = {}

count = 0
for file in pdb_files:
    X, label = coordinate_embedding(file)
    #all_diagrams[label] = ripser(X, maxdim=2, do_cocycles=True, thresh=5.)
    temp = ripser(X, maxdim=2, thresh=5.)
    del temp['dperm2all']
    all_diagrams[label] = temp

    count += 1
    if count % 5 == 0:
        print("{0:d} / {1:d}".format(count, len(pdb_files)))

with open(pickle_filename, 'wb') as h:
    pickle.dump(all_diagrams, h,protocol=pickle.HIGHEST_PROTOCOL)
