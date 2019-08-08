# Computes persistence diagrams from ripser using an ???? embedding
# Filters by proteins for which we have alpha-complex-based thresholds
# computed. (~200 left out)

import pickle
import glob
import numpy as np
from ripser import ripser
import ntpath

from embeddings import coordinate_embedding, adjusted_lorentz_metric, lorentz_kernel, compute_etaij
from local_file_locations import pdb_dir

pickle_template = 'diagrams.pkl'
thresholds_pkl = 'thresholds.pickle' # Courtesy of V, this is in the
                                     # root directory of the shared Box
                                     # folder

pdb_files = glob.glob(pdb_dir+'*.pdb')

save_pickles = True

# Computes only this many diagrams: mostly for debugging. Set to np.Inf
# to compute all of them.
max_diagrams = np.Inf

elements = ['S']
exclusion_elements = ['H']

with open(thresholds_pkl, 'rb') as h:
    thresholds = pickle.load(h)

# Options: euclidean or lorentz
metric = 'lorentz'

## Definition of metrics
metric_info = {\
        'euclidean': {},
        'lorentz'  : {\
                        'tau': 1,
                        'nu': 3,
                        'atom_list': ['H', 'C', 'O', 'N', 'S'],
                        'VDWrad_dict': {
                                         'H': 1.2,
                                         'C': 1.7,
                                         'O': 1.52,
                                         'N': 1.54,
                                         'S': 1.8\
                                        }
                     }
              }

if metric == 'euclidean':
    metricfun = 'euclidean' # Ripser accepts this keyword arg
elif metric == 'lorentz':
    metricfun = lambda A1, A2: adjusted_lorentz_metric(A1, A2,
                                tau=metric_info[metric]['tau'],
                                nu=metric_info[metric]['nu'])

# Loop over elements
for element in elements:

    print('Element ' + element + ':')

    # Dict storage for diagrams
    diagrams = {}
    count = 0

    # Loop over all pdb files
    for fname in pdb_files:

        label = ntpath.basename(fname).split('.')[0]
        threshs = thresholds['threshold'][thresholds['name']==label]

        # Skip if threshold not available
        if threshs.size == 0: # skip: we don't have a threshold
            print('Skipping label {0:s}: no threshold available'.format(label))
            continue

        # Grab embedding
        X,label = coordinate_embedding(fname, filterbyelement=element)

        if X.size == 0:
            print('Skipping element {1:s}, label {0:s}: no atoms of specified type'.format(label, element))
            continue

        # Compute PD threshold
        thresh = threshs[threshs.keys()[0]]
        if metric is 'euclidean':
            pass # thresh = thresh
        else: # Pass through lorentz kernel metric
            etaij = compute_etaij(X[0,:], X[0,:]) # All atoms should be the same
            thresh = 1 - lorentz_kernel(thresh, etaij, nu=metric_info['lorentz']['nu'])

        # Compute diagram
        temp = ripser(X, maxdim=2, thresh=thresh, do_cocycles=True, metric=metricfun)

        # ripser also returns the distance matrix, which is huge and unnecessary
        del temp['dperm2all']
        diagrams[label] = temp

        # Log of progress
        count += 1
        if count % 5 == 0:
            print("{0:d} / {1:d}".format(count, min(len(pdb_files), max_diagrams)))

        if count >= max_diagrams:
            break

    if save_pickles:
        pickle_filename = pickle_template.split('.')[0] + '_' + element + '.' + pickle_template.split('.')[1]
        with open(pickle_filename, 'wb') as h:
            pickle.dump(diagrams, h,protocol=pickle.HIGHEST_PROTOCOL)
