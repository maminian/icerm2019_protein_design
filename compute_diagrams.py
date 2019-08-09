# Computes persistence diagrams from ripser using an ???? embedding
# Filters by proteins for which we have alpha-complex-based thresholds
# computed. (~200 left out)

import pickle
import glob
import ntpath
import pdb

import numpy as np

import pandas as pd
from ripser import ripser

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
max_diagrams = np.inf

# Maximum homology dimension
maxdim = 2

# Specifies which atoms to use in computation of the diagrams
#atom_lists = ['C', 'H', 'N', 'O', 'S',
atom_lists = [ 'S',
               'CNOS', 'HNOS', 'CHNS', 'CHOS', 'CH', 'CHNOS', 'NOS']

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


cols = ['name', 'secstruct', 'rd', 'number']
cols.extend(['H'+str(dim)+'_dgm' for dim in range(maxdim+1)])
cols.extend(['H'+str(dim)+'_cocycles' for dim in range(maxdim+1)])
cols.extend(['num_edges', 'idx_perm', 'r_cover'])

# Loop over elements
for atoms in atom_lists:

    # Pandas storage for diagrams
    diagrams = pd.DataFrame(columns=cols)
    count = 0

    # Loop over all pdb files
    for fname in pdb_files:

        label = ntpath.basename(fname).split('.')[0]
        #threshs = thresholds['threshold'][thresholds['name']==label]

        ## Skip if threshold not available
        #if threshs.size == 0: # skip: we don't have a threshold
        #    print('Skipping label {0:s}: no threshold available'.format(label))
        #    continue

        # Grab embedding
        X,label = coordinate_embedding(fname, filterbyelements=atoms)

        if X.size == 0:
            print('Skipping atoms {1:s}, label {0:s}: no atoms of specified type'.format(label, atoms))
            temp = {}
            for col in cols:
                temp[col] = None
        else:

            # Compute PD threshold
            #thresh = threshs[threshs.keys()[0]]
            #if metric is 'euclidean':
            #    pass # thresh = thresh
            #else: # Pass through lorentz kernel metric
            #    etaij = compute_etaij(X[0,:], X[0,:]) # All atoms should be the same
            #    thresh = 1 - lorentz_kernel(thresh, etaij, nu=metric_info['lorentz']['nu'])

            # Compute homology
            temp = ripser(X, maxdim=maxdim, thresh=1., do_cocycles=True, metric=metricfun)

            # "Unravel" diagrams and cocycles by dimension
            for dim in range(maxdim+1):
                temp['H'+str(dim)+'_dgm'] = temp['dgms'][dim]
                temp['H'+str(dim)+'_cocycles'] = temp['cocycles'][dim]

            del temp['dgms']
            del temp['cocycles']

            # Check for valid distance matrix
            if metric == 'lorentz':
                assert np.all(np.all(temp['dperm2all'] >= 0)) & np.all(np.all(temp['dperm2all'] <= 1)), "Distance matrix contains invalid values"
            elif metric == 'euclidean':
                assert np.all(temp['dperm2all'] >= 0), "Distance matrix contains invalid values"

            # the distance matrix is huge and unnecessary
            del temp['dperm2all']

        # adding other identifying labels: assume filename 
        # with anatomy SSSS_rdN_PPPP.pdb, where SSSS is a secondary
        # structure, N is a round id, and PPPP is a number
        parts = label.split('_')
        temp['secstruct'] = parts[0]
        temp['rd'] = int(parts[1][2:])
        temp['number'] = int(parts[2])
        temp['name'] = label

        diagrams.loc[len(diagrams)] = temp

        # Log of progress
        count += 1
        if count % 1 == 0:
            print("{0:d} / {1:d}".format(count, min(len(pdb_files), max_diagrams)))

        if count >= max_diagrams:
            break

    if save_pickles:
        pickle_filename = pickle_template.split('.')[0] + '_' + atoms + '.' + pickle_template.split('.')[1]
        with open(pickle_filename, 'wb') as h:
            pickle.dump(diagrams, h,protocol=pickle.HIGHEST_PROTOCOL)
