import glob
from os import path
import pickle
from embeddings import lorentz_kernel
from diagram_utils import compute_all_diagram_statistics
from local_file_locations import dgm_pickle_dir
import pdb

nu  = 3.
tau = 1.

VDWrads = {
             'H': 1.2,
             'C': 1.7,
             'O': 1.52,
             'N': 1.54,
             'S': 1.8\
            }

pkl_files = glob.glob(path.join(dgm_pickle_dir, '*.pkl'))

pkl_template = 'stats'
save_pickles = True

# Setup for recomputing thresholds
thresholds_pkl = 'thresholds.pickle'
with open(thresholds_pkl, 'rb') as h:
    data = pickle.load(h)

for fname in pkl_files:
    element = fname.split('.')[-2].split('_')[-1]
    if element == 'S':
        continue    # Sulfur has many empty diagrams that cause issues
                    # in the code, don't have time to fix this now.

    etaij = 2*tau*VDWrads[element]
    thresholds = {}
    for label in data['name']:
        thresh = data['threshold'][data['name']==label]
        thresh = thresh[list(thresh.keys())[0]]
        thresholds[label] =  1 - lorentz_kernel(thresh, etaij, nu=nu)

    data = None

    with open(fname, 'rb') as h:
        dgms = pickle.load(h)

    allstats = compute_all_diagram_statistics(dgms, thresholds)
    pname = pkl_template + '_' + element + '.pkl'
    allstats.to_pickle(pname)
