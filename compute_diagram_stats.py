import glob
from os import path
import pickle
import pandas as pd
from embeddings import lorentz_kernel
from diagram_utils import compute_all_diagram_statistics
from local_file_locations import dgm_pickle_dir
import pdb

save_pickles = True

pkl_files = glob.glob(path.join(dgm_pickle_dir, '*.pkl'))
pkl_template = 'stats'

for fname in pkl_files:
    atoms = fname.split('.')[-2].split('_')[-1]

    if atoms == 'CNOS':
        continue

    print('Computing stats for atoms {0:s}'.format(atoms))

    # Maximum radius
    maxval = 1.

    with open(fname, 'rb') as h:
        dgms = pickle.load(h)

    allstats = compute_all_diagram_statistics(dgms, maxval)
    pname = pkl_template + '_' + atoms + '.pkl'
    if save_pickles:
        allstats.to_pickle(pname)
