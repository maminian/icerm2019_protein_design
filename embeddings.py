# This assumes pdb file atom columns are:
#
# 0:  ATOM (verbatim...dunno)
# 1:  number (int) -- label of atom
# 2:  dunno (str)
# 3:  residue label (str, 3-letter label)
# 4:  A (verbatim...dunno)
# 5:  residue index (int)
# 6:  x-coordinate
# 7:  y-coordinate
# 8:  z-coordinate
# 9:  1.000 (verbatim...dunno)
# 10: 0.000 (verbatim...dunno)
# 11: Element (str)

import ntpath
import numpy as np

def coordinate_embedding(pdbfile):
    """
    Extracts the coordinate embedding of a protein: just a matrix of
    atom locations.
    """

    label = ntpath.basename(pdbfile).split('.')[0]

    f = open(pdbfile, 'r')
    lines = f.readlines()
    f.close()

    Natoms = 0
    for line in lines:
        if line.split()[0] != 'TER':
            Natoms +=1
        else:
            break

    X = np.zeros([Natoms, 3],dtype=float)

    for atomindex in range(Natoms):
        lsplit = lines[atomindex].split()
        X[atomindex,0] = float(lsplit[6])
        X[atomindex,1] = float(lsplit[7])
        X[atomindex,2] = float(lsplit[8])

    return X, label
