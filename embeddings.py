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

import pdb
import ntpath
import numpy as np

atom_list = ['H', 'C', 'O', 'N', 'S']
VDW_list = [ 1.2, 1.7, 1.52, 1.54, 1.8]
def coordinate_embedding(pdbfile, filterbyelements=None):
    """
    Like the coordinate embedding, but also saves atom type as an index
    of atom_list.
    [ xcoord, ycoord, zcoord, atomindex]
    """

    label = ntpath.basename(pdbfile).split('.')[0]

    f = open(pdbfile, 'r')
    lines = f.readlines()
    f.close()

    not_TER = lambda inp: inp != 'TER'
    if filterbyelements is None:
        validatom = lambda inp: True
    else:
        validatom = lambda inp: inp in filterbyelements

    Natoms = 0      # Total number of atoms in protein
    Nvalidatoms = 0 # Number of atoms of a certain element type
    for line in lines:
        if not_TER(line.split()[0]):
            Natoms +=1
            if validatom(line.split()[11]):
                Nvalidatoms +=1
        else:
            break

    X = np.zeros([Nvalidatoms, 4],dtype=float)

    count = 0
    for atomindex in range(Natoms):
        lsplit = lines[atomindex].split()
        if validatom(lsplit[11]):
            X[count,0] = float(lsplit[6])
            X[count,1] = float(lsplit[7])
            X[count,2] = float(lsplit[8])
            X[count,3] = atom_list.index(lsplit[11])

            count +=1

    return X, label

def compute_etaij(A1, A2, tau=1.):
    return tau*(VDW_list[int(A1[3])] + VDW_list[int(A2[3])])

def adjusted_lorentz_metric(A1, A2, tau=1., nu=3.):
    """
    Computes adjusted Lorentz distance.
    """

    r = np.linalg.norm(A1[:3] - A2[:3])
    return 1 - lorentz_kernel(r, compute_etaij(A1, A2,tau=tau), nu=nu)

def lorentz_kernel(r, etaij, nu=3.):
    return (1 + (r/etaij)**nu)**(-1.)
