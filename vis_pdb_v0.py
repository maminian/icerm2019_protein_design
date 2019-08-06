from matplotlib import pyplot
import glob
import os

#pdb_dir = '../data/pdbs/pdb_files/' #adjust as necessary
from local_file_locations import *
pdb_files = glob.glob(pdb_dir+'*.pdb')

def read_pdb(fname):
    '''
    Purpose: read the given pdb file for atomic coordinates
    and associated metadata on those rows.
    '''
    f = open(fname, 'r')
    lines = f.readlines()
    f.close()

    atoms = []
    # don't know how to make pandas drop lines after a criteria has been reached upon read.
    for line in lines:
        lsplit = line.split()
        if lsplit[0] == '#': # extra stuff we don't care about
            break
        elif len(lsplit) == 1: #I don't know
            break
        else:
            # it looks like the xyz coordinates are in 6,7,8. Atom name in 11.
            # fix types... certainly a cleaner way to do this.
            lsplit[1] = int(lsplit[1])
            lsplit[5] = int(lsplit[5])
            lsplit[6] = float(lsplit[6])
            lsplit[7] = float(lsplit[7])
            lsplit[8] = float(lsplit[8])
            lsplit[9] = float(lsplit[9]) # don't know what this is
            lsplit[10] = float(lsplit[10]) # don't know what this is
            atoms.append( lsplit )
    #
    return atoms
#


# atom color conventions according to https://en.wikipedia.org/wiki/CPK_coloring
atom_colors = {'H': [1,1,1],
               'C': [0,0,0],
               'N': [0,0,1],
               'O': [1,0,0],
              }

# van der waals radii (in Angstroms) according to Table 9 ("Final Model")
# of the paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3658832/
#
# Note pyplot scatter "size" seems to be in units**2 (yes, I know), so these values
# will be squared in the scatter command (in 3D).

vdw_radii = {'H': 1.1,
             'C': 1.6,
             'N': 1.49,
             'O': 1.54
            }

#################

prot_num = 14000 # happens to be an HHH
atoms = read_pdb(pdb_files[prot_num])

# visualization
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

base_size = 50 # arbitrary; choose to make scatterplot visible.

for row in atoms:
    aname = row[11]
    xa,ya,za = row[6:9]
    # note: size still seems to correspond to units**2 in 3D plots.
    ax.scatter(xa,ya,za, c=[atom_colors.get(aname,[1,0,1])], s=base_size*vdw_radii.get(aname, 1)**2, alpha=0.5)
ax.set_title(os.path.basename(pdb_files[prot_num]))

fig.show()
