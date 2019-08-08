from matplotlib import pyplot
import glob
import os
import pandas
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import metrics

pdb_dir = '../data/pdbs/pdb_files/' #adjust as necessary
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

def get_mst_pairs(distmat):
    '''
    from example at:

    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.cs$

    Inputs: distance matrix
    Outputs: pairs; a list of pairs of indices, to be
        used as (e.g.)

            [ax.plot(arr[p,0], arr[p,1]) for p in pairs]

    wow
    '''
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    X = csr_matrix(distmat)
    Tcsr = minimum_spanning_tree(X)
    ii,jj = np.where( Tcsr.toarray() )
    return [[i,j] for i,j in zip(ii,jj)]
#


# atom color conventions according to https://en.wikipedia.org/wiki/CPK_coloring
atom_colors = {'H': [1,1,1],
#               'C': [0,0,0],
               'C': [0.2,0.2,0.2],
               'N': [0,0,1],
               'O': [1,0,0],
               'S': [1,1,0]
              }

# van der waals radii (in Angstroms) according to Table 12 ("Consistent van der Waals Radii for All Main-Group Elements")
# of the paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3658832/
#
# Note pyplot scatter "size" seems to be in units**2, so these values
# will be squared in the scatter command (in 3D).

vdw_radii = {'H': 1.1,
             'C': 1.7,
             'N': 1.55,
             'O': 1.52,
             'S': 1.80
            }

#################

#prot_num = 14004 # happens to be an HHH
prot_num = 1000 # looks like it has a significant H1 (?) featuer.
atoms = read_pdb(pdb_files[prot_num])

colnames = ['thing0','idx','thing1','thing2','thing3','residual','x','y','z','thing4','thing5','atom']
prot_df = pandas.DataFrame(data=atoms, columns=colnames)

# inefficient; whatever.
# partition data based on the residual (secondary structure) to improve visualization.
ec = {g: np.where(g==prot_df['residual'].values)[0] for g in np.unique(prot_df['residual'].values)}

# visualization

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

# it looks like the xyz coordinates are in 6,7,8. Atom name in 11.

include=['C', 'H', 'O','N']

for j,(k,v) in enumerate( ec.items() ):
    secondary = prot_df.iloc[v]
#    subset = np.where(secondary['atom']=='C')[0]
    subset = np.arange(secondary.shape[0])


    # try another approach - construct minimal spanning trees on each residual; hope this is a
    # better representation of the thing than plotting sequential points.
    dm = metrics.pairwise_distances(secondary.iloc[subset][['x','y','z']].values)
    mst_pairs = get_mst_pairs(dm)
    for pair in mst_pairs:
        sec_subset = secondary.iloc[subset[pair]]
        atom1,atom2 = sec_subset['atom'].values

        vec = sec_subset[['x','y','z']].values
        halfway = np.mean(vec,axis=0)
        #ax.plot(vec[:,0],vec[:,1],vec[:,2], lw=2, c=colors[j%len(colors)])

        if atom1 in include:
            ax.plot([vec[0,0],halfway[0]], [vec[0,1],halfway[1]], [vec[0,2],halfway[2]], c=atom_colors[atom1], lw=2*vdw_radii[atom1])
        if atom2 in include:
            ax.plot([halfway[0],vec[1,0]], [halfway[1],vec[1,1]], [halfway[2],vec[1,2]], c=atom_colors[atom2], lw=2*vdw_radii[atom2])
#



ax.set_title(os.path.basename(pdb_files[prot_num]), c='w')

ax.set_facecolor('k')

if False:
    dmm = get_mst_pairs(metrics.pairwise_distances(prot_df[['x','y','z']].values))
    for pair in dmm:
        coords = prot_df[['x','y','z']].iloc[pair].values
        ax.plot(coords[:,0], coords[:,1], coords[:,2], c=[1,0.8,1], lw=4, alpha=0.2)
#

fig.show()
