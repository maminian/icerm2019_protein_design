# Utilities that operate on persistence diagrams

import numpy as np
import scipy as sp
from pandas import DataFrame
from persim import PersImage
import pdb

def bottleneck_distance_matrix(dgms):
    """
    Computes a symmetric matrix between input persistence diagrams.

    Input
    ------
    dgms: dict, whose keys index full diagram outputs from ripser

    Output
    ------
    D: distance matrix, len(dgms) x len(dgms)
    labels: List of labels that index D
    """

    D = np.zeros([len(dgms), len(dgms)])

    labels = list(dgms.keys())

    for (index1,label1) in enumerate(labels):
        for (index2,label2) in enumerate(labels):
            #print((index1,index2))
            dgm1 = dgms[label1]['dgms'][1]
            dgm2 = dgms[label2]['dgms'][1]
            D[index1,index2] = persim.bottleneck(dgm1, dgm2)

    return D, labels

def compute_stats(array):
    """
    Computes ``desirable`` statistics for a vector.
    """

    if array is None: # Return stat names
        return ['min', 'max', 'median', 'mean', 'var', \
                'mom2', 'mom3', 'momcen3', 'mom4', 'momcen4']
    else:
        return np.array([np.min(array),
                         np.max(array),
                         np.median(array),
                         np.mean(array),
                         np.var(array),
                         np.sum(array**2),
                         np.sum(array**3),
                         np.sum(array**4),
                         sp.stats.moment(array, moment=3),
                         sp.stats.moment(array, moment=4)])

def compute_betti_curve(dgm):
    """
    Computes the Betti curve defined by the (2D) array of births and
    deaths dgm.
    """

    b = dgm[:,0] # births
    d = dgm[:,1] # deaths

    bsort = np.sort(b)
    dsort = np.sort(d)

    # bcurve is a temporary variable storing all incrememts and
    # decrements of the curve, including repeated ones.
    bcurve = np.zeros([2*bsort.size, 2])
    bcurve[:bsort.size,0] = bsort
    bcurve[:bsort.size,1] = 1.
    bcurve[bsort.size:,0] = dsort
    bcurve[bsort.size:,1] = -1.

    bcurve[bcurve[:,0].argsort()]

    # Find repeated entries
    u,c = np.unique(bcurve[:,0], return_counts=True)
    removeinds = []
    qrepeat = []
    for q in range(u.size):
        if c[q] > 1:
            # Collect repeated locations
            removeinds.append([u[q], np.where(bcurve[:,0]==u[q])])
            removeinds[-1].append( np.sum(bcurve[removeinds[-1][1],1]) )
            qrepeat.append(q)

    betticurve = np.zeros([u.size, 2])
    allremoved_inds = []
    for bind,q in enumerate(qrepeat):
        betticurve[bind,0] = u[q]
        betticurve[bind,1] = removeinds[bind][2]
        allremoved_inds.extend(removeinds[bind][1][0])
    if len(qrepeat) == 0:
        bind = -1;
    bind += 1

    remaining_inds = np.setdiff1d(np.arange(2*bsort.size), allremoved_inds)
    for cind,q in enumerate(remaining_inds):
        betticurve[bind+cind,:] = bcurve[q,:]

    betticurve = betticurve[betticurve[:,0].argsort()]
    betticurve[:,1] = np.cumsum(betticurve[:,1])

    return betticurve

def integrate_binned_betti_curve(betticurve, Nbins, threshold):
    """
    Computes the integral of the Betti curve defined by the array
    betticurve. (This is an output from compute_betti_curve.) This
    integral is computed over Nbins bins that are equisized from 0 to
    threshold.

    The integrals over bins along with the bin edges are returned.
    """

    # We are running horizontally through the Betti curve. This
    # is the last value of the Betti curve observed
    leftval = 0

    bettiints = np.zeros(Nbins)
    bin_edges = np.linspace(0, threshold, Nbins+1)

    for binid in range(Nbins):
        left  = bin_edges[binid]
        right = bin_edges[binid+1]

        # Find betti curve increments that occur in this
        # interval
        betticurvelocs = np.sort(np.where(np.logical_and(betticurve[:,0] >= left, betticurve[:,0] <= right))[0])
        prevloc = left
        for loc in betticurvelocs:
            nextloc = betticurve[loc,0]
            bettiints[binid] += (nextloc - prevloc) * betticurve[loc,1]
            prevloc = nextloc
            leftval = betticurve[loc,1]

        # And handle case when no locs fall in bin.
        if betticurvelocs.size == 0:
            bettiints[binid] = (right-left) * leftval
        elif prevloc < right: # Take care of last value if locs don't line up with bin edge
            bettiints[binid] += (right - prevloc) * betticurve[loc,1]

    return bettiints, bin_edges

temp_names = ['min', 'max', 'median', 'mean', 'var', 'mom2', 'mom3', 'momcen3', 'mom4', 'momcen4']
Nbins = 100 # Number of bins for the betti curve

bstats = ['b'+name for name in temp_names]
binds0 = 0
binds1 = binds0 + len(bstats)

dstats = ['d'+name for name in temp_names]
dinds0 = binds1
dinds1 = dinds0 + len(dstats)

pstats = ['p'+name for name in temp_names]
pinds0 = dinds1
pinds1 = pinds0 + len(pstats)

miscstats = ['bdcor', 'bdcencor', 'totalper', 'bmaxper', 'dmaxper', 'bminper', 'dminper', 'Nfeatures']
minds0 = pinds1
minds1 = minds0 + len(miscstats)

bettistats = ['bci_' + '{0:02d}'.format(ind) for ind in range(Nbins)]
betti0 = minds1
betti1 = betti0 + len(bettistats)

stat_names = []
stat_names.extend(bstats)
stat_names.extend(dstats)
stat_names.extend(pstats)
stat_names.extend(miscstats)
stat_names.extend(bettistats)

# birth stats + death stats + pers stats + misc stats + betti curve
Nstats = betti1

def clean_births_deaths(bd, threshold, dim):
    """
    If dim == 0, this removes the feature with the first death value
    of inf, and replaces the remaining inf's with the threshold value.

    If dim > 0, this replaces all inf death values with the threshold.
    """

    b = bd[:,0] # births
    d = bd[:,1] # deaths

    if dim == 0: # Remove first instance of inf death
        infinds = np.where(np.isinf(d))[0]
        if len(infinds) > 0:
            infind = infinds[0]
            #rest = infinds[1:]
        d = np.delete(d, infind)
        b = np.delete(b, infind)

    # Replace infs with maximum threshold
    d[np.where(np.isinf(d))[0]] = threshold

    bd = np.zeros([b.size, 2])
    bd[:,0] = b
    bd[:,1] = d

    return bd

def compute_single_diagram_statistics(dgm, threshold=None, dim=0):
    """
    Computes stats of a single persistence diagram dgm, input as an
    array with 2 columns.
    """

    if dgm is None: # Return list of stat names
        basenames = compute_stats(None)
        stat_names = []
        # Birth stats
        stat_names.append(['b'+name for name in basenames])
        # Death stats
        stat_names.append(['d'+name for name in basenames])
        # Persistence stats
        stat_names.append(['p'+name for name in basenames])

        # Miscllaneous stats
        stat_names.append(['bdcor', 'bdcencor', 'totalper', 'bmaxper', 'dmaxper', 'bminper', 'dminper', 'Nfeatures'])
        # Betti curve
        stat_names.append(['betticurve'])
        # Persistence image
        stat_names.append(['PI'])
        return stat_names

    # Else:

    Nbins = 100 # Number of betti curve bins
    stats = []

    dgm = clean_births_deaths(dgm, threshold, dim)
    b = dgm[:,0]
    d = dgm[:,1]

    if b.size > 0:

        p = d - b # persistence
        stats.append(compute_stats(b))
        stats.append(compute_stats(d))
        stats.append(compute_stats(p))
        #stats[binds0:binds1] = compute_stats(b)
        #stats[dinds0:dinds1] = compute_stats(d)
        #stats[pinds0:pinds1] = compute_stats(p)

        bmean = stats[0][3]
        dmean = stats[1][3]
        #bmean = stats[binds0 + 3]
        #dmean = stats[dinds0 + 3]

        maxfeatind = np.argmax(p)
        minfeatind = np.argmin(p)
        stats.append([np.sum(b*d),
                                np.sum((b-bmean)*(d-dmean)),
                                np.sum(p),
                                b[maxfeatind],
                                d[maxfeatind],
                                b[minfeatind],
                                d[minfeatind],
                                b.size
                     ])
        #stats[minds0:minds1] = [np.sum(b*d),
        #                        np.sum((b-bmean)*(d-dmean)),
        #                        np.sum(p),
        #                        b[maxfeatind],
        #                        d[maxfeatind],
        #                        b[minfeatind],
        #                        d[minfeatind],
        #                        b.size]

        betticurve = compute_betti_curve(dgm)

        ## Compute integral of Betti curve over bin
        bettiints, bin_edges = integrate_binned_betti_curve(betticurve, Nbins, threshold)

        stats.append([bettiints])
        #stats[betti0:betti1] = bettiints

        pim = PersImage(spread=0.05, pixels=[10,10], verbose=False)
        img = pim.transform(dgm)
        stats.append([img])

    else:
        stats = None

    return stats, threshold/Nbins

def compute_all_diagram_statistics(dgms, maxval, maxdgm=np.Inf):
    """
    Computes statistics for all persistence diagrams that are input dict form.

    dgms is a pandas dataframe where each entry contains the diagram for
    a particular protein.

    Only computes maxdgm diagrams. (Useful for debugging.)

    Returns a pandas dataframe.
    """

    verbosity = 100

    Nproteins = dgms.shape[0]
    numdims = 0
    for col in dgms.keys():
        HN = col.split('_')[0]
        if HN[0] == 'H':
            numdims = max(numdims, int(HN[1])+1)

    # Create names for all the dimension as well
    idcols = ['name', 'secstruct', 'rd', 'number']
    all_names = idcols.copy()
    stat_names = compute_single_diagram_statistics(None)
    Hnames = []
    for dim in range(numdims):
        Hnames.append([])
        for names in stat_names:
            Hnames[dim].append(['H{0:d}_'.format(dim) + name for name in names])
            all_names.extend(Hnames[dim][-1])
    all_names.append('bettibin_size')

    # Allocate data frame
    df = DataFrame(columns=all_names)

    # Sloppy way to add rows to df with nans
    npnans = np.zeros(df.shape[1])
    npnans[:] = np.nan

    count = 0
    for index, protein in dgms.iterrows():
        # Append row to data frame
        df.loc[index] = npnans

        # Assign metadata
        for col in idcols:
            df.loc[index, col] = protein[col]

        for dim in range(numdims):

            # Compute diagram statistics
            dgmcolname = 'H{0:d}'.format(dim) + '_dgm'
            if protein[dgmcolname] is not None:
                stats, dthresh = compute_single_diagram_statistics(protein[dgmcolname], maxval, dim)
            else:
                stats = None
                dthresh = maxval
            if stats is None: # empty diagram
                print('H{0:d} diagram for {1:s} is empty'.format(dim, protein['name']))
                pass
            else:
                # Assign stats to appropriate locations
                for batchind, names in enumerate(Hnames[dim]):
                    for statind, name in enumerate(names):
                        df.loc[index, name] = stats[batchind][statind]

        df.loc[index, 'bettibin_size'] = dthresh

        count += 1
        if count >= maxdgm:
            break
        if (count % verbosity) == 0:
            print("Computed {0:d}/{1:d} proteins".format(count, Nproteins))

    return df
