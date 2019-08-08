# Utilities that operate on persistence diagrams

import numpy as np
import scipy as sp
from pandas import DataFrame
import persim
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

stat_names = []
stat_names.extend(['b'+name for name in temp_names])
stat_names.extend(['d'+name for name in temp_names])
stat_names.extend(['p'+name for name in temp_names])
stat_names.extend(['bdcor', 'bdcencor', 'totalper', 'bmaxper', 'dmaxper', 'bminper', 'dminper', 'Nfeatures'])
stat_names.extend(['bci_' + '{0:02d}'.format(ind) for ind in range(Nbins)])
stat_names.extend(['bcbin_' + '{0:03d}'.format(ind) for ind in range(Nbins+1)])

# birth stats + death stats + pers stats + misc stats + betti curve
Nstats = 10 + 10 + 10 + 8 + (Nbins + Nbins + 1)

def compute_single_diagram_statistics(dgm, threshold, order):
    """
    Computes stats of a single persistence diagram, input as an array
    with 2 columns.

    If order == 0, this removes the feature with the first death value
    of inf, and replaces the rest with the threshold value.

    If order > 0, this replaces all inf death values with the threshold.
    """

    Nbins = 100 # Number of betti curve bins
    stats = np.zeros(Nstats)

    b = dgm[:,0] # births
    d = dgm[:,1] # deaths

    if order == 0:
        # Remove first instance of inf death
        infinds = np.where(np.isinf(d))[0]
        if len(infinds) > 0:
            infind = infinds[0]
            rest = infinds[1:]
        d = np.delete(d, infind)
        b = np.delete(b, infind)

    # Replace infs with maximum threshold
    d[np.where(np.isinf(d))[0]] = threshold

    dgm = np.zeros([b.size, 2])
    dgm[:,0] = b
    dgm[:,1] = d

    p = d - b # persistence
    stats[:10] = compute_stats(b)
    stats[10:20] = compute_stats(d)
    stats[20:30] = compute_stats(p)

    bmean = stats[3]
    dmean = stats[13]
    maxfeatind = np.argmax(p)
    minfeatind = np.argmax(p)
    stats[30:38] = [np.sum(b*d),
                    np.sum((b-bmean)*(d-dmean)),
                    np.sum(p),
                    b[maxfeatind],
                    d[maxfeatind],
                    b[minfeatind],
                    d[minfeatind],
                    b.size]

    betticurve = compute_betti_curve(dgm)

    ## Compute integral of Betti curve over bin
    bettiints, bin_edges = integrate_binned_betti_curve(betticurve, Nbins, threshold)

    stats[38:(38+Nbins)] = bettiints
    stats[(38+Nbins):(38+Nbins+Nbins+1)] = bin_edges

    return stats

def compute_all_diagram_statistics(dgms, threshs):
    """
    Computes statistics for all persistence diagrams that are input dict form.

    dgms is a dict whose keys are labels for the particular protein, and
    each entry in the dict is a verbatim output from ripser.ripser.

    threshs has the same structure: a dict of thresholds for each label.

    Returns a pandas dataframe
    """

    verbosity = 5

    data = {}
    data['label'] = list(dgms.keys())
    Nproteins = len(data['label'])
    numdims = len(dgms[list(dgms.keys())[0]]['dgms'])

    dfblank = DataFrame(data)
    dffull = DataFrame()

    npnans = np.zeros([Nproteins])
    npnans[:] = np.nan

    dfblank['threshold'] = npnans
    dfblank['order'] = npnans
    # Allocate data frame rows/cols
    for name in stat_names:
        dfblank[name] = npnans

    thresholds = np.zeros(Nproteins)

    stats = np.zeros([Nproteins, Nstats])

    for order in range(numdims):

        df = dfblank.copy()
        df['order'] = order

        count = 0
        for ind,label in enumerate(data['label']):

            #thresholds[ind] = threshs['threshold'][threshs['name']==label]
            thresholds[ind] = threshs[label]

            stats[ind, :] = compute_single_diagram_statistics(dgms[label]['dgms'][order], thresholds[ind], order)

            count += 1
            if (count % verbosity) == 0:
                print("Computed order {2:d}, count {0:d}/{1:d}".format(count, Nproteins, order))

            df['threshold'][ind] = thresholds[ind]

        # Insert into appropriate place in df
        df[stat_names] = stats
        dffull = dffull.append(df)

    return dffull
