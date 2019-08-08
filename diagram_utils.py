# Utilities that operate on persistence diagrams

import numpy as np
import persim

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
