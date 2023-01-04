import numpy as np


def generate_parts_visibilities(parts_masks):
    parts_visibilities = np.zeros((len(parts_masks),9))

    for i in range(len(parts_visibilities)):
        for j in range(1, 9):
            parts_visibilities[i, j] = parts_masks[i, j].max()
            if parts_visibilities[i,j] > 0.4:
                parts_visibilities[i, j] = 1.
            else:
                parts_visibilities[i,j] = 0.

        parts_visibilities[i, 0] = parts_visibilities[i, 1:].max()

    return parts_visibilities

def generate_parts_visibilities_sigmoid(parts_masks, a, b):
    parts_mask_maxes = parts_masks.max(axis=(2,3))
    parts_visibilities = 1./(1+np.exp(-a*(parts_mask_maxes - b)))
    parts_visibilities[:,0] = parts_visibilities[:,1:].max(axis=1)
    return parts_visibilities

def generate_parts_visibilities_multiple_thresholds(parts_masks, V):
    parts_visibilities = np.zeros((len(parts_masks), 9))

    for i in range(len(parts_visibilities)):
        for j in range(1, 9):
            parts_visibilities[i, j] = parts_masks[i, j].max()
            if parts_visibilities[i, j] > V[j-1]:
                parts_visibilities[i, j] = 1.
            else:
                parts_visibilities[i, j] = 0.

        parts_visibilities[i, 0] = parts_visibilities[i, 1:].max()

    return parts_visibilities