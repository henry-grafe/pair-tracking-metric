

import numpy as np
import matplotlib.pyplot as plt


def generate_error_information_curve(distances_matrix_flat, pids_mask_flat, N=5000):
    reverse_pids_mask_flat = np.flip(pids_mask_flat)
    reverse_pids_mask_flat = np.logical_not(reverse_pids_mask_flat)

    pids_mask_cumulation = np.cumsum(pids_mask_flat) / pids_mask_flat.sum()
    reverse_pids_mask_cumulation = np.cumsum(reverse_pids_mask_flat) / reverse_pids_mask_flat.sum()

    #print(pids_mask_flat.shape, reverse_pids_mask_flat.shape)

    percentiles = np.linspace(0, 1, N)
    n_useful_links = np.zeros(percentiles.shape)
    error_rate = np.zeros(percentiles.shape)
    actual_ratio = np.zeros(percentiles.shape)

    n_valid = pids_mask_flat.sum()
    n_invalid = reverse_pids_mask_flat.sum()
    ratio = n_valid / n_invalid
    percentile_center = 1 / (1 + ratio)
    ranking_center = int(percentile_center * ratio * (n_valid + n_invalid))
    #print(percentile_center, ranking_center)
    #print(n_valid, n_invalid, len(percentiles), len(pids_mask_flat))
    index_pids = 0
    index_reverse_pids = 0
    for i in range(len(percentiles)):
        index_pids = int(percentiles[i] * len(pids_mask_flat) * ratio)

        index_reverse_pids = int(percentiles[i] * len(reverse_pids_mask_flat) * (1 - ratio))

        if index_pids + index_reverse_pids > len(pids_mask_flat) - 1:
            break

        n_useful_links[i] = pids_mask_cumulation[index_pids] * n_valid + reverse_pids_mask_cumulation[
            index_reverse_pids] * n_invalid
        error_rate[i] = 1. - (n_useful_links[i]) / (index_pids + index_reverse_pids + 2)
        actual_ratio[i] = (pids_mask_cumulation[index_pids] * n_valid) / (
                    reverse_pids_mask_cumulation[index_reverse_pids] * n_invalid)
        #print(index_pids, index_reverse_pids, percentiles[i], pids_mask_cumulation[index_pids],reverse_pids_mask_cumulation[index_reverse_pids])

    n_useful_links = n_useful_links / len(pids_mask_flat)

    return error_rate, n_useful_links, actual_ratio, pids_mask_flat, ratio

