import generate_artificial_data
import numpy as np
import matplotlib.pyplot as plt
from pAP_generation import Extracted_features_framework

obj = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"],mode='part_based')
obj.is_distances_matrix_computed = True

distances_matrix_flat, pids_mask_flat = obj.get_lists(verbose=False)

print(distances_matrix_flat.shape, distances_matrix_flat)
print(pids_mask_flat.shape, pids_mask_flat)

args = distances_matrix_flat.argsort()
pids_mask_flat = pids_mask_flat[args]
reverse_pids_mask_flat = np.flip(pids_mask_flat)
reverse_pids_mask_flat = np.logical_not(reverse_pids_mask_flat)

pids_mask_cumulation = np.cumsum(pids_mask_flat)/pids_mask_flat.sum()
reverse_pids_mask_cumulation = np.cumsum(reverse_pids_mask_flat)/reverse_pids_mask_flat.sum()

print(pids_mask_flat.shape, reverse_pids_mask_flat.shape)

plt.plot(pids_mask_cumulation[::100])
plt.plot(reverse_pids_mask_cumulation[::100])
plt.show()

N = 100000

percentiles = np.linspace(0,1,N)
n_useful_links = np.zeros(percentiles.shape)
error_rate = np.zeros(percentiles.shape)
actual_ratio = np.zeros(percentiles.shape)

n_valid = pids_mask_flat.sum()
n_invalid = reverse_pids_mask_flat.sum()
ratio = n_valid/n_invalid
integ = np.zeros(100)
c=0
for l in np.linspace(1.057,1.062,100):
    print(c)
    index_pids=0
    index_reverse_pids=0
    for i in range(len(percentiles)):

        index_pids = int(percentiles[i]*len(pids_mask_flat)*ratio*l)

        index_reverse_pids = int(percentiles[i]*len(reverse_pids_mask_flat)*(1-ratio*l))

        if index_pids + index_reverse_pids > len(pids_mask_flat) - 1 :
            break

        n_useful_links[i] = pids_mask_cumulation[index_pids]*n_valid + reverse_pids_mask_cumulation[index_reverse_pids]*n_invalid
        error_rate[i] = 1. - (n_useful_links[i])/(index_pids + index_reverse_pids + 2)
        actual_ratio[i] = (pids_mask_cumulation[index_pids]*n_valid)/(reverse_pids_mask_cumulation[index_reverse_pids]*n_invalid)
        #print(index_pids, index_reverse_pids, percentiles[i], pids_mask_cumulation[index_pids], reverse_pids_mask_cumulation[index_reverse_pids])
    integ[c] = np.abs(actual_ratio-ratio).sum()/float(len(n_useful_links))
    c += 1
    """
    plt.subplot(2,2,1)
    plt.plot(error_rate, n_useful_links/len(pids_mask_flat), '.')
    plt.xlabel("error rate = 1 - precision")
    plt.ylabel("number of useful links used in tracking algorithm")
    plt.title("Useful links in algorithm selecting x % of links with biggest distance and x % of links with smalled distance")
    plt.subplot(2,2,2)
    plt.plot(n_useful_links/len(pids_mask_flat), actual_ratio,'.')
    plt.xlabel("number of useful links used in tracking algorithm")
    plt.ylabel("ratio in the selected pairs between pairs of same and different identity")
    plt.title("ratio in the selected pairs between pairs of same and different identity vs numb of links")
    plt.subplot(2,2,3)
    plt.plot(error_rate, actual_ratio,'.')
    plt.xlabel("error rate = 1 - precision")
    plt.ylabel("ratio in the selected pairs between pairs of same and different identity")
    plt.title("ratio in the selected pairs between pairs of same and different identity vs error rate")
    """

index_pids=0
index_reverse_pids=0
for i in range(len(percentiles)):

    index_pids = int(percentiles[i]*len(pids_mask_flat)*ratio*0.)

    index_reverse_pids = int(percentiles[i]*len(reverse_pids_mask_flat)*(1-ratio*0.))

    if index_pids + index_reverse_pids > len(pids_mask_flat) - 1 :
        break

    n_useful_links[i] = pids_mask_cumulation[index_pids]*n_valid + reverse_pids_mask_cumulation[index_reverse_pids]*n_invalid
    error_rate[i] = 1. - (n_useful_links[i])/(index_pids + index_reverse_pids + 2)
    actual_ratio[i] = (pids_mask_cumulation[index_pids]*n_valid)/(reverse_pids_mask_cumulation[index_reverse_pids]*n_invalid)
    print(index_pids, index_reverse_pids, percentiles[i], pids_mask_cumulation[index_pids], reverse_pids_mask_cumulation[index_reverse_pids])
"""
plt.subplot(2,2,1)
plt.plot(error_rate, n_useful_links/len(pids_mask_flat), '.')
plt.xlabel("error rate = 1 - precision")
plt.ylabel("number of useful links used in tracking algorithm")
plt.title("Useful links in algorithm selecting x % of links with biggest distance and x % of links with smalled distance")
plt.subplot(2,2,2)
plt.plot(n_useful_links/len(pids_mask_flat), actual_ratio,'.')
plt.plot([0,1],[ratio, ratio],'k-')
plt.xlabel("number of useful links used in tracking algorithm")
plt.ylabel("ratio in the selected pairs between pairs of same and different identity")
plt.title("ratio in the selected pairs between pairs of same and different identity vs numb of links")
plt.subplot(2,2,3)
plt.plot(error_rate, actual_ratio,'.')
plt.plot([error_rate.min(),error_rate.max()],[ratio, ratio],'k-')
plt.xlabel("error rate = 1 - precision")
plt.ylabel("ratio in the selected pairs between pairs of same and different identity")
plt.title("ratio in the selected pairs between pairs of same and different identity vs error rate")

plt.subplot(2,2,4)
"""
plt.plot(np.linspace(1.057,1.062,100), integ, '.-')


plt.show()