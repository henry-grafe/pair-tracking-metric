import numpy as np
import matplotlib.pyplot as plt
from Extracted_features_framework_part_based import Extracted_features_framework
import new_metric
import new_metric_plot

extracted_features_framework = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"], is_distances_matrix_computed = True, is_confidence_matrix_computed=True)

#print("mAP = {0:2.2f} %".format(extracted_features_framework.compute_mAP(verbose=False)*100))
pids = extracted_features_framework.pids

print('computing distance matrix flat and pids mask flat')
distances_matrix_flat, pids_mask_flat, confidence_matrix_flat = (extracted_features_framework.get_lists_and_confidence_special_selection(7000,0,verbose=True))
print(distances_matrix_flat.shape)
print(pids_mask_flat)
print(confidence_matrix_flat)
plt.hist(distances_matrix_flat[pids_mask_flat],bins=100,density=False)
plt.hist(distances_matrix_flat[np.logical_not(pids_mask_flat)],bins=300,density=False, alpha = 0.5)
plt.show()
print('generating curve')
error_rate, n_useful_links, actual_ratio, pids_mask_flat, ratio = new_metric.generate_error_information_curve(distances_matrix_flat, pids_mask_flat, N=15000)
print('plotting curve')
new_metric_plot.plot_error_information_curve(error_rate, n_useful_links, actual_ratio, pids_mask_flat, ratio)