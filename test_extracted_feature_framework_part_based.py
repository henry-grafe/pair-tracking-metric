import numpy as np
import matplotlib.pyplot as plt
from Extracted_features_framework_part_based import Extracted_features_framework
import new_metric
import new_metric_plot
import ranking_improvement_side_information

extracted_features_framework = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"], is_distances_matrix_computed = True, is_confidence_matrix_computed=True)

#print("mAP = {0:2.2f} %".format(extracted_features_framework.compute_mAP(verbose=False)*100))
pids = extracted_features_framework.pids

print('computing distance matrix flat and pids mask flat')
distances_matrix_flat, pids_mask_flat, confidence_matrix_flat = (extracted_features_framework.get_lists_and_confidence_special_selection(7000,0,verbose=True))
args = distances_matrix_flat.argsort()
pids_mask_flat = pids_mask_flat[args]
distances_matrix_flat = distances_matrix_flat[args]
confidence_matrix_flat = confidence_matrix_flat[args]

print('generating curve')
error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(distances_matrix_flat, pids_mask_flat, N=15000)
print('plotting curve')
new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio)

confidences = np.linspace(0,1,9)
print(confidences)
for i in range(len(confidences)):
    plt.subplot(3,3,i+1)
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(
        distances_matrix_flat, pids_mask_flat, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio,
                                                              pids_mask_flat_sorted, ratio)

    distances_matrix_flat_reordered, pids_mask_flat_reordered, confidence_matrix_flat_reordered = ranking_improvement_side_information.ranking_improvement_algorithm(distances_matrix_flat, pids_mask_flat, confidence_matrix_flat, 0.1, confidences[i])
    print('generating curve')
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(distances_matrix_flat_reordered, pids_mask_flat_reordered, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio, label=str(confidences[i]))
    plt.ylim(0,0.0014)
    plt.legend()
plt.legend()
plt.show()

