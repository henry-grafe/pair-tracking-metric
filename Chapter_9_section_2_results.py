import numpy as np
import matplotlib.pyplot as plt
from Extracted_features_framework_part_based import Extracted_features_framework
import new_metric
import new_metric_plot
import ranking_improvement_side_information

extracted_features_framework = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"], is_distances_matrix_computed = True, is_confidence_matrix_computed=True)

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
plt.show()

sigma = 0.5
confidences = [0.376,0.51,0.626]

for i in range(len(confidences)):
    plt.subplot(1,3,i+1)
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(
        distances_matrix_flat, pids_mask_flat, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio,
                                                              pids_mask_flat_sorted, ratio,label="Original curve")

    distances_matrix_flat_reordered, pids_mask_flat_reordered, confidence_matrix_flat_reordered = ranking_improvement_side_information.ranking_improvement_algorithm(distances_matrix_flat, pids_mask_flat, confidence_matrix_flat, sigma, confidences[i])
    print('generating curve')
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(distances_matrix_flat_reordered, pids_mask_flat_reordered, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio, label="Modified Curve")
    plt.ylim(-0.0002,0.0018)
    plt.title(f"Certainty threshold = {confidences[i]}")
    plt.legend()
plt.legend()
plt.show()