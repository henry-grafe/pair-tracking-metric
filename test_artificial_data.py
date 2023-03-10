import numpy as np
import matplotlib.pyplot as plt
import new_metric
import new_metric_plot
import generate_artificial_data
import ranking_improvement_side_information

print("generating artificial pair and distance list")
artificial_pids_mask, artificial_distance_matrix, artificial_confidence_matrix = generate_artificial_data.generate_distances_list_with_confidence_coefficients(
    [1,4],[1,1],[1,1],valid_set_size=10000,imbalance=3e-3)

args = np.argsort(artificial_distance_matrix)
artificial_pids_mask = artificial_pids_mask[args]
artificial_distance_matrix = artificial_distance_matrix[args]
artificial_confidence_matrix =artificial_confidence_matrix[args]

print('generating curve')
error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(artificial_distance_matrix, artificial_pids_mask, N=15000)
print('plotting curve')
new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio)
plt.show()


confidences = np.linspace(0,1,9)
print(confidences)
for i in range(len(confidences)):
    plt.subplot(3,3,i+1)
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(
        artificial_distance_matrix, artificial_pids_mask, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio,
                                                              pids_mask_flat_sorted, ratio)

    distances_matrix_flat_reordered, pids_mask_flat_reordered, confidence_matrix_flat_reordered = ranking_improvement_side_information.ranking_improvement_algorithm(artificial_distance_matrix, artificial_pids_mask, artificial_confidence_matrix, 0.3, confidences[i])
    print('generating curve')
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(distances_matrix_flat_reordered, pids_mask_flat_reordered, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio, label=str(confidences[i]))
    plt.ylim(0,0.005)
    plt.legend()
plt.legend()
plt.show()