import numpy as np
import matplotlib.pyplot as plt
import new_metric
import new_metric_plot
import generate_artificial_data
import ranking_improvement_side_information

print("generating artificial pair and distance list")
artificial_pids_mask, artificial_distance_matrix, artificial_confidence_matrix = generate_artificial_data.generate_distances_list_with_confidence_coefficients(
    [5,8],[1,1],[1,1],valid_set_size=10000,imbalance=3e-3)

plt.hist(artificial_distance_matrix[artificial_pids_mask], bins=60, density=True,label="same target")
plt.hist(artificial_distance_matrix[np.logical_not(artificial_pids_mask)], bins=60, density=True, alpha=0.7,label="different targets")
plt.legend()
plt.title("Artificial data : distances distributions")
plt.xlabel("distance")
plt.show()

print("sorting pairs by distance")
args = np.argsort(artificial_distance_matrix)
artificial_pids_mask = artificial_pids_mask[args]
artificial_distance_matrix = artificial_distance_matrix[args]
artificial_confidence_matrix =artificial_confidence_matrix[args]

print('generating curve')
error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(artificial_distance_matrix, artificial_pids_mask, N=15000)
print('plotting curve')
new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio)
plt.show()

sigma = 0.99

confidences = [0.1,0.2,0.3]
print(confidences)
for i in range(len(confidences)):
    plt.subplot(1,3,i+1)
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(
        artificial_distance_matrix, artificial_pids_mask, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio,
                                                              pids_mask_flat_sorted, ratio, label="Original ranking")

    distances_matrix_flat_reordered, pids_mask_flat_reordered, confidence_matrix_flat_reordered = ranking_improvement_side_information.ranking_improvement_algorithm(artificial_distance_matrix, artificial_pids_mask, artificial_confidence_matrix, sigma, confidences[i])
    print('generating curve')
    error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio = new_metric.generate_error_information_curve(distances_matrix_flat_reordered, pids_mask_flat_reordered, N=15000)
    print('plotting curve')
    new_metric_plot.plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat_sorted, ratio, label="Modified ranking")
    plt.ylim(-0.0005,0.005)
    plt.title(f"Certainty threshold = {confidences[i]}")
    plt.legend()
plt.legend()
plt.show()