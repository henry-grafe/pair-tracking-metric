import numpy as np
import matplotlib.pyplot as plt

def ranking_improvement_algorithm(distances_matrix_flat, pids_mask_flat, confidence_matrix_flat, percentile_move_limit, confidence_threshold):
    args = distances_matrix_flat.argsort()
    pids_mask_flat = pids_mask_flat[args]
    confidence_matrix_flat = confidence_matrix_flat[args]
    distances_matrix_flat = distances_matrix_flat[args]

    reverse_pids_mask_flat = np.flip(pids_mask_flat)
    indexes = np.arange(len(pids_mask_flat))

    n_valid = pids_mask_flat.sum()
    n_invalid = len(pids_mask_flat) - n_valid

    ratio = n_valid / n_invalid
    percentile_center = 1 / (1 + ratio)
    ranking_center = int(percentile_center * ratio * (n_valid + n_invalid))
    print(f"n_valid = {n_valid}, n_invalid = {n_invalid}")
    print(f"ranking center = {ranking_center}")

    # compute the number of pairs under which a pair with low confidence will be relocated to the center
    N_valid_percentile_move_limit = int(percentile_move_limit * ratio * (n_valid + n_invalid))
    # compute the number of pairs under which a pair with low confidence will be relocated to the center
    N_invalid_percentile_move_limit = int(percentile_move_limit * (n_valid + n_invalid))

    indexes_flat_left = indexes[:ranking_center]
    indexes_flat_right = indexes[ranking_center:]
    print(indexes_flat_left)
    print(indexes_flat_right)

    indexes_flat_left_zone_to_move =  indexes_flat_left[:N_valid_percentile_move_limit]
    indexes_flat_left_zone_not_to_move = indexes_flat_left[N_valid_percentile_move_limit:]
    print("indexes flat left zones")
    print(indexes_flat_left_zone_to_move)
    print(indexes_flat_left_zone_not_to_move)

    indexes_flat_right_zone_to_move = indexes_flat_right[(len(indexes_flat_right)-N_invalid_percentile_move_limit):]
    indexes_flat_right_zone_not_to_move = indexes_flat_right[:(len(indexes_flat_right)-N_invalid_percentile_move_limit)]
    print("indexes flat right zones")
    print(indexes_flat_right_zone_to_move)
    print(indexes_flat_right_zone_not_to_move)

    indexes_flat_left_elements_to_move = indexes_flat_left_zone_to_move[confidence_matrix_flat[indexes_flat_left_zone_to_move] < confidence_threshold]
    print(f"indexes on the left (valid) to move  = {len(indexes_flat_left_elements_to_move)}")

    indexes_flat_left_elements_not_to_move = indexes_flat_left_zone_to_move[np.logical_not(confidence_matrix_flat[indexes_flat_left_zone_to_move] < confidence_threshold)]
    print("indexes flat left elements to move and not to move")
    print(indexes_flat_left_elements_to_move)
    print(indexes_flat_left_elements_not_to_move)

    indexes_flat_right_elements_to_move = indexes_flat_right_zone_to_move[confidence_matrix_flat[indexes_flat_right_zone_to_move] < confidence_threshold]
    indexes_flat_right_elements_not_to_move = indexes_flat_right_zone_to_move[np.logical_not(confidence_matrix_flat[indexes_flat_right_zone_to_move] < confidence_threshold)]
    print("indexes flat right elements to move and not to move")
    print(indexes_flat_right_elements_to_move)
    print(indexes_flat_right_elements_not_to_move)

    indexes_flat_left_uncertain_pairs_moved = np.append(np.append(indexes_flat_left_elements_not_to_move, indexes_flat_left_zone_not_to_move),indexes_flat_left_elements_to_move)
    indexes_flat_right_uncertain_pairs_moved = np.append(np.append(indexes_flat_right_elements_to_move, indexes_flat_right_zone_not_to_move),indexes_flat_right_elements_not_to_move)

    indexes_reordered_ranking = np.append(indexes_flat_left_uncertain_pairs_moved, indexes_flat_right_uncertain_pairs_moved)

    distances_matrix_flat_reordered = distances_matrix_flat[indexes_reordered_ranking]
    pids_mask_flat_reordered = pids_mask_flat[indexes_reordered_ranking]
    confidence_matrix_flat_reordered = confidence_matrix_flat[indexes_reordered_ranking]

    return distances_matrix_flat_reordered, pids_mask_flat_reordered, confidence_matrix_flat_reordered









