import numpy as np
import pickle
import torch
import time
from part_visibilities_generator import generate_parts_visibilities
import matplotlib.pyplot as plt

"""
    compute the distance matrix based on the part feature vectors and the visibility coefficients
    
    Args :
        files_working_directory : the directory where the feature vectors, parts mask and visibility coefficients are
        extracted_features_filename : list the names of the files where the feature vectors are. One file for the query 
        feature vectors and one file for the gallery feature vectors.
    
    Returns : 
        distances_matrix : the distance matrix
"""
def compute_distances_matrix_part_based(files_working_directory ,extracted_features_filename, verbose=False):
    total_features, _, _, total_parts_visibilities, total_parts_masks = pickle.load(open(files_working_directory + extracted_features_filename[0], 'rb'))
    for i in range(1, len(extracted_features_filename)):
        temp_f, _, _, temp_part_visibility, temp_parts_masks = pickle.load(open(files_working_directory + extracted_features_filename[i], 'rb'))
        total_features = np.append(total_features, temp_f, axis=0)
        total_parts_visibilities = np.append(total_parts_visibilities, temp_part_visibility, axis=0)
        total_parts_masks = np.append(total_parts_masks, temp_parts_masks, axis=0)
        del temp_f

    total_parts_visibilities = generate_parts_visibilities(total_parts_masks)

    for i in range(len(total_features)):
        for j in range(len(total_features[0])):
            total_features[i,j] = total_features[i,j]/np.linalg.norm(total_features[i,j])
        print(i, total_parts_visibilities[i])

    n_pieces = 3
    total_features = torch.from_numpy(total_features)









    pieces = [total_features[:, :, :int(total_features.shape[2] / n_pieces)].to('cuda')]
    for i in range(1, n_pieces - 1):
        pieces.append(total_features[:, :, (int((total_features.shape[2] / n_pieces) * i)):(
            int((total_features.shape[2] / n_pieces) * (i + 1)))].to('cuda'))
    pieces.append(total_features[:, :, (int((total_features.shape[2] / n_pieces) * (n_pieces - 1))):].to('cuda'))
    #
    """
    for piece in pieces:
        print(piece.shape)
    print(total_features.shape)
    print("done")
    """
    distances_matrix = torch.zeros((total_features.shape[0], total_features.shape[0]), dtype=torch.float32).to('cpu')
    total_parts_visibilities = torch.from_numpy(total_parts_visibilities).to('cuda')
    # print("computing distance matrix...")
    K = 1
    b = time.time()
    for i in range(distances_matrix.shape[0]):

        if i % 50 == 0:
            e = time.time()
            time_spent = e - b
            estimated_speed = float(i + 1e-12) / (1e-12 + time_spent)
            estimated_time = float(distances_matrix.shape[0] - i) / estimated_speed
            hours = int(estimated_time / 3600)
            minutes = int((estimated_time - 3600 * hours) / 60.)
            seconds = estimated_time - minutes * 60. - hours * 3600.
            if verbose:
                print(
                    f"\r processing line {i} / {distances_matrix.shape[0]}, {hours}h{minutes}m{seconds:.2f}s remaining",
                    end='')
        if i % K == 0:
            intermediary_distance_row = torch.zeros((K, distances_matrix.shape[1])).to('cuda')

        sum = 0
        for piece in pieces:
            intermediary = torch.sum(torch.mul(piece, piece[i, :, :]), axis=2)
            sum = sum + intermediary

        t_parts_vis = torch.mul(total_parts_visibilities, total_parts_visibilities[i, :])
        sum = torch.mul(sum, t_parts_vis)
        sum = torch.sum(sum, axis=1)/torch.sum(t_parts_vis, axis=1)
        intermediary_distance_row[(i % K), :] = torch.pow(sum, 1.)
        # input(intermediary_distance_row[(i%K),:])

        if i % K == 0:
            intermediary_distance_row = intermediary_distance_row.to('cpu')
            distances_matrix[i:(i + K)] = intermediary_distance_row
    # print("done")

    distances_matrix = - np.array(distances_matrix)
    return distances_matrix


"""
    compute the distances matrix based on the part feature vectors of one body part

    Args :
        files_working_directory : the directory where the feature vectors, parts mask and visibility coefficients are
        extracted_features_filename : list the names of the files where the feature vectors are. One file for the query 
        feature vectors and one file for the gallery feature vectors.

    Returns : 
        distances_matrix : the distances matrix of all the parts, limited to all the query-gallery distances for memory reasons
"""
def compute_parts_distances_matrix_part_based(files_working_directory ,extracted_features_filename, verbose=False):
    total_features, _, _, total_parts_visibilities, total_parts_masks = pickle.load(open(files_working_directory + extracted_features_filename[0], 'rb'))
    for i in range(1, len(extracted_features_filename)):
        temp_f, _, _, temp_part_visibility, temp_parts_masks = pickle.load(open(files_working_directory + extracted_features_filename[i], 'rb'))
        total_features = np.append(total_features, temp_f, axis=0)
        total_parts_visibilities = np.append(total_parts_visibilities, temp_part_visibility, axis=0)
        total_parts_masks = np.append(total_parts_masks, temp_parts_masks, axis=0)
        del temp_f

    total_parts_visibilities = generate_parts_visibilities(total_parts_masks)

    for i in range(len(total_features)):
        for j in range(len(total_features[0])):
            total_features[i,j] = total_features[i,j]/np.linalg.norm(total_features[i,j])
        print(i, total_parts_visibilities[i])

    n_pieces = 3
    total_features = torch.from_numpy(total_features)









    pieces = [total_features[:, :, :int(total_features.shape[2] / n_pieces)].to('cuda')]
    for i in range(1, n_pieces - 1):
        pieces.append(total_features[:, :, (int((total_features.shape[2] / n_pieces) * i)):(
            int((total_features.shape[2] / n_pieces) * (i + 1)))].to('cuda'))
    pieces.append(total_features[:, :, (int((total_features.shape[2] / n_pieces) * (n_pieces - 1))):].to('cuda'))
    #
    """
    for piece in pieces:
        print(piece.shape)
    print(total_features.shape)
    print("done")
    """
    distances_matrix = torch.zeros((total_features.shape[0], total_features.shape[0],9), dtype=torch.float32).to('cpu')
    total_parts_visibilities = torch.from_numpy(total_parts_visibilities).to('cuda')
    # print("computing distance matrix...")
    K = 1
    b = time.time()
    for i in range(distances_matrix.shape[0]):

        if i % 50 == 0:
            e = time.time()
            time_spent = e - b
            estimated_speed = float(i + 1e-12) / (1e-12 + time_spent)
            estimated_time = float(distances_matrix.shape[0] - i) / estimated_speed
            hours = int(estimated_time / 3600)
            minutes = int((estimated_time - 3600 * hours) / 60.)
            seconds = estimated_time - minutes * 60. - hours * 3600.
            if verbose:
                print(
                    f"\r processing line {i} / {distances_matrix.shape[0]}, {hours}h{minutes}m{seconds:.2f}s remaining",
                    end='')
        if i % K == 0:
            intermediary_distance_row = torch.zeros((K, distances_matrix.shape[1],9)).to('cuda')

        sum = 0
        for piece in pieces:
            intermediary = torch.sum(torch.mul(piece, piece[i, :, :]), axis=2)
            sum = sum + intermediary

        intermediary_distance_row[(i % K), :, :] = torch.pow(sum, 1.)
        # input(intermediary_distance_row[(i%K),:])

        if i % K == 0:
            intermediary_distance_row = intermediary_distance_row.to('cpu')
            distances_matrix[i:(i + K)] = intermediary_distance_row
    # print("done")

    distances_matrix = - np.array(distances_matrix)
    return distances_matrix[:3368,3368:,:]


"""
    compute the distance matrix based on the part feature vectors

    Args :
        files_working_directory : the directory where the feature vectors, parts mask and visibility coefficients are
        extracted_features_filename : list the names of the files where the feature vectors are. One file for the query 
        feature vectors and one file for the gallery feature vectors.

    Returns : 
        distances_matrix : the distance matrix
"""
def compute_distances_matrix_simple(files_working_directory ,extracted_features_filename, verbose=False):
    K = 1
    total_features, _, _ = pickle.load(open(files_working_directory + extracted_features_filename[0], 'rb'))
    for i in range(1, len(extracted_features_filename)):
        temp_f, _, _ = pickle.load(open(files_working_directory + extracted_features_filename[i], 'rb'))
        total_features = np.append(total_features,temp_f, axis=0)
        del temp_f



    total_features = total_features[::K]

    n_pieces = 3
    total_features = torch.from_numpy(total_features)

    #print("piecing...")

    pieces = [total_features[:, :int(total_features.shape[1] / n_pieces)].to('cuda')]
    for i in range(1, n_pieces - 1):
        pieces.append(total_features[:, (int((total_features.shape[1] / n_pieces) * i)):(
            int((total_features.shape[1] / n_pieces) * (i + 1)))].to('cuda'))
    pieces.append(total_features[:, (int((total_features.shape[1] / n_pieces) * (n_pieces - 1))):].to('cuda'))
    #
    """
    for piece in pieces:
        print(piece.shape)
    print(total_features.shape)
    print("done")
    """
    distances_matrix = torch.zeros((total_features.shape[0], total_features.shape[0]), dtype=torch.float32).to('cpu')

    #print("computing distance matrix...")
    K = 1
    b = time.time()
    for i in range(distances_matrix.shape[0]):

        if i % 50 == 0:
            e = time.time()
            time_spent = e - b
            estimated_speed = float(i + 1e-12) / (1e-12 + time_spent)
            estimated_time = float(distances_matrix.shape[0] - i) / estimated_speed
            hours = int(estimated_time / 3600)
            minutes = int((estimated_time - 3600 * hours) / 60.)
            seconds = estimated_time - minutes * 60. - hours * 3600.
            if verbose:
                print(f"\r processing line {i} / {distances_matrix.shape[0]}, {hours}h{minutes}m{seconds:.2f}s remaining",
                  end='')
        if i % K == 0:
            intermediary_distance_row = torch.zeros((K, distances_matrix.shape[1])).to('cuda')

        sum = 0
        for piece in pieces:
            intermediary = torch.sum(torch.pow(piece - piece[i, :], 2), axis=1)
            sum = sum + intermediary

        intermediary_distance_row[(i % K), :] = torch.pow(sum, 0.5)
        # input(intermediary_distance_row[(i%K),:])

        if i % K == 0:
            intermediary_distance_row = intermediary_distance_row.to('cpu')
            distances_matrix[i:(i + K)] = intermediary_distance_row
    #print("done")

    distances_matrix = np.array(distances_matrix)
    return distances_matrix

"""
    compute two matrix that are the same size as the distances matrix, and contain the pids and camids equality masks
    
"""
def generate_pids_camids_equality_mask(pids, camids):

    pids_mask = np.zeros((len(pids), len(pids)),dtype='bool')
    camids_mask = np.zeros((len(camids), len(camids)),dtype='bool')
    #print(camids.shape, pids.shape)
    for i in range(len(pids)):
        pids_mask[i] = (pids == pids[i])
        camids_mask[i] = (camids == camids[i])

    #print(camids_mask[7800,10000], camids_mask[10000,7800])
    return pids_mask, camids_mask

"""

"""
def generate_pids_camids_equality_mask_for_tracking(pids, camids):

    pids_mask = np.zeros((len(pids), len(pids)),dtype='bool')
    camids_mask = np.zeros((len(camids), len(camids)),dtype='bool')
    #print(camids.shape, pids.shape)
    for i in range(len(pids)):
        pids_mask[i] = (pids == pids[i])
        if pids[i] == 0:
            pids_mask[i] = False
            pids_mask[i,i] = True

        camids_mask[i] = (camids == camids[i])

    #print(camids_mask[7800,10000], camids_mask[10000,7800])
    return pids_mask, camids_mask

"""
     compute the confidence coefficients for each pairs
"""
def generate_confidence_coefficients_matrix(parts_visibility, verbose=False):
    confidence_matrix = np.zeros((19281, 19281),dtype='float')
    for i in range(len(confidence_matrix)):
        if verbose and i%100==0:
            print(f"{i}/19281 lines done")

        part_visibility_inter = parts_visibility[:,1:] * parts_visibility[i,1:]
        confidence_matrix[i,:] = part_visibility_inter.sum(axis=1)/8.
        #for j in range(len(confidence_matrix[0])):
        #    confidence_matrix[i, j] = (parts_visibility[i,1:]*parts_visibility[j,1:]).sum()/8.

    return confidence_matrix