import numpy as np
import pickle
import torch
import time
import matplotlib.pyplot as plt

from big_matrix_tools import compute_distances_matrix_simple, compute_distances_matrix_part_based, generate_pids_camids_equality_mask, compute_parts_distances_matrix_part_based, generate_pids_camids_equality_mask_for_tracking, generate_confidence_coefficients_matrix
import part_visibilities_generator
from scipy.interpolate import interp1d
from scipy.integrate import quad

from sklearn.metrics import roc_auc_score
import random_selection_generation

class Extracted_features_framework():
    """
        args:
            files_working_directory : the path to the directory where the feature are stored. This will also be the
            directory where the distance matrix will be written to once they are computed, so please have at least ~5 GB
            free space on your hard drive.

            extracted_features_filename : a list with the two names of the feature vectors files. One for the query
            features, one for the gallery features

            is_distances_matrix_computed : set to True if the distance matrix file is already in files_working_directory

            is_confidence_matrix_computed : set to True if the confidence matrix file is already in files_working_directory

    """
    def __init__(self,  files_working_directory ,extracted_features_filename, is_distances_matrix_computed = False, is_confidence_matrix_computed = False):
        self.files_working_directory = files_working_directory
        self.extracted_features_filename = extracted_features_filename
        self.is_distances_matrix_computed = is_distances_matrix_computed
        self.is_confidence_matrix_computed = is_confidence_matrix_computed

        self.pids = np.zeros(0, dtype='int')
        self.camids = np.zeros(0, dtype='int')
        self.parts_visibility = np.zeros((0, 9))
        self.parts_masks = np.zeros((0, 9, 16, 8))

        for filename in extracted_features_filename:
            temp_f, temp_pids, temp_camids, temp_parts_visibility, temp_parts_masks = pickle.load(
                open(files_working_directory + filename, 'rb'))

            self.features_dim = temp_f.shape[1:]
            self.parts_masks_dim = temp_parts_masks.shape[1:]

            del temp_f

            self.parts_masks = np.append(self.parts_masks, temp_parts_masks, axis=0)
            self.pids = np.append(self.pids, temp_pids)
            self.camids = np.append(self.camids, temp_camids)
            # print(temp_parts_visibility.shape)
            self.parts_visibility = np.append(self.parts_visibility, temp_parts_visibility, axis=0)


    """
    Returns : the features vectors of the query appended to the feature vectors of the gallery
    """
    def get_features(self):
        total_features, _, _, total_parts_visibilities, total_parts_masks = pickle.load(
            open(self.files_working_directory + self.extracted_features_filename[0], 'rb'))
        for i in range(1, len(self.extracted_features_filename)):
            temp_f, _, _, temp_part_visibility, temp_parts_masks = pickle.load(
                open(self.files_working_directory + self.extracted_features_filename[i], 'rb'))
            total_features = np.append(total_features, temp_f, axis=0)
            total_parts_visibilities = np.append(total_parts_visibilities, temp_part_visibility, axis=0)
            total_parts_masks = np.append(total_parts_masks, temp_parts_masks, axis=0)
            del temp_f
        return total_features


    def compute_distances_matrix(self, verbose=False):
        if self.is_distances_matrix_computed:
            distances_matrix = pickle.load(open(self.files_working_directory + "distances_matrix.pickle", 'rb'))
        else:

            distances_matrix = compute_distances_matrix_part_based(self.files_working_directory,
                                                                   self.extracted_features_filename, verbose=verbose)

            pickle.dump(distances_matrix, open(self.files_working_directory + "distances_matrix.pickle", 'wb'))
            self.is_distances_matrix_computed = True

        return distances_matrix

    """
        Returns a flattened list of all the possible pairs of detection for a selection of detections. 
        Pairs of images that are of the same person and the same camera are eliminated from the lists
        
        Returns : 
            distances_matrix_flat : the distance for each of the pairs
            
            pids_mask_flat : the list indicating which pair is of the same target and which isn't.
            True if the pair is of the same targets, False if the pair is of different targets.
            
            if (return_slection==True) selection : the indexes of each of the person's image that are in the selection of detections
            
    """
    def get_lists_special_selection(self, verbose=False, return_selection=False):
        if verbose:
            print("computing distances matrix...")


        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        selection = random_selection_generation.generate_simple_random_selection_tracking(self.pids,7000,2000)
        print(len(selection), selection)
        input("next")
        distances_matrix = distances_matrix[selection,:]
        distances_matrix = distances_matrix[:,selection]

        if verbose:
            print("computing pids and camids masks...")




        pids_mask, camids_mask = generate_pids_camids_equality_mask_for_tracking(self.pids, self.camids)

        pids_mask = pids_mask[selection, :]
        pids_mask = pids_mask[:, selection]
        camids_mask = camids_mask[selection, :]
        camids_mask = camids_mask[:, selection]

        if verbose:
            print("computing valid identities masks (excluding same identities if they have same camids)...")

        valid_pairs_mask = np.copy(pids_mask)

        valid_pairs_mask[camids_mask] = False

        if verbose:
            plt.subplot(2, 2, 1)
            plt.title("distance matrix")
            plt.imshow(distances_matrix)

            plt.subplot(2, 2, 2)
            plt.title("pids mask")
            plt.imshow(pids_mask)

            plt.subplot(2, 2, 3)
            plt.title("camids mask")
            plt.imshow(camids_mask)

            plt.subplot(2, 2, 4)
            plt.title("valid pairs mask")
            plt.imshow(valid_pairs_mask)

            plt.show()

        total_flat_matrix_length = int(len(selection)*(len(selection)-1)/2)
        distances_matrix_flat = np.zeros(total_flat_matrix_length,dtype=distances_matrix.dtype)
        pids_mask_flat = np.zeros(total_flat_matrix_length,dtype=pids_mask.dtype)
        camids_mask_flat = np.zeros(total_flat_matrix_length,dtype=camids_mask.dtype)
        current_index = 0
        for i in range(len(distances_matrix)-1):
            if verbose and i%100==0:
                print(i)
            length_line = len(distances_matrix) - 1 - i
            distances_matrix_flat[current_index:(current_index+length_line)] = distances_matrix[i, (i + 1):]
            pids_mask_flat[current_index:(current_index+length_line)] = pids_mask[i, (i + 1):]
            camids_mask_flat[current_index:(current_index+length_line)] = camids_mask[i, (i + 1):]
            current_index += length_line
        print(len(pids_mask_flat), selection.shape)


        distances_matrix_flat = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        pids_mask_flat = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]


        if return_selection:
            return distances_matrix_flat, pids_mask_flat, selection
        else:
            return distances_matrix_flat, pids_mask_flat

    """
            Returns a flattened list of all the possible pairs of detection for a selection of detections and the
            confidence coefficients associated. Pairs of images that are of the same person and the same camera
            are eliminated from the lists

            Returns : 
                distances_matrix_flat : the distance for each of the pairs

                pids_mask_flat : the list indicating which pair is of the same target and which isn't.
                True if the pair is of the same targets, False if the pair is of different targets.
                
                confidence_matrix_flat : confidence coefficient for each pair

                if (return_slection==True) selection : the indexes of each of the person's image that are in the selection of detections

    """
    def get_lists_and_confidence_special_selection(self, selection_size, garbage_detections_size, verbose=False, return_selection=False):
        if verbose:
            print("computing distances matrix...")


        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        selection = random_selection_generation.generate_simple_random_selection_tracking(self.pids, selection_size, garbage_detections_size)
        distances_matrix = distances_matrix[selection,:]
        distances_matrix = distances_matrix[:,selection]

        if verbose:
            print("computing pids and camids masks...")




        pids_mask, camids_mask = generate_pids_camids_equality_mask_for_tracking(self.pids, self.camids)

        pids_mask = pids_mask[selection, :]
        pids_mask = pids_mask[:, selection]
        camids_mask = camids_mask[selection, :]
        camids_mask = camids_mask[:, selection]

        if verbose:
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix(verbose=verbose)

        confidence_matrix = confidence_matrix[selection, :]
        confidence_matrix = confidence_matrix[:, selection]

        plt.imshow(confidence_matrix)
        plt.show()


        if verbose:
            print("computing valid identities masks (excluding same identities if they have same camids)...")

        valid_pairs_mask = np.copy(pids_mask)

        valid_pairs_mask[camids_mask] = False

        if verbose:
            plt.subplot(2, 2, 1)
            plt.title("distance matrix")
            plt.imshow(distances_matrix)

            plt.subplot(2, 2, 2)
            plt.title("pids mask")
            plt.imshow(pids_mask)

            plt.subplot(2, 2, 3)
            plt.title("camids mask")
            plt.imshow(camids_mask)

            plt.subplot(2, 2, 4)
            plt.title("valid pairs mask")
            plt.imshow(valid_pairs_mask)

            plt.show()

        total_flat_matrix_length = int(len(selection)*(len(selection)-1)/2)
        distances_matrix_flat = np.zeros(total_flat_matrix_length,dtype=distances_matrix.dtype)
        pids_mask_flat = np.zeros(total_flat_matrix_length,dtype=pids_mask.dtype)
        camids_mask_flat = np.zeros(total_flat_matrix_length,dtype=camids_mask.dtype)
        confidence_matrix_flat = np.zeros(total_flat_matrix_length,dtype=confidence_matrix.dtype)
        current_index = 0
        for i in range(len(distances_matrix)-1):

            length_line = len(distances_matrix) - 1 - i
            distances_matrix_flat[current_index:(current_index+length_line)] = distances_matrix[i, (i + 1):]
            pids_mask_flat[current_index:(current_index+length_line)] = pids_mask[i, (i + 1):]
            camids_mask_flat[current_index:(current_index+length_line)] = camids_mask[i, (i + 1):]
            confidence_matrix_flat[current_index:(current_index+length_line)] = confidence_matrix[i, (i + 1):]
            current_index += length_line
        print(len(pids_mask_flat), selection.shape)


        #distances_matrix_flat = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        #confidence_matrix_flat = confidence_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        #pids_mask_flat = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        print(pids_mask_flat.sum(), len(pids_mask_flat))
        if return_selection:
            return distances_matrix_flat, pids_mask_flat, confidence_matrix_flat, selection
        else:
            return distances_matrix_flat, pids_mask_flat, confidence_matrix_flat

    """
        compute the confidence coefficients for each pairs
    """
    def compute_confidence_coefficients_matrix(self, verbose=False):

        if (not self.is_confidence_matrix_computed):
            confidence_matrix = generate_confidence_coefficients_matrix(self.parts_visibility, verbose=verbose)
            pickle.dump(confidence_matrix, open(self.files_working_directory + "confidence_matrix.pickle",'wb'))

        else:
            if verbose:
                print("loading confidence matrix...")
            confidence_matrix = pickle.load(open(self.files_working_directory + "confidence_matrix.pickle",'rb'))

        return confidence_matrix


    def compute_confidence_coefficients_matrix_with_selection(self, selection_1, selection_2, verbose=False):
        confidence_matrix = np.zeros((len(selection_1), len(selection_2)),dtype='float')
        parts_visibility_1 = self.parts_visibility[selection_1]
        parts_visibility_2 = self.parts_visibility[selection_2]
        for i in range(len(confidence_matrix)):
            if verbose and i%100==0:
                print(f"{i} lines of the confidence matrix done")
            for j in range(len(confidence_matrix[0])):
                confidence_matrix[i, j] = (parts_visibility_1[i,:8]*parts_visibility_2[j,:8]).sum()/8.


        return confidence_matrix

    """
    Compute the mAP for market1501 dataset using the distances matrix
    
    Returns :
        the computed mAP
    """
    def compute_mAP(self, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle",'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)


        if verbose:
            plt.subplot(2, 2, 1)
            plt.title("distance matrix")
            plt.imshow(distances_matrix)

            plt.subplot(2, 2, 2)
            plt.title("pids mask")
            plt.imshow(pids_mask)

            plt.subplot(2, 2, 3)
            plt.title("camids mask")
            plt.imshow(camids_mask)

            plt.show()

        if verbose:
            print("computing mAP...")

        APs = np.zeros(3368,dtype='float')
        for i in range(len(APs)):
            temp_distances = distances_matrix[i,3368:]

            temp_camids = camids_mask[i,3368:]
            temp_pids = pids_mask[i,3368:]

            argselect = np.logical_not(np.logical_and(temp_pids, temp_camids))
            temp_distances = temp_distances[argselect]
            temp_pids = temp_pids[argselect]


            temp_args = np.argsort(temp_distances)
            temp_pids = temp_pids[temp_args]


            N_TP = np.sum(temp_pids)
            N_G = len(temp_pids)

            cumulated_validity_div_k = np.cumsum(temp_pids) / np.arange(1, N_G + 1)
            APs[i] = (np.sum(cumulated_validity_div_k[temp_pids]) / N_TP)
            if verbose:
                print(APs[i], N_G, N_TP)

        del distances_matrix
        del pids_mask
        del camids_mask

        mAP = APs.mean()
        if verbose:
            print(mAP)
            plt.hist(APs, bins=50, range=[0, 1])
            plt.show()

        return mAP, APs