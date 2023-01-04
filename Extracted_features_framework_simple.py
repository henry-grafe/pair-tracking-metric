import numpy as np
import pickle

import matplotlib.pyplot as plt

from big_matrix_tools import compute_distances_matrix_simple, generate_pids_camids_equality_mask, generate_pids_camids_equality_mask_for_tracking


import random_selection_generation

class Extracted_features_framework():
    def __init__(self,  files_working_directory ,extracted_features_filename, is_distances_matrix_computed = False):
        self.files_working_directory = files_working_directory
        self.extracted_features_filename = extracted_features_filename
        self.is_distances_matrix_computed = is_distances_matrix_computed

        self.pids = np.zeros(0, dtype='int')
        self.camids = np.zeros(0, dtype='int')

        for filename in extracted_features_filename:
            temp_f, temp_pids, temp_camids = pickle.load(open(files_working_directory + filename, 'rb'))

            self.features_dim = temp_f.shape[1:]

            del temp_f

            self.pids = np.append(self.pids, temp_pids)
            self.camids = np.append(self.camids, temp_camids)


    def get_features(self):
        """
        Obtain the feature stored in self.files_working_directory + self.extracted_features_filename


        Returns: All the feature vectors contained in the file

        """
        #total_features, _, _, total_parts_visibilities, total_parts_masks = pickle.load(
        #    open(self.files_working_directory + self.extracted_features_filename[0], 'rb'))
        total_features, total_parts_visibilities, total_parts_masks = pickle.load(
            open(self.files_working_directory + self.extracted_features_filename[0], 'rb'))
        for i in range(1, len(self.extracted_features_filename)):
            #temp_f, _, _, temp_part_visibility, temp_parts_masks = pickle.load(
            #    open(self.files_working_directory + self.extracted_features_filename[i], 'rb'))
            temp_f, temp_part_visibility, temp_parts_masks = pickle.load(
                open(self.files_working_directory + self.extracted_features_filename[i], 'rb'))
            total_features = np.append(total_features, temp_f, axis=0)
            total_parts_visibilities = np.append(total_parts_visibilities, temp_part_visibility, axis=0)
            total_parts_masks = np.append(total_parts_masks, temp_parts_masks, axis=0)
            del temp_f
        return total_features


    def compute_distances_matrix(self, verbose=False):
        """

        Compute the distance matrix from the feature vectors in
        Obtains the features outputted by get_features(self)
        and compute the distances matrix. If there are N feature vectors, the distance matrix is N*N

        Returns: the distance matrix

        """
        if self.is_distances_matrix_computed:
            distances_matrix = pickle.load(open(self.files_working_directory + "distances_matrix.pickle", 'rb'))
        else:

            distances_matrix = compute_distances_matrix_simple(self.files_working_directory,
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

    def get_lists(self, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

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

        distances_matrix_flat = distances_matrix.reshape((3368 * (19281-3368),))
        pids_mask_flat = pids_mask.reshape((3368 * (19281-3368),))
        camids_mask_flat = camids_mask.reshape((3368 * (19281-3368),))



        distances_matrix_flat = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        pids_mask_flat = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]

        return distances_matrix_flat, pids_mask_flat

    def get_lists_and_confidence_special_pair_selection(self, verbose=False):
        if verbose:
            print("computing distances matrix...")


        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        selection_1, selection_2 = random_selection_generation.generate_double_complementary_random_selection_1d(19281,[3368,19281 - 3368])
        distances_matrix = distances_matrix[selection_1,:]
        distances_matrix = distances_matrix[:,selection_2]
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")




        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)

        pids_mask = pids_mask[selection_1, :]
        pids_mask = pids_mask[:, selection_2]
        camids_mask = camids_mask[selection_1, :]
        camids_mask = camids_mask[:, selection_2]

        if verbose:
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix_with_selection(selection_1, selection_2, verbose)


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

        distances_matrix_flat = distances_matrix.reshape((3368 * (19281-3368),))
        pids_mask_flat = pids_mask.reshape((3368 * (19281-3368),))
        camids_mask_flat = camids_mask.reshape((3368 * (19281-3368),))
        confidence_matrix_flat = confidence_matrix.reshape((3368 * (19281-3368),))



        distances_matrix_flat = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        confidence_matrix_flat = confidence_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        pids_mask_flat = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]



        return distances_matrix_flat, pids_mask_flat, confidence_matrix_flat

    def compute_mAP(self, verbose=False):
        """
        Compute the mAP for the given feature vectors
        Args:
            verbose: bool, plot the limited distance matrix

        Returns: the mAP of the dataset

        """

        if verbose:
            print("computing distances matrix...")

        # Computing the distance matrix, in order after to evaluate the rankings
        distances_matrix = self.compute_distances_matrix(verbose=verbose)

        if verbose:
            print("computing pids and camids masks...")

        # Compute a matrix the same size of the distance matrix,
        # True if two detections have the same pid for the pids_mask, same for the camids
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

        # compute the average precisions for each query rankings one by one
        APs = np.zeros(3368,dtype='float')
        for i in range(len(APs)):
            # Select in the matrix the distances from the query to all the gallery image
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

        return mAP