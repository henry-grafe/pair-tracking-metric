import numpy as np
import pickle
import torch
import time
import matplotlib.pyplot as plt

from big_matrix_tools import compute_distances_matrix_simple, compute_distances_matrix_part_based, generate_pids_camids_equality_mask, compute_parts_distances_matrix_part_based
import part_visibilities_generator
from scipy.interpolate import interp1d
from scipy.integrate import quad

from sklearn.metrics import roc_auc_score
import random_selection_generation

class Extracted_features_framework():
    def __init__(self,  files_working_directory ,extracted_features_filename, mode="simple"):
        self.files_working_directory = files_working_directory
        self.extracted_features_filename = extracted_features_filename
        self.is_distances_matrix_computed = False
        self.is_confidence_matrix_computed = False
        self.is_parts_distances_matrix_computed = False
        self.is_alternate_distances_matrix_computed = False
        self.alternate_distances_matrix_name = "sigmoid_10_visibilities_distances_matrix.pickle"

        if mode == "simple":
            self.mode = "simple"
            self.init_simple(files_working_directory ,extracted_features_filename)

        if mode =="part_based":
            self.mode = "part_based"
            self.init_part_based(files_working_directory, extracted_features_filename)

        else:
            print("Init mode not recognized")


        self.load_parts_distances_matrix = False
        self.parts_distances_matrix = None


    def init_simple(self, files_working_directory ,extracted_features_filename):
        self.pids = np.zeros(0,dtype='int')
        self.camids = np.zeros(0,dtype='int')

        for filename in extracted_features_filename:
            temp_f, temp_pids, temp_camids = pickle.load(open(files_working_directory + filename,'rb'))

            self.features_dim = temp_f.shape[1:]

            del temp_f

            self.pids = np.append(self.pids, temp_pids)
            self.camids = np.append(self.camids, temp_camids)






    def init_part_based(self, files_working_directory, extracted_features_filename):
        self.pids = np.zeros(0, dtype='int')
        self.camids = np.zeros(0, dtype='int')
        self.parts_visibility = np.zeros((0,9))
        self.parts_masks = np.zeros((0,9,16,8))

        for filename in extracted_features_filename:

            temp_f, temp_pids, temp_camids, temp_parts_visibility, temp_parts_masks = pickle.load(open(files_working_directory + filename, 'rb'))

            self.features_dim = temp_f.shape[1:]
            self.parts_masks_dim = temp_parts_masks.shape[1:]

            del temp_f

            self.parts_masks = np.append(self.parts_masks, temp_parts_masks, axis=0)
            self.pids = np.append(self.pids, temp_pids)
            self.camids = np.append(self.camids, temp_camids)
            #print(temp_parts_visibility.shape)
            self.parts_visibility = np.append(self.parts_visibility, temp_parts_visibility, axis=0)


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
            if self.mode == "simple":
                distances_matrix = compute_distances_matrix_simple(self.files_working_directory,
                                                                   self.extracted_features_filename, verbose=verbose)
            if self.mode == "part_based":
                distances_matrix = compute_distances_matrix_part_based(self.files_working_directory,
                                                                   self.extracted_features_filename, verbose=verbose)

            pickle.dump(distances_matrix, open(self.files_working_directory + "distances_matrix.pickle", 'wb'))
            self.is_distances_matrix_computed = True

        return distances_matrix

    def compute_parts_distances_matrix(self, verbose=False):

        if self.parts_distances_matrix is None:
            if self.is_parts_distances_matrix_computed:
                distances_matrix = pickle.load(open(self.files_working_directory + "parts_distances_matrix.pickle", 'rb'))
            else:
                if self.mode == "simple":
                    print("Parts distances matrix cant be computed on simple mode !")
                    return False
                if self.mode == "part_based":
                    distances_matrix = compute_parts_distances_matrix_part_based(self.files_working_directory,self.extracted_features_filename, verbose=verbose)

                pickle.dump(distances_matrix, open(self.files_working_directory + "parts_distances_matrix.pickle", 'wb'))
                self.is_parts_distances_matrix_computed = True

            if self.load_parts_distances_matrix:

                self.parts_distances_matrix = distances_matrix

            return self.parts_distances_matrix
        else:
            return self.parts_distances_matrix

    def compute_alternate_distances_matrix(self, a, b, verbose=False):
        #print(verbose)
        if self.is_alternate_distances_matrix_computed:
            distances_matrix = pickle.load(open(self.files_working_directory + self.alternate_distances_matrix_name, 'rb'))
        else:
            if self.mode == "simple":
                print("Alternate distances matrix cant be computed on simple mode !")
                return False
            if self.mode == "part_based":
                parts_distances_matrix = self.compute_parts_distances_matrix(verbose=verbose)
                new_visibility_coefs = part_visibilities_generator.generate_parts_visibilities_sigmoid(self.parts_masks, a, b)

                distances_matrix = np.zeros(parts_distances_matrix.shape[:2])

                for i in range(len(parts_distances_matrix)):
                    #print(new_visibility_coefs.shape, new_visibility_coefs[3368:].shape, parts_distances_matrix.shape, parts_distances_matrix[i, :, :].shape)
                    inter = new_visibility_coefs[3368:] * new_visibility_coefs[i]
                    distances_matrix[i,:] = (inter * parts_distances_matrix[i, :, :]).sum(axis=1) / (inter.sum(axis=1))
                    """
                    for j in range(len(parts_distances_matrix[0])):
                        inter = new_visibility_coefs[i]*new_visibility_coefs[j+3368]
                        distances_matrix[i,j] = (inter*parts_distances_matrix[i,j,:]).sum()/(inter.sum())
                    input((distances_matrix2[i] == distances_matrix[i]).sum())
                    """
                    #print(verbose)
                    if verbose:
                        print(i)


            pickle.dump(distances_matrix, open(self.files_working_directory + self.alternate_distances_matrix_name, 'wb'))
            self.is_alternate_distances_matrix_computed = True

        return distances_matrix

    def compute_alternate_distances_matrix_select_visibility_threshold(self, V, verbose=False):
        #print(verbose)
        if self.is_alternate_distances_matrix_computed:
            distances_matrix = pickle.load(open(self.files_working_directory + self.alternate_distances_matrix_name, 'rb'))
        else:
            if self.mode == "simple":
                print("Alternate distances matrix cant be computed on simple mode !")
                return False
            if self.mode == "part_based":
                parts_distances_matrix = self.compute_parts_distances_matrix(verbose=verbose)
                new_visibility_coefs = part_visibilities_generator.generate_parts_visibilities_multiple_thresholds(self.parts_masks, V)

                distances_matrix = np.zeros(parts_distances_matrix.shape[:2])

                for i in range(len(parts_distances_matrix)):
                    #print(new_visibility_coefs.shape, new_visibility_coefs[3368:].shape, parts_distances_matrix.shape, parts_distances_matrix[i, :, :].shape)
                    inter = new_visibility_coefs[3368:] * new_visibility_coefs[i]
                    distances_matrix[i,:] = (inter * parts_distances_matrix[i, :, :]).sum(axis=1) / (inter.sum(axis=1))
                    """
                    for j in range(len(parts_distances_matrix[0])):
                        inter = new_visibility_coefs[i]*new_visibility_coefs[j+3368]
                        distances_matrix[i,j] = (inter*parts_distances_matrix[i,j,:]).sum()/(inter.sum())
                    input((distances_matrix2[i] == distances_matrix[i]).sum())
                    """
                    #print(verbose)
                    if verbose:
                        print(i)


            pickle.dump(distances_matrix, open(self.files_working_directory + self.alternate_distances_matrix_name, 'wb'))
            self.is_alternate_distances_matrix_computed = True

        return distances_matrix

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

    def get_lists_and_confidence(self, verbose=False):
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
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix(verbose=verbose)


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


    def get_lists_and_confidence_alternate_distances(self, a, b, verbose=False):
        if verbose:
            print("computing distances matrix...")
        print(verbose)
        distances_matrix = self.compute_alternate_distances_matrix(a, b, verbose=verbose)
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix(verbose=verbose)


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

    def get_lists_and_confidence_alternate_distances_multiple_thresholds(self, V, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_alternate_distances_matrix_select_visibility_threshold(V, verbose=verbose)
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix(verbose=verbose)


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

    def get_lists_and_confidence_specific_part(self, part_index, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_parts_distances_matrix(verbose=verbose)[:,:,part_index]
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing confidence mask...")

        confidence_matrix = self.compute_confidence_coefficients_matrix(verbose=verbose)


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

    def compute_confidence_coefficients_matrix(self, verbose=False):

        if (not self.is_confidence_matrix_computed):
            confidence_matrix = np.zeros((3368, 19281-3368),dtype='float')
            for i in range(len(confidence_matrix)):
                if verbose:
                    print(i)
                for j in range(len(confidence_matrix[0])):
                    confidence_matrix[i, j] = (self.parts_visibility[i,:8]*self.parts_visibility[3368+j,:8]).sum()/8.
            if verbose:
                plt.imshow(confidence_matrix)
                plt.colorbar()
                plt.show()
            pickle.dump(confidence_matrix, open(self.files_working_directory + "confidence_matrix.pickle",'wb'))
            return confidence_matrix
        else:
            confidence_matrix = pickle.load(open(self.files_working_directory + "confidence_matrix.pickle",'rb'))
            if verbose:
                plt.imshow(confidence_matrix)
                plt.colorbar()
                plt.show()
            return confidence_matrix

    def compute_confidence_coefficients_matrix_with_selection(self, selection_1, selection_2, verbose=False):

        if (not self.is_confidence_matrix_computed):
            confidence_matrix = np.zeros((3368, 19281-3368),dtype='float')
            parts_visibility_1 = self.parts_visibility[selection_1]
            parts_visibility_2 = self.parts_visibility[selection_2]
            for i in range(len(confidence_matrix)):
                if verbose:
                    print(i)
                for j in range(len(confidence_matrix[0])):
                    confidence_matrix[i, j] = (parts_visibility_1[i,:8]*parts_visibility_2[j,:8]).sum()/8.



            if verbose:
                plt.imshow(confidence_matrix)
                plt.colorbar()
                plt.show()
            pickle.dump(confidence_matrix, open(self.files_working_directory + "confidence_matrix_selection.pickle",'wb'))
            return confidence_matrix
        else:
            confidence_matrix = pickle.load(open(self.files_working_directory + "confidence_matrix_selection.pickle",'rb'))
            if verbose:
                plt.imshow(confidence_matrix)
                plt.colorbar()
                plt.show()
            return confidence_matrix

    def compute_pAP(self, verbose=False):
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


        distances_matrix_flat_2 = distances_matrix.reshape((3368 * (19281 - 3368),))
        pids_mask_flat_2 = pids_mask.reshape((3368 * (19281 - 3368),))
        camids_mask_flat_2 = camids_mask.reshape((3368 * (19281 - 3368),))


        #interindex = np.logical_not(np.logical_and(np.logical_not(pids_mask_flat_2), np.logical_not(camids_mask_flat_2)))
        interindex = np.logical_not(camids_mask_flat_2)
        distances_matrix_flat_2 = distances_matrix_flat_2[interindex]
        camids_mask_flat_2 = camids_mask_flat_2[interindex]
        pids_mask_flat_2 = pids_mask_flat_2[interindex]


        distances_matrix_flat_2 = distances_matrix_flat_2[np.logical_not(np.logical_and(pids_mask_flat_2, camids_mask_flat_2))]
        pids_mask_flat_2 = pids_mask_flat_2[np.logical_not(np.logical_and(pids_mask_flat_2, camids_mask_flat_2))]



        if verbose:
            print("sorting distances...")
        args = distances_matrix_flat.argsort()
        args_2 = distances_matrix_flat_2.argsort()
        if verbose:
            print("reordering pids...")
        pids_mask_flat = pids_mask_flat[args]
        pids_mask_flat_2 = pids_mask_flat_2[args_2]

        if verbose:
            print("deleting superfluous objects from memory...")
        del args
        del args_2
        del camids_mask
        del camids_mask_flat
        del camids_mask_flat_2
        del pids_mask
        del distances_matrix
        del distances_matrix_flat
        del distances_matrix_flat_2

        if verbose:
            print("computing pAP1...")

        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)

        if verbose:
            a = np.cumsum(pids_mask_flat)#[np.logical_not(np.array(pids_mask_flat, dtype='bool'))]
            #a = a[::50]
            #print(a.max(), pids_mask_flat, a)
            FPs = (np.arange(1, N_G + 1) - a) / (N_G - N_TP)
            FPs = 100 * FPs / FPs.max()
            TPs = (a / N_TP)
            TPs = 100 * TPs / TPs.max()
            precision = a/np.arange(1, N_G + 1)
            recall = (a / N_TP)
            plt.plot(recall[np.logical_and(recall>0.675, recall<0.680)], precision[np.logical_and(recall>0.675, recall<0.680)],'.')
            eps = 1e-1
            plt.xlim(0 - eps, 1 + eps)
            plt.ylim(0 - eps, 1 + eps)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.title("precision-recall curve, Forward pass")
            plt.show()

            plt.plot(FPs, TPs)
            plt.plot(np.linspace(0,100,10), np.linspace(0,100,10),color = "orange")


            plt.xlim((0 - eps)* 100, (1 + eps) * 100)
            plt.ylim((0 - eps) * 100, (1 + eps) * 100)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC")
            print("computing ROC AUC...")
            print(roc_auc_score(pids_mask_flat, np.flip(np.linspace(0,1,len(pids_mask_flat)))))
            print("done")
            plt.show()

            #plt.plot(FPs,TPs,'r.')
            #plt.plot(100 * np.linspace(0,1,len(a)), 100 * np.linspace(0,1,len(a)))
            print(np.sum(TPs)/N_G, N_G, np.sum(TPs))
            #print(roc_auc_score(pids_mask_flat, np.ones(len(pids_mask_flat), dtype='bool')))
            #plt.show()
        pAP1 = np.sum(cumulated_pairs_validity_div_k[pids_mask_flat]) / N_TP
        #======================================================================
        pids_mask_flat = np.flip(np.logical_not(pids_mask_flat))
        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)
        print('inter2')
        if verbose:
            a = np.cumsum(pids_mask_flat)  # [np.logical_not(np.array(pids_mask_flat, dtype='bool'))]
            # a = a[::50]
            # print(a.max(), pids_mask_flat, a)
            FPs = (np.arange(1, N_G + 1) - a) / (N_G - N_TP)
            FPs = 100 * FPs / FPs.max()
            TPs = (a / N_TP)
            TPs = 100 * TPs / TPs.max()
            # plt.plot(FPs,TPs,'r.')
            # plt.plot(100 * np.linspace(0,1,len(a)), 100 * np.linspace(0,1,len(a)))
            print(np.sum(TPs) / N_G, N_G, np.sum(TPs))
            print(100-np.sum(FPs) / N_G, N_G, np.sum(FPs))
            # print(roc_auc_score(pids_mask_flat, np.ones(len(pids_mask_flat), dtype='bool')))
            # plt.show()
        # ======================================================================



        if verbose:
            print("computing pAP2...")

        pids_mask_flat = pids_mask_flat_2


        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)

        if verbose:
            a = np.array(np.cumsum(pids_mask_flat))#[np.logical_not(np.array(pids_mask_flat, dtype='bool'))],dtype='float')
            #a = a[::50]
            #print(a.max(), pids_mask_flat, a)
            FPs =  (np.arange(1,N_G+1) - a)/(N_G - N_TP)
            FPs = 100*FPs/FPs.max()
            TPs =  (a/N_TP)
            TPs = 100 * TPs / TPs.max()
            precision = a / np.arange(1, N_G + 1)
            recall = (a / N_TP)
            plt.plot(recall, precision,'-')
            eps = 1e-1
            plt.xlim(0-eps, 1+eps)
            plt.ylim(0-eps, 1+eps)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.title("precision-recall curve, Backward pass")
            plt.show()

            plt.plot(FPs, TPs)
            plt.plot(np.linspace(0,100,10), np.linspace(0,100,10),color = "orange")

            plt.xlim((0 - eps) * 100, (1 + eps) * 100)
            plt.ylim((0 - eps) * 100, (1 + eps) * 100)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC")

            print("computing ROC AUC...")
            print(roc_auc_score(pids_mask_flat, np.flip(np.linspace(0, 1, len(pids_mask_flat)))))
            print("done")


            plt.show()
            #plt.show()

        pAP2 = np.sum(cumulated_pairs_validity_div_k[pids_mask_flat]) / N_TP

        return pAP1, pAP2
        # print(np.sum(cumulated_pairs_validity_div_k[pairs_validity_flattened_sorted]) / N_TP)

    def compute_pAP_alternate_distance_select_visibility_thresholds(self,V, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_alternate_distances_matrix_select_visibility_threshold(V, verbose=verbose)

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


        distances_matrix_flat_2 = distances_matrix.reshape((3368 * (19281 - 3368),))
        pids_mask_flat_2 = pids_mask.reshape((3368 * (19281 - 3368),))
        camids_mask_flat_2 = camids_mask.reshape((3368 * (19281 - 3368),))


        #interindex = np.logical_not(np.logical_and(np.logical_not(pids_mask_flat_2), np.logical_not(camids_mask_flat_2)))
        interindex = np.logical_not(camids_mask_flat_2)
        distances_matrix_flat_2 = distances_matrix_flat_2[interindex]
        camids_mask_flat_2 = camids_mask_flat_2[interindex]
        pids_mask_flat_2 = pids_mask_flat_2[interindex]


        distances_matrix_flat_2 = distances_matrix_flat_2[np.logical_not(np.logical_and(pids_mask_flat_2, camids_mask_flat_2))]
        pids_mask_flat_2 = pids_mask_flat_2[np.logical_not(np.logical_and(pids_mask_flat_2, camids_mask_flat_2))]



        if verbose:
            print("sorting distances...")
        args = distances_matrix_flat.argsort()
        args_2 = distances_matrix_flat_2.argsort()
        if verbose:
            print("reordering pids...")
        pids_mask_flat = pids_mask_flat[args]
        pids_mask_flat_2 = pids_mask_flat_2[args_2]

        if verbose:
            print("deleting superfluous objects from memory...")
        del args
        del args_2
        del camids_mask
        del camids_mask_flat
        del camids_mask_flat_2
        del pids_mask
        del distances_matrix
        del distances_matrix_flat
        del distances_matrix_flat_2

        if verbose:
            print("computing pAP1...")

        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)

        if verbose:
            a = np.cumsum(pids_mask_flat)#[np.logical_not(np.array(pids_mask_flat, dtype='bool'))]
            #a = a[::50]
            #print(a.max(), pids_mask_flat, a)
            FPs = (np.arange(1, N_G + 1) - a) / (N_G - N_TP)
            FPs = 100 * FPs / FPs.max()
            TPs = (a / N_TP)
            TPs = 100 * TPs / TPs.max()
            precision = a/np.arange(1, N_G + 1)
            recall = (a / N_TP)
            plt.plot(recall[np.logical_and(recall>0.675, recall<0.680)], precision[np.logical_and(recall>0.675, recall<0.680)],'.')
            eps = 1e-1
            plt.xlim(0 - eps, 1 + eps)
            plt.ylim(0 - eps, 1 + eps)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.title("precision-recall curve, Forward pass")
            plt.show()

            plt.plot(FPs, TPs)
            plt.plot(np.linspace(0,100,10), np.linspace(0,100,10),color = "orange")


            plt.xlim((0 - eps)* 100, (1 + eps) * 100)
            plt.ylim((0 - eps) * 100, (1 + eps) * 100)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC")
            print("computing ROC AUC...")
            print(roc_auc_score(pids_mask_flat, np.flip(np.linspace(0,1,len(pids_mask_flat)))))
            print("done")
            plt.show()

            #plt.plot(FPs,TPs,'r.')
            #plt.plot(100 * np.linspace(0,1,len(a)), 100 * np.linspace(0,1,len(a)))
            print(np.sum(TPs)/N_G, N_G, np.sum(TPs))
            #print(roc_auc_score(pids_mask_flat, np.ones(len(pids_mask_flat), dtype='bool')))
            #plt.show()
        pAP1 = np.sum(cumulated_pairs_validity_div_k[pids_mask_flat]) / N_TP
        #======================================================================
        pids_mask_flat = np.flip(np.logical_not(pids_mask_flat))
        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)
        print('inter2')
        if verbose:
            a = np.cumsum(pids_mask_flat)  # [np.logical_not(np.array(pids_mask_flat, dtype='bool'))]
            # a = a[::50]
            # print(a.max(), pids_mask_flat, a)
            FPs = (np.arange(1, N_G + 1) - a) / (N_G - N_TP)
            FPs = 100 * FPs / FPs.max()
            TPs = (a / N_TP)
            TPs = 100 * TPs / TPs.max()
            # plt.plot(FPs,TPs,'r.')
            # plt.plot(100 * np.linspace(0,1,len(a)), 100 * np.linspace(0,1,len(a)))
            print(np.sum(TPs) / N_G, N_G, np.sum(TPs))
            print(100-np.sum(FPs) / N_G, N_G, np.sum(FPs))
            # print(roc_auc_score(pids_mask_flat, np.ones(len(pids_mask_flat), dtype='bool')))
            # plt.show()
        # ======================================================================



        if verbose:
            print("computing pAP2...")

        pids_mask_flat = pids_mask_flat_2


        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)

        if verbose:
            a = np.array(np.cumsum(pids_mask_flat))#[np.logical_not(np.array(pids_mask_flat, dtype='bool'))],dtype='float')
            #a = a[::50]
            #print(a.max(), pids_mask_flat, a)
            FPs =  (np.arange(1,N_G+1) - a)/(N_G - N_TP)
            FPs = 100*FPs/FPs.max()
            TPs =  (a/N_TP)
            TPs = 100 * TPs / TPs.max()
            precision = a / np.arange(1, N_G + 1)
            recall = (a / N_TP)
            plt.plot(recall, precision,'-')
            eps = 1e-1
            plt.xlim(0-eps, 1+eps)
            plt.ylim(0-eps, 1+eps)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.title("precision-recall curve, Backward pass")
            plt.show()

            plt.plot(FPs, TPs)
            plt.plot(np.linspace(0,100,10), np.linspace(0,100,10),color = "orange")

            plt.xlim((0 - eps) * 100, (1 + eps) * 100)
            plt.ylim((0 - eps) * 100, (1 + eps) * 100)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC")

            print("computing ROC AUC...")
            print(roc_auc_score(pids_mask_flat, np.flip(np.linspace(0, 1, len(pids_mask_flat)))))
            print("done")


            plt.show()
            #plt.show()

        pAP2 = np.sum(cumulated_pairs_validity_div_k[pids_mask_flat]) / N_TP

        return pAP1, pAP2
        # print(np.sum(cumulated_pairs_validity_div_k[pairs_validity_flattened_sorted]) / N_TP)

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

    def compute_gAP(self, verbose=False):
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
            print("computing gAP...")

        APs = np.zeros(len(distances_matrix),dtype='float')
        for i in range(len(APs)):
            temp_distances = distances_matrix[i]

            temp_camids = camids_mask[i]
            temp_pids = pids_mask[i]

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

        gAP = APs.mean()
        if verbose:
            print(gAP)
            plt.hist(APs, bins=50, range=[0, 1])
            plt.show()

        return gAP, APs

    def plot_pair_distances_distributions(self, verbose=False):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        distances_matrix = (distances_matrix-distances_matrix.min())/(distances_matrix.max()-distances_matrix.min())
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

        valid_pairs = np.sum(pids_mask_flat)
        print(f"There is {valid_pairs} valid pairs, which make {100*valid_pairs / len(pids_mask_flat):.4f} % of all pairs")
        plt.pie([valid_pairs, len(pids_mask_flat) - valid_pairs], labels=["SAME ID", "DIFFERENT ID"])
        plt.show()



        plt.title("Pairs distances distributions")
        plt.xlabel("distance (cosine distance)")
        plt.ylabel("density")


        inter = distances_matrix_flat[np.logical_and(pids_mask_flat, camids_mask_flat)]
        plt.hist(inter, bins=100, label="same identities, same cameras", density=True, range=[0,1])
        print(f"same identities, same cameras distribution mean is {inter.mean():.2f}, std is {inter.std():.2f}")

        inter = distances_matrix_flat[np.logical_and(np.logical_not(pids_mask_flat), np.logical_not(camids_mask_flat))]
        data = plt.hist(inter, bins=100, label="different identities, different cameras", density=True, range=[0,1])
        print(f"data is {data}")
        print(f"different identities, different cameras distribution mean is {inter.mean():.2f}, std is {inter.std():.2f}")

        inter = distances_matrix_flat[np.logical_and(np.logical_not(pids_mask_flat), camids_mask_flat)]
        plt.hist(inter, bins=100, label="different identities, same cameras", density=True, range=[0,1])
        print(f"different identities, same cameras distribution mean is {inter.mean():.2f}, std is {inter.std():.2f}")

        inter = distances_matrix_flat[np.logical_and(pids_mask_flat, np.logical_not(camids_mask_flat))]
        plt.hist(inter, bins=100, label="same identities, different cameras", density=True, range=[0,1])
        print(f"same identities, different cameras distribution mean is {inter.mean():.2f}, std is {inter.std():.2f}")

        plt.plot(np.ones(5)*0.2416, np.linspace(0,10,5),'k-',label="Distance threshold for maximal f1-score")

        plt.legend()
        plt.show()

    def plot_f1_score_vs_threshold(self, verbose = False, plot = True):
        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        distances_matrix = (distances_matrix - distances_matrix.min()) / (
                    distances_matrix.max() - distances_matrix.min())
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


        if verbose:
            print("sorting distances...")
        args = distances_matrix_flat.argsort()
        if verbose:
            print("reordering pids...")
        pids_mask_flat = pids_mask_flat[args]

        if verbose:
            print("reordering distances...")

        distances_matrix_flat = distances_matrix_flat[args]

        if verbose:
            print("deleting superfluous objects from memory...")
        del args
        del camids_mask
        del camids_mask_flat
        del pids_mask
        del distances_matrix

        if verbose:
            print("computing pAP1...")

        N_G = float(pids_mask_flat.shape[0])
        N_TP = float(np.sum(pids_mask_flat))

        cumulated_pairs_validity_div_k = np.cumsum(pids_mask_flat) / np.arange(1, N_G + 1)



        if plot:
            a = np.cumsum(pids_mask_flat)#[np.logical_not(np.array(pids_mask_flat, dtype='bool'))]
            #a = a[::50]
            #print(a.max(), pids_mask_flat, a)
            FPs = (np.arange(1, N_G + 1) - a) / (N_G - N_TP)
            FPs = 100 * FPs / FPs.max()
            TPs = (a / N_TP)
            TPs = 100 * TPs / TPs.max()
            precision = a/np.arange(1, N_G + 1)
            recall = (a / N_TP)

            f1_score = 2*precision*recall/(precision + recall)

            plt.plot(distances_matrix_flat[::50], f1_score[::50],'-')
            eps = 1e-1
            plt.xlim(0 - eps, 1 + eps)
            plt.ylim(0 - eps, 1 + eps)
            plt.xlabel("distance threshold (normalized)")
            plt.ylabel("f1-score")
            plt.title("f1-score in function of distance threshold")
            plt.show()


    def weighted_pair_AUC_generation(self, verbose=False, plot=False, argu=2, tolerance = 0.):


        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        distances_matrix = (distances_matrix - distances_matrix.min()) / (
                distances_matrix.max() - distances_matrix.min())
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing valid identities masks (excluding same identities if they have same camids)...")

        valid_pairs_mask = np.copy(pids_mask)

        valid_pairs_mask[camids_mask] = False


        distances_matrix_flat = distances_matrix.reshape((3368 * (19281-3368),))
        pids_mask_flat = pids_mask.reshape((3368 * (19281-3368),))
        camids_mask_flat = camids_mask.reshape((3368 * (19281-3368),))



        distance_list = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        validity_list = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]

        valid_pairs_list = distance_list[validity_list]
        valid_pairs_mean = valid_pairs_list.mean()
        valid_pairs_std = valid_pairs_list.std()
        del valid_pairs_list
        invalid_pairs_list = distance_list[np.logical_not(validity_list)]
        invalid_pairs_mean = invalid_pairs_list.mean()
        invalid_pairs_std = invalid_pairs_list.std()
        del invalid_pairs_list

        def g_valid(d, threshold):
            ret = np.zeros(d.shape)
            ret[d < (threshold + tolerance)] = 1.
            # ret[d > threshold] = argu*(d[d>threshold]-threshold) + 1
            ret[d > (threshold + tolerance)] = 1 * np.exp(
                (argu/valid_pairs_std) * (d[d > (threshold + tolerance)] - (threshold + tolerance)))
            return ret

        def g_invalid(d, threshold):
            ret = np.zeros(d.shape)
            ret[d > (threshold - tolerance)] = 1.
            # ret[d < threshold] = -argu * np.exp(d[d<threshold] - threshold) + 1
            ret[d < (threshold - tolerance)] = 1 * np.exp(
                -(argu/invalid_pairs_std) * (d[d < (threshold - tolerance)] - (threshold - tolerance)))
            return ret

        if plot:
            t = np.linspace(0, 1, 1000)

            plt.subplot(1, 2, 1)
            plt.plot(t, g_valid(t, 0.5))
            plt.subplot(1, 2, 2)
            plt.plot(t, g_invalid(t, 0.5))
            plt.show()

        thresholds_list = np.linspace(0,1,300)
        TPR_list = np.zeros(thresholds_list.shape)
        FPR_list = np.zeros(thresholds_list.shape)

        for i in range(len(thresholds_list)):
            current_threshold = thresholds_list[i]
            TP_distances_list = distance_list[np.logical_and(distance_list < current_threshold, validity_list)]
            FN_distances_list = distance_list[np.logical_and(distance_list >= current_threshold, validity_list)]
            TN_distances_list = distance_list[np.logical_and(distance_list >= current_threshold, np.logical_not(validity_list))]
            FP_distances_list = distance_list[np.logical_and(distance_list < current_threshold, np.logical_not(validity_list))]

            g_TP_distances_list = g_valid(TP_distances_list, current_threshold)
            g_FN_distances_list = g_valid(FN_distances_list, current_threshold)
            g_TN_distances_list = g_invalid(TN_distances_list, current_threshold)
            g_FP_distances_list = g_invalid(FP_distances_list, current_threshold)

            modified_TP_list = g_TP_distances_list / ((g_TP_distances_list.sum() + g_FN_distances_list.sum())/(len(TP_distances_list) + len(FN_distances_list)))
            modified_FN_list = g_FN_distances_list / ((g_TP_distances_list.sum() + g_FN_distances_list.sum())/(len(TP_distances_list) + len(FN_distances_list)))

            modified_TN_list = g_TN_distances_list / ((g_TN_distances_list.sum() + g_FP_distances_list.sum())/(len(TN_distances_list) + len(FP_distances_list)))
            modified_FP_list = g_FP_distances_list / ((g_TN_distances_list.sum() + g_FP_distances_list.sum())/(len(TN_distances_list) + len(FP_distances_list)))

            TPR_list[i] = modified_TP_list.sum() / (modified_TP_list.sum() + modified_FN_list.sum())
            FPR_list[i] = modified_FP_list.sum() / (modified_FP_list.sum() + modified_TN_list.sum())

            if verbose:
                print(f"step {i+1}/{len(thresholds_list)}, {TPR_list[i]}, {FPR_list[i]}")

        if plot:
            plt.plot(FPR_list, TPR_list, 'o-')
            plt.show()

        FPR_lens = np.diff(FPR_list)
        TPR_averages = 0.5*(TPR_list[:-1] + TPR_list[1:])
        ROC_AUC = (FPR_lens*TPR_averages).sum()
        return FPR_list, TPR_list, ROC_AUC


    def weighted_pair_f1_score_vs_threshold_generation(self, verbose=False, plot=False, argu=2, tolerance = 0.):

        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        distances_matrix = (distances_matrix - distances_matrix.min()) / (
                distances_matrix.max() - distances_matrix.min())
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing valid identities masks (excluding same identities if they have same camids)...")

        valid_pairs_mask = np.copy(pids_mask)

        valid_pairs_mask[camids_mask] = False


        distances_matrix_flat = distances_matrix.reshape((3368 * (19281-3368),))
        pids_mask_flat = pids_mask.reshape((3368 * (19281-3368),))
        camids_mask_flat = camids_mask.reshape((3368 * (19281-3368),))



        distance_list = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        validity_list = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]

        valid_pairs_list = distance_list[validity_list]
        valid_pairs_mean = valid_pairs_list.mean()
        valid_pairs_std = valid_pairs_list.std()
        del valid_pairs_list
        invalid_pairs_list = distance_list[np.logical_not(validity_list)]
        invalid_pairs_mean = invalid_pairs_list.mean()
        invalid_pairs_std = invalid_pairs_list.std()
        del invalid_pairs_list

        print(invalid_pairs_mean,valid_pairs_mean,invalid_pairs_std,valid_pairs_std)
        input('next')

        if verbose:
            print("computing distances matrix...")

        distances_matrix = self.compute_distances_matrix(verbose=verbose)
        distances_matrix = distances_matrix[:3368, 3368:]
        distances_matrix = (distances_matrix - distances_matrix.min()) / (
                distances_matrix.max() - distances_matrix.min())
        #distances_matrix = pickle.load(open("D:/Pair-reID_data/test_dir/distances_matrix.pickle", 'rb'))
        if verbose:
            print("computing pids and camids masks...")

        pids_mask, camids_mask = generate_pids_camids_equality_mask(self.pids, self.camids)
        pids_mask, camids_mask = pids_mask[:3368, 3368:], camids_mask[:3368, 3368:]

        if verbose:
            print("computing valid identities masks (excluding same identities if they have same camids)...")

        valid_pairs_mask = np.copy(pids_mask)

        valid_pairs_mask[camids_mask] = False


        distances_matrix_flat = distances_matrix.reshape((3368 * (19281-3368),))
        pids_mask_flat = pids_mask.reshape((3368 * (19281-3368),))
        camids_mask_flat = camids_mask.reshape((3368 * (19281-3368),))



        distance_list = distances_matrix_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]
        validity_list = pids_mask_flat[np.logical_not(np.logical_and(pids_mask_flat, camids_mask_flat))]

        valid_pairs_list = distance_list[validity_list]
        valid_pairs_mean = valid_pairs_list.mean()
        valid_pairs_std = valid_pairs_list.std()
        del valid_pairs_list
        invalid_pairs_list = distance_list[np.logical_not(validity_list)]
        invalid_pairs_mean = invalid_pairs_list.mean()
        invalid_pairs_std = invalid_pairs_list.std()
        del invalid_pairs_list


        def g_valid(d, threshold):
            ret = np.zeros(d.shape)
            ret[d < (threshold + tolerance)] = 1.
            # ret[d > threshold] = argu*(d[d>threshold]-threshold) + 1
            ret[d > (threshold + tolerance)] = 1 * np.exp(
                (argu/valid_pairs_std) * (d[d > (threshold + tolerance)] - (threshold + tolerance)))
            return ret

        def g_invalid(d, threshold):
            ret = np.zeros(d.shape)
            ret[d > (threshold - tolerance)] = 1.
            # ret[d < threshold] = -argu * np.exp(d[d<threshold] - threshold) + 1
            ret[d < (threshold - tolerance)] = 1 * np.exp(
                -(argu/invalid_pairs_std) * (d[d < (threshold - tolerance)] - (threshold - tolerance)))
            return ret

        if False:
            t = np.linspace(0, 1, 1000)

            plt.subplot(1, 2, 1)
            plt.plot(t, g_valid(t, 0.5))
            plt.subplot(1, 2, 2)
            plt.plot(t, g_invalid(t, 0.5))
            plt.show()

        thresholds_list = np.linspace(0,0.5,300)

        precisions_list = np.zeros(thresholds_list.shape)
        recalls_list = np.zeros(thresholds_list.shape)

        for i in range(len(thresholds_list)):
            current_threshold = thresholds_list[i]
            TP_distances_list = distance_list[np.logical_and(distance_list < current_threshold, validity_list)]
            FN_distances_list = distance_list[np.logical_and(distance_list >= current_threshold, validity_list)]
            TN_distances_list = distance_list[np.logical_and(distance_list >= current_threshold, np.logical_not(validity_list))]
            FP_distances_list = distance_list[np.logical_and(distance_list < current_threshold, np.logical_not(validity_list))]

            g_TP_distances_list = g_valid(TP_distances_list, current_threshold)
            g_FP_distances_list = g_invalid(FP_distances_list, current_threshold)

            modified_TP_list = g_TP_distances_list / ((g_TP_distances_list.sum() + g_FP_distances_list.sum())/(len(TP_distances_list) + len(FP_distances_list)))
            modified_FP_list = g_FP_distances_list / ((g_TP_distances_list.sum() + g_FP_distances_list.sum())/(len(TP_distances_list) + len(FP_distances_list)))

            #input(modified_TP_list)
            #input(modified_FP_list)

            precisions_list[i] = modified_TP_list.sum() / (modified_TP_list.sum() + modified_FP_list.sum())
            recalls_list[i] = len(TP_distances_list) / (len(TP_distances_list) + len(FN_distances_list))

            if verbose:
                print(f"step {i+1}/{len(thresholds_list)}, precision = {precisions_list[i]*100:.6f} %, recall = {recalls_list[i]*100:.6f} %")

        f1_scores_list = 2./((1./precisions_list)+(1./recalls_list))


        if plot:
            plt.plot(thresholds_list, f1_scores_list, '-')
            #plt.show()


        return precisions_list, recalls_list, f1_scores_list


def weighted_pair_AUC_generation(distances_list, pairs_validity, verbose=False, plot=False, argu=2, tolerance=0.):
    distance_list = distances_list
    validity_list = pairs_validity

    valid_pairs_list = distance_list[validity_list]
    valid_pairs_mean = valid_pairs_list.mean()
    valid_pairs_std = valid_pairs_list.std()
    del valid_pairs_list
    invalid_pairs_list = distance_list[np.logical_not(validity_list)]
    invalid_pairs_mean = invalid_pairs_list.mean()
    invalid_pairs_std = invalid_pairs_list.std()
    del invalid_pairs_list

    def g_valid(d, threshold):
        ret = np.zeros(d.shape)
        ret[d < (threshold + tolerance)] = 1.
        # ret[d > threshold] = argu*(d[d>threshold]-threshold) + 1
        ret[d > (threshold + tolerance)] = 1 * np.exp(
            (argu / valid_pairs_std) * (d[d > (threshold + tolerance)] - (threshold + tolerance)))
        return ret

    def g_invalid(d, threshold):
        ret = np.zeros(d.shape)
        ret[d > (threshold - tolerance)] = 1.
        # ret[d < threshold] = -argu * np.exp(d[d<threshold] - threshold) + 1
        ret[d < (threshold - tolerance)] = 1 * np.exp(
            -(argu / invalid_pairs_std) * (d[d < (threshold - tolerance)] - (threshold - tolerance)))
        return ret

    if plot:
        t = np.linspace(0, 1, 1000)

        plt.subplot(1, 2, 1)
        plt.title("g^{valid} function")
        plt.xlabel("distance")
        plt.ylabel("g(d)")
        plt.plot(t, g_valid(t, 0.5))
        plt.subplot(1, 2, 2)
        plt.title("g^{invalid} function")
        plt.xlabel("distance")
        plt.ylabel("g(d)")
        plt.plot(t, g_invalid(t, 0.5))
        plt.show()

    thresholds_list = np.linspace(0, 1, 300)
    TPR_list = np.zeros(thresholds_list.shape)
    FPR_list = np.zeros(thresholds_list.shape)

    for i in range(len(thresholds_list)):
        current_threshold = thresholds_list[i]
        TP_distances_list = distance_list[np.logical_and(distance_list < current_threshold, validity_list)]
        FN_distances_list = distance_list[np.logical_and(distance_list >= current_threshold, validity_list)]
        TN_distances_list = distance_list[
            np.logical_and(distance_list >= current_threshold, np.logical_not(validity_list))]
        FP_distances_list = distance_list[
            np.logical_and(distance_list < current_threshold, np.logical_not(validity_list))]

        g_TP_distances_list = g_valid(TP_distances_list, current_threshold)
        g_FN_distances_list = g_valid(FN_distances_list, current_threshold)
        g_TN_distances_list = g_invalid(TN_distances_list, current_threshold)
        g_FP_distances_list = g_invalid(FP_distances_list, current_threshold)

        modified_TP_list = g_TP_distances_list / ((g_TP_distances_list.sum() + g_FN_distances_list.sum()) / (
                len(TP_distances_list) + len(FN_distances_list)))
        modified_FN_list = g_FN_distances_list / ((g_TP_distances_list.sum() + g_FN_distances_list.sum()) / (
                len(TP_distances_list) + len(FN_distances_list)))

        modified_TN_list = g_TN_distances_list / ((g_TN_distances_list.sum() + g_FP_distances_list.sum()) / (
                len(TN_distances_list) + len(FP_distances_list)))
        modified_FP_list = g_FP_distances_list / ((g_TN_distances_list.sum() + g_FP_distances_list.sum()) / (
                len(TN_distances_list) + len(FP_distances_list)))

        TPR_list[i] = modified_TP_list.sum() / (modified_TP_list.sum() + modified_FN_list.sum())
        FPR_list[i] = modified_FP_list.sum() / (modified_FP_list.sum() + modified_TN_list.sum())

        if verbose:
            print(f"step {i + 1}/{len(thresholds_list)}, {TPR_list[i]}, {FPR_list[i]}")

    if plot:
        plt.plot(FPR_list, TPR_list, 'o-')
        plt.show()

    FPR_lens = np.diff(FPR_list)
    TPR_averages = 0.5 * (TPR_list[:-1] + TPR_list[1:])
    ROC_AUC = (FPR_lens * TPR_averages).sum()
    return FPR_list, TPR_list, ROC_AUC
