import matplotlib.pyplot as plt
import numpy as np
from Extracted_features_framework_part_based import Extracted_features_framework
from big_matrix_tools import compute_distances_matrix_simple, compute_distances_matrix_part_based, generate_pids_camids_equality_mask
import scipy.special
import scipy.optimize
import cv2
import big_matrix_tools
import os

class Pair_plotting():
    def __init__(self,verbose=False):
        self.extracted_feature_framework = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"],is_distances_matrix_computed=True, is_confidence_matrix_computed=True)

        if verbose:
            print("loading features...")
        self.features = self.extracted_feature_framework.get_features()
        if verbose:
            print("computing distance and pids list for histogram...")
        self.distances_matrix_flat_for_hist, self.pids_mask_flat_for_hist, self.confidence_matrix_flat, self.selection = self.extracted_feature_framework.get_lists_and_confidence_special_selection(7000,2000,verbose=verbose, return_selection=True)

        if verbose:
            print("computing distances matrix...")
        self.distances_matrix = self.extracted_feature_framework.compute_distances_matrix(verbose=verbose)
        self.distances_matrix = self.distances_matrix[self.selection, :]
        self.distances_matrix = self.distances_matrix[:, self.selection]
        print(self.distances_matrix.shape)
        if verbose:
            print("generating pids and camids equality masks")

        self.pids_matrix, self.camids_matrix = big_matrix_tools.generate_pids_camids_equality_mask_for_tracking(self.extracted_feature_framework.pids, self.extracted_feature_framework.camids)
        self.pids_matrix, self.camids_matrix = self.pids_matrix[self.selection, :], self.camids_matrix[self.selection, :]
        self.pids_matrix, self.camids_matrix = self.pids_matrix[:,self.selection], self.camids_matrix[:,self.selection]

        ind2_matrix, ind1_matrix = np.meshgrid(self.selection, self.selection)

        print(ind1_matrix, ind2_matrix)
        if verbose:
            print("flattening...")
        self.distances_matrix_flat = self.distances_matrix.flatten()
        self.pids_matrix_flat = self.pids_matrix.flatten()
        self.camids_matrix_flat = self.camids_matrix.flatten()
        self.ind1_matrix_flat = ind1_matrix.flatten()
        self.ind2_matrix_flat = ind2_matrix.flatten()
        if verbose:
            print("sorting...")
        self.args = self.distances_matrix_flat.argsort()
        if verbose:
            print("reordering...")
        self.distances_matrix_flat = self.distances_matrix_flat[self.args]
        self.pids_matrix_flat = self.pids_matrix_flat[self.args]
        self.camids_matrix_flat = self.camids_matrix_flat[self.args]
        self.ind1_matrix_flat = self.ind1_matrix_flat[self.args]
        self.ind2_matrix_flat = self.ind2_matrix_flat[self.args]

    def find_pairs_with_given_specs(self, desired_distance, desired_pids_equality):
        if desired_pids_equality == "S":
            desired_pids_equality = True
        else:
            desired_pids_equality = False

        restricted_matrix_selection = np.logical_and((self.pids_matrix_flat == desired_pids_equality), np.logical_not(
            np.logical_and(self.pids_matrix_flat, self.camids_matrix_flat)))

        distances_matrix_flat_temp = self.distances_matrix_flat[restricted_matrix_selection]
        ind1_matrix_flat_temp = self.ind1_matrix_flat[restricted_matrix_selection]
        ind2_matrix_flat_temp = self.ind2_matrix_flat[restricted_matrix_selection]
        current_distance = self.distances_matrix_flat[0]
        c = 1
        while current_distance < desired_distance:
            current_distance = distances_matrix_flat_temp[c]
            c += 1
            if c % 100 == 0:
                print(current_distance, desired_distance)
        ind1 = ind1_matrix_flat_temp[c]
        ind2 = ind2_matrix_flat_temp[c]
        return ind1, ind2

    def plot_pair_with_details(self, ind1, ind2):
        img1 = cv2.imread("D:\\Pair-reID_data\\reference_repertory\\" + str(ind1) + ".jpg")
        plt.subplot(3, 2, 1)
        plt.imshow(img1)


        img2 = cv2.imread("D:\\Pair-reID_data\\reference_repertory\\" + str(ind2) + ".jpg")
        plt.subplot(3, 2, 2)
        plt.imshow(img2)
        plt.subplot(3, 2, 5)
        plt.hist(self.distances_matrix_flat_for_hist[self.pids_mask_flat_for_hist], bins=100, density=True)
        plt.hist(self.distances_matrix_flat_for_hist[np.logical_not(self.pids_mask_flat_for_hist)], bins=100, density=True,
                 alpha=0.7)
        corresponding_ind1 = np.argmax(self.selection == ind1)
        corresponding_ind2 = np.argmax(self.selection == ind2)
        print(self.distances_matrix[corresponding_ind1, corresponding_ind2], self.pids_matrix[corresponding_ind1, corresponding_ind2], self.camids_matrix[corresponding_ind1, corresponding_ind2],
              ind1, ind2)
        pid1, pid2 = self.extracted_feature_framework.pids[ind1], self.extracted_feature_framework.pids[ind2]
        print(f"pids of the two images : {pid1} and {pid2}")
        current_distance = self.distances_matrix[corresponding_ind1, corresponding_ind2]
        plt.plot([current_distance, current_distance], [0, 1], 'r-')
        dist_short = str(current_distance).split(".")[-1]
        dir_name = str(pid1)+"_"+str(pid2)+"_"+dist_short
        os.mkdir("D:\\Pair-reID_data\\images_plot\\"+dir_name)
        cv2.imwrite("D:\\Pair-reID_data\\images_plot\\"+dir_name+"\\img1.jpg", img1[:,:,[2,1,0]])
        cv2.imwrite("D:\\Pair-reID_data\\images_plot\\"+dir_name+"\\img2.jpg", img2[:,:,[2,1,0]])

        plt.show()


plotter = Pair_plotting(verbose=True)
while True:
    ind1, ind2 = plotter.find_pairs_with_given_specs(float(input("desired distance : ")), input("Same (S) or Different (D) : "))
    plotter.plot_pair_with_details(ind1,ind2)