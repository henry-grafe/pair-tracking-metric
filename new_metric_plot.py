import numpy as np
import matplotlib.pyplot as plt

def plot_error_information_curve(error_rate, n_useful_links, actual_ratio, pids_mask_flat, ratio):
    plt.plot(n_useful_links, error_rate, '.', label='new metric curve')
    plt.ylabel("error rate")
    plt.xlabel("quantity")
    plt.title("New metric")
    plt.show()

def plot_error_information_curve_without_show(error_rate, n_useful_links, actual_ratio, pids_mask_flat, ratio, label=None):
    plt.plot(n_useful_links, error_rate, '.', label=label)
    plt.ylabel("error rate")
    plt.xlabel("quantity")
    plt.title("New metric")