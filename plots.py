import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def radial_error_vs_ere_graph(all_radial_errors, all_expected_radial_errors, save_path, n_bin=36):

    # Bin the ere and calculate the radial error for each bin
    binned_eres = []
    binned_errors = []
    sorted_indices = np.argsort(all_expected_radial_errors)
    for l in range(int(len(all_expected_radial_errors) / n_bin)):
        binned_indices = sorted_indices[l * n_bin: (l + 1) * n_bin]
        binned_eres.append(np.mean(np.take(all_expected_radial_errors, binned_indices)))
        binned_errors.append(np.mean(np.take(all_radial_errors, binned_indices)))
    correlation = np.corrcoef(binned_eres, binned_errors)[0, 1]

    # Plot graph
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    plt.xlabel('Expected Radial Error (ERE)', fontsize=14)
    plt.ylabel('True Radial Error', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.text(0.5, 0.075, "CORRELATION={:.2f}".format(correlation), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large', transform=ax.transAxes)
    ax.scatter(binned_eres, binned_errors, c='lime', edgecolors='black', zorder=3)
    plt.savefig(save_path)
    plt.close()


def roc_outlier_graph(all_radial_errors, all_expected_radial_errors, save_path, outlier_threshold=4.0):
    outliers = all_radial_errors > outlier_threshold

    fpr, tpr, thresholds = roc_curve(outliers, all_expected_radial_errors)
    auc = roc_auc_score(outliers, all_expected_radial_errors)

    # Plot graph
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    plt.xlabel("False Positive Rate (FPR)", fontsize=14)
    plt.ylabel("True Positive Rate (TPR)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot([0, 1], [0, 1], c='black', linestyle='dashed')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True)

    plt.plot(fpr, tpr, c="blue")
    plt.text(0.42, 0.075, 'Area Under Curve={:.2f}'.format(auc), backgroundcolor=(0.8, 0.8, 0.8, 0.8), size='x-large',
             transform=ax.transAxes)
    plt.savefig(save_path)
    plt.close()


# At the moment this assumes all images have the same resolution
def reliability_diagram(all_radial_errors, all_mode_probabilities, save_path,
                        n_of_bins=10, x_max=0.15, pixel_size=0.30234375, do_not_save=False):

    bins = np.linspace(0, x_max, n_of_bins + 1)
    bins[-1] = 1.1
    widths = x_max / n_of_bins
    radius = math.sqrt((pixel_size**2) / math.pi)
    correct_predictions = all_radial_errors < radius

    # a 10 length array with values adding to 19
    count_for_each_bin, _ = np.histogram(all_mode_probabilities, bins=bins)

    # total confidence in each bin
    total_confidence_for_each_bin, _, bin_indices \
        = stats.binned_statistic(all_mode_probabilities, all_mode_probabilities, 'sum', bins=bins)

    no_of_correct_preds = np.zeros(len(bins) - 1)
    for bin_idx, pred_correct in zip(bin_indices, correct_predictions):
        no_of_correct_preds[bin_idx - 1] += pred_correct

    # get confidence of each bin
    avg_conf_for_each_bin = total_confidence_for_each_bin / count_for_each_bin.astype(float)
    avg_acc_for_each_bin = no_of_correct_preds / count_for_each_bin.astype(float)

    n = float(np.sum(count_for_each_bin))
    ece = 0.0
    for i in range(len(bins) - 1):
        ece += count_for_each_bin[i] / n * np.abs(avg_acc_for_each_bin[i] - avg_conf_for_each_bin[i])
    ece *= 100

    # save plot
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)

    plt.subplots_adjust(left=0.15)
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(zorder=0)
    plt.xlim(0.0, x_max)
    plt.ylim(0.0, x_max * 2)
    plt.bar(bins[:-1], avg_acc_for_each_bin, align='edge', width=widths, color='blue', edgecolor='black', label='Accuracy', zorder=3)
    plt.bar(bins[:-1], avg_conf_for_each_bin, align='edge', width=widths, color='lime', edgecolor='black', alpha=0.5,
            label='Gap', zorder=3)
    plt.legend(fontsize=20, loc="upper left", prop={'size': 16})
    plt.text(0.71, 0.075, 'ECE={:.2f}'.format(ece), backgroundcolor='white', fontsize='x-large', transform=ax.transAxes)

    if not do_not_save:
        plt.savefig(save_path)

    plt.close()

    return ece
