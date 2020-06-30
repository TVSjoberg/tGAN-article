from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from definitions import RESULT_DIR
from gan_thesis.data.load_data import Dataset
from gan_thesis.models.general.utils import save_json


def association(dataset, split=False):
    data = dataset.data
    discrete_columns, continuous_columns = dataset.get_columns()
    columns = data.columns.to_list()

    if not split:
        association_matrix = np.zeros(shape=(len(columns), len(columns)))
        for i in range(len(columns)):
            for j in range(i):
                if (columns[i] in continuous_columns) and (columns[j] in continuous_columns):
                    association_matrix[i, j] = pearsonr(data.iloc[:, i], data.iloc[:, j])[0]
                if (columns[i] in discrete_columns) and (columns[j] in discrete_columns):
                    association_matrix[i, j] = normalized_mutual_info_score(data.iloc[:, i], data.iloc[:, j])
                if (columns[i] in continuous_columns) and (columns[j] in discrete_columns):
                    bin_nmi = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j], bin_axis=[True, False])
                    association_matrix[i, j] = bin_nmi
                    association_matrix[j, i] = bin_nmi

        return pd.DataFrame(association_matrix, index=columns, columns=columns)
    else:
        contcont_matrix = np.ones(shape=(len(continuous_columns), len(continuous_columns)))
        catcat_matrix = np.ones(shape=(len(discrete_columns), len(discrete_columns)))
        contcat_matrix = np.ones(shape=(len(continuous_columns), len(discrete_columns)))
        for i in range(len(columns)):
            for j in range(i):
                if (columns[i] in continuous_columns) and (columns[j] in continuous_columns):
                    contcont_matrix[i, j] = pearsonr(data.iloc[:, i], data.iloc[:, j])[0]
                if (columns[i] in discrete_columns) and (columns[j] in discrete_columns):
                    catcat_matrix[i, j] = normalized_mutual_info_score(data.iloc[:, i], data.iloc[:, j])
                if (columns[i] in continuous_columns) and (columns[j] in discrete_columns):
                    bin_nmi = mutual_info_score_binned(data.iloc[:, i], data.iloc[:, j], bin_axis=[True, False])
                    contcat_matrix[i, j] = bin_nmi
        return pd.DataFrame(contcont_matrix, index=continuous_columns, columns=continuous_columns), pd.DataFrame(
            catcat_matrix, index=discrete_columns, columns=discrete_columns), pd.DataFrame(contcat_matrix,
                                                                                           index=continuous_columns,
                                                                                           columns=discrete_columns),


def mutual_info_score_binned(x, y, bin_axis=None, bins=100):
    if bin_axis is None:
        bin_axis = [True, False]  # Bin x, don't bin y

    x = pd.cut(x, bins=bins) if bin_axis[0] else x
    y = pd.cut(y, bins=bins) if bin_axis[1] else y
    return normalized_mutual_info_score(x, y)


def association_difference(real=None, samples=None, association_real=None, association_samples=None):
    if (association_real is None) or (association_samples is None):
        association_real = association(real)
        association_samples = association(samples)

    return np.sum(np.abs(association_real.to_numpy().flatten() - association_samples.to_numpy().flatten()))


def plot_association(real_dataset, samples, dataset, model, force=True):
    association_real = association(real_dataset)
    samples_dataset = Dataset(None, None, samples, real_dataset.info, None)
    association_samples = association(samples_dataset)

    mask = np.triu(np.ones_like(association_real, dtype=np.bool))

    colormap = sns.diverging_palette(20, 220, n=256)

    plt.figure(figsize=(20, 10))
    plt.suptitle(model.upper() + ' Association')
    plt.subplot(1, 2, 1)
    plt.title('Real')
    sns.heatmap(association_real,
                vmin=-1,
                vmax=1,
                mask=mask,
                annot=False,
                cmap=colormap)

    plt.subplot(1, 2, 2)
    plt.title('Samples')
    sns.heatmap(association_samples,
                vmin=-1,
                vmax=1,
                mask=mask,
                annot=False,
                cmap=colormap)

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist, model)
    filepath = os.path.join(basepath, '{0}_{1}_association.png'.format(dataset, model))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    return association_difference(association_real=association_real, association_samples=association_samples)


def plot_all_association(complete_dataset, dataset, force=True, pass_tgan=True):
    alist = dataset.split(sep='-', maxsplit=1)
    base_path = os.path.join(RESULT_DIR, *alist)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_path = os.path.join(base_path, 'real_{0}_association.csv'.format(dataset))
    if os.path.exists(file_path):
        association_real = pd.read_csv(file_path)
        association_real = association_real.iloc[:, 1:]
        association_real = association_real.set_index(association_real.columns)
        print('loaded real association matrix')
    else:
        association_real = association(complete_dataset)
        association_real.to_csv(file_path)
    n_col = len(association_real.columns.to_list())

    diff = {}

    file_path = os.path.join(base_path, 'wgan_{0}_association.csv'.format(dataset))
    if os.path.exists(file_path):
        association_wgan = pd.read_csv(file_path)
        association_wgan = association_wgan.iloc[:, 1:]
        association_wgan = association_wgan.set_index(association_wgan.columns)
        print('loaded WGAN association matrix')

    else:
        samples_wgan = complete_dataset.samples.get('wgan')
        samples_dataset = Dataset(None, None, samples_wgan, complete_dataset.info, None)
        association_wgan = association(samples_dataset)
        association_wgan.to_csv(os.path.join(base_path, 'wgan_{0}_association.csv'.format(dataset)))
    diff['wgan'] = association_difference(association_real=association_real,
                                          association_samples=association_wgan)
    diff['wgan_norm'] = diff['wgan'] / (0.5 * len(association_real.columns.to_list()) * (len(association_real.columns.to_list()) - 1))

    file_path = os.path.join(base_path, 'ctgan_{0}_association.csv'.format(dataset))
    if os.path.exists(file_path):
        association_ctgan = pd.read_csv(file_path)
        association_ctgan = association_ctgan.iloc[:, 1:]
        association_ctgan = association_ctgan.set_index(association_ctgan.columns)
        print('loaded CTGAN association matrix')
    else:
        samples_ctgan = complete_dataset.samples.get('ctgan')
        samples_dataset = Dataset(None, None, samples_ctgan, complete_dataset.info, None)
        association_ctgan = association(samples_dataset)
        association_ctgan.to_csv(os.path.join(base_path, 'ctgan_{0}_association.csv'.format(dataset)))
    diff['ctgan'] = association_difference(association_real=association_real,
                                           association_samples=association_ctgan)
    diff['ctgan_norm'] = diff['ctgan'] / (
                0.5 * len(association_real.columns.to_list()) * (len(association_real.columns.to_list()) - 1))

    file_path = os.path.join(base_path, 'tgan_{0}_association.csv'.format(dataset))
    if pass_tgan:
        if os.path.exists(file_path):
            association_tgan = pd.read_csv(file_path)
            association_tgan = association_tgan.iloc[:, 1:]
            association_tgan = association_tgan.set_index(association_tgan.columns)
            print('loaded TGAN association matrix')
        else:
            samples_tgan = complete_dataset.samples.get('tgan')
            samples_dataset = Dataset(None, None, samples_tgan, complete_dataset.info, None)
            association_tgan = association(samples_dataset)
            association_tgan.to_csv(os.path.join(base_path, 'tgan_{0}_association.csv'.format(dataset)))
        diff['tgan'] = association_difference(association_real=association_real,
                                              association_samples=association_tgan)
        diff['tgan_norm'] = diff['tgan'] / (
                0.5 * len(association_real.columns.to_list()) * (len(association_real.columns.to_list()) - 1))


    colormap = sns.diverging_palette(20, 220, n=256)
    mask = np.triu(np.ones_like(association_real, dtype=np.bool))

    if pass_tgan:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 6))
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    cbar_ax = fig.add_axes([.94, .5, .02, .4])

    ax1.set_title('Real')
    ax1.set_aspect('equal')
    chart = sns.heatmap(association_real,
                        vmin=-1,
                        vmax=1,
                        mask=mask,
                        annot=False,
                        cmap=colormap,
                        ax=ax1,
                        cbar=False)

    chart.set_yticklabels(labels=chart.get_yticklabels(), rotation=0)

    ax2.set_title('WGAN')
    ax2.set_aspect('equal')

    sns.heatmap(association_wgan,
                vmin=-1,
                vmax=1,
                mask=mask,
                annot=False,
                cmap=colormap,
                ax=ax2,
                cbar=False)

    ax3.set_title('CTGAN')
    ax3.set_aspect('equal')

    if pass_tgan:
        sns.heatmap(association_ctgan,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                    annot=False,
                    cmap=colormap,
                    ax=ax3,
                    cbar=False)
    else:
        sns.heatmap(association_ctgan,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                    annot=False,
                    cmap=colormap,
                    ax=ax3,
                    cbar=True,
                    cbar_ax=cbar_ax)

    if pass_tgan:
        ax4.set_title('TGAN')
        ax4.set_aspect('equal')

        sns.heatmap(association_tgan,
                    vmin=-1,
                    vmax=1,
                    mask=mask,
                    annot=False,
                    cmap=colormap,
                    ax=ax4,
                    cbar=True,
                    cbar_ax=cbar_ax)

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout()

    alist = dataset.split(sep='-', maxsplit=1)
    dataset = alist[0]
    basepath = os.path.join(RESULT_DIR, *alist)
    filepath = os.path.join(basepath, '{0}_all_association.png'.format(dataset))
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    if os.path.isfile(filepath) and force:
        os.remove(filepath)

    plt.savefig(filepath)
    plt.close()

    filepath = os.path.join(basepath, '{0}_euclidian_distance.json'.format(dataset))
    save_json(diff, filepath)
