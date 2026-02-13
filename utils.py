from kmeans_gpu import kmeans
import torch
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score, f1_score, \
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from Cluster_gpu import *
import matplotlib.pyplot as plt
from scipy.special import comb

device = 'cuda'


def eva(X, ture_labels, predict_labels, show_details=False):
    predict_labels_flat = predict_labels.reshape(-1)
    true_label_flat = ture_labels.reshape(-1)
    X_reshaped = X.reshape(X.shape[0], -1)
    if isinstance(X_reshaped, torch.Tensor):
        X_reshaped_tensor = X_reshaped  # No need to convert
    else:
        X_reshaped_tensor = torch.from_numpy(X_reshaped)
    X_reshaped = X_reshaped_tensor.cpu().detach().numpy()


    ari = adjusted_rand_score(true_label_flat, predict_labels_flat)
    acc = calculate_acc(true_label_flat, predict_labels_flat)
    dcv = calculate_dcv(true_label_flat, predict_labels_flat)
    f1 = calculate_f1(true_label_flat, predict_labels_flat)
    pre = calculate_precision(true_label_flat, predict_labels_flat)
    rec = calculate_recall(true_label_flat, predict_labels_flat)
    nmi = cal_nmi(true_label_flat, predict_labels_flat)
    silhouette = 0
    davies_bouldin = 0
    ch_index = 0
    dunn = 0
    # silhouette = 0
    # davies_bouldin = 0
    # ch_index = 0

    if show_details:
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"Calinski-Harabasz Index: {ch_index:.4f}")

    return {
        "acc":acc,
        "dcv":dcv,
        "f1":f1,
        "pre":pre,
        "rec":rec,
        "nmi":nmi,
        "silhouette_score": silhouette,
        "davies_bouldin_index": davies_bouldin,
        "calinski_harabasz_index": ch_index,
        "Dunn": dunn,
        "ari": ari
    }


def dunn_index(X, labels):

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return np.nan


    min_inter_distance = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]
            if len(cluster_i) == 0 or len(cluster_j) == 0:
                continue
            distances = pairwise_distances(cluster_i, cluster_j)
            current_min = np.min(distances)
            if current_min < min_inter_distance:
                min_inter_distance = current_min


    max_intra_diameter = -np.inf
    for i in range(n_clusters):
        cluster = X[labels == i]
        if len(cluster) < 2:
            continue
        distances = pairwise_distances(cluster)
        max_dist = np.max(distances)
        if max_dist > max_intra_diameter:
            max_intra_diameter = max_dist

    if max_intra_diameter == 0:
        return np.nan

    return min_inter_distance / max_intra_diameter

def calculate_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    f1_scores = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fp = np.sum(cost_matrix[:, col_ind[i]]) - tp
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall > 0:
            f1_scores.append(2 * precision * recall / (precision + recall))

    f1 = np.mean(f1_scores)
    return f1


def calculate_precision(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    precisions = []
    for j in range(len(col_ind)):
        tp = cost_matrix[row_ind[j], col_ind[j]]
        fp = np.sum(cost_matrix[:, col_ind[j]]) - tp
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))

    precision = np.mean(precisions)
    return precision


def calculate_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    recalls = []
    for i in range(len(row_ind)):
        tp = cost_matrix[row_ind[i], col_ind[i]]
        fn = np.sum(cost_matrix[row_ind[i], :]) - tp
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))

    recall = np.mean(recalls)
    return recall

def calculate_acc(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    matching = cost_matrix[row_ind, col_ind].sum()
    acc = matching / y_true.size
    return acc


def calculate_dcv(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")


    n = len(y_true)
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)


    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=int)


    for i, true_label in enumerate(classes_true):
        for j, pred_label in enumerate(classes_pred):
            contingency[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))


    a = contingency.sum(axis=1)
    b = contingency.sum(axis=0)


    sum_comb_a = sum(comb(n_i, 2) for n_i in a)
    sum_comb_b = sum(comb(n_j, 2) for n_j in b)
    sum_comb = sum(comb(n_ij, 2) for n_ij in contingency.flatten())


    total_comb = comb(n, 2)


    expected_index = sum_comb_a * sum_comb_b / total_comb


    max_index = (sum_comb_a + sum_comb_b) / 2

    if max_index == expected_index:
        ari = 0.0
    else:
        ari = (sum_comb - expected_index) / (max_index - expected_index)

    return ari




def cal_nmi(x, y):
    # Compute normalized mutual information I(x,y)/sqrt(H(x)*H(y)) of two discrete variables x and y.
    n = len(x)
    x = x.reshape(1, n)
    y = y.reshape(1, n)

    # Using flatten and min/max on entire arrays
    l = min(x.min(), y.min())
    x = x - l + 1
    y = y - l + 1
    k = max(x.max(), y.max())

    idx = np.arange(n)
    Mx = np.zeros((n, k), dtype=int)
    My = np.zeros((n, k), dtype=int)
    Mx[idx, x.flatten() - 1] = 1
    My[idx, y.flatten() - 1] = 1
    Pxy = (Mx.T @ My) / n
    Pxy = Pxy[Pxy > 0]
    Hxy = -np.sum(Pxy * np.log2(Pxy))

    Px = np.mean(Mx, axis=0)
    Py = np.mean(My, axis=0)
    Px = Px[Px > 0]
    Py = Py[Py > 0]

    Hx = -np.sum(Px * np.log2(Px))
    Hy = -np.sum(Py * np.log2(Py))

    MI = Hx + Hy - Hxy

    # Avoid division by zero
    if Hx > 0 and Hy > 0:
        z = np.sqrt((MI / Hx) * (MI / Hy))
    else:
        z = 0

    return max(0, z)

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import skfuzzy as fuzz

def clustering(X, true_label, num_cluster, draw=0):
    k_predicted_labels = sf_kmeans(X, 'cuda', num_cluster).numpy()
    predict_labels = k_predicted_labels.reshape(-1)
    label_encoder = LabelEncoder()
    true_label = label_encoder.fit_transform(true_label)
    metrics = eva(X, true_label, predict_labels, show_details=False)
    # metrics1 = eva(X, true_label, k_predicted_labels, show_details=False)

    if draw == 1:
        # ====================================================================UMAP
        # reducer = UMAP(n_components=2, random_state=42)
        # combined_data = np.vstack([X])
        # reducer.fit(combined_data)
        #
        # combined_embedding = reducer.fit_transform(combined_data)
        # data_projected = combined_embedding[:len(X)]
        #
        # plt.figure(figsize=(14, 8))
        #
        # plt.scatter(data_projected[:, 0], data_projected[:, 1],
        #         c=predict_labels,
        #         cmap='tab20', alpha=0.6, s=20)
        #
        #
        # plt.title("Visualization - UMAP")
        # plt.colorbar(label='Cluster ID')
        # plt.show()
        # ====================================================================UMAP

        # ====================================================================TSNE
        reducer = TSNE(n_components=2, random_state=4)
        combined_data = np.vstack([X])
        combined_embedding = reducer.fit_transform(combined_data)
        data_projected = combined_embedding[:len(X)]

        plt.figure(figsize=(14, 8))

        plt.scatter(
            data_projected[:, 0],
            data_projected[:, 1],
            c=predict_labels,
            cmap='tab20',
            alpha=0.6,
            s=20
        )

        plt.grid(True, linestyle='--', alpha=0.3, color='gray')

        plt.savefig('WHB-3.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.show()
        # ====================================================================TSNE

        # ====================================================================MDS

        # reducer = MDS(n_components=2, random_state=42)
        # combined_data = np.vstack([X])
        #
        # combined_embedding = reducer.fit_transform(combined_data)
        #
        # data_projected = combined_embedding[:len(X)]
        #
        # plt.figure(figsize=(14, 8))
        #
        # plt.scatter(data_projected[:, 0], data_projected[:, 1],
        #         c=predict_labels,
        #         cmap='tab20', alpha=0.6, s=20)
        #
        #
        # plt.title("Visualization - MDS")
        # plt.colorbar(label='Cluster ID')
        # plt.show()
        # ====================================================================MDS


    slt = metrics["silhouette_score"]
    dbi = metrics["davies_bouldin_index"]
    CH = metrics["calinski_harabasz_index"]
    acc = metrics["acc"]
    dcv = metrics["dcv"]
    f1 = metrics["f1"]
    pre = metrics["pre"]
    rec = metrics["rec"]
    nmi = metrics["nmi"]
    dunn = metrics["Dunn"]
    ari = metrics["ari"]

    return acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, dunn, ari, predict_labels

def clustering1(X, true_label, num_cluster):
    X = X.detach().numpy()
    k_predicted_labels = sf_kmeans(X, 'cuda', num_cluster).numpy()
    predict_labels = k_predicted_labels.reshape(-1)
    label_encoder = LabelEncoder()
    true_label = label_encoder.fit_transform(true_label)
    metrics = eva(X, true_label, predict_labels, show_details=False)


    slt = metrics["silhouette_score"]
    dbi = metrics["davies_bouldin_index"]
    CH = metrics["calinski_harabasz_index"]
    acc = metrics["acc"]
    dcv = metrics["dcv"]
    f1 = metrics["f1"]
    pre = metrics["pre"]
    rec = metrics["rec"]
    nmi = metrics["nmi"]
    dunn = metrics["Dunn"]


    return acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, dunn, predict_labels
