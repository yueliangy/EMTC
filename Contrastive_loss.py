import torch
import numpy as np
from Con_calculate import *
from sklearn.cluster import KMeans
from Cluster_gpu import *
import torch.nn.functional as F



def calculate_contrastive(fused_data, soft_assignments):
    N, f = fused_data.shape

    similarity_matrix = F.cosine_similarity(fused_data.unsqueeze(1), fused_data.unsqueeze(0), dim=2)  # (N, N)

    loss = 0.0
    for i in range(N):
        pos_idx = torch.argmax(soft_assignments[i])
        pos_similarity = similarity_matrix[i, pos_idx]


        neg_similarities = similarity_matrix[i]
        neg_similarities[pos_idx] = 0


        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities], dim=0)

        loss += F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long).to(fused_data.device))

    loss /= N
    return loss

def calculate_contrastive1(fused_data, predicted_labels):
    N, f = fused_data.shape
    similarity_matrix = F.cosine_similarity(fused_data.unsqueeze(1), fused_data.unsqueeze(0), dim=2)  # (N, N)

    loss = 0.0
    for i in range(N):
        pos_idx = predicted_labels[i]
        pos_similarity = similarity_matrix[i, pos_idx]
        neg_similarities = similarity_matrix[i]
        neg_similarities[pos_idx] = 0
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities], dim=0)
        loss += F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long).to(fused_data.device))

    loss /= N
    return loss


def multi_view_contrastive_loss(fused_data_list, predicted_labels, temperature=0.5, mode='features'):

    num_views = len(fused_data_list)
    num_samples = fused_data_list[0].size(0)

    if len(predicted_labels.shape) > 1 and predicted_labels.shape[1] > 1:
        labels = torch.argmax(predicted_labels, dim=1)
    else:
        labels = torch.tensor(predicted_labels)

    total_loss = 0.0

    if mode == 'labels':

        for i in range(num_views):
            for j in range(i + 1, num_views):

                features_i = fused_data_list[i]
                features_j = fused_data_list[j]


                features_i = F.normalize(features_i, dim=1)
                features_j = F.normalize(features_j, dim=1)


                similarity_matrix = torch.matmul(features_i, features_j.T) / temperature

                label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()


                diag_mask = torch.eye(num_samples, dtype=torch.bool, device=features_i.device)
                label_mask = label_mask.masked_fill(diag_mask, 0)


                positives = similarity_matrix * label_mask
                positive_loss = -positives.sum() / (label_mask.sum() + 1e-8)


                negatives = similarity_matrix * (1 - label_mask)
                negative_loss = torch.logsumexp(negatives, dim=1).mean()


                total_loss += positive_loss + negative_loss


        total_loss /= (num_views * (num_views - 1) / 2)

    elif mode == 'features':

        for i in range(num_views):
            for j in range(i + 1, num_views):

                features_i = fused_data_list[i]
                features_j = fused_data_list[j]


                features_i = F.normalize(features_i, dim=1)
                features_j = F.normalize(features_j, dim=1)


                similarity_matrix = torch.matmul(features_i, features_j.T) / temperature


                positives = torch.diag(similarity_matrix)


                negatives = similarity_matrix


                diag_mask = torch.eye(num_samples, dtype=torch.bool, device=features_i.device)
                negatives = negatives.masked_fill(diag_mask, -1e9)


                numerator = torch.exp(positives)
                denominator = torch.exp(negatives).sum(dim=1)
                loss = -torch.log(numerator / denominator).mean()

                total_loss += loss


        total_loss /= (num_views * (num_views - 1) / 2)

    return total_loss


def calculate_intra_cluster_loss(pooled_features, predict_labels):

    intra_losses = []

    for view_features in pooled_features:

        if isinstance(view_features, np.ndarray):
            view_features = torch.from_numpy(view_features).float()


        unique_labels = torch.unique(predict_labels)
        view_loss = 0.0


        for label in unique_labels:

            mask = (predict_labels == label)
            cluster_samples = view_features[mask]

            if len(cluster_samples) == 0:
                continue


            cluster_center = cluster_samples.mean(dim=0)


            distances = torch.norm(cluster_samples - cluster_center, dim=1)


            normalized_distances = distances / torch.sqrt(torch.tensor(view_features.shape[1], dtype=torch.float))


            view_loss += normalized_distances.mean()


        if len(unique_labels) > 0:
            view_loss /= len(unique_labels)
            intra_losses.append(view_loss)


    if intra_losses:
        intra_cluster_loss = torch.stack(intra_losses).mean()
    else:
        intra_cluster_loss = torch.tensor(0.0)

    return intra_cluster_loss

