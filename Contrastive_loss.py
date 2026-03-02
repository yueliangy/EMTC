import torch
import numpy as np
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
    """
    Implements the Clustering-Guided MEV Contrastive Learning (CMC) loss.
    """
    # [Fix] Ensure predicted_labels is a Tensor and on the correct device
    if not isinstance(predicted_labels, torch.Tensor):
        predicted_labels = torch.tensor(predicted_labels)

    # Ensure it's on the same device as the data
    predicted_labels = predicted_labels.to(fused_data.device)

    device = fused_data.device
    N = fused_data.shape[0]

    # 1. Normalize features for Cosine Similarity
    features = F.normalize(fused_data, dim=1)

    # 2. Compute Similarity Matrix (N, N)
    # Temperature parameter (tau) is typically 0.5 or 0.07 in contrastive learning
    temperature = 0.5
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # 3. Construct Masks (Cluster Alignment)
    # predicted_labels contains cluster IDs.
    labels = predicted_labels.view(-1, 1)

    # mask: (N, N) where mask[i, j] = 1 if sample i and j are in the same cluster
    mask = torch.eq(labels, labels.T).float()

    # Mask out self-contrast (diagonal)
    # We must exclude the sample itself from being its own positive pair
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(N).view(-1, 1).to(device),
        0
    )

    # 'mask' now indicates valid Positive Pairs (same cluster, different sample)
    mask = mask * logits_mask

    # 4. Compute Loss
    # Numerical stability: subtract max per row
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Denominator: sum of exp(sim) for ALL other samples (Pos + Neg)
    # Note: We use logits_mask to exclude self-similarity from the denominator
    exp_logits = torch.exp(logits) * logits_mask
    log_prob_sum = torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # Numerator: log(exp(sim)) for Positive pairs
    # Since we want -log( sum(exp(pos)) / sum(exp(all)) )
    # This is equivalent to: - ( log(sum(exp(pos))) - log(sum(exp(all))) )

    # Compute sum of exp(pos)
    exp_pos = (mask * exp_logits).sum(1, keepdim=True)
    log_prob_pos = torch.log(exp_pos + 1e-8)

    # Calculate loss for each sample
    # Formula: - log ( sum_pos / sum_all ) = - (log_prob_pos - log_prob_sum)
    # We only compute loss for samples that actually HAVE positive pairs (mask.sum(1) > 0)
    # If a sample is the only one in its cluster, mask.sum(1) will be 0, leading to valid_mask=0

    per_sample_loss = - (log_prob_pos - log_prob_sum)

    # Handle cases where a cluster has only 1 sample (no positive pairs)
    valid_mask = (mask.sum(1) > 0).float()
    loss = (per_sample_loss.flatten() * valid_mask).sum() / (valid_mask.sum() + 1e-8)

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


