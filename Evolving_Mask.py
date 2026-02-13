import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist


def MASK(X, missing_rate, num_view=2, important_indices=None, flag=0, alpha=0.5):

    device = X.device
    num_samples, seq_len, feature_dim = X.shape
    H_v = []

    for view in range(num_view):
        view_masks = []
        for sample_idx in range(num_samples):
            sample_important_idx = important_indices[view][sample_idx] if (
                        flag != 0 and important_indices is not None) else None

            sample_mask = add_mixed_missing_mask(
                seq_len=seq_len,
                feature_dim=feature_dim,
                missing_rate=missing_rate,
                important_idx=sample_important_idx,
                alpha=alpha
            )
            view_masks.append(sample_mask)

        view_masks_tensor = torch.stack(view_masks).to(device)
        view_data = X * view_masks_tensor.float()
        H_v.append(view_data)

    return H_v


def add_mixed_missing_mask(seq_len, feature_dim, missing_rate=0.7, max_continuous_length=5, important_idx=None, alpha=0.5):


    def create_mask(flag):
        total_elements = seq_len * feature_dim
        total_missing = int(total_elements * missing_rate)
        continuous_missing = total_missing // 2
        scattered_missing = total_missing - continuous_missing

        if important_idx is not None and flag == 1:
            mask = torch.zeros((seq_len, feature_dim), dtype=torch.bool)
            mask[important_idx] = 1
            return mask


        mask = torch.ones((seq_len, feature_dim), dtype=torch.bool)
        available_starts = list(range(seq_len))


        while continuous_missing > 0 and available_starts:
            cont_len = min(max_continuous_length, continuous_missing // feature_dim)
            if cont_len <= 0:
                break
            start = np.random.choice(available_starts[:seq_len - cont_len + 1], replace=False)
            mask[start:start + cont_len, :] = False
            for t in range(start, start + cont_len):
                if t in available_starts:
                    available_starts.remove(t)
            continuous_missing -= cont_len * feature_dim


        if scattered_missing > 0:
            available_positions = torch.where(mask.flatten())[0].numpy()
            scatter_pos = np.random.choice(available_positions, scattered_missing, replace=False)
            mask_flat = mask.flatten()
            mask_flat[scatter_pos] = False
            mask = mask_flat.reshape(seq_len, feature_dim)

        return mask

    mask_non_important = create_mask(flag=1)
    mask_random = create_mask(flag=0)

    non_important_zeros = (mask_non_important == 0)
    num_to_set_1 = int((1 - alpha) * non_important_zeros.sum().item())
    if num_to_set_1 > 0:
        zero_indices = np.argwhere(non_important_zeros.numpy())
        selected_indices = np.random.choice(len(zero_indices), num_to_set_1, replace=False)
        mask_non_important[zero_indices[selected_indices][:, 0], zero_indices[selected_indices][:, 1]] = True

    random_zeros = (mask_random == 0)
    num_to_set_1 = int(alpha * random_zeros.sum().item())
    if num_to_set_1 > 0:
        zero_indices = np.argwhere(random_zeros.numpy())
        selected_indices = np.random.choice(len(zero_indices), num_to_set_1, replace=False)
        mask_random[zero_indices[selected_indices][:, 0], zero_indices[selected_indices][:, 1]] = True

    m1 = mask_non_important | mask_random
    return m1



def add_mixed_missing_mask1(seq_len, feature_dim, missing_rate=0.7, max_continuous_length=3):


    def create_mask():

        total_missing_elements = int(seq_len * feature_dim * missing_rate)
        continuous_missing_elements = total_missing_elements // 2
        scattered_missing_elements = total_missing_elements - continuous_missing_elements


        mask = torch.ones((seq_len, feature_dim), dtype=torch.bool)


        available_start_indices = list(range(seq_len))


        while continuous_missing_elements > 0 and available_start_indices:

            continuous_length = min(max_continuous_length, continuous_missing_elements // feature_dim)

            if continuous_length <= 0:
                break


            start_time_idx = np.random.choice(available_start_indices[:seq_len - continuous_length + 1], replace=False)
            mask[start_time_idx:start_time_idx + continuous_length, :] = False


            for idx in range(start_time_idx, start_time_idx + continuous_length):
                if idx in available_start_indices:
                    available_start_indices.remove(idx)


            continuous_missing_elements -= continuous_length * feature_dim


        scattered_indices = np.random.choice(seq_len * feature_dim, scattered_missing_elements, replace=False)
        for idx in scattered_indices:
            i = idx // feature_dim
            j = idx % feature_dim
            mask[i, j] = False

        return mask


    m1 = create_mask()

    return m1


def random_masking(data, missing_rate=0.2):

    n_samples, seq_len, feature_dims = data.shape
    mask = np.random.rand(n_samples, seq_len, feature_dims) < missing_rate
    masked_data = data.copy()
    masked_data[mask] = 0
    return masked_data


def contiguous_masking(data, missing_rate=0.2, max_continuous_length=5):

    n_samples, seq_len, feature_dims = data.shape
    masked_data = data.copy()

    for i in range(n_samples):

        total_missing_steps = int(seq_len * missing_rate)

        start_idx = np.random.randint(0, seq_len - total_missing_steps)

        masked_data[i, start_idx:start_idx + total_missing_steps, :] = 0

    return masked_data


def temporal_dependency_masking(data, missing_rate=0.2):

    n_samples, seq_len, feature_dims = data.shape
    masked_data = data.copy()

    for i in range(n_samples):

        num_missing_features = int(feature_dims * missing_rate)

        missing_feature_indices = np.random.choice(feature_dims, num_missing_features, replace=False)

        masked_data[i, :, missing_feature_indices] = 0

    return masked_data


