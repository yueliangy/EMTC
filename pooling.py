import numpy as np
import torch
import torch.nn.functional as F

def global_max_pooling(x_whole_list):

    pooled_list = []

    for x in x_whole_list:

        pooled_output = F.max_pool1d(x.permute(0, 2, 1), kernel_size=x.shape[1])

        pooled_output = pooled_output.squeeze(dim=2)
        pooled_list.append(pooled_output.cpu())

    return pooled_list


def pooling(x_whole_list):

    pooled_list = []

    for x_view in x_whole_list:

        num_samples, seq_len, feature_dims = x_view.shape

        var_per_sample = torch.var(x_view, dim=2).sum(dim=1)  # (num_samples,)

        var_min = var_per_sample.min()
        var_max = var_per_sample.max()

        time_steps = torch.arange(seq_len, device='cuda').float()

        pooled_samples = []
        for i in range(num_samples):

            alpha = (var_per_sample[i] - var_min) / (var_max - var_min + 1e-8)


            logits = alpha * time_steps
            weights = torch.softmax(logits, dim=0)

            # max_logit = logits.max()
            # shifted_logits = logits - max_logit
            # weights = torch.exp(shifted_logits)
            # # weights = torch.exp(alpha * time_steps)
            # weights = weights / weights.sum()


            pooled = torch.sum(x_view[i] * weights.view(-1, 1), dim=0)

            pooled_samples.append(pooled)



        pooled_list.append(torch.stack(pooled_samples).cpu())

    return pooled_list


def max_pooling_multiview(view_features_list, pool_dim=1):

    pooled_views = []
    for view_feats in view_features_list:

        pooled_view = torch.max(view_feats, dim=pool_dim)[0]  # [batch_size, hidden_dim]
        pooled_views.append(pooled_view)

    pooled_features = sum(
        pooled_views[i] for i in range(len(pooled_views))
    ) / len(pooled_views)

        # torch.cat(pooled_views, dim=-1)  # [batch_size, num_views * hidden_dim]
    return pooled_features.cpu().detach().numpy(), pooled_views

