import torch
import torch.nn as nn
import torch.nn.functional as F


class ATIN_op(nn.Module):
    def __init__(self, num_vars, seq_len, kernel_size=3, eta=0.5):
        super().__init__()
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.eta = eta

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=kernel_size,
                stride=1,
                padding=0
            ) for _ in range(num_vars)
        ])

        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        all_timestamps = []

        for var in range(self.num_vars):
            var_seq = x[:, var:var + 1, :]
            var_feat = self.conv_layers[var](var_seq)
            num_subseq = var_feat.shape[2]
            var_feat = var_feat.transpose(1, 2)

            scores = self.attention(var_feat)

            top_k = int(num_subseq * self.eta)
            top_indices = scores.topk(top_k, dim=1).indices
            top_indices = top_indices.squeeze(-1)

            all_timestamps.append(top_indices)

        all_timestamps = torch.stack(all_timestamps, dim=1)
        return all_timestamps

