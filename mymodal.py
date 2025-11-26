import torch
import copy
from torch import nn
import numpy as np
from cov import DilatedConvEncoder
from Evolving_Mask import *
from CAE import *
import math
from ATIN import ATIN_op

device = 'cuda'

class BertInterpHead(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.activation = nn.ReLU()
        self.project = nn.Linear(4 * hidden_dim, input_dim)

    def forward(self, first_token_tensor):
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.project(pooled_output)
        return pooled_output


class EMTC(nn.Module):
    def __init__(self, input_dims, output_dims, seq_len, num_views=2, hidden_dims=64, depth=10, dropout=0.2, top_k=2):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_views = num_views
        self.top_k = top_k

        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            # input_dims,
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.ATINS = ATIN_op(
            num_vars=input_dims,
            seq_len=seq_len,
            kernel_size=3,
            eta=0.3
        )

        self.view_encoders = nn.ModuleDict()

        for i in range(self.num_views):
            self.view_encoders[f'view_{i}'] = DilatedConvEncoder(
                hidden_dims,
                [hidden_dims] * depth + [output_dims],
                kernel_size=5
            )


        self.repr_dropout = nn.Dropout(p=0.1)
        self.interphead = BertInterpHead(input_dims, output_dims)


        self.decoder_rnn_cell = nn.GRUCell(input_size=output_dims, hidden_size=hidden_dims)

        self.feature_mapper = nn.Linear(self.input_dims, 1)

        self.cross_view_decoders = nn.ModuleDict()

        self.attention_head = nn.Sequential(
            nn.Linear(output_dims, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

        # create cross view decoder
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                self.cross_view_decoders[f'{i}_to_{j}'] = nn.Sequential(
                    nn.Linear(output_dims, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, output_dims)
                )
                self.cross_view_decoders[f'{j}_to_{i}'] = nn.Sequential(
                    nn.Linear(output_dims, hidden_dims * 2),
                    nn.LayerNorm(hidden_dims * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dims * 2, output_dims)
                )

        # Create decoder for every view
        self.view_decoders = nn.ModuleDict()
        for i in range(self.num_views):
            self.view_decoders[f'view_{i}_decoder'] = nn.Sequential(
                nn.Linear(output_dims, hidden_dims * 2),
                nn.LayerNorm(hidden_dims * 2),
                nn.ReLU(),
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.LayerNorm(hidden_dims),
                nn.ReLU(),
                nn.Linear(hidden_dims, input_dims)
            )

        self.attn = nn.MultiheadAttention(embed_dim=output_dims, num_heads=1, dropout=dropout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_v, h_v, train_data):  # X: B x T x input_dims
        sample_num, seq_len, _ = H_v[0].shape
        x_whole_list = []

        processed_H_v = []
        for view in H_v:
            if not isinstance(view, torch.Tensor):
                view = torch.tensor(view, dtype=torch.float32, requires_grad=True, device='cuda')
            else:
                view = view.float().to('cuda')
                view.requires_grad_(True)  # Set requires_grad=True
            processed_H_v.append(view)
        H_v = processed_H_v

        processed_h_v = []
        for view in h_v:
            if not isinstance(view, torch.Tensor):
                view = torch.tensor(view, dtype=torch.float32, requires_grad=True, device='cuda')
            else:
                view = view.float().to('cuda')
                view.requires_grad_(True)
            processed_h_v.append(view)
        h_v = processed_h_v

        for i, view in enumerate(H_v):
            view = torch.tensor(view, dtype=torch.float32)
            x_whole = self.input_fc(view)
            x_whole = x_whole.transpose(1, 2)  # B x T x Ch -> B x Ch x T
            x_whole = self.view_encoders[f'view_{i}'](x_whole)
            x_whole = x_whole.transpose(1, 2)  # B x Ch x T -> B x T x Co
            x_whole = self.repr_dropout(x_whole)
            x_whole_list.append(x_whole)

        envolve_indices = []
        for view in H_v:
            view = view.transpose(1, 2)
            important_indices = self.ATINS(view).transpose(1, 2)
            envolve_indices.append(important_indices.cpu().numpy())



        cross_loss = 0
        if self.training:
            for i in range(self.num_views):
                for j in range(i + 1, self.num_views):

                    view_i_emb = x_whole_list[i]
                    view_j_recon = self.cross_view_decoders[f'{i}_to_{j}'](view_i_emb)


                    view_j_emb = x_whole_list[j]
                    view_i_recon = self.cross_view_decoders[f'{j}_to_{i}'](view_j_emb)

                    cross_loss += F.mse_loss(view_j_recon, x_whole_list[j].detach()) / sample_num
                    cross_loss += F.mse_loss(view_i_recon, x_whole_list[i].detach()) / sample_num

            cross_loss /= (self.num_views * (self.num_views - 1))

            reconstruction_loss = 0
            for i in range(self.num_views):

                view_recon = self.view_decoders[f'view_{i}_decoder'](x_whole_list[i])
                view_recon = view_recon.float()
                train_data = train_data.float()
                view_recon_loss = F.mse_loss(view_recon, train_data.detach())
                reconstruction_loss += view_recon_loss

        if self.training:
            return x_whole_list, envolve_indices,  cross_loss, reconstruction_loss
        else:
            return x_whole_list

    def pooling(self, x_whole_list):

        pooled_list = []

        for x_view in x_whole_list:

            x_view = torch.tensor(x_view).to('cuda')
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


                pooled = torch.sum(x_view[i] * weights.view(-1, 1), dim=0)
                pooled_samples.append(pooled)

            pooled_list.append(torch.stack(pooled_samples).cpu().numpy())

        return pooled_list

    def get_important_indices_from_emb(self, x_emb, attention_head, remain_ratio=0.3):

        with torch.no_grad():
            attn_x_score = attention_head(x_emb)


        batch_size, num_subseq, _ = x_emb.shape
        num_important = math.ceil(remain_ratio * num_subseq)


        _, important_indices = torch.topk(
            attn_x_score,
            k=num_important,
            dim=1,
            largest=True,
            sorted=True
        )

        important_indices = important_indices.squeeze(-1)

        return important_indices



