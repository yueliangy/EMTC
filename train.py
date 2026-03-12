import warnings
warnings.filterwarnings("ignore")
import os
import time
import math
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Self-defined
from utils import *
from mymodal import EMTC
from Evolving_Mask import *
from pooling import *
from Contrastive_loss import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="EMTC Model Training")

    parser.add_argument('--dataname', type=str, default="AtrialFibrillation", help="Name of the dataset")
    parser.add_argument('--seeds', nargs='+', type=int, default=[2022, 2023, 2024, 2025, 2026],
                        help="List of random seeds")

    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--epoch_num', type=int, default=200, help="Number of training epochs")

    parser.add_argument('--output_dims', type=int, default=64, help="Output dimensions")
    parser.add_argument('--hidden_dims', type=int, default=1, help="Hidden dimensions")
    parser.add_argument('--depth', type=int, default=4, help="Depth of the model")
    parser.add_argument('--num_view', type=int, default=3, help="Number of views")

    parser.add_argument('--missing_rate', type=float, default=0.1, help="Missing rate for mask")
    parser.add_argument('--quan', type=float, default=0.9, help="mask parameter")
    parser.add_argument('--eta', type=float, default=0.5, help="Threshold parameter")
    parser.add_argument('--gama', type=float, default=1.0, help="Gama parameter")
    parser.add_argument('--temperature', type=float, default=100.0, help="Temperature for contrastive loss")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    acc_list, f1_list, nmi_list, ari_list = [], [], [], []

    # Loading/Proccessing Dta
    data_path = f'./New_Data/{args.dataname}/{args.dataname}.npy'
    data = np.load(data_path, allow_pickle=True).item()
    train_X, train_Y, data_test, label_test = data['train_X'], data['train_Y'], data['test_X'], data['test_Y']

    all_X = np.concatenate((train_X, data_test), axis=0)
    all_Y = np.concatenate((train_Y, label_test), axis=0)

    sample_num, seq_len, feature_dim = all_X.shape

    scaler = StandardScaler()
    all_X_reshaped = all_X.reshape(-1, feature_dim)
    all_X_normalized = scaler.fit_transform(all_X_reshaped)
    data_train = all_X_normalized.reshape(sample_num, seq_len, feature_dim)
    label_train = all_Y

    num_cluster = len(np.unique(label_train))
    print(f"Dataset: {args.dataname} | Data shape: {data_train.shape} | Clusters: {num_cluster}")

    train_data_tensor = torch.tensor(data_train, dtype=torch.float32, device=device)

    for seed in args.seeds:
        setup_seed(seed)

        contrastive_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True))
        rec_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True))
        cross_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True))

        model = EMTC(input_dims=feature_dim, output_dims=args.output_dims, seq_len=seq_len,
                     hidden_dims=args.hidden_dims, depth=args.depth, num_views=args.num_view, eta=args.eta).to(device)

        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr},
            {'params': [contrastive_weight, rec_weight, cross_weight], 'lr': 0.005}
        ])

        best_acc, best_f1, best_nmi, best_ari = -1, -1, -1, -1
        envolve_indices = None
        flag = 0

        # Initializing pseudo lables
        predict_labels = clustering1(train_data_tensor.reshape(sample_num, -1), num_cluster)

        for epoch in range(args.epoch_num + 1):
            epoch_start_time = time.time()
            model.train()
            optimizer.zero_grad()

            H_v = MASK(train_data_tensor, missing_rate=args.missing_rate, num_view=args.num_view,
                       important_indices=envolve_indices, flag=flag, alpha=args.quan)
            h_v = [train_data_tensor for _ in range(args.num_view)]
            flag = 1

            x_whole_list, envolve_indices, cross_loss, recon_loss = model(H_v, h_v, train_data_tensor)

            pooled_features = pooling(x_whole_list)
            fused_data = sum(pooled_features) / args.num_view

            if epoch % 5 == 0:
                predict_labels = clustering1(fused_data, num_cluster)

            con_loss = calculate_contrastive1(fused_data, predict_labels)
            loss = con_loss + rec_weight * recon_loss + cross_weight * cross_loss

            print(f"Seed-{seed} Epoch-{epoch} | Loss: {loss.item():.4f} | Time: {time.time() - epoch_start_time:.2f}s")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    R = model(H_v, h_v, train_data_tensor)
                    pooled_features_eval = pooling(R[0] if isinstance(R, tuple) else R)
                    fused_data_eval = sum(pooled_features_eval) / args.num_view

                    acc, f1, nmi, ari, p = clustering(fused_data_eval.cpu().numpy(), label_train, num_cluster)

                    if acc >= best_acc:
                        best_acc, best_f1, best_nmi, best_ari = acc, f1, nmi, ari

        print(f"--- Seed {seed} Best ACC: {best_acc:.4f} ---")
        acc_list.append(best_acc)
        f1_list.append(best_f1)
        nmi_list.append(best_nmi)
        ari_list.append(best_ari)

    print("\n" + "=" * 30)
    print(f"Final Results over {len(args.seeds)} seeds for dataset '{args.dataname}':")
    print(f"ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"F1:  {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"NMI: {np.mean(nmi_list):.4f} ± {np.std(nmi_list):.4f}")
    print(f"ARI: {np.mean(ari_list):.4f} ± {np.std(ari_list):.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()






