import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
import numpy as np
import torch
from utils import *
from tqdm import tqdm
from torch import optim
from mymodal import EMTC
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from Evolving_Mask import *
import math
from Con_calculate import *
from utils2 import clustering2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from DeepDPM import *
from pooling import *
from Contrastive_loss import *
from SVS import *
import os
from View_score import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = 'cuda'
from torch import nn
import time
from SubSpace import StaticMaskingStrategies
patience = 500
counter = 0

acc_list = []
dcv_list = []
f1_list = []
pre_list = []
nmi_list = []
slt_list = []
dbi_list = []
CH_list = []
dunn_list = []
Rec_list = []
ncc_list = []
cps_list = []
ari_list = []

dataname = "Epilepsy"

data = np.load(f'./New_Data/{dataname}/{dataname}.npy',allow_pickle=True).item()
train_X,train_Y,data_test,label_test = data['train_X'],data['train_Y'],data['test_X'],data['test_Y']
data_train = data_test
label_train = label_test

sample_num, seq_len, feature_dim = data_train.shape
sample_num1, seq_len1, feature_dim1 = data_test.shape
data_reshaped = data_train.reshape(-1, feature_dim)
data_reshaped1 = data_test.reshape(-1, feature_dim1)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_reshaped)
data_normalized1 = scaler.fit_transform(data_reshaped1)
data_train = data_normalized.reshape(sample_num, seq_len, feature_dim)
data_test = data_normalized1.reshape(sample_num1, seq_len1, feature_dim1)
print(data_train.shape)


unique_labels, counts = np.unique(label_train, return_counts=True)
proportions = counts / counts.sum()
num_cluster = len(np.unique(label_train))
print(num_cluster)
input_dims = feature_dim


lr = 0.1        #
epoch_num = 200
output_dims = 10    #
hidden_dims = 1
depth = 4
num_view = 3   #
missing_rate = 0.3
gama = 1
temperature = 100
quan = 0.85      #

for seed in [5]:

    setup_seed(seed)
    train_data = data_train
    test_data = data_test
    sample_num, seq_len, feature_dim = train_data.shape
    sample_num1, seq_len1, feature_dim1 = test_data.shape

    best_acc = 0
    best_dcv = 0
    best_f1 = 0
    best_pre = 0
    best_rec = 0
    best_nmi = 0
    best_slt = 0
    best_dbi = 0
    best_CH = 0
    best_dunn = 0
    best_ncc = 0
    best_cps = 0
    best_wkl = 0
    best_wrec = 0
    best_wcon = 0
    best_alpha = 0
    best_dz = 0
    best_ari = 0
    k_best_acc = 0
    k_best_dunn = 0
    k_best_slt = 0
    k_best_CH = 0
    k_best_dbi = 0


    num = int(math.sqrt(train_data.shape[0]))
    envolve_indices = None
    flag = 0


    # view_weights = nn.Parameter(torch.ones(num_view, requires_grad=True))
    contrastive_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    rec_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    cross_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, requires_grad=True))
    model = EMTC(input_dims=input_dims, output_dims=output_dims, seq_len=seq_len, hidden_dims=hidden_dims, depth=depth, num_views=num_view)
    # masker = StaticMaskingStrategies(mask_ratio=0.3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.add_param_group({'params': [contrastive_weight, rec_weight, cross_weight], "lr":0.005})
    # optimizer.add_param_group({'params': [view_weights], "lr":0.005})


    # GPU
    model.to(device)
    train_data = torch.tensor(train_data).to(device)
    sample_size = train_data.shape[0]
    acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, dunn, predict_labels = clustering1(train_data.reshape(train_data.shape[0], -1).cpu(),
                                                                                                    label_train,
                                                                                                    num_cluster)

    for epoch in range(epoch_num+1):
        epoch_start_time = time.time()

        optimizer.zero_grad()

        H_v = MASK(torch.tensor(train_data), missing_rate=missing_rate, num_view=num_view, important_indices=envolve_indices, flag=flag, alpha=quan)
        # H_v = [torch.tensor(masked_x) for _ in range(num_view)]
        h_v = [torch.tensor(train_data) for _ in range(num_view)]
        H_v1 = H_v
        h_v1 = h_v
        flag = 1

        model.train()
        #################################################
        x_whole_list, envolve_indices, cross_loss, recon_loss = model(H_v, h_v, train_data)

        # envolve_indices = None

        pooled_features = pooling(x_whole_list)

        pooled_features = [
            torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p
            for p in pooled_features
        ]

        fused_data = sum(
            pooled_features[i].float()
            for i in range(len(pooled_features))
        ) / num_view


        # fused_data = pooled_features[0].float()

        wea = False

        acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, dunn, predict_labels = clustering1(fused_data, label_train, num_cluster)
        # slt, dbi, CH, dunn, ari, nmi, predict_labels = train_cluster_net(fused_data.detach().numpy(), label_test, wea)


        # con_loss = multi_view_contrastive_loss(pooled_features, predict_labels, temperature)

        con_loss = calculate_contrastive1(fused_data, predict_labels)

        loss = con_loss + cross_weight * cross_loss + rec_weight * recon_loss

        # cross_weight * cross_loss + contrastive_weight * con_loss + rec_weight * recon_loss

        print(f"epoch-{epoch}")
        print(f"loss: {loss}")
        # print(f"con_loss{con_loss} recon_loss{recon_loss}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        training_time = time.time() - epoch_start_time
        #
        print(f"training time: {training_time:.4f}s")


        if epoch % 1 == 0:
            model.eval()
            wea = True
            with torch.no_grad():
                # inference_start = time.time()
                R = model(H_v1, h_v1, train_data)
                # inference_time = time.time() - inference_start
                # print(f"inference time: {inference_time:.4f}s")

                print("evaluating===============================")
                ###########################################################

                pooled_features = pooling(R)

                pooled_features = [
                    torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p
                    for p in pooled_features
                ]

                fused_data1 = sum(
                    # view_weights[i].float() *
                    pooled_features[i].float()
                    for i in range(len(pooled_features))
                ) / num_view

                acc, dcv, f1, pre, rec, nmi, slt, dbi, CH, dunn, ari, p = clustering(fused_data1, label_test, num_cluster)
                print("-----------------------")
                print(f"ari{float(ari):.4f} nmi{float(nmi):.4f}")
                print(f"ACC{float(acc):.4f} f1{float(f1):.4f}")
                print("-----------------------")

                if acc >= best_acc:
                    best_acc = acc
                    best_dcv = dcv
                    best_f1 = f1
                    best_pre = pre
                    best_rec = rec
                    best_nmi = nmi
                    best_slt = slt
                    best_dbi = dbi
                    best_CH = CH
                    best_dunn = dunn
                    best_ari = ari
                    counter = 0
                else:
                    counter += 1

                # import matplotlib.pyplot as plt
                # from sklearn.manifold import TSNE
                # from umap import UMAP
                # import numpy as np
                # from sklearn.preprocessing import LabelEncoder
                #
                # tsne = TSNE(n_components=2, random_state=5)
                # R_2d = tsne.fit_transform(fused_data1)
                #
                # fig, ax1 = plt.subplots(figsize=(7, 7))
                #
                # ax1.set_facecolor('#EAEAF1')
                #
                # title_font = {'family': 'Calibri', 'size': 17}
                # cmap = plt.cm.get_cmap('tab20', len(np.unique(p)))
                #
                # le = LabelEncoder()
                # label_train_numeric = le.fit_transform(p)
                #
                #
                # for label in ax1.get_xticklabels():
                #     label.set_fontname('Calibri')
                #     label.set_fontsize(24)
                # for label in ax1.get_yticklabels():
                #     label.set_fontname('Calibri')
                #     label.set_fontsize(24)
                #
                # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'x', '|', '_', '1', '2', '3',
                #            '4']
                #
                # unique_labels = np.unique(label_train_numeric)
                # ax1.grid(True, linestyle='-', alpha=1, color='white', zorder=1)
                #
                # for i, label in enumerate(unique_labels):
                #     mask = (label_train_numeric == label)
                #     marker = markers[i % len(markers)]
                #
                #     ax1.scatter(R_2d[mask, 0], R_2d[mask, 1],
                #                 c=[cmap(label)],
                #                 cmap=cmap,
                #                 marker=marker,
                #                 s=80,
                #                 alpha=0.8,
                #                 linewidth=0.5,
                #                 label=f'Class {label}',
                #                 zorder=3)
                #
                # ax1.set_xlabel(ax1.get_xlabel(), fontfamily='Calibri')
                # ax1.set_ylabel(ax1.get_ylabel(), fontfamily='Calibri')
                #
                #
                #
                # ax1.set_axisbelow(True)
                #
                # plt.tight_layout()
                #
                # plt.savefig(f'./fig/{dataname}_EMTC.pdf',
                #             format='pdf', bbox_inches='tight')
                #
                # plt.show()
                # plt.close(fig)



    acc_list.append(best_acc)
    dcv_list.append(best_dcv)
    f1_list.append(best_f1)
    pre_list.append(best_pre)
    Rec_list.append(best_rec)
    nmi_list.append(best_nmi)
    slt_list.append(best_slt)
    dbi_list.append(best_dbi)
    CH_list.append(best_CH)
    # ncc_list.append(best_ncc)
    cps_list.append(best_cps)
    dunn_list.append(best_dunn)
    ari_list.append(best_ari)


acc_list = np.array(acc_list)
dcv_list = np.array(dcv_list)
f1_list = np.array(f1_list)
pre_list = np.array(pre_list)
Rec_list = np.array(Rec_list)
nmi_list = np.array(nmi_list)
slt_list = np.array(slt_list)
dbi_list = np.array(dbi_list)
CH_list = np.array(CH_list)
dunn_list = np.array(dunn_list)
ari_list = np.array(ari_list)


print(f"acc: {acc_list.mean():.4f} ± {acc_list.std():.4f}")
print(f"f1: {f1_list.mean():.4f} ± {f1_list.std():.4f}")
print(f"nmi: {nmi_list.mean():.4f} ± {nmi_list.std():.4f}")
print(f"ari: {ari_list.mean():.4f} ± {ari_list.std():.4f}")






