import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing as pp

import torch_sparse
import csv

class Unigencoder(nn.Module):
    @staticmethod
    def d_expansion(data):
        V, E = data.edge_index
        # print(data.edge_index.shape)
        edge_dict = {}
        for i in range(data.edge_index.shape[1]):
            if E[i].item() not in edge_dict:
                edge_dict[E[i].item()] = []
            edge_dict[E[i].item()].append(V[i].item())

        def cosine_similarity_dense_small(x):
            norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
            sim = norm.mm(norm.t())
            return sim

        # def cosine_similarity_sparse(mat):
        #     col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
        #     return col_normed_mat.T * col_normed_mat

        sim = cosine_similarity_dense_small(data.x)
        # sim_index = torch.where(sim > 0.4)
        # sim_set = set()
        # for i in range(len(sim_index[0])):
        #     # print(i)
        #     # if sim_index[0][i] != sim_index[1][i]:
        #     sim_set.add((sim_index[0][i].item(), sim_index[1][i].item()))
        # print(len(sim_set))
        # add self-loops
        N_vertex = V.max() + 1
        N_hyperedge = data.edge_index.shape[1]
        self_set = set()
        for key, val in edge_dict.items():
            if len(val) == 1:
                self_set.add(val[0])
        # print(len(self_set))
        if len(self_set) < N_vertex:
            print(len(self_set))
            count = 0
            for i in range(N_vertex):
                if i not in self_set:
                    edge_dict[N_hyperedge + count].append(i)

        print(sim.shape)
        threshold = 0.5
        for i in range(N_hyperedge):
            neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            if len(neighbor_indices) > 1:
                idx = torch.tensor(neighbor_indices)
                new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
                new_sim[torch.eye(len(idx)).bool()] = 0
                del_idx = (new_sim < threshold).all(dim=1)
                keep_idx = ~del_idx
                # res = torch.masked_select(idx, del_idx)
                res = torch.masked_select(idx, keep_idx)

        # print(edge_dict)
        # for i, j in pairs:
        #     if j not in edge_dict:
        #         edge_dict[j] = []
        #     edge_dict[j].append(i)
        # E = E + V.max()  # [V | E]
        # N_vertex = V.max() + 1
        #
        pv_rows = []
        pv_cols = []
        # n_count = 0
        # for i in range(N_vertex):
        #     pv_rows.append(i)
        #     pv_cols.append(i)
        threshold = 0.01
        for i in range(N_hyperedge):
            neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            res_idx = torch.tensor(neighbor_indices)
            if len(neighbor_indices) > 1:
                idx = torch.tensor(neighbor_indices)
                new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
                new_sim[torch.eye(len(idx)).bool()] = 0
                del_idx = (new_sim < threshold).all(dim=1)
                keep_idx = ~del_idx
                # res = torch.masked_select(idx, del_idx)
                res_idx = torch.masked_select(idx, keep_idx)
            # for i in range(N_hyperedge):
            #     neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            #     # pv_rows.append(i)
            #     # pv_cols.append(i)
            for p in res_idx:
                # pv_rows.append(n_count)
                pv_rows.append(i)
                pv_cols.append(p)
                # n_count += 1
        pv_rows = torch.tensor(pv_rows)
        pv_cols = torch.tensor(pv_cols)
        pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
        pv_values = torch.ones_like(pv_rows, dtype=torch.float32)
        Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[len(pv_rows), N_vertex])
        PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, len(pv_rows)])

        data.Pv = Pv
        data.PvT = PvT
        # Pv_col_sum = torch.sparse.sum(Pv, dim=0)
        # Pv_diag_indices = Pv_col_sum.indices()[0]
        # Pv_diag_values = Pv_col_sum.values()
        # Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
        #                                   torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # # print(Pv.shape, Pv_diag.shape)
        # Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)
        #
        # PvT_row_sum = torch.sparse.sum(PvT, dim=1)
        # PvT_diag_indices = PvT_row_sum.indices()[0]
        # PvT_diag_values = PvT_row_sum.values()
        # PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
        #                                    torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # # print(PvT.shape, PvT_diag.shape)
        # PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
        # # print(Pv)
        # # print(PvT)
        # data.Pv = Pv_col_norm
        # data.PvT = PvT_row_norm
        return data

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(Unigencoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, data):
        x = data.x
        Pv, PvT = data.Pv, data.PvT
        x = torch.spmm(Pv, x)
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = torch.spmm(PvT, x)
        return x

    def flops(self, x):
        num_samples = np.prod(x.shape[:-1])
        flops = num_samples * self.in_channels # first normalization
        flops += num_samples * self.in_channels * self.hidden_channels # first linear layer
        flops += num_samples * self.hidden_channels # first relu layer

        # flops for each layer
        per_layer = num_samples * self.hidden_channels * self.hidden_channels
        per_layer += num_samples * self.hidden_channels # relu + normalization
        flops += per_layer * (len(self.lins) - 2)

        flops += num_samples * self.out_channels * self.hidden_channels # last linear layer

        return flops


class PlainUnigencoder(nn.Module):
    @staticmethod
    def d_expansion(data, threshold, norm_type, init_val, init_type, sample_rate):
        V, E = data.edge_index
        # print(data.edge_index.shape)
        edge_dict = {}
        for i in range(data.edge_index.shape[1]):
            if E[i].item() not in edge_dict:
                edge_dict[E[i].item()] = []
            edge_dict[E[i].item()].append(V[i].item())
        # print(edge_dict)
        # def cosine_similarity_sparse(mat):
        #     col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
        #     return col_normed_mat.T * col_normed_mat

        # sim_index = torch.where(sim > threshold)
        # sim_set = set()
        # for i in range(len(sim_index[0])):
        #     # print(i)
        #     # if sim_index[0][i] != sim_index[1][i]:
        #     sim_set.add((sim_index[0][i].item(), sim_index[1][i].item()))
        # print(len(sim_set))

        # add self-loops
        N_vertex = V.max() + 1
        N_hyperedge = data.edge_index.shape[1]
        N_hy = len(edge_dict)
        self_set = set()
        print(len(edge_dict), N_hyperedge)
        for key, val in edge_dict.items():
            if len(val) == 1:
                self_set.add(val[0])
        # print(len(self_set))
        if len(self_set) < N_vertex:
            # print(len(self_set))
            count = 0
            for i in range(N_vertex):
                if i not in self_set:
                    edge_dict[N_hy + count] = []
                    edge_dict[N_hy+count].append(i)
            count+=1
        print(len(edge_dict), len(self_set))
        # print(sim.shape)
        # threshold = 0.15
        # for i in range(N_hyperedge):
        #     neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
        #     if len(neighbor_indices) > 1:
        #         idx = torch.tensor(neighbor_indices)
        #         new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
        #         new_sim[torch.eye(len(idx)).bool()] = 0
        #         del_idx = (new_sim < threshold).all(dim=1)
        #         keep_idx = ~del_idx
        #         # res = torch.masked_select(idx, del_idx)
        #         res = torch.masked_select(idx, keep_idx)

        # print(edge_dict)
        # for i, j in pairs:
        #     if j not in edge_dict:
        #         edge_dict[j] = []
        #     edge_dict[j].append(i)
        # E = E + V.max()  # [V | E]
        # N_vertex = V.max() + 1
        #

        node_num_egde = {}
        for key, val in edge_dict.items():
            for v in val:
                if v not in node_num_egde:
                    node_num_egde[v] = 0
                else:
                    node_num_egde[v] += 1

        pv_rows = []
        pv_cols = []
        # n_count = 0
        # for i in range(N_vertex):
        #     pv_rows.append(i)
        #     pv_cols.append(i)
        # threshold = -1
        n_count = 0
        if threshold >= -1.0:
            def cosine_similarity_dense_small(x):
                norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
                sim = norm.mm(norm.t())
                return sim
            sim = cosine_similarity_dense_small(data.x)
            # print(sim.shape)
            for i in range(len(edge_dict)):
                neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
                res_idx = torch.tensor(neighbor_indices)
                # sim_mean = 0
                if len(neighbor_indices) > 1:
                    # continue
                    idx = torch.tensor(neighbor_indices)
                    new_sim = torch.index_select(torch.index_select(sim, 0, idx), 1, idx)
                    new_sim[torch.eye(len(idx)).bool()] = 0.0
                    # print(torch.sum(new_sim))
                    sim_mean = torch.sum(new_sim) / (len(idx)*(len(idx)-1))
                    # print(sim_mean)
                    # del_idx = (new_sim <= threshold).all(dim=1)
                    # # print(del_idx)
                    # keep_idx = ~del_idx
                    # new_del_res = torch.masked_select(idx, del_idx)
                    # res_keep_idx = torch.masked_select(idx, keep_idx)
                    # print(len(res_keep_idx), len(new_del_res))
                    # if len(res_keep_idx) > 1:
                    #     for q, p in enumerate(res_keep_idx):
                    #         pv_rows.append(n_count)
                    #         # pv_rows.append(i)
                    #         pv_cols.append(p)
                    #         if q == (len(res_keep_idx) - 1):
                    #             n_count += 1
                    # if len(new_del_res) > 1:
                    #     for q, p in enumerate(new_del_res):
                    #         pv_rows.append(n_count)
                    #         # pv_rows.append(i)
                    #         pv_cols.append(p)
                    #         # if q == (len(res_idx)-1):
                    #         n_count += 1
                    if sim_mean > threshold:
                        for q, p in enumerate(res_idx):
                            pv_rows.append(n_count)
                            # pv_rows.append(i)
                            pv_cols.append(p)
                            if q == (len(res_idx)-1):
                                n_count += 1
                    # else:
                    #     for q, p in enumerate(res_idx):
                    #         pv_rows.append(n_count)
                    #         # pv_rows.append(i)
                    #         pv_cols.append(p)
                    #         # if q == (len(res_idx)-1):
                    #         n_count += 1
                else:
                    for q, p in enumerate(res_idx):
                        pv_rows.append(n_count)
                        # pv_rows.append(i)
                        pv_cols.append(p)
                        # if q == (len(res_idx)-1):
                        n_count += 1
            n_count2 = n_count
        elif -1 > threshold >= -2:
            # position = np.random.choice([i for i in range(N_hyperedge)], int(N_hyperedge*0.03), replace=False)
            # for i in position:
            # candidate_list = [0, 1]
            # p = 0.0
            # probabilities = [1-p, p]
            # for i in range(N_hyperedge):
            #     res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
            #     # if len(res_idx) > 1 and sample == 0:
            #     # if sample == 0:
            #     #     continue
            #     # if i < 100:
            #     #     print(len(res_idx))
            #     for q, p in enumerate(res_idx):
            #         pv_rows.append(n_count)
            #         pv_cols.append(p)
            #         if q == (len(res_idx)-1):
            #             n_count += 1
            n_count2 = n_count
            for i in range(len(edge_dict)):
                res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
                # if len(res_idx) > 1 and sample == 0:
                # if sample == 0:
                #     continue
                # if i < 100:
                #     print(len(res_idx))
                for q, p in enumerate(res_idx):
                    pv_rows.append(n_count2)
                    pv_cols.append(p)
                    # if q == (len(res_idx)-1):
                    n_count2 += 1
        else:
            n_count2 = n_count
            pv_init_val = []
            for i in range(len(edge_dict)):
                res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
                for q, p in enumerate(res_idx):
                    if len(res_idx) == 1:
                        if res_idx[0] in node_num_egde:
                            if init_type == 1:
                                pv_init_val.append(node_num_egde[res_idx[0]]*init_val)
                            else:
                                pv_init_val.append(init_val)
                            # print()
                        else:
                            pv_init_val.append(1)
                    else:
                        pv_init_val.append(1)
                    pv_rows.append(n_count2)
                    pv_cols.append(p)
                    if q == (len(res_idx)-1):
                        n_count2 += 1
        pv_rows = torch.tensor(pv_rows)
        pv_cols = torch.tensor(pv_cols)
        pv_indices = torch.stack([pv_rows, pv_cols], dim=0)
        pv_values = torch.tensor(pv_init_val, dtype=torch.float32)
        # pv_values = torch.ones_like(pv_rows, dtype=torch.float32)
        # Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[len(pv_rows), N_vertex])
        Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[n_count2, N_vertex])
        PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, n_count2])
        # PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, len(pv_rows)])

        # data.Pv = Pv
        # data.PvT = PvT
        if norm_type == 0:
            Pv_col_sum = torch.sparse.sum(Pv, dim=1)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                           torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

            PvT_row_sum = torch.sparse.sum(PvT, dim=1)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                           torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
        elif norm_type == 1:
            Pv_col_sum = torch.sparse.sum(Pv, dim=0)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

            PvT_row_sum = torch.sparse.sum(PvT, dim=1)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT_diag, PvT)
        elif norm_type == 2:
            Pv_col_sum = torch.sparse.sum(Pv, dim=0)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv, Pv_diag)

            PvT_row_sum = torch.sparse.sum(PvT, dim=0)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
            # print(PvT.shape, PvT_diag.shape)
            PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
        elif norm_type == 3:
            Pv_col_sum = torch.sparse.sum(Pv, dim=1)
            Pv_diag_indices = Pv_col_sum.indices()[0]
            Pv_diag_values = Pv_col_sum.values()
            Pv_diag_values = torch.reciprocal(Pv_diag_values)
            Pv_diag = torch.sparse_coo_tensor(torch.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                              torch.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
            # print(Pv.shape, Pv_diag.shape)
            Pv_col_norm = torch.sparse.mm(Pv_diag, Pv)

            PvT_row_sum = torch.sparse.sum(PvT, dim=0)
            PvT_diag_indices = PvT_row_sum.indices()[0]
            PvT_diag_values = PvT_row_sum.values()
            PvT_diag_values = torch.reciprocal(PvT_diag_values)
            PvT_diag = torch.sparse_coo_tensor(torch.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                               torch.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))

            PvT_row_norm = torch.sparse.mm(PvT, PvT_diag)
        else:
            Pv_col_norm = Pv
            PvT_row_norm = PvT
        # print(Pv)
        # print(PvT)
        data.Pv = Pv_col_norm
        data.PvT = PvT_row_norm
        print(data.Pv.shape, data.x.shape)
        row = [data.Pv.shape, data.PvT.shape]
        with open("data_4_29_{}_{}.csv".format(norm_type, threshold), mode="a", newline="") as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # Write the data to the CSV file row by row
            writer.writerow(row)
        return data

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(PlainUnigencoder, self).__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.cls = nn.Linear(out_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, data):
        x = data.x
        Pv, PvT = data.Pv, data.PvT
        x = torch.spmm(Pv, x)
        for i, lin in enumerate(self.lins[:-1]):
            # x = torch.spmm(Pv, x)
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # x = torch.spmm(PvT, x)
            # if i == 1:
            #     x = torch.spmm(PvT, x)
        x = self.lins[-1](x)
        x = torch.spmm(PvT, x)
        return x
