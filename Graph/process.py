import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import itertools
import scipy.sparse as sp
import torch as th
import torch.nn.functional as F
import copy

from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor, sys_normalized_adjacency_i


# adapted from geom-gcn
def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def full_load_citation(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, labels, train_mask, val_mask, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def full_load_data(dataset_name, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    g = adj

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    adj = sys_normalized_adjacency(g)
    adj_i = sys_normalized_adjacency_i(g)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)

    return adj, adj_i, features, labels, train_mask, val_mask, test_mask, num_features, num_labels


def full_load_data_new(dataset_name, norm_type, init_val, init_type, pro=1, splits_file_path=None):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels, _, _, _ = full_load_citation(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj)
    else:
        graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])

        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
    features = preprocess_features(features)
    g = adj

    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    num_features = features.shape[1]
    num_labels = len(np.unique(labels))
    assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    adj = sys_normalized_adjacency(g)
    adj_i = sys_normalized_adjacency_i(g)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_i = sparse_mx_to_torch_sparse_tensor(adj_i)

    pairs = G.edges
    pairs = [[i, j] for i, j in pairs]

    tuple_list = [tuple(sub_list) for sub_list in pairs]
    unique_set = set()
    for tpl in tuple_list:
        if tpl not in unique_set and tpl[::-1] not in unique_set:
            unique_set.add(tpl)
    new_pairs = [list(tpl) for tpl in unique_set]

    # node_dict = {}
    # for i in range(len(new_pairs)):
    #     if new_pairs[i][0] not in node_dict:
    #         node_dict[new_pairs[i][0]] = []
    #     else:
    #         node_dict[new_pairs[i][0]].append(new_pairs[i][1])
    # edge_dict = node_dict

    labels_dict = {}
    for i in range(len(labels)):
        if labels[i].item() not in labels_dict:
            labels_dict[labels[i].item()] = []
        else:
            labels_dict[labels[i].item()].append(i)

    # print(labels_dict)
    edge_dict = {}
    for i in range(len(new_pairs)):
        if i not in edge_dict:
            edge_dict[i] = []
        if new_pairs[i][0] != new_pairs[i][1]:
            edge_dict[i].append(new_pairs[i][0])
            edge_dict[i].append(new_pairs[i][1])
        else:
            edge_dict[i].append(new_pairs[i][0])
    # print(edge_dict)
    # print(len(new_pairs))
    # random add some egde


    N_vertex = features.shape[0]
    n_edge = len(edge_dict)
    n_cc = 0
    for i in range(n_edge):
        if len(edge_dict[i]) > 1:
            label = labels[edge_dict[i][0]].item()
            # if len(labels_dict[label]) > 2:
                # pro = 0.9
            np.random.seed(i)
            sample = np.random.choice([0, 1], p=[pro, 1 - pro])

            if sample == 0:
                # continue
                # origin_label = set()
                # origin_label.add(labels[edge_dict[i][0]].item())
                # origin_label.add(labels[edge_dict[i][1]].item())
                # all_label = set(np.unique(labels))
                # sub_label = all_label - origin_label
                #
                # cho_lst = []
                # # label = np.random.choice(list(sub_label), 1, replace=False)[0]
                # # print(sub_label)
                # for lab in sub_label:
                #     for k in labels_dict[lab]:
                #         cho_lst.append(k)
                # # print(cho_lst)

                cho_lst = labels_dict[label]
                if edge_dict[i][0] in cho_lst:
                    cho_lst.remove(edge_dict[i][0])
                if edge_dict[i][1] in cho_lst:
                    cho_lst.remove(edge_dict[i][1])
                if len(cho_lst) > 3:
                    pos = np.random.choice(cho_lst, 3, replace=False)
                    for p in pos:
                        edge_dict[i].append(p)

            # if sample == 0:
            #     # continue
            #     origin_label = set()
            #     origin_label.add(labels[edge_dict[i][0]].item())
            #     origin_label.add(labels[edge_dict[i][1]].item())
            #     all_label = set(np.unique(labels))
            #     sub_label = all_label - origin_label
            #     label = np.random.choice(list(sub_label), 1, replace=False)[0]
            #     # print(labels[edge_dict[i][0]].item(), labels[edge_dict[i][1]].item())
            #     # print(all_label, sub_label, origin_label)
            #     # print(label)
            #     cho_lst = labels_dict[label]
            #     node_lst = copy.deepcopy(edge_dict[i])
            #     if edge_dict[i][0] in cho_lst:
            #         cho_lst.remove(edge_dict[i][0])
            #     if edge_dict[i][1] in cho_lst:
            #         cho_lst.remove(edge_dict[i][1])
            #     if len(cho_lst) > 1:
            #         pos = np.random.choice(cho_lst, 1, replace=False)
            #         for p in pos:
            #             node_lst.append(p)
            #             # edge_dict[i].append(p)
            #     # print(len(node_lst))
            #     edge_dict[i] = []
            #     for pair in itertools.combinations(node_lst, 2):
            #         # print(n_edge + n_cc)
            #         edge_dict[n_edge + n_cc] = []
            #         edge_dict[n_edge + n_cc].append(pair[0])
            #         edge_dict[n_edge + n_cc].append(pair[1])
            #         n_cc += 1
            #         # src_node.append(pair[0])
            #         # targ_node.append(pair[1])
    # print(edge_dict)

    # cal edge_homophily
    src_node = []
    targ_node = []
    for i in range(len(edge_dict)):
        if len(edge_dict[i]) > 1:
            node_lst = edge_dict[i]
            # print(node_lst)
            # hyperedge
            for pair in itertools.combinations(node_lst, 2):
                src_node.append(pair[0])
                targ_node.append(pair[1])
            # print(len(node_lst))

            # edge
            # src_node.append(node_lst[0])
            # targ_node.append(node_lst[1])
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    ignore_negative = True
    if ignore_negative:
        # edge_hom = np.mean(matching[labeled_mask])
        edge_hom = np.mean(np.array(matching[labeled_mask]).astype(float))
    else:
        edge_hom = np.mean(np.array(matching).astype(float))
    print('edge_homophily:', edge_hom)    # cal edge_homophilyinit_type
    src_node = []
    targ_node = []
    for i in range(len(edge_dict)):
        if len(edge_dict[i]) > 1:
            node_lst = edge_dict[i]
            # print(node_lst)
            # hyperedge
            for pair in itertools.combinations(node_lst, 2):
                src_node.append(pair[0])
                targ_node.append(pair[1])
            # print(len(node_lst))

            # edge
            # src_node.append(node_lst[0])
            # targ_node.append(node_lst[1])
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    ignore_negative = True
    if ignore_negative:
        # edge_hom = np.mean(matching[labeled_mask])
        edge_hom = np.mean(np.array(matching[labeled_mask]).astype(float))
    else:
        edge_hom = np.mean(np.array(matching).astype(float))
    print('edge_homophily:', edge_hom)


    # print(pairs)
    # def cosine_similarity_dense_small(x):
    #     norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
    #     sim = norm.mm(norm.t())
    #     return sim
    # sim = cosine_similarity_dense_small(features)
    # print(sim.shape)
    #     position = np.random.choice([j for j in range(N_vertex)], 1, replace=False)
    # n_count = 0
    # len_edge = len(edge_dict)
    # for i in range(len(edge_dict)):
    #     if len(edge_dict[i]) > 1:
    #         idx = edge_dict[i][0]
    #         idx_2 = edge_dict[i][1]
    #         new_sim = sim[idx]
    #         new_sim[idx] = -1.0
    #         new_sim_2 = sim[idx_2]
    #         new_sim_2[idx_2] = -1.0
    #         # print(new_sim)
    #         max_index = th.argmax(new_sim)
    #         max_index_2 = th.argmax(new_sim_2)
    #         # print(idx, max_index, new_sim[max_index])
    #         sim_0 = float(new_sim[max_index])
    #         # print(sim_0)
    #         # edge_dict[len_edge + n_count].append(idx)
    #         new_sim[max_index] = -1.0
    #         next_max_index = th.argmax(new_sim)
    #         sim_1 = new_sim[next_max_index]
    #         # print(sim_0, sim_1)
    #         if sim_0 > 0.6 and max_index != edge_dict[i][1] and max_index_2 != edge_dict[i][0]:
    #             edge_dict[i].append(max_index)
    #             edge_dict[i].append(max_index_2)
    #         # if sim_1 > 0.6 and next_max_index != edge_dict[i][1]:
    #         #     edge_dict[i].append(next_max_index)
    #         if sim_0 > 0.5 and sim_1 > 0.5:
    #             edge_dict[len_edge + n_count] = []
    #             edge_dict[len_edge + n_count].append(max_index)
    #             edge_dict[len_edge + n_count].append(next_max_index)
    #             n_count += 1


    N_hyperedge = len(edge_dict)
    self_set = set()
    for key, val in edge_dict.items():
        if len(val) == 1:
            self_set.add(val[0])
    # print(len(edge_dict), N_vertex)
    # print(len(edge_dict))
    if len(self_set) < N_vertex:
        # print(len(self_set))
        count_0 = 0
        for i in range(N_vertex):
            if i not in self_set:
                edge_dict[N_hyperedge + count_0] = []
                edge_dict[N_hyperedge + count_0].append(i)
                count_0 += 1
        print(count_0)
    vertex_num_egde = {}
    for key, val in edge_dict.items():
        for v in val:
            if v not in vertex_num_egde:
                vertex_num_egde[v] = 0
            else:
                vertex_num_egde[v] += 1
    # print(len(vertex_num_egde))


    print(len(edge_dict), len(self_set))
    pv_rows = []
    pv_cols = []
    n_count = 0
    threshold = -5
    if threshold >= -1.0:
        def cosine_similarity_dense_small(x):
            norm = F.normalize(x, p=2., dim=-1)  # [N, out_channels]
            sim = norm.mm(norm.t())
            return sim

        sim = cosine_similarity_dense_small(features)
        # print(sim.shape)
        for i in range(len(edge_dict)):
            neighbor_indices = np.array(edge_dict.get(i, []), dtype=np.int32)
            # print(i, neighbor_indices)
            res_idx = th.tensor(neighbor_indices)
            # sim_mean = 0
            if len(neighbor_indices) > 1:
                # continue
                idx = th.tensor(neighbor_indices)
                new_sim = th.index_select(th.index_select(sim, 0, idx), 1, idx)
                new_sim[th.eye(len(idx)).bool()] = 0.0
                # print(torch.sum(new_sim))
                sim_mean = th.sum(new_sim) / (len(idx) * (len(idx) - 1))
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
                        if q == (len(res_idx) - 1):
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
        # pro = 0.5
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
            # sample = np.random.choice([0, 1], p=[pro, 1 - pro])
            # if len(res_idx) > 1 and sample == 0:
            #     continue
            # if sample == 0:
            #     continue
            # if i < 100:
            #     print(len(res_idx))
            for q, p in enumerate(res_idx):
                pv_rows.append(n_count2)
                pv_cols.append(p)
                if q == (len(res_idx)-1):
                    n_count2 += 1
    else:
        n_count2 = n_count
        pv_init_val = []
        for i in range(len(edge_dict)):
            res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
            for q, p in enumerate(res_idx):
                if len(res_idx) == 1:
                    if res_idx[0] in vertex_num_egde:
                        if init_type == 1:
                            pv_init_val.append(vertex_num_egde[res_idx[0]]*init_val)
                        else:
                            pv_init_val.append(init_val)
                        # print()
                    else:
                        pv_init_val.append(1)
                else:
                    pv_init_val.append(1)
                pv_rows.append(n_count2)
                pv_cols.append(p)
                if q == (len(res_idx) - 1):
                    n_count2 += 1
    # for i in range(len(edge_dict)):
    #     res_idx = np.array(edge_dict.get(i, []), dtype=np.int32)
    #     for q, p in enumerate(res_idx):
    #         pv_rows.append(n_count)
    #         pv_cols.append(p)
    #         if q == (len(res_idx) - 1):
    #             n_count += 1
    pv_rows = th.tensor(pv_rows)
    pv_cols = th.tensor(pv_cols)
    pv_indices = th.stack([pv_rows, pv_cols], dim=0)
    # print(pv_init_val)
    pv_values = th.tensor(pv_init_val, dtype=th.float32)
    # pv_values = th.ones_like(pv_rows, dtype=th.float32)
    # Pv = torch.sparse_coo_tensor(pv_indices, pv_values, size=[len(pv_rows), N_vertex])
    # PvT = torch.sparse_coo_tensor(torch.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, len(pv_rows)])
    Pv = th.sparse_coo_tensor(pv_indices, pv_values, size=[n_count2, N_vertex])
    PvT = th.sparse_coo_tensor(th.stack([pv_cols, pv_rows], dim=0), pv_values, size=[N_vertex, n_count2])
    # data.Pv = Pv
    # data.PvT = PvT
    # norm_type = args.norm_type
    if norm_type == 0:
        Pv_col_sum = th.sparse.sum(Pv, dim=1)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = th.reciprocal(Pv_diag_values)
        Pv_diag = th.sparse_coo_tensor(th.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                          th.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        Pv_col_norm = th.sparse.mm(Pv_diag, Pv)

        PvT_row_sum = th.sparse.sum(PvT, dim=1)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = th.reciprocal(PvT_diag_values)
        PvT_diag = th.sparse_coo_tensor(th.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                           th.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = th.sparse.mm(PvT_diag, PvT)
    elif norm_type == 1:
        Pv_col_sum = th.sparse.sum(Pv, dim=0)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = th.reciprocal(Pv_diag_values)
        Pv_diag = th.sparse_coo_tensor(th.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                          th.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = th.sparse.mm(Pv, Pv_diag)

        PvT_row_sum = th.sparse.sum(PvT, dim=1)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = th.reciprocal(PvT_diag_values)
        PvT_diag = th.sparse_coo_tensor(th.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                           th.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = th.sparse.mm(PvT_diag, PvT)
    elif norm_type == 2:
        Pv_col_sum = th.sparse.sum(Pv, dim=0)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = th.reciprocal(Pv_diag_values)
        Pv_diag = th.sparse_coo_tensor(th.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                          th.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = th.sparse.mm(Pv, Pv_diag)

        PvT_row_sum = th.sparse.sum(PvT, dim=0)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = th.reciprocal(PvT_diag_values)
        PvT_diag = th.sparse_coo_tensor(th.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                           th.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))
        # print(PvT.shape, PvT_diag.shape)
        PvT_row_norm = th.sparse.mm(PvT, PvT_diag)
    elif norm_type == 3:
        Pv_col_sum = th.sparse.sum(Pv, dim=1)
        Pv_diag_indices = Pv_col_sum.indices()[0]
        Pv_diag_values = Pv_col_sum.values()
        Pv_diag_values = th.reciprocal(Pv_diag_values)
        Pv_diag = th.sparse_coo_tensor(th.stack([Pv_diag_indices, Pv_diag_indices]), Pv_diag_values,
                                          th.Size([Pv_col_sum.shape[0], Pv_col_sum.shape[0]]))
        # print(Pv.shape, Pv_diag.shape)
        Pv_col_norm = th.sparse.mm(Pv_diag, Pv)

        PvT_row_sum = th.sparse.sum(PvT, dim=0)
        PvT_diag_indices = PvT_row_sum.indices()[0]
        PvT_diag_values = PvT_row_sum.values()
        PvT_diag_values = th.reciprocal(PvT_diag_values)
        PvT_diag = th.sparse_coo_tensor(th.stack([PvT_diag_indices, PvT_diag_indices]), PvT_diag_values,
                                           th.Size([PvT_row_sum.shape[0], PvT_row_sum.shape[0]]))

        PvT_row_norm = th.sparse.mm(PvT, PvT_diag)
    else:
        Pv_col_norm = Pv
        PvT_row_norm = PvT
    print(Pv.shape, features.shape)
    Pv, PvT = Pv_col_norm, PvT_row_norm
    # features = th.matmul(Pv, features)
    return Pv, PvT, adj, adj_i, features, labels, train_mask, val_mask, test_mask, num_features, num_labels
