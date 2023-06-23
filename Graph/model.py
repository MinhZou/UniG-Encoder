import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 2. / math.sqrt(self.in_features + self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FSGNN(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout):
        super(FSGNN, self).__init__()
        self.fc2 = nn.Linear(nhidden * nlayers, nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat, int(nhidden)) for _ in range(nlayers)])
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)

    def forward(self, list_mat, layer_norm):

        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind], tmp_out)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class FSGNN2(nn.Module):
    def __init__(self, nfeat_node, nfeat_graph, nlayers, nhidden, nclass, dropout):
        super(FSGNN2, self).__init__()
        self.fc2 = nn.Linear(nhidden * nlayers * 2, nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1_graph = nn.ModuleList([nn.Linear(nfeat_graph, int(nhidden)) for _ in range(nlayers)])
        self.fc1_node = nn.ModuleList([nn.Linear(nfeat_node, int(nhidden)) for _ in range(nlayers)])
        self.att = nn.Parameter(torch.ones(2*nlayers))
        self.sm = nn.Softmax(dim=0)

    def forward(self, list_mat_node, list_mat_graph, layer_norm):

        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat_node):
            tmp_out = self.fc1_node[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind], tmp_out)
            list_out.append(tmp_out)

        for ind, mat in enumerate(list_mat_graph):
            tmp_out = self.fc1_graph[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            new_ind = ind + len(list_mat_node)
            tmp_out = torch.mul(mask[new_ind], tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class FSGNNSNG(nn.Module):
    def __init__(self, nfeat_node, nfeat_graph, nlayers_node, nlayers_graph, nhidden, nclass, init_type, dropout):
        super(FSGNNSNG, self).__init__()
        self.fc2 = nn.Linear(nhidden * (nlayers_node + nlayers_graph), nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1_graph = nn.ModuleList([nn.Linear(nfeat_graph, int(nhidden)) for _ in range(nlayers_graph)])
        self.fc1_node = nn.ModuleList([nn.Linear(nfeat_node, int(nhidden)) for _ in range(nlayers_node)])
        if init_type == 'oo':
            self.att = nn.Parameter(torch.cat([torch.ones(nlayers_node), torch.ones(nlayers_graph)], dim=-1))
        elif init_type == 'zo':
            self.att = nn.Parameter(torch.cat([torch.zeros(nlayers_node), torch.ones(nlayers_graph)], dim=-1))
        elif init_type == 'oz':
            self.att = nn.Parameter(torch.cat([torch.ones(nlayers_node), torch.zeros(nlayers_graph)], dim=-1))
        elif init_type == 'zz':
            self.att = nn.Parameter(torch.cat([torch.zeros(nlayers_node), torch.zeros(nlayers_graph)], dim=-1))
        else:
            # p = 0.5  # Probability of a 1
            p = nlayers_node / (nlayers_node + nlayers_graph)
            # Generate a tensor of random numbers and apply threshold to obtain binary values
            data = torch.rand(nlayers_node+nlayers_graph) < p
            self.att = nn.Parameter(data.float())
        self.sm = nn.Softmax(dim=0)

    def forward(self, list_mat_node, list_mat_graph, layer_norm):

        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat_node):
            tmp_out = self.fc1_node[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind], tmp_out)
            list_out.append(tmp_out)

        for ind, mat in enumerate(list_mat_graph):
            tmp_out = self.fc1_graph[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            new_ind = ind + len(list_mat_node)
            tmp_out = torch.mul(mask[new_ind], tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)


class FSGNN_Large(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dp1, dp2):
        super(FSGNN_Large, self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat, int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden * nlayers, nhidden)
        self.fc3 = nn.Linear(nhidden, nclass)
        self.dropout1 = dp1
        self.dropout2 = dp2
        self.act_fn = nn.ReLU()

        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)

    def forward(self, list_adj, layer_norm, st=0, end=0):

        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj), mask)

        list_out = list()
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end, :].cuda()
            tmp_out = self.wt1[ind](mat)
            if layer_norm:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind], tmp_out)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)

        out = self.act_fn(final_mat)
        out = F.dropout(out, self.dropout1, training=self.training)
        out = self.fc2(out)

        out = self.act_fn(out)
        out = F.dropout(out, self.dropout2, training=self.training)
        out = self.fc3(out)

        return F.log_softmax(out, dim=1)


# class MLP(nn.Module):
#         def __init__(self, nfeat, nhid, nclass, dropout):
#             super(MLP, self).__init__()
#
#             self.mlp1 = nn.Linear(nfeat, nhid)
#             self.mlp2 = nn.Linear(nhid, nhid)
#             self.mlp3 = nn.Linear(nhid, nclass)
#             self.dropout = dropout
#
#         def forward(self, x, adj, Pv, PvT):
#             x = torch.spmm(Pv, x)
#             x = F.relu(self.mlp1(x))
#             x = F.dropout(x, self.dropout, training=self.training)
#
#             # x = F.relu(self.mlp2(x))
#             # x = F.dropout(x, self.dropout, training=self.training)
#             x = self.mlp3(x)
#             x = torch.spmm(PvT, x)
#             # x = torch.matmul(PvT, x)
#             return F.log_softmax(x, dim=1)


class UniG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers, dropout):
        super(UniG, self).__init__()
        self.lins = nn.ModuleList()

        self.lins.append(nn.Linear(nfeat, nhid))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(nhid, nhid))
        self.lins.append(nn.Linear(nhid, nclass))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj, Pv, PvT):
        x = torch.spmm(Pv, x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        x = torch.spmm(PvT, x)
        return F.log_softmax(x, dim=1)



class GCNMLP(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCNMLP, self).__init__()
        self.mlp1 = nn.Linear(nfeat, nhid)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.mlp2 = nn.Linear(nhid, nhid)
        self.mlp3 = nn.Linear(nhid, nclass)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, PvT):
        x = torch.mm(PvT.T, x)
        x = F.relu(self.mlp1(x))
        # x = self.mlp1(x)
        # x = torch.mm(PvT, x)

        # x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x)

        # x = torch.mm(PvT.T, x)
        # x = F.relu(self.mlp2(x))
        # x = F.dropout(x, self.dropout, training=self.training)

        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        #
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        # x = torch.mm(PvT.T, x)
        x = self.mlp3(x)
        # x = F.relu(self.mlp3(x))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(PvT, x)

        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    pass
