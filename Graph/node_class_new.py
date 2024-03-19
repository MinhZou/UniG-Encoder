from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import uuid
import csv
import optuna

import torch.optim as optim

from model import *
from process import *
from utils import *

# Training settings


def main(args):
    print("==========================")
    print(f"Dataset: {args.dataset}, Dropout: {args.dropout}, 'num_layers:'{args.num_layers}")
    print(f"lr: {args.lr}, weight_decay: {args.weight_decay}")

    cudaid = "cuda:" + str(args.dev)
    device = torch.device(cudaid)
    checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
    # checkpt_file = 'pretrained/' + '1' + '.pt'

    def train_step(model, optimizer, labels, list_mat, list_mat_graph, idx_train, Pv, PvT):
        model.train()
        optimizer.zero_grad()
        # output = model(list_mat, list_mat_graph, layer_norm)
        output = model(list_mat[0], list_mat_graph[0], Pv, PvT)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item()


    def validate_step(model, labels, list_mat, list_mat_graph, idx_val, Pv, PvT):
        model.eval()
        with torch.no_grad():
            # output = model(list_mat, list_mat_graph, layer_norm)
            output = model(list_mat[0], list_mat_graph[0], Pv, PvT)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(), acc_val.item()


    def test_step(model, labels, list_mat, list_mat_graph, idx_test, Pv, PvT):
        try:
            model.load_state_dict(torch.load(checkpt_file))
        except:
            return float("inf"), 0
        model.eval()
        with torch.no_grad():
            # output = model(list_mat, list_mat_graph, layer_norm)
            output = model(list_mat[0], list_mat_graph[0], Pv, PvT)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
            # print(mask_val)
            return loss_test.item(), acc_test.item()


    def train(datastr, splitstr):
        Pv, PvT, adj, adj_i, features, labels, idx_train, \
        idx_val, idx_test, num_features, num_labels = full_load_data_new(datastr,
                                                                         args.norm_type,
                                                                         args.init_val,
                                                                         args.init_type,
                                                                         pro=1,
                                                                         splits_file_path=splitstr)
        features = features.to(device)
        Pv = Pv.to(device)
        PvT = PvT.to(device)

        adj = adj.to(device)

        list_mat = [features]
        list_mat_graph = [adj]

        # model = FSGNN(nfeat=num_features,
        #               nlayers=len(list_mat),
        #               nhidden=args.hidden,
        #               nclass=num_labels,
        #               dropout=args.dropout).to(device)

        model = UniG(nfeat=num_features,
                    nhid=args.hidden,
                    nclass=num_labels,
                    num_layers=args.num_layers,
                    dropout=args.dropout).to(device)

        # model = GCNMLP(nfeat=num_features,
        #             nhid=args.hidden,
        #             nclass=num_labels,
        #             dropout=args.dropout).to(device)
        # model = FSGNN2(nfeat_node=num_features,
        #                nfeat_graph=num_features_graph,
        #                nlayers=len(list_mat),
        #                nhidden=args.hidden,
        #                nclass=num_labels,
        #                dropout=args.dropout).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        bad_counter = 0
        best = float("inf")
        for epoch in range(args.epochs):
            loss_tra, acc_tra = train_step(model, optimizer, labels, list_mat, list_mat_graph, idx_train, Pv, PvT)
            loss_val, acc_val = validate_step(model, labels, list_mat, list_mat_graph, idx_val, Pv, PvT)
            # Uncomment following lines to see loss and accuracy values
            '''
            if(epoch+1)%1 == 0:
    
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    'acc:{:.2f}'.format(acc_tra*100),
                    '| val',
                    'loss:{:.3f}'.format(loss_val),
                    'acc:{:.2f}'.format(acc_val*100))
            '''

            if loss_val < best:
                best = loss_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break
        test_out = test_step(model, labels, list_mat, list_mat_graph, idx_test, Pv, PvT)
        acc = test_out[1]

        return acc * 100


    t_total = time.time()
    acc_list = []

    for i in range(10):
        datastr = args.dataset
        splitstr = 'splits/' + args.dataset + '_split_0.6_0.2_' + str(i) + '.npz'
        accuracy_data = train(datastr, splitstr)
        acc_list.append(accuracy_data)

        # # print(i,": {:.2f}".format(acc_list[-1]))

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    # print("Test acc.:{:.2f}".format(np.mean(acc_list)))
    print(f"Test accuracy: {np.mean(acc_list)}, {np.round(np.std(acc_list), 2)}")
    test_mean = np.mean(acc_list)
    test_std = np.round(np.std(acc_list), 2)
    row = [args.method, args.dataset, args.num_layers,
           args.lr, args.hidden, args.weight_decay, args.dropout, args.init_val, args.init_type, test_mean, test_std]
    with open("output_23_5_1_{}_{}_2.csv".format(args.dataset, str(args.method)), mode="a", newline="") as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        # Write the data to the CSV file row by row
        writer.writerow(row)
    return np.mean(acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # Default seed same as GCNII
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--num_layers', type=int, default=0, help='Number of layers.')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimensions.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--dataset', default='cora', help='dateset')  # cora texas wisconsin cornell citeseer pubmed
    parser.add_argument('--dev', type=int, default=7, help='device id')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay layer-1')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate 2 fully connected layers')
    parser.add_argument('--norm_type', type=int, default=1, help='norm type')
    parser.add_argument('--init_val', type=float, default=1, help='init val')
    parser.add_argument('--init_type', type=int, default=1, help='init type')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.method = 'UniG'
    main(args)
