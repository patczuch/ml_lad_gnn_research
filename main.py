import os.path as osp
import time
import torch
import os

from typing import Literal
from models.base import PureNet
from models.lad_base import STnet, Tenet
from torch_geometric.data import DataLoader
from utils import evaluate_func

import numpy as np
import datetime
import random
import argparse

# Open Graph Benchmark datasets (for graph property prediction, with different scale (small, medium, large) and different domains (molecular, knowledge, etc.)
OGB_GRAPH_PROPERTY_DATASETS = {'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-ppa', 'ogbg-code2'}     # More info: https://ogb.stanford.edu/docs/graphprop/


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accuracies = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1]) if i < n_bins - 1 else (confidences > bins[i]) & (confidences <= bins[i+1])
        bin_count = np.sum(mask)
        bin_counts[i] = int(bin_count)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            bin_accs[i] = bin_acc
            bin_confs[i] = bin_conf
            ece += (bin_count / len(probs)) * abs(bin_conf - bin_acc)
    return ece


def compute_brier(probs: np.ndarray, labels: np.ndarray):
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels.astype(int)] = 1.0
    sq = (probs - one_hot) ** 2
    sample_brier = np.sum(sq, axis=1)
    return float(np.mean(sample_brier))


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden feature_map dim (default: 64)')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='gnn layer numbers (default: 2)')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='gat heads num (default: 4)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('--with_bn', type=bool, default=True,
                        help='if with bn (default: True)')
    parser.add_argument('--with_bias', type=bool, default=True,
                        help='if with bias (default: True)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight decay of optimizer (default: 5e-5)')
    parser.add_argument('--scheduler_patience', type=int, default=50,
                        help='scheduler patience (default: 50)')
    parser.add_argument('--scheduler_factor', type=float, default=0.1,
                        help='scheduler factor (default: 0.1)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='the weight of distill loss(default: 0.1)')
    parser.add_argument('--tau', type=float, default=0.1,
                        help='linear attention temprature(default: 0.1)')
    parser.add_argument('--early_stop', type=int, default=7,
                        help='early stoping epoches (default:7)')
    parser.add_argument('--train_mode', type=str, default="T",
                        help='train mode T,S,P')
    parser.add_argument('--checkpoints_path', type=str, default="checkpoints",
                        help='teacher model save file path')
    parser.add_argument('--result_path', type=str, default="results",
                        help='three type models results save path')
    parser.add_argument('--backbone', type=str, default="GCN",
                        help='backbone models: GAT, GCN, GIN')
    parser.add_argument('--runs', type=int, default=1, help='ten-fold cross validation')
    parser.add_argument('--summary_csv', type=str, default=osp.join('results', 'summary.csv'),
                        help='Path to the global summary CSV (semicolon separated)')
    parser.add_argument('--epoch_logs_dir', type=str, default=osp.join('results', 'epoch_logs'),
                        help='Directory to store per-epoch logs')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoints_path = f'{args.checkpoints_path}/{args.backbone}'
    result_path = f'{args.result_path}/{args.backbone}'
    # ensure extra results dirs
    os.makedirs(osp.dirname(args.summary_csv), exist_ok=True)
    os.makedirs(args.epoch_logs_dir, exist_ok=True)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dataset_type: Literal['ogb-g', 'ogb-n', 'ogb-l', 'torch-dataset'] = ...
    is_from_ogb = lambda d_type: d_type in {'ogb-g', 'ogb-n', 'ogb-l'}

    # Check dataset parameter (if ogb, and if yes - what type)
    if args.dataset in OGB_GRAPH_PROPERTY_DATASETS:
        from ogb.graphproppred import PygGraphPropPredDataset as Dataset, Evaluator
        dataset_type = 'ogb-g'
    else:   # Dataset from outside the OGB
        from torch_geometric.datasets import TUDataset as Dataset
        dataset_type = 'torch-dataset'
        path = osp.join(osp.dirname(osp.realpath(__file__)), './data', args.dataset)
        dataset = Dataset(path, name=args.dataset, use_node_attr=True, use_edge_attr=True).shuffle()
    
    if is_from_ogb(dataset_type):
        try:
            from torch_geometric.data.data import DataEdgeAttr, Data, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
            safe_globals = torch.serialization.safe_globals
        except Exception:
            dataset = Dataset(name=args.dataset, root='data/').shuffle()
        else:
            with safe_globals([DataEdgeAttr, Data, DataTensorAttr, GlobalStorage]):
                dataset = Dataset(name=args.dataset, root='data/').shuffle()

    print(f"Dataset: {args.dataset}, Type: {dataset_type}, Number of graphs: {len(dataset)}")
    
    # pdb.set_trace()

    ##graph features process
    x_flag = True
    if dataset[0].x == None:  ## no node features: node degree as node feature, degree as node feature
        x_flag = False
        graphs = []
        tagset = set([])
        num_features = 1
        for graph in dataset:
            x1 = list(torch.bincount(graph.edge_index[0]).numpy())
            tagset = tagset.union(set(x1))
        tagset = list(tagset)
        tag2index = {tagset[i]: i for i in range(len(tagset))}
        for graph in dataset:
            x1 = torch.bincount(graph.edge_index[0])
            x1 = (x1 - torch.min(x1)) / (torch.max(x1) - torch.min(x1) + 0.00000001)
            # ensure node features are float (torch.bincount produces int tensors)
            graph.x = x1.view(-1, 1).float()
            graphs.append(graph)
        n = (len(dataset) + 9) // 10
        input_dim = num_features
        num_classes = dataset.num_classes
        del (dataset)
    else:
        n = (len(dataset) + 9) // 10
        input_dim = dataset.num_features
        num_classes = dataset.num_classes

    if args.train_mode == 'P':
        model = PureNet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers,
                      gat_heads=args.gat_heads, dropout=args.dropout, with_bn=args.with_bn,
                      with_bias=args.with_bias).to(device)

    elif args.train_mode == 'S' or args.train_mode == 'O':
        model = STnet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers,
                      gat_heads=args.gat_heads, dropout=args.dropout, with_bn=args.with_bn,
                      with_bias=args.with_bias).to(device)

    else:
        model = Tenet(nfeat=input_dim, nhid=args.nhid, nclass=num_classes, gnn=args.backbone, nlayers=args.nlayers,
                      gat_heads=args.gat_heads, dropout=args.dropout, tau=args.tau, with_bn=args.with_bn,
                      with_bias=args.with_bias).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=args.scheduler_patience,
                                                           factor=args.scheduler_factor)

    nll_loss = torch.nn.NLLLoss()
    mse_loss = torch.nn.MSELoss()
    test_acc_all = []
    auc_all, f1_all = [], []
    ece_all, brier_all = [], []

    if args.train_mode == 'S':
        num_k = args.runs
    else:
        num_k = 1

    for idd in range(num_k):
        print("=========================" + str(idd + 1) + " on runs " + str(num_k) + "=========================")
        if dataset_type == 'ogb-g': #       (ogb)
            split_idx = dataset.get_idx_split()
            from torch.utils.data import Subset
            train_dataset = Subset(dataset, split_idx["train"])
            val_dataset = Subset(dataset, split_idx["valid"])
            test_dataset = Subset(dataset, split_idx["test"])
        else:                       #       (classic from pytorch)
            if x_flag:
                dataset = dataset.shuffle()
                test_dataset = dataset[:n]
                val_dataset = dataset[n:2 * n]
                train_dataset = dataset[2 * n:]                
            else:
                random.shuffle(graphs)
                test_dataset = graphs[:n]
                val_dataset = graphs[n:2 * n]
                train_dataset = graphs[2 * n:]
        
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


        best_val_loss = float('inf')
        best_test_acc = 0.0
        wait = None
        best_test_acc, best_auc, best_f1 = 0.0, 0.0, 0.0
        best_ece, best_brier = 1.0, 1.0

        # prepare per-epoch log file (one file per training)
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = osp.join(args.epoch_logs_dir, str(args.backbone))
        os.makedirs(log_dir, exist_ok=True)
        epoch_log_path = osp.join(
            log_dir,
            f"{args.dataset}_{args.backbone}_{args.train_mode}_{ts}.csv"
        )
        with open(epoch_log_path, 'w', encoding='utf-8') as ef:
            # header metadata
            ef.write(f"# dataset: {args.dataset}\n")
            ef.write(f"# type: {args.backbone}\n")
            ef.write(f"# train_mode: {args.train_mode}\n")
            ef.write(f"epoch;train_loss;train_acc;val_loss;val_acc;test_loss;test_acc;AUC;F1;ECE;Brier\n")

        # model training
        epochs_run = 0
        for epoch in range(args.epochs):
            # Training the model
            s_time = time.time()
            train_loss = 0.
            train_corrects = 0
            model.train()

            if args.train_mode == 'S':
                # load teacher model (safe unpickling on PyTorch >=2.6)
                # Prefer using the safe_globals context manager to allowlist custom classes
                try:
                    safe_globals = torch.serialization.safe_globals
                except Exception:
                    # fallback: try loading with weights_only=False (use only if you trust the file)
                    teacher_model = torch.load(f'{checkpoints_path}/{args.dataset}_teacher.pth', weights_only=False).to(device)
                else:
                    with safe_globals([STnet, Tenet, PureNet]):
                        teacher_model = torch.load(f'{checkpoints_path}/{args.dataset}_teacher.pth', weights_only=False).to(device)
                teacher_model.eval()

            for i, data in enumerate(train_loader):
                s = time.time()
                data = data.to(device)
                # ensure node features are float
                if hasattr(data, 'x') and data.x is not None:
                    data.x = data.x.float()
                optimizer.zero_grad()

                inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)
                labs = torch.repeat_interleave(data.y, inds)

                if args.train_mode == 'P':
                    out, st_map = model(data.x, data.edge_index, data.batch)  ##pure model
                    loss_classification = nll_loss(out, data.y.view(-1))
                    loss = loss_classification
                elif args.train_mode == 'S':
                    # pdb.set_trace()
                    out, st_map = model(data.x, data.edge_index, data.batch)  ##student model
                    _, te_map = teacher_model(data.x, labs, data.edge_index, data.batch)
                    loss_distill = mse_loss(te_map, st_map)
                    loss_classification = nll_loss(out, data.y.view(-1))
                    loss = loss_classification + args.alpha * loss_distill
                else:
                    out, _ = model(data.x, labs, data.edge_index, data.batch)  ##teacher model input
                    loss_classification = nll_loss(out, data.y.view(-1))
                    loss = loss_classification

                loss.backward()
                train_loss += loss.item()
                train_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()
                optimizer.step()

            train_loss /= len(train_loader)
            train_acc = train_corrects / len(train_dataset)
            scheduler.step(train_loss)

            # Validation
            val_loss = 0.
            val_corrects = 0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    data = data.to(device)
                    # ensure node features are float
                    if hasattr(data, 'x') and data.x is not None:
                        data.x = data.x.float()

                    inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)
                    labs = torch.repeat_interleave(data.y, inds)

                    if args.train_mode == 'P':
                        out, st_map = model(data.x, data.edge_index, data.batch)  ##pure model
                        loss = nll_loss(out, data.y.view(-1))
                    elif args.train_mode == 'S':
                        out, st_map = model(data.x, data.edge_index, data.batch)  ##student model
                        _, te_map = teacher_model(data.x, labs, data.edge_index, data.batch)
                        loss_distill = mse_loss(te_map, st_map)
                        loss_classification = nll_loss(out, data.y.view(-1))
                        loss = loss_classification + args.alpha * loss_distill
                    else:
                        out, _ = model(data.x, labs, data.edge_index, data.batch)  ##teacher model input
                        loss = nll_loss(out, data.y.view(-1))

                    val_loss += loss.item()
                    val_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_corrects / len(val_dataset)

            # Test
            test_loss = 0.
            test_corrects = 0
            model.eval()
            y_preds = []
            y_tures = []
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    data = data.to(device)
                    # ensure node features are float
                    if hasattr(data, 'x') and data.x is not None:
                        data.x = data.x.float()
                    inds = torch.tensor([data.ptr[i + 1] - data.ptr[i] for i in range(data.y.shape[0])]).to(device)
                    labs = torch.repeat_interleave(data.y, inds)

                    if args.train_mode == 'S' or args.train_mode == 'P':
                        out, _ = model(data.x, data.edge_index, data.batch)  ##student model
                        loss = nll_loss(out, data.y.view(-1))

                    else:
                        out, _ = model(data.x, labs, data.edge_index, data.batch)  ##teacher model input
                        loss = nll_loss(out, data.y.view(-1))

                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    y_preds.append(probs)
                    y_tures.append(data.y.view(-1).cpu().numpy())

                    test_loss += loss.item()
                    test_corrects += out.max(dim=1)[1].eq(data.y.view(-1)).sum().item()

            test_loss /= len(test_loader)
            test_acc = test_corrects / len(test_dataset)
            auc, f1 = evaluate_func(y_preds, y_tures, args.dataset)

            try:
                y_pred_all = np.concatenate(y_preds, axis=0)
                y_true_all = np.concatenate(y_tures, axis=0).astype(int)
                if y_pred_all.ndim == 1:
                    y_pred_all = y_pred_all.reshape(-1, 1)
                n_test = len(y_true_all)
                n_bins = int(np.clip(np.sqrt(n_test), 5, 50))
                ece_val = compute_ece(y_pred_all, y_true_all, n_bins=n_bins)
                brier_val = compute_brier(y_pred_all, y_true_all)
            except Exception:
                ece_val, brier_val = float('nan'), float('nan')

            log = '[*] Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.2f}, Val Loss: {:.3f}, ' \
                  'Val Acc: {:.2f}, Test Loss: {:.3f}, Test Acc: {:.2f}, AUC:{:.3f}, F1:{:.3f}, ECE:{:.4f}, Brier:{:.4f}' \
                .format(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, auc, f1, ece_val, brier_val)
            print(log)

            # append to per-epoch log
            with open(epoch_log_path, 'a', encoding='utf-8') as ef:
                ef.write(
                    f"{epoch};{train_loss:.6f};{train_acc:.6f};{val_loss:.6f};{val_acc:.6f};{test_loss:.6f};{test_acc:.6f};{auc:.6f};{f1:.6f};{(ece_val if not np.isnan(ece_val) else float('nan')):.6f};{(brier_val if not np.isnan(brier_val) else float('nan')):.6f}\n"
                )

            if not np.isnan(ece_val) and ece_val < best_ece:
                best_ece = ece_val

            if best_test_acc < test_acc:
                best_test_acc = test_acc

            if best_auc < auc:
                best_auc = auc

            if best_f1 < f1:
                best_f1 = f1

            if not np.isnan(brier_val) and best_brier > brier_val:
                best_brier = brier_val


            # Early-Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.train_mode == 'P':
                    torch.save(model, f'{checkpoints_path}/{args.dataset}_pure.pth')  #pure model save
                elif args.train_mode == 'T':
                    torch.save(model, f'{checkpoints_path}/{args.dataset}_teacher.pth')  #teacher model save
                else:
                    torch.save(model, f'{checkpoints_path}/{args.dataset}_student.pth')  #student model save
                wait = 0
            else:
                wait += 1
            # early stopping
            if wait == args.early_stop:
                print('======== Early stopping! ========')
                # saving the model with best validation loss
                epochs_run = epoch + 1
                break
            epochs_run = epoch + 1

        test_acc_all.append(best_test_acc)
        auc_all.append(best_auc)
        f1_all.append(best_f1)
        ece_all.append(best_ece)
        brier_all.append(best_brier)

    # pdb.set_trace()
    top_acc = np.asarray(test_acc_all)
    test_avg = np.mean(top_acc)
    test_std = np.std(top_acc)

    print("test_avg_acc: {:.5f}, test_std_acc: {:.5f}, AUC: {:.5f}, f1: {:.5f}, ECE: {:.5f}, Brier: {:.5f}".format(
        test_avg, test_std, max(auc_all) if auc_all else float('nan'), max(f1_all) if f1_all else float('nan'),
        min(ece_all) if ece_all else float('nan'), min(brier_all) if brier_all else float('nan')))

    if not is_from_ogb(dataset_type):
        # delete file if args.train_mode == T
        if args.train_mode == 'T':
            if os.path.exists(f'{result_path}/{args.dataset}_ACC_result.txt'):
                os.remove(f'{result_path}/{args.dataset}_ACC_result.txt')
    #
    #     with open(f'{result_path}/{args.dataset}_ACC_result.txt', 'a+') as f:
    #         f.write(str(datetime.datetime.now().strftime(
    #             '%Y-%m-%d %H:%M:%S')) + f" Train Mode: {args.train_mode} test_avg_acc: {test_avg:.4f}, test_std_acc: {test_std:.4f}, AUC: {max(auc_all) if auc_all else float('nan'):.4f}, F1: {max(f1_all) if f1_all else float('nan'):.4f}, ECE: {min(ece_all) if ece_all else float('nan'):.4f}, Brier: {min(brier_all) if brier_all else float('nan'):.4f}")
    #         f.write("\n")

    # Append to global summary CSV (semicolon separated)
    try:
        header = "dataset;type;train_mode;test_avg_acc;test_std_acc;AUC;F1;ECE;Brier;num_epochs\n"
        row = ";".join([
            str(args.dataset),
            str(args.backbone),
            str(args.train_mode),
            f"{test_avg:.6f}",
            f"{test_std:.6f}",
            f"{(max(auc_all) if auc_all else float('nan')):.6f}",
            f"{(max(f1_all) if f1_all else float('nan')):.6f}",
            f"{(min(ece_all) if ece_all else float('nan')):.6f}",
            f"{(min(brier_all) if brier_all else float('nan')):.6f}",
            str(epochs_run if 'epochs_run' in locals() and epochs_run else args.epochs)
        ]) + "\n"
        # create file if not exists with header
        write_header = not os.path.exists(args.summary_csv) or os.path.getsize(args.summary_csv) == 0
        with open(args.summary_csv, 'a', encoding='utf-8') as sf:
            if write_header:
                sf.write(header)
            sf.write(row)
    except Exception as e:
        print(f"[WARN] Failed to append to summary CSV: {e}")
    
    if is_from_ogb(dataset_type):
        evaluator = Evaluator(name=args.dataset)
        # print(len(y_tures), y_tures[0].shape)
        # print(len(y_preds), y_preds[0].shape)

        # delete file if args.train_mode == T
        if args.train_mode == 'T':
            if os.path.exists(f'{result_path}/{args.dataset}_OGB_result.txt'):
                os.remove(f'{result_path}/{args.dataset}_OGB_result.txt')

        if dataset_type == 'ogb-g':
            y_true_all = np.concatenate(y_tures, axis=0).reshape(-1, 1)

            y_pred_all = np.concatenate(y_preds, axis=0)
            if y_pred_all.shape[1] == 2:
                y_pred_all = y_pred_all[:, 1].reshape(-1, 1)
            else:
                y_pred_all = y_pred_all.reshape(-1, 1)
        try:
            result_dict = evaluator.eval({"y_true": y_true_all, "y_pred": y_pred_all})
            print("OGB Evaluator Test: {}".format(result_dict))
            with open(f'{result_path}/{args.dataset}_OGB_result.txt', 'a+') as f:
                f.write(str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + f" Train Mode: {args.train_mode}")
                for key, val in result_dict.items():
                    f.write(f", {key}: {val:.4f}")
                f.write(f", ECE: {min(ece_all) if ece_all else float('nan'):.4f}, Brier: {min(brier_all) if brier_all else float('nan'):.4f}")
                f.write("\n")
        except:
            print("And error with ogb evaluator has occurred")
            print(evaluator.expected_input_format) 
            print(evaluator.expected_output_format)

if __name__ == '__main__':
    main()