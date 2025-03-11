import os
import time
import logging
from argparse import ArgumentParser
from torch import optim
import torch.optim.lr_scheduler as lrs
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import random
from warmup_scheduler import GradualWarmupScheduler
from esm import Alphabet
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from PLM_GNN.model import CNN_Transformer,GNN,Conbine_two
from PLM_GNN.dataset import SequenceData,GraphData1,GraphData2,GraphData
from PLM_GNN.utils import  label2index, viz_conf_matrix
from PLM_GNN.trainer import train, test, set_seed, EarlyStopping,train_esm,train_geo,test_geo,test_esm,train_all,test_all
import torch_geometric
from torch_geometric.data import Batch
import esm
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PLM_GNN.losss import *



def main(args):
    set_seed(args.seed)
    log_dir = os.path.join(args.log_dir, f'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=os.path.join(log_dir, "training.log"), encoding='utf-8', mode='w+')],
        format="%(asctime)s %(levelname)s:%(message)s", datefmt="%F %A %T", level=logging.INFO)
    writer = SummaryWriter(log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if args.mode == "geo":
        model = GNN(node_input_dim=1024+184, edge_input_dim=450, hidden_dim=args.hid_dim, num_layers=args.num_layers, dropout=args.dropout_rate, augment_eps=args.augment_eps, task=args.mode, device=device, return_embedding=False)
    elif args.mode == "esm":
        model = CNN_Transformer(1280, 33, hid_dim=args.hid_dim, num_layers=args.num_layers,
                            heads=args.num_heads, dropout_rate=args.dropout_rate, num_classes=7)    
    elif args.mode == "all": 
        model = Conbine_two(1280, 33, hid_dim=args.hid_dim, num_layers=args.num_layers,
                            heads=args.num_heads, dropout_rate=args.dropout_rate, num_classes=5,attn_dropout=0.05, return_embedding1=True,return_embedding2=False,node_input_dim=1024 + 184,edge_input_dim=450,augment_eps=0.15,task='all',device=0)  
    else:
        raise ValueError('Invalid model type!')
    model.to(device)


    alphabet = Alphabet.from_architecture("roberta_large")
   
    train_dataset1 = SequenceData(fasta_path=os.path.join(args.data_dir, 'shuffled.fasta'),
                                transform=label2index)
    train_dataset2 = GraphData('./data/shuffled.fasta', args)
    
    valid_dataset1 = SequenceData(fasta_path=os.path.join(args.data_dir, 'valid_set.fasta'),
                                transform=label2index)
    valid_dataset2 = GraphData('./data/valid_set.fasta', args)
    
    test_dataset1 = SequenceData(fasta_path=os.path.join(args.data_dir, 'test_set.fasta'),
                                transform=label2index)
    test_dataset2 = GraphData('./data/test_set.fasta', args)


    cnn_train_loader = TorchDataLoader(train_dataset1, batch_size=args.batch_size,collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers, shuffle=False)
    gnn_train_loader = GeoDataLoader(train_dataset2, batch_size = args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
    
    cnn_valid_loader = TorchDataLoader(valid_dataset1, batch_size=args.batch_size,collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers, shuffle=False)
    gnn_valid_loader = GeoDataLoader(valid_dataset2, batch_size = args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
    
    cnn_test_loader = TorchDataLoader(test_dataset1, batch_size=32,collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers, shuffle=False)
    gnn_test_loader = GeoDataLoader(test_dataset2, batch_size =32, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
    
    
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler is None:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warm_epochs)
        return [optimizer], [scheduler]
    else:
        if args.lr_scheduler == 'step':
            after_scheduler = lrs.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
        elif args.lr_scheduler == 'cosine':
            after_scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr_decay_min_lr)
        else:
            raise ValueError('Invalid lr_scheduler type!')
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=args.warm_epochs, after_scheduler=after_scheduler)

    early_stopping = EarlyStopping(
        patience=args.patience, checkpoint_dir=log_dir, mode=args.mode)

    # Training epochs
    for epoch in tqdm(range(args.max_epochs), desc='Training', unit='epoch'):
        start_time = time.time()
        if args.mode == 'esm':
            train_loss, train_acc = train_esm(model, cnn_train_loader, criterion, optimizer, device)
            valid_loss, valid_metrics = test_esm(model, cnn_valid_loader, criterion, device)
        elif args.mode == 'geo':
            train_loss, train_acc = train_geo(model, gnn_train_loader, criterion, optimizer, device)
            valid_loss, valid_metrics = test_geo(model, gnn_valid_loader, criterion, device)
        elif args.mode == 'all':
            train_loss, train_acc = train_all(model, cnn_train_loader,gnn_train_loader,criterion, optimizer, device)
            valid_loss, valid_metrics = test_all(model, cnn_valid_loader,gnn_valid_loader, criterion, device)

        scheduler.step()

        end_time = time.time()
        epoch_secs = end_time - start_time

        valid_acc = valid_metrics['Accuracy']
        valid_f1 = valid_metrics['F1-score']
        valid_map = valid_metrics['AUPRC']

        logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
        logging.info(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        logging.info(f'Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
        logging.info(f'Valid F1: {valid_f1:.3f} | Valid mAP: {valid_map:.3f}')

        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
        writer.add_scalar('Valid/Loss', valid_loss, epoch+1)
        for key, value in valid_metrics.items():
            writer.add_scalar('Valid/' + key, value, epoch+1)

        early_stopping(valid_f1, model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at Epoch {epoch+1}")
            break

    if args.mode == 'esm':
        model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_esm.pt')))
        valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test_esm(model, cnn_valid_loader, criterion, device, True)
        _, test_final_metrics, test_truth, test_pred = test_esm(model, cnn_test_loader, criterion, device, True)

        logging.info(f'Best Valid Loss: {valid_best_loss:.3f} | Acc: {valid_best_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {valid_best_metrics["F1-score"]:.3f} | AUPRC: {valid_best_metrics["AUPRC"]:.3f}'
                    f' Precision: {valid_best_metrics["Precision"]:.3f} | Recall: {valid_best_metrics["Recall"]:.3f}'
                    f' AUC: {valid_best_metrics["AUC"]:.3f}')
        logging.info(f'Final Test Acc: {test_final_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {test_final_metrics["F1-score"]:.3f} | AUPRC: {test_final_metrics["AUPRC"]:.3f}'
                    f' Precision: {test_final_metrics["Precision"]:.3f} | Recall: {test_final_metrics["Recall"]:.3f}'
                    f' AUC: {test_final_metrics["AUC"]:.3f}')

        for key, value in valid_best_metrics.items():
            writer.add_scalar('Valid/Best ' + key, value)
        for key, value in test_final_metrics.items():
            writer.add_scalar('Test/Final ' + key, value)

        valid_cm = confusion_matrix(valid_truth, valid_pred)
        test_cm = confusion_matrix(test_truth, test_pred)

        labels = ['VFC0272','VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271'] 


        _ = viz_conf_matrix(valid_cm, labels, filename='confusion_matrix_train_esm.png')

        _ = viz_conf_matrix(test_cm, labels, filename='confusion_matrix_test_esm.png')
        
    elif args.mode == 'geo':
        model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_geo.pt')))
        valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test_geo(model, gnn_valid_loader, criterion, device, True)
        _, test_final_metrics, test_truth, test_pred = test_geo(model, gnn_test_loader, criterion, device, True)

        logging.info(f'Best Valid Loss: {valid_best_loss:.3f} | Acc: {valid_best_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {valid_best_metrics["F1-score"]:.3f} | AUPRC: {valid_best_metrics["AUPRC"]:.3f}'
                    f' Precision: {valid_best_metrics["Precision"]:.3f} | Recall: {valid_best_metrics["Recall"]:.3f}'
                    f' AUC: {valid_best_metrics["AUC"]:.3f}')
        logging.info(f'Final Test Acc: {test_final_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {test_final_metrics["F1-score"]:.3f} | AUPRC: {test_final_metrics["AUPRC"]:.3f}'
                    f' Precision: {test_final_metrics["Precision"]:.3f} | Recall: {test_final_metrics["Recall"]:.3f}'
                    f' AUC: {test_final_metrics["AUC"]:.3f}')

        for key, value in valid_best_metrics.items():
            writer.add_scalar('Valid/Best ' + key, value)
        for key, value in test_final_metrics.items():
            writer.add_scalar('Test/Final ' + key, value)

        valid_cm = confusion_matrix(valid_truth, valid_pred)
        test_cm = confusion_matrix(test_truth, test_pred)

        labels = ['VFC0272','VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271'] 


        _ = viz_conf_matrix(valid_cm, labels, filename='confusion_matrix_train_geo.png')

        _ = viz_conf_matrix(test_cm, labels, filename='confusion_matrix_test_geo.png')
    elif args.mode == 'all':
        model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoint_all.pt')))
        valid_best_loss, valid_best_metrics, valid_truth, valid_pred = test_all(model, cnn_valid_loader,gnn_valid_loader, criterion, device, True)
        _, test_final_metrics, test_truth, test_pred = test_all(model, cnn_test_loader,gnn_test_loader, criterion, device, True)

        logging.info(f'Best Valid Loss: {valid_best_loss:.3f} | Acc: {valid_best_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {valid_best_metrics["F1-score"]:.3f} | AUPRC: {valid_best_metrics["AUPRC"]:.3f}'
                    f' Precision: {valid_best_metrics["Precision"]:.3f} | Recall: {valid_best_metrics["Recall"]:.3f}'
                    f' AUC: {valid_best_metrics["AUC"]:.3f}')
        logging.info(f'Final Test Acc: {test_final_metrics["Accuracy"]*100:.2f}% |'
                     f' F1: {test_final_metrics["F1-score"]:.3f} | AUPRC: {test_final_metrics["AUPRC"]:.3f}'
                    f' Precision: {test_final_metrics["Precision"]:.3f} | Recall: {test_final_metrics["Recall"]:.3f}'
                    f' AUC: {test_final_metrics["AUC"]:.3f}')

        for key, value in valid_best_metrics.items():
            writer.add_scalar('Valid/Best ' + key, value)
        for key, value in test_final_metrics.items():
            writer.add_scalar('Test/Final ' + key, value)

        valid_cm = confusion_matrix(valid_truth, valid_pred)
        test_cm = confusion_matrix(test_truth, test_pred)

        labels = ['VFC0272','VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271'] 

        _ = viz_conf_matrix(valid_cm, labels, filename='confusion_matrix_train_all.png')

        _ = viz_conf_matrix(test_cm, labels, filename='confusion_matrix_test_all.png')
        
        
if __name__ == '__main__':

    parser = ArgumentParser(description="Train PLM_GNN model.")
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help="bacth size used in training. (default: 32)")
    parser.add_argument('--num_workers', default=0, type=int,
                        help="num. of workers used in dataloader")
    parser.add_argument('--seed', default=42, type=int, 
                        help="random seed used in training. (default: 42)")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="learning rate. (default: 1e-4)")
    parser.add_argument('--warm_epochs', default=1, type=int,
                        help="num. of epochs under warm start. (default: 1)")
    parser.add_argument('--patience', default=5, type=int,
                        help="patience for early stopping. (default: 5")
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str,
                        help="learning rate scheduler. [step, cosine]")
    parser.add_argument('--lr_decay_steps', default=10, type=int, 
                        help="step of learning rate decay. (default: 10)")
    parser.add_argument('--lr_decay_rate', default=0.5, type=float,
                        help="ratio of learning rate decay. (default: 0.5)")
    parser.add_argument('--lr_decay_min_lr', default=5e-6, type=float,
                        help="minimum value of learning rate. (default: 5e-6)")

    parser.add_argument('--max_epochs', default=100, type=int,
                        help="maximum num. of epochs. (default: 30")
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="path to data. (default: ./data)")
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help="weight decay for regularization. (default: 1e-5)")
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help="path to the logging directory. (default: ./logs)")

    parser.add_argument('--hid_dim', default=256, type=int,
                        help="hidden dimension in the model. (default: 256)")
    parser.add_argument('--num_layers', default=1, type=int,
                        help="num. of training transformer layers (default: 1)")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="num. of attention heads (default: 4)")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="dropout rate (default: 0.4)")
    parser.add_argument('--r', default=15, type=float,
                        help="")
    parser.add_argument('--mode', default='esm', type=str,
                        help="")
    parser.add_argument('--augment_eps', default=0.15, type=float,
                        help="")


    args = parser.parse_args()

    main(args)
