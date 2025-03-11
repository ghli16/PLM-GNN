import os
import random
import time
from argparse import ArgumentParser
import esm
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from esm import Alphabet, FastaBatchedDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
from PLM_GNN.dataset import SequenceData,GraphData
from PLM_GNN.model import Conbine_two
from PLM_GNN.utils import label2index,metrics


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def predict(model, fasta, batch_size, device, outdir, pos_labels):
    predicted_labels = ['VFC0272','VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271']
    print(f'Loading FASTA Dataset from {fasta}')
    
    test_dataset1 = SequenceData(fasta_path='./data/test_set.fasta',
                                transform=label2index)
    test_dataset2 = GraphData('./data/test_set.fasta', args)

    alphabet = Alphabet.from_architecture("roberta_large")
    
    
    
    cnn_train_loader = TorchDataLoader(test_dataset1, batch_size=args.batch_size,collate_fn=alphabet.get_batch_converter(), num_workers=args.num_workers, shuffle=False)
    gnn_train_loader = GeoDataLoader(test_dataset2, batch_size = args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
    
    model.eval()
    probs = []
    preds = []
    names = []
    lengths = []
    seq_records = []
    truth = []
    with torch.no_grad():
        for (cnn_batch, data) in zip(cnn_train_loader, gnn_train_loader):

            labels, strs, toks = cnn_batch
            toks = toks.to(device)
            data = data.to(device)
            if save_attn:
                out, attn = model(strs, toks,structure_feature)
                attn = attn.cpu().numpy()
            else:
                out = model(strs,toks,data.X, data.node_feat, data.edge_index, data.seq, data.batch)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)
            y = torch.tensor(labels, device=device).long()
            truth.append(y.detach().cpu().numpy())

            probs.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            
            
            for i, str in enumerate(strs):

                # name = labels[i].split()[0]
                name = data.name[i]
                pred_label = predicted_labels[pred[i].cpu().numpy()]

                if pred_label in pos_labels:
                    record = SeqRecord(Seq(str), id=name, description=f'putative type {pred_label} secreted protein')
                    seq_records.append(record)
                names.append(name)
                lengths.append(len(str))

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    truth = np.concatenate(truth)
    metrics_dict = metrics(truth, preds, probs)   
    print(metrics_dict)
    
    
    probs_VFC0001 = probs[:, 1]
    probs_VFC0086 = probs[:, 2]
    probs_VFC0204 = probs[:, 3]
    probs_VFC0235 = probs[:, 4]
    probs_VFC0258 = probs[:, 5]
    probs_VFC0271 = probs[:, 6]
    probs_VFC0272 = probs[:, 0]
    systems = list(map(lambda x: predicted_labels[x], preds))
    scores = [prob[idx] for prob, idx in zip(probs, preds)]
    
    ['VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271','VFC0272']
    
    result = pd.DataFrame({'name': names, 'system': systems, 'score': scores,'VFC0272': probs_VFC0272 ,'VFC0001': probs_VFC0001, 'VFC0086': probs_VFC0086,'VFC0204': probs_VFC0204, 'VFC0235': probs_VFC0235,'VFC0258': probs_VFC0258, 'VFC0271': probs_VFC0271, 'length': lengths})
    
    result = result.round(4)

    effector = result[result['system'].isin(pos_labels)]
    effector.to_csv(os.path.join(outdir, 'results.csv'), index=False)


def main(args):

    set_seed(42)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Using device {device} for prediction')
    start_time = time.time()

    model = Conbine_two(1280, 33, hid_dim=256, num_layers=1,
                            heads=4, dropout_rate=0.2, num_classes=7,attn_dropout=0.05, return_embedding1=True,return_embedding2=False,node_input_dim=1024 + 184,edge_input_dim=450,augment_eps=0.15,task='all',device=0)  
    model.to(device)
    if args.no_cuda:
        model.load_state_dict(torch.load(args.model_location, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(args.model_location))

    predict(model, args.batch_size, device,
            args.out_dir, args.secretion_systems)

    end_time = time.time()
    secs = end_time - start_time

    print(f'It took {secs:.1f}s to finish the prediction')


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Predict.")
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help='bacth size used in prediction. (default: 1)')

    parser.add_argument('--model_location', default='./runs/all_wc_256/log/checkpoint_all.pt', type=str,
                        help='path to the model weights.')
    parser.add_argument('--secretion_systems', nargs='+', default=['VFC0001', 'VFC0086','VFC0204', 'VFC0235','VFC0258', 'VFC0271','VFC0272'],
                        help="types of secreted proteins requiring prediction. (default: I II III IV VI)")
    parser.add_argument('--out_dir', default='./test_set.fasta', type=str,
                        help='output directory of prediction results.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='add when CUDA is not available.')
    parser.add_argument('--r', default=15, type=int,
                        help="")
    parser.add_argument('--mode', default='all', type=str,
                        help="")
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="")
    parser.add_argument('--num_workers', default=8, type=int,
                        help="num. of workers used in dataloader")
    args = parser.parse_args()

    main(args)



