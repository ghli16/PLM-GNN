import argparse
import gc
import multiprocessing
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from Bio import pairwise2
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

def get_prottrans(fasta_file,output_path, gpu):
    """
    get ProtTrans embeddings
    """
    num_cores = 2
    multiprocessing.set_start_method("forkserver")
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    ID_list = []
    seq_list = []
    with open(fasta_file, "r") as f:
        lines = f.readlines()
    for line in lines:
            if line[0] == ">":
                ID_list.append(line[1:-1])
            else:
                seq_list.append(" ".join(list(line.strip())))

    # Replace it with your own path
    model_path = "./Prot-T5-XL-U50"
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path)
    gc.collect()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model = model.to(device)
    print(next(model.parameters()).device)
    batch_size = 4

    for i in tqdm(range(0, len(ID_list), batch_size)):
        if i + batch_size <= len(ID_list):
            batch_ID_list = ID_list[i:i + batch_size]
            batch_seq_list = seq_list[i:i + batch_size]
        else:
            batch_ID_list = ID_list[i:]
            batch_seq_list = seq_list[i:]
        

        # Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
        batch_seq_list = [re.sub(r"[UZOB]", "X", sequence) for sequence in batch_seq_list]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            seq_emd = torch.tensor(seq_emd)
            torch.save(seq_emd, os.path.join(output_path, batch_ID_list[seq_num] + '.tensor'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get ProtTrans embeddings")
    parser.add_argument('--fasta_file', type=str, default='tss_all_line.fasta', help='Path to the FASTA file')
    parser.add_argument('--output_path', type=str, default='TSS_fea', help='Output directory for embeddings')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: None for CPU)')
    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    get_prottrans(args.fasta_file, args.output_path, args.gpu)
            
        
