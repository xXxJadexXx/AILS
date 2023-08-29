import pandas as pd
import os
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils import canonicalize_smiles




class MyData(Dataset):

    def __init__(self, smiles, encoder, seq_len, chars):

        self.smiles = smiles
        self.chars = chars

        self.encoder = encoder
        self.seq_len = seq_len



    def __len__(self):
        return len(self.smiles)#(len(self.smiles_raw)//self.seq_len) - self.seq_len

    def __getitem__(self, idx):
        while idx >= len(self):
            idx = idx - len(self)

        if torch.is_tensor(idx):
            idx = idx.to_list()


        #J Z E
        #Start Filler End
        smile = self.smiles[idx]
        smile = "J" + smile

        while len(smile) < 120:
            smile = smile + "Z"

        smile = smile + "E"

        item = self.encoder(smile[0:-1])
        target = self.encoder(smile[1:])


        return item, target, idx

def smile_edit(out_str):


    #end = out_str.find("E")
    #out_str = out_str[0:end]

    #out_str = out_str.replace("A", "")
    #out_str = out_str.replace("B", "")

    bracks1 = out_str.count("(")
    bracks11 = out_str.count(")")
    bracks2 = out_str.count("[")
    bracks22 = out_str.count("]")
    ring1 = out_str.count("1")
    ring2 = out_str.count("2")
    ring3 = out_str.count("3")

    r_idx = [index for index, letter in enumerate(out_str) if letter == "r"]
    for r in r_idx:

        if r == 0 or (not out_str[r-1] == "B"):
            out_str = out_str[0:r] + "B" + out_str[r:-1]
        else:
            continue

    if bracks1 > bracks11:
        dif = bracks1 - bracks11
        out_str = out_str + dif*")"
    elif bracks1 == 1:
        out_str = out_str.replace("(", "")
    elif bracks11 == 1:
        out_str = out_str.replace(")", "")


    if bracks2 > bracks22:
        dif = bracks2 - bracks22
        out_str = out_str + dif*"]"
    elif bracks2 == 1:
        out_str = out_str.replace("[", "")
    elif bracks22 == 1:
        out_str = out_str.replace("]", "")

    if ring1 % 2 != 0:
        out_str = out_str + "1"
    if ring2 % 2 != 0:
        out_str = out_str + "2"
    if ring3 % 2 != 0:
        out_str = out_str + "3"

    return out_str

def gen_mol(model, smiles, seq_len, decoder, encoder):

    model = model.float()
    batch = 1

    sample = torch.randint(0, len(smiles), [batch])
    sample = smiles[sample][0:seq_len-1]
    sample = "J" + sample + (119-seq_len)*"Z"
    #mol = sample

    sample = encoder(sample)#torch.tensor([encoder(smile[0:seq_len-1]) for smile in sample])


    sample = torch.tensor(sample).unsqueeze(0).float()


    hidden = [torch.zeros(model.n_layers, batch, model.n_hidden),
              torch.zeros(model.n_layers, batch, model.n_hidden)]



    out, _ = model(sample, hidden)
    last_str = decoder(out)#[decoder(smile) for smile in out]
    out_str = smile_edit(last_str)


    return out_str

