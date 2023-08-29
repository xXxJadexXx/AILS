import pandas as pd
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from data import *
from training import *
#from training_GAN import *
from model import *
from rdkit import Chem

#from rdkit import RDLogger
#RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")

#from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance

#np.random.seed(1234)




use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Device:", device)
results_path = "./results"
learningrate = 0.0001
weight_decay = 0.00001
batch_size = 128
epochs = 10



data = pd.read_csv("./smiles_train.txt", header=None)

#smiles_can = [w for w in canonical_smiles(data[0]) if w is not None]
all_smiles = np.asarray(data).reshape((len(data)))


train_ids = range(64*700*2, 64*700*8)
val_ids = range(len(data)//3*2, len(data)//3*2 + 64*90)

train_smiles = all_smiles[train_ids]
val_smiles = all_smiles[val_ids]

print(f"Length of Training-set: {len(train_smiles)}")
print(f"Length of Validation-set: {len(val_smiles)}")



#with open("./smiles_data", "rU") as fd:
    #chars = [c for line in fd for c in line if c != " "]
#chars = list(set(chars))

#hardcoded for efficiency
#added some elements with more than one letter
#J = Start
#Z = Filler
#E = End
#All characters not appearing in the periodic table

chars = ['J', 'Z', 'E', 'i', 'I', '%', '6', 'N', '-', 'r', '3', 'C', 'B', '4', 'l', 'e', '+', '[', '0', '(', 'O', ')',
         'c', ']', 'P', '7', 'H', 'p', 'n', '#', 'b', '=', '2', '5', 's', '8', 'S', '1', 'o', '9', 'Br', 'Cl', 'Se', 'Sn', 'Sb', 'Si', 'Al']

smi2index = dict((c, i) for i, c in enumerate(chars))
#smi2index.update({-1 : "\n"})
index2smi = dict((i, c) for i, c in enumerate(chars))
#index2smi.update({"\n" : -1})


def smiles_encoder(smiles, maxlen=120):
    X = np.zeros((maxlen, len(chars)))

    for i, c in enumerate(smiles):
        X[i, smi2index[c]] = 1

    return X


def smiles_decoder(X):

    if len(X.shape) > 2:
        smi = []
        for smile in X:
            smi.append(smiles_decoder(smile))
        return smi

    smi = ''
    idx = torch.argmax(X, dim=-1)

    for j, i in enumerate(idx):
        if torch.is_tensor(i):
            i = i.item()
        smi += index2smi[i]

    return smi


def smiles_decoder_fcd(X):

    if len(X.shape) > 2:
        smi = []
        for smile in X:
            smi.append(smiles_decoder_fcd(smile))
        return smi

    smi = ''
    idx = torch.argmax(X, dim=-1)

    for j, i in enumerate(idx):

        if torch.is_tensor(i):
            i = i.item()
        if i == 0 or i == 1:
            continue
        elif i == 2:
            return smi
        else:
            smi += index2smi[i]

    return smi



def smiles_decoder_gen(X):
    smi = ''
    #idx = torch.argmax(X, dim=-1)
    idx = []
    X = X.squeeze()

    for i in X:

        e = 0.8
        rand = torch.rand(1)

        if rand > e:
            distr = torch.distributions.categorical.Categorical(logits=i)
            id = distr.sample()
            idx.append(id)
        else:
            id = torch.argmax(i)
            idx.append(id)


    for j, i in enumerate(idx):

        if torch.is_tensor(i):
            i = i.item()
        if i == 0 or i == 1:
            continue
        elif i == 2:
            return smi
        else:
            smi += index2smi[i]

    return smi



train_dat = MyData(train_smiles, smiles_encoder, seq_len=7, chars=chars)
val_dat = MyData(val_smiles, smiles_encoder, seq_len=7, chars=chars)

trainloader = torch.utils.data.DataLoader(train_dat, batch_size=batch_size,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(val_dat, batch_size=batch_size,
                                          shuffle=False)


n_layers = 2
n_hidden = 128
model = MoleculeRNN(len(chars), n_hidden, n_layers, device=device)
#adv = Adversarial()
#model.load_state_dict(torch.load("/home/nicole/Dokumente/AI_Life_Science/03 Notebook - Evaluation of generative models-20230427/best_model.pt"))
#model = model.float()


#train(model, adv, smiles_decoder_fcd, trainloader, valloader, learningrate, epochs, device, batch_size)

train(model, smiles_decoder_fcd, trainloader, valloader, learningrate, epochs, device, batch_size)




model = MoleculeRNN(len(chars), n_hidden, n_layers, device=device)
model.load_state_dict(torch.load("/home/nicole/Dokumente/AI_Life_Science/03 Notebook - Evaluation of generative models-20230427/last_model.pt"))
model = model.float()


generated_smiles = []


while len(generated_smiles) < 10001:

    out_str = gen_mol(model, all_smiles, seq_len=3, decoder=smiles_decoder_gen, encoder=smiles_encoder)

    try:

        out_smile = Chem.MolToSmiles(Chem.MolFromSmiles(out_str))
        if not out_str in generated_smiles and len(out_str) > 5:
            generated_smiles.append(out_str)

        print("Yay")

    except:
        continue



with open('generated.txt', 'w') as f:
    for smile in generated_smiles:
        # write each item on a new line
        f.write("%s\n" % smile)

print("Finished")

