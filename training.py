import pandas as pd
import os
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from utils import canonicalize_smiles
import fcd_torch




def evaluate_model(model, decoder, batch_size, dataloader: torch.utils.data.DataLoader, device):

    model.to(device)

    loss_funct = torch.nn.CrossEntropyLoss()

    hidden = [torch.zeros(model.n_layers, batch_size, model.n_hidden),
              torch.zeros(model.n_layers, batch_size, model.n_hidden)]
    hidden[0] = hidden[0].to(device)
    hidden[1] = hidden[1].to(device)

    losses = []
    with torch.no_grad():
        for data in dataloader:

            hidden = [torch.zeros(model.n_layers, batch_size, model.n_hidden),
                      torch.zeros(model.n_layers, batch_size, model.n_hidden)]
            hidden[0] = hidden[0].to(device)
            hidden[1] = hidden[1].to(device)

            inputs, targets, ids = data

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            outputs, hidden = model(inputs, hidden)

            target_ids = torch.argmax(targets, dim=2).long()

            #out_str = decoder(outputs)
            #in_str = decoder(inputs)

            #valid = list(filter(None, canonicalize_smiles(out_str)))
            #loss2 = 1 / (len(valid) + 1)


            loss = loss_funct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1)) #+ loss2

            losses.append(loss)

    return sum(losses) / len(losses)



def train(model, smiles_decoder, trainloader, valloader, learningrate,
          epochs, device, batch_size):



    #loss_funct = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_funct = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learningrate,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                  base_lr=learningrate,
                                                  max_lr=0.008,
                                                  cycle_momentum=False)

    model.to(device)

    train_losses = []
    val_losses = []
    update = 0
    best_val = np.inf


    for epoch in range(epochs):

        print("_______________________")
        print("EPOCH")
        print(epoch)
        print("_______________________")

        for data in trainloader:

            hidden = [torch.zeros(model.n_layers, batch_size, model.n_hidden).to(device),
                      torch.zeros(model.n_layers, batch_size, model.n_hidden).to(device)]

            inputs, targets, ids = data

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            optimizer.zero_grad()

            outputs, hidden = model(inputs, hidden)

            target_ids = torch.argmax(targets, dim=2).long()

            #out_str = smiles_decoder(outputs)
            #in_str = smiles_decoder(inputs)

            #valid = list(filter(None, canonicalize_smiles(out_str[0:batch_size//2])))

            #loss2 = 2/(len(valid)+1)

            loss = loss_funct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1)) #+ loss2


            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1

            if update % 50 == 0:

                val_loss = evaluate_model(model, smiles_decoder, batch_size, valloader, device)
                print("Validation loss:")
                print(val_loss)
                val_losses.append(val_loss)

                if val_loss < best_val:
                    torch.save(model.state_dict(), './best_model.pt')
                    best_val = val_loss


    #plt.figure(1)
    #plt.plot(torch.tensor(val_losses).cpu().detach())
    #plt.title("Val-Loss")
    #plt.savefig("Validation_loss.png")

    #plt.figure(2)
    #plt.plot(torch.tensor(train_losses).cpu().detach())
    #plt.title("Train-Loss")
    #plt.savefig("Train_loss.png")

    torch.save(model.state_dict(), './last_model.pt')
