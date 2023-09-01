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


    losses = []
    with torch.no_grad():
        for data in dataloader:

            inputs, targets, ids = data

            hidden = [torch.zeros(model.n_layers, inputs.shape[0], model.n_hidden).to(device),
                      torch.zeros(model.n_layers, inputs.shape[0], model.n_hidden).to(device)]

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            outputs, _ = model(inputs, hidden)

            target_ids = torch.argmax(targets, dim=2).long()

            #out_str = decoder(outputs)
            #in_str = decoder(inputs)


            loss = loss_funct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1)) #+ loss2

            losses.append(loss)

    return sum(losses) / len(losses)



def train(model, smiles_decoder, trainloader, valloader, learningrate,
          epochs, device, batch_size):



    #loss_funct = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_funct = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learningrate)

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

            inputs, targets, ids = data

            hidden = [torch.zeros(model.n_layers, inputs.shape[0], model.n_hidden).to(device),
                      torch.zeros(model.n_layers, inputs.shape[0], model.n_hidden).to(device)]

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            optimizer.zero_grad()

            outputs, _ = model(inputs, hidden)

            target_ids = torch.argmax(targets, dim=2).long()

            #out_str = smiles_decoder(outputs[0})
            #in_str = smiles_decoder(inputs[0])


            loss = loss_funct(outputs.view(-1, outputs.size(-1)), target_ids.view(-1)) #+ loss2


            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            update += 1

            if update % 100 == 0:

                val_loss = evaluate_model(model, smiles_decoder, batch_size, valloader, device)
                print("Validation loss:")
                print(val_loss)
                val_losses.append(val_loss)

                if val_loss < best_val:
                    torch.save(model.state_dict(), './best_model.pt')
                    best_val = val_loss




    torch.save(model.state_dict(), './last_model.pt')

    plt.figure(1)
    plt.plot(torch.tensor(val_losses).cpu().detach())
    plt.title("Val-Loss")
    plt.savefig("Validation_loss.png")

    plt.figure(2)
    plt.plot(torch.tensor(train_losses).cpu().detach())
    plt.title("Train-Loss")
    plt.savefig("Train_loss.png")

