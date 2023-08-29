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




def evaluate_model(model, adversarial, batch_size, dataloader: torch.utils.data.DataLoader, device):

    model.to(device)

    loss_funct = torch.nn.CrossEntropyLoss()
    loss_adv = torch.nn.BCELoss()

    hidden = [torch.zeros(model.n_layers, batch_size, model.n_hidden),
              torch.zeros(model.n_layers, batch_size, model.n_hidden)]
    hidden[0] = hidden[0].to(device)
    hidden[1] = hidden[1].to(device)

    label_true = torch.ones(size=(batch_size, 1)).to(device)
    label_wr = torch.zeros(size=(batch_size, 1)).to(device)

    losses = []
    with torch.no_grad():
        for i in range(batch_size):

            hidden = [torch.zeros(model.n_layers, batch_size, model.n_hidden).to(device),
                      torch.zeros(model.n_layers, batch_size, model.n_hidden).to(device)]

            """
            Main Model
            """

            noise = torch.randn(batch_size, 120, 64).to(device)
            outputs, hidden = model(noise, hidden)

            loss = loss_adv(adversarial(outputs), label_true)

            losses.append(loss.item())

    return sum(losses) / len(losses)



def train(model, adversarial, smiles_decoder, trainloader, valloader, learningrate,
          epochs, device, batch_size):



    #loss_funct = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_adv = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learningrate, weight_decay=1e-4)
    optimizer_adv = torch.optim.Adam(params=adversarial.parameters(), lr=learningrate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,
                                                  base_lr=learningrate,
                                                  max_lr=0.001,
                                                  cycle_momentum=False)

    model.to(device)
    adversarial.to(device)

    train_losses = []
    val_losses = []
    update = 0
    best_val = np.inf

    label_true = torch.ones(size=(batch_size, 1)).to(device)
    label_wr = torch.zeros(size=(batch_size, 1)).to(device)

    latent = 64

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


            """
            Adversarial
            """
            optimizer_adv.zero_grad()

            loss_real = loss_adv(adversarial(inputs), label_true)

            noise = torch.randn(batch_size, 120, latent).to(device)
            outputs, _ = model(noise, hidden)

            loss_wrong = loss_adv(adversarial(outputs), label_wr)

            loss = loss_wrong + loss_real

            loss.backward()

            optimizer_adv.step()


            """
            Main Model
            """
            optimizer.zero_grad()

            noise = torch.randn(batch_size, 120, latent).to(device)
            outputs, hidden = model(noise, hidden)

            loss = loss_adv(adversarial(outputs), label_true)

            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1

            if update % 50 == 0:

                val_loss = evaluate_model(model, adversarial, batch_size, valloader, device)
                print("Validation loss:")
                print(val_loss)
                val_losses.append(val_loss)

                if val_loss < best_val:
                    torch.save(model.state_dict(), './best_model.pt')
                    best_val = val_loss


    #plt.figure(1)
    #plt.plot(val_losses)
    #plt.title("Val-Loss")
    #plt.savefig("Validation_loss.png")

    #plt.figure(2)
    #plt.plot(train_losses)
    #plt.title("Train-Loss")
    #plt.savefig("Train_loss.png")

    torch.save(model.state_dict(), './last_model.pt')
