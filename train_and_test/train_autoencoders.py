import os

import torch
import torch.nn as nn
import torch.utils.data as data

from data_processing.data_pipeline import data_processing
from model.AutoEncoderDecoder import AutoEncoderDecoder


def train(sp_model, encoder_model, device, train_loader, criterion, optimizer, scheduler, epoch, input_dict):
    sp_model[0].eval()
    encoder_model.train()
    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        if batch_idx not in input_dict.keys():
            split_activations = sp_model[0](spectrograms)
            input_dict[batch_idx] = split_activations
        else:
            split_activations = input_dict[batch_idx]
        output = encoder_model(split_activations)

        loss = criterion(output, split_activations)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Autoencoder Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / len(train_loader), loss.item()))


def train_autoencoders(sp_model, hparams, train_dataset):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=hparams['batch_size'],
                                   shuffle=True,
                                   collate_fn=lambda x: data_processing(x, 'train'),
                                   **kwargs)

    model = AutoEncoderDecoder(hparams['input_layers'], hparams['hidden_layers'], hparams['output_layers'],
                               hparams['leaky_relu'])
    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])

    criterion = nn.MSELoss().to(device)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                    steps_per_epoch=int(len(train_loader)),
                                                    epochs=hparams['epochs'],
                                                    anneal_strategy='linear')

    input_dict = dict()
    for epoch in range(1, hparams['epochs'] + 1):
        train(sp_model, model, device, train_loader, criterion, optimizer, scheduler, epoch, input_dict)

    return model
