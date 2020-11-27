import torch.nn.functional as F
import torch
import torch.utils.data as data
import torch.nn as nn
import os

from data_processing.data_pipeline import data_processing
from model.base_model import SpeechRecognitionModel
from train_and_test.test import evaluate_model
from train_and_test.utils import IterMeter


def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                       100. * batch_idx / len(train_loader), loss.item()))


def train_model(hparams, train_dataset, test_dataset):
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

    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = torch.optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=28).to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                                    steps_per_epoch=int(len(train_loader)),
                                                    epochs=hparams['epochs'],
                                                    anneal_strategy='linear')

    iter_meter = IterMeter()
    for epoch in range(1, hparams['epochs'] + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)

    evaluate_model(hparams, model, None, test_dataset)
    return model
