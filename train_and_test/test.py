import os

import torch.nn.functional as F
import torch.utils.data as data

from data_processing.data_pipeline import *
from data_processing.utils import *
from encoder.base_encoder import EncoderDecoder


def run_entire_model(model, device, test_loader, criterion):
    test_loss = 0
    test_cer, test_wer = [], []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1)  # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)
            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def run_node1_client_data(test_loader, device, sp_model):
    client_data = dict()  # In the fomat "1":[shape, labels, label_length, input_lengths]
    with torch.no_grad:
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            intermediate_activations = sp_model[0].forward(spectrograms)  # Pass the input through head model
            if i not in client_data.keys():
                client_data[i] = list()
            client_data[i].append(intermediate_activations.shape)
            client_data[i].append(labels)
            client_data[i].append(label_lengths)
            client_data[i].append(input_lengths)

    return client_data


def run_node0(sp_model, device, test_loader, encoder_decoder):
    with torch.no_grad:
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            intermediate_activations = sp_model[0].forward(spectrograms)  # Pass the input through head model
            intermediate = encoder_decoder.compress(intermediate_activations)
            # Send this intermediate value to server


def run_node1_on_receive(sp_model, encoder_decoder, client_data, criterion, test_cer, test_wer):
    ##########################################################
    batch_idx = 1;  # Get index of the current received batch
    intermediate = None;  # Get data sent from node 0
    ##########################################################
    batch_data = client_data[batch_idx]
    reconstructed_output = encoder_decoder.decompress(intermediate, batch_data[0])

    with torch.no_grad:
        output = sp_model[1].forward(reconstructed_output)

        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)  # (time, batch, n_class)

        loss = criterion(output, batch_data[1], batch_data[3], batch_data[2])
        # test_loss += loss.item() / len(test_loader)

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), batch_data[1], batch_data[2])
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    return loss.item()


def run_node1(test_loader, sp_model, encoder_decoder, client_data, criterion):
    test_loss = 0
    test_cer, test_wer = [], []

    # Run this while we are receiving the inputs
    while (1):
        loss = run_node1_on_receive(sp_model, encoder_decoder, client_data, criterion, test_cer, test_wer)
        test_loss += loss / len(test_loader)

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def test(model, sp_model, device, test_loader, criterion, encoder_decoder: EncoderDecoder, rank):
    print('\nevaluating...')

    if model is not None:
        run_entire_model(model, device, test_loader, criterion)
        return

    if rank == 0:
        run_node0(sp_model, device, test_loader, encoder_decoder)
    elif rank == 1:
        client_data = run_node1_client_data(test_loader, device, sp_model)
        run_node1(test_loader, sp_model, encoder_decoder, client_data, criterion)


def evaluate_model(hparams, model, sp_model, test_dataset, encoder_decoder: EncoderDecoder, rank):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, 'valid'),
                                  **kwargs)
    criterion = nn.CTCLoss(blank=28).to(device)

    test(model, sp_model, device, test_loader, criterion, encoder_decoder, rank)
