import os

import torch.nn.functional as F
import torch.utils.data as data

from data_processing.data_pipeline import *
from data_processing.utils import *
from encoder.base_encoder import EncoderDecoder
from encoder.huffman import Huffman


def test(model, sp_model, device, test_loader, criterion, encoder_decoder: EncoderDecoder):
    print('\nevaluating...')
    test_loss = 0
    test_cer, test_wer = [], []

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            if model is not None:
                output = model(spectrograms)  # (batch, time, n_class)
            else:
                intermediate_activations = sp_model[0].forward(spectrograms)  # Pass the input through head model
                intermediate = encoder_decoder.compress(intermediate_activations)
                # Send this intermediate value to server
                reconstructed_output = encoder_decoder.decompress(intermediate)
                output = sp_model[1].forward(reconstructed_output)

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


def evaluate_model(hparams, model, sp_model, test_dataset, encoder_decoder: EncoderDecoder):
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

    test(model, sp_model, device, test_loader, criterion, encoder_decoder)
