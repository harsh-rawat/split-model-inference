import torch.nn.functional as F

from data_processing.data_pipeline import *
from data_processing.utils import *
from distributed_setup.client import get_data


def run_node1_on_receive(received_data, intermediate, model, sp_model, encoder_decoder, criterion, test_cer,
                         test_wer):
    with torch.no_grad():
        if model is None:
            # reconstructed_output = encoder_decoder.decompress(intermediate, batch_data[0])
            reconstructed_output = intermediate
            output = sp_model[1].forward(reconstructed_output)
        else:
            # reconstructed_output = encoder_decoder.decompress(intermediate, batch_data[0])
            output = model(intermediate)

        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        loss = criterion(output, received_data[1], received_data[3], received_data[2])

        decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), received_data[1], received_data[2])
        for j in range(len(decoded_preds)):
            test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
            test_wer.append(wer(decoded_targets[j], decoded_preds[j]))

    return loss.item()


def run_node1(test_loader_len, model, sp_model, encoder_decoder, criterion, server_socket):
    test_loss = 0
    test_cer, test_wer = [], []

    conn, addr = server_socket.accept()
    batch_idx = 0
    total_batches = test_loader_len
    # Run this while we are receiving the inputs
    while batch_idx < total_batches:
        received_data = get_data(conn, batch_idx)
        # Assuming that received data is of the format - [intermediate, labels, label_length, input_length]
        intermediate = received_data[0]

        loss = run_node1_on_receive(received_data, intermediate, model, sp_model, encoder_decoder, criterion,
                                    test_cer, test_wer)

        test_loss += loss / test_loader_len
        batch_idx += 1

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
