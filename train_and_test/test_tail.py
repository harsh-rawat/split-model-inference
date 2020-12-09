import torch.nn.functional as F

from data_processing.data_pipeline import *
from data_processing.utils import *
from distributed_setup.client import get_data
from utilities.Timer import Timer


def run_node1_on_receive(batch_idx, received_data, intermediate, model, sp_model, encoder_decoder, criterion, test_cer,
                         test_wer, timers):
    with torch.no_grad():
        if model is None:
            timers[0].record(batch_idx)
            reconstructed_output = encoder_decoder.decompress(intermediate)
            timers[1].record(batch_idx)
            output = sp_model[1].forward(reconstructed_output)
        else:
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
    end_to_end_timer = Timer('End to end', '')
    network_latency_timer = Timer('Network Latency', '')
    decompression_start_timer = Timer('Decompression start compute', '')
    decompression_end_timer = Timer('Decompression end compute', '')
    timers = [decompression_start_timer, decompression_end_timer]

    batch_idx = 0
    total_batches = 1
    # Run this while we are receiving the inputs
    conn, addr = server_socket.accept()
    while batch_idx < total_batches:
        received_data = get_data(conn, batch_idx, network_latency_timer)
        # Assuming that received data is of the format - [intermediate, labels, label_length, input_length]
        intermediate = received_data[0]
        loss = run_node1_on_receive(batch_idx, received_data, intermediate, model, sp_model, encoder_decoder, criterion,
                                    test_cer, test_wer, timers)

        test_loss += loss / test_loader_len
        batch_idx += 1
        end_to_end_timer.record(batch_idx)

    avg_cer = sum(test_cer) / len(test_cer)
    avg_wer = sum(test_wer) / len(test_wer)

    print(
        'Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))
    end_to_end_timer.print()
    network_latency_timer.print()
    decompression_end_timer.find_difference(decompression_start_timer)
