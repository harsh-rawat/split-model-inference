from data_processing.data_pipeline import *
from distributed_setup.server import *
from utilities.Timer import Timer


def run_node0(model, sp_model, device, test_loader, encoder_decoder, s):
    network_latency_timer = Timer('Network Latency', '')
    compression_start_timer = Timer('Compression start compute', '')
    compression_end_timer = Timer('Compression end compute', '')
    inference_start_timer = Timer('Inference Start', '')
    inference_end_timer = Timer('Inference End', '')
    end_to_end_timer = Timer('End to end', '')

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            if i == 1:
                break
            end_to_end_timer.record(i)
            inference_start_timer.record(i)
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            if model is None:
                # Pass the input through head model
                intermediate_activations = sp_model[0].forward(spectrograms)
                compression_start_timer.record(i)
                intermediate, shape, coder = encoder_decoder.compress(intermediate_activations)
                compression_end_timer.record(i)
            else:
                # shape and coder are None here
                intermediate, shape, coder = spectrograms, None, None

            data_to_send = [intermediate, labels, label_lengths, input_lengths, shape, coder]
            print("data size: {}".format(len(data_to_send)))
            inference_end_timer.record(i)
            network_latency_timer.record(i)
            # Send this intermediate value to server
            bytes_sent = send_data(s, data_to_send)
            print("Bytes sent for batch {} are {}".format(i, bytes_sent))

    end_to_end_timer.print()
    network_latency_timer.print()
    compression_start_timer.print()
    compression_end_timer.print()
    inference_start_timer.print()
    inference_end_timer.print()