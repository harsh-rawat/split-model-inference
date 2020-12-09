from data_processing.data_pipeline import *
from distributed_setup.server import *


def run_node0(model, sp_model, device, test_loader, encoder_decoder, s):
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            if i == 1:
                break
            spectrograms, labels, input_lengths, label_lengths = _data
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            if model is None:
                # Pass the input through head model
                intermediate_activations = sp_model[0].forward(spectrograms)
                intermediate, shape, coder = encoder_decoder.compress(intermediate_activations)
            else:
                # shape and coder are None here
                intermediate, shape, coder = spectrograms

            data_to_send = [intermediate, labels, label_lengths, input_lengths, shape, coder]
            # Send this intermediate value to server
            send_data(s, data_to_send)
