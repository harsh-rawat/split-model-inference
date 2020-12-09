from encoder.base_encoder import EncoderDecoder
from model.save_load_model import load_autoencoder_model


class AutoEncoders(EncoderDecoder):
    def __init__(self, file_path, input_size=512, hidden_size=128, output_size=32, leaky_relu=0.2):
        self.autoencoder = load_autoencoder_model(file_path, input_size, hidden_size, output_size, leaky_relu)

    def compress(self, x):
        return (self.autoencoder.encoder(x), None, None)

    def decompress(self, x):
        return self.autoencoder.decoder(x)
