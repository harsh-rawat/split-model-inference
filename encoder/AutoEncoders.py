import torch.nn as nn

from encoder.base_encoder import EncoderDecoder
from model.save_load_model import load_autoencoder_model


class AutoEncoderDecoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, output_size=32, leaky_relu=0.2):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(leaky_relu)
        # self.sigmoid = nn.Sigmoid()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        self.layer_3 = nn.Linear(output_size, hidden_size)
        self.layer_4 = nn.Linear(hidden_size, input_size)

        self._initialize_weights()

    def forward(self, x):
        x = self.leaky_relu(self.layer_1(x))
        x = self.leaky_relu(self.layer_2(x))
        x = self.leaky_relu(self.layer_3(x))
        x = self.leaky_relu(self.layer_4(x))
        return x

    def encoder(self, x):
        x = self.leaky_relu(self.layer_1(x))
        x = self.leaky_relu(self.layer_2(x))
        return x

    def decoder(self, x):
        x = self.leaky_relu(self.layer_3(x))
        x = self.leaky_relu(self.layer_4(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.02)
                nn.init.constant_(m.bias.data, 0)


class AutoEncoders(EncoderDecoder):
    def __init__(self, file_path, input_size=512, hidden_size=128, output_size=32, leaky_relu=0.2):
        self.autoencoder = load_autoencoder_model(file_path, input_size, hidden_size, output_size, leaky_relu)

    def compress(self, x):
        return self.autoencoder.encoder(x)

    def decompress(self, x):
        return self.autoencoder.decoder(x)
