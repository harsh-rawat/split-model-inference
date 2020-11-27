import numpy as np
import torch
from dahuffman import HuffmanCodec

from encoder.base_encoder import EncoderDecoder


# Huffman encoding
class Huffman(EncoderDecoder):
    def __init__(self):
        self.dim = None
        self.codec = None

    def compress(self, x):
        # note down x's dimensions using x.shape. Save this in some variable. Assume x has dimension of
        # (4,3,2) => 24 values
        self.dim = x.shape
        # Convert pytorch tensor into numpy array
        tensor = x.numpy()
        # Convert numpy array into a single array
        a = np.array(tensor)
        b = a.ravel()

        # encode it. Assume you converted 24 values into 6 values.
        # encode with Huffman coding
        self.coded = HuffmanCodec.from_data(b)
        encoded = self.coded.encode(b)
        # return it
        return encoded

    def decompress(self, x):
        # x is an array of 6 values
        # decompress it. So convert 6 values to 24 values.
        decoded = self.coded.decode(x)

        # Decoded is a list now
        decode_numpy = np.asarray(decoded)
        decode_tensor = torch.from_numpy(decode_numpy)

        # convert x into numpy n-d array using saved shape of tensor. Now the shape of numpy array would be (4,3,2)
        ans = torch.reshape(decode_tensor, self.dim)
        return ans
