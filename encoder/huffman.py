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
        print("Huffman encode")
        self.dim = x.shape
        print("in huffman, shape = {}".format(x.shape))
        # Convert pytorch tensor into numpy array
        tensor = x.numpy()
        # Convert numpy array into a single array
        a = np.array(tensor)
        b = a.ravel()

        # encode it. Assume you converted 24 values into 6 values.
        # encode with Huffman coding
        self.codec = HuffmanCodec.from_data(b)
        encoded = self.codec.encode(b)
        # return it
        return encoded, self.dim, self.codec

    def decompress(self, x, dim=None, codec=None):
        # x is an array of 6 values
        # decompress it. So convert 6 values to 24 values.
        print("Huffman decode")
        print("in huffman decode, shape = {}".format(x.shape))
        if codec is not None:
            self.codec = codec
        decoded = self.codec.decode(x)

        # Decoded is a list now
        decode_numpy = np.asarray(decoded)
        decode_tensor = torch.from_numpy(decode_numpy)

        # convert x into numpy n-d array using saved shape of tensor. Now the shape of numpy array would be (4,3,2)
        if dim is not None:
            self.dim = dim
        ans = torch.reshape(decode_tensor, self.dim)
        return ans
