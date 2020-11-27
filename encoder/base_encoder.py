from abc import ABC, abstractmethod


class EncoderDecoder(ABC):
    @abstractmethod
    def compress(self, x):
        pass

    @abstractmethod
    def decompress(self, x):
        pass
