import argparse
import torchaudio

from model.save_load_model import *
from train_and_test.test import evaluate_model
from train_and_test.train import train_model
from encoder.huffman import Huffman


def get_encoder(encoder_type):
    if encoder_type == 'huffman':
        print('Huffman Encoder is being used!')
        return Huffman()
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for split model inference')
    parser.add_argument('-test', action='store_true', default=False, required=False, help='Test mode')
    parser.add_argument('-path', metavar='base-path', action='store', default="./", required=False,
                        help='The base path for the project')
    parser.add_argument('-batch', metavar='Batch Size', action='store', default=10, required=False,
                        help='Batch size to be used in training set')
    parser.add_argument('-epochs', metavar='Epochs', action='store', default=10, required=False,
                        help='No of Epochs for training')
    parser.add_argument('-savefile', metavar='Save File', action='store', default='model.pth', required=False,
                        help='File for saving the checkpoint')
    parser.add_argument('-encoder', metavar='Encoder type', action='store', default='huffman', required=False,
                        help='Encoder to be used encoding in split model inference')

    args = parser.parse_args()

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride": 2,
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": int(args.batch),
        "epochs": int(args.epochs)
    }

    train_dataset = None
    if not args.test:
        train_dataset = torchaudio.datasets.LIBRISPEECH("{}/dataset".format(args.path), url='train-clean-100', download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("{}/dataset".format(args.path), url='test-clean', download=True)

    save_filepath = '{}/{}'.format(args.path, args.savefile)
    if args.test:
        model = load_model(save_filepath, hparams)
        sp_model = load_split_model(save_filepath, hparams)

        encoder = get_encoder(args.encoder)

        print('Evaluating complete model without any splitting')
        evaluate_model(hparams, model, None, test_dataset, encoder)
        print('Evaluating split model')
        evaluate_model(hparams, None, sp_model, test_dataset, encoder)
    else:
        model = train_model(hparams, train_dataset, test_dataset)
        save_model(model, save_filepath)
