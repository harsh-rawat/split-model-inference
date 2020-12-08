import argparse
from pathlib import Path

import torchaudio

from encoder.AutoEncoders import AutoEncoders
from encoder.huffman import Huffman
from model.save_load_model import *
from train_and_test.test import evaluate_model
from train_and_test.train import train_model
from train_and_test.train_autoencoders import train_autoencoders


def get_encoder(encoder_type, encoder_path):
    if encoder_type == 'huffman':
        print('Huffman Encoder is being used!')
        return Huffman()
    elif encoder_type == 'autoencoder':
        print('AutoEncoder is being used!')
        return AutoEncoders(encoder_path)
    else:
        return None


def create_folder(path):
    directory = Path(path)
    if not directory.exists() or not directory.is_dir():
        directory.mkdir(parents=True)


def run_server():
    pass


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
    parser.add_argument('-encoderpath', metavar='Path of saved autoencoder model', action='store',
                        default='autoencoder.pth', required=False, help='Path of the saved models of autoencoder and '
                                                                        'decoder')
    parser.add_argument('-rank', metavar='Rank of node', action='store', default=0, required=True,
                        help='Rank of the node')
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
        "epochs": int(args.epochs),
        "input_layers": 512,
        "hidden_layers": 128,
        "output_layers": 32,
        "leaky_relu": 0.2
    }

    if args.rank < 0 or args.rank > 1:
        raise Exception('Rank is incorrect. It should be either 0 or 1!')

    base_dataset_directory = "{}/dataset".format(args.path)
    create_folder(base_dataset_directory)
    train_dataset = None
    if not args.test:
        train_dataset = torchaudio.datasets.LIBRISPEECH(base_dataset_directory, url='train-clean-100',
                                                        download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH(base_dataset_directory, url='test-clean', download=True)

    save_filepath = '{}/{}'.format(args.path, args.savefile)
    encoder_base_path = '{}/{}'.format(args.path, args.encoderpath)
    if args.test:
        if args.rank == 0:
            run_server()
        model = load_model(save_filepath, hparams)
        sp_model = load_split_model(save_filepath, hparams)

        encoder = get_encoder(args.encoder, encoder_base_path)

        # print('Evaluating complete model without any splitting')
        # evaluate_model(hparams, model, None, test_dataset, encoder)
        print('Evaluating split model')
        evaluate_model(hparams, None, sp_model, test_dataset, encoder, args.rank)
    else:
        if args.encoder == 'autoencoder':
            sp_model = load_split_model(save_filepath, hparams)
            model = train_autoencoders(sp_model, hparams, train_dataset)
            save_model(model, encoder_base_path)
        else:
            model = train_model(hparams, train_dataset, test_dataset)
            save_model(model, save_filepath)
