import argparse

from model.save_load_model import *
from setup import install_requirements
from train_and_test.test import evaluate_model
from train_and_test.train import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for split model inference')
    parser.add_argument('-test', action='store_true', default=False, required=False, help='Test mode')
    parser.add_argument('-path', metavar='base-path', action='store', default='/tmp', required=False,
                        help='The base path for the project')
    parser.add_argument('-batch', metavar='Batch Size', action='store', default=10, required=False,
                        help='Batch size to be used in training set')
    parser.add_argument('-epochs', metavar='Epochs', action='store', default=10, required=False,
                        help='No of Epochs for training')
    parser.add_argument('-savefile', metavar='Save File', action='store', default='model.pth', required=False,
                        help='File for saving the checkpoint')

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
        "batch_size": args.batch,
        "epochs": args.epochs
    }

    install_requirements()
    save_filepath = '{}/{}'.format(args.path, args.savefile)
    if args.test:
        model = load_model(save_filepath, hparams)
        sp_model = load_split_model(save_filepath, hparams)
        print('Evaluating complete model without any splitting')
        evaluate_model(hparams, model, None)
        print('Evaluating split model')
        evaluate_model(hparams, None, sp_model)
    else:
        model = train_model(hparams)
        save_model(model, save_filepath)
