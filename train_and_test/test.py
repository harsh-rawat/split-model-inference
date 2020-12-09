import os

import torch.utils.data as data

from data_processing.data_pipeline import *
from distributed_setup.client import *
from distributed_setup.server import *
from encoder.base_encoder import EncoderDecoder
from train_and_test.test_entire_model import run_entire_model
from train_and_test.test_head import run_node0
from train_and_test.test_tail import run_node1


def test(model, sp_model, device, test_loader, criterion, encoder_decoder: EncoderDecoder, rank, host, port):
    print('\nevaluating...')

    if model is not None:
        run_entire_model(model, device, test_loader, criterion)
        return

    if rank == 0:
        s = set_client_connection(host, port)
        run_node0(sp_model, device, test_loader, encoder_decoder, s)
        s.close()
    elif rank == 1:
        server_socket = set_server_connection(port)
        run_node1(len(test_loader), sp_model, encoder_decoder, criterion, server_socket)


def evaluate_model(hparams, model, sp_model, test_dataset, encoder_decoder: EncoderDecoder, rank, host, port):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=hparams['batch_size'],
                                  shuffle=False,
                                  collate_fn=lambda x: data_processing(x, 'valid'),
                                  **kwargs)
    criterion = nn.CTCLoss(blank=28).to(device)

    test(model, sp_model, device, test_loader, criterion, encoder_decoder, rank, host, port)
