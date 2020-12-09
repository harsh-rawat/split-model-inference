import pickle
import socket

from utilities.Timer import Timer


def set_client_connection(host, port):
    server_socket = socket.socket()
    server_socket.connect((host, port))
    print('Client has been set up!')
    return server_socket


def get_data(socket_connection, batch_idx, network_latency_timer: Timer):
    print('fetching data')
    data = []
    while True:
        packet = socket_connection.recv(4096)
        if not packet:
            break
        data.append(packet)
    network_latency_timer.record(batch_idx)

    data_arr = pickle.loads(b"".join(data))
    print(data_arr)
    return data_arr
