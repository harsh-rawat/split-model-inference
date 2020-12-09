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
    i = 0
    size = 0
    while True:
        i += 1
        packet = socket_connection.recv(4096)
        size += len(packet)
        if packet[-1] == 68 and packet[-2] == 78 and packet[-3] == 69:
            data.append(packet[0:-4])
            break
        data.append(packet)
    network_latency_timer.record(batch_idx)

    data_arr = pickle.loads(b"".join(data))
    print('Sending ack from client for batch {}'.format(batch_idx))
    ack_text = "This is ACK for batch " + str(batch_idx)
    socket_connection.send(ack_text.encode())

    return data_arr
