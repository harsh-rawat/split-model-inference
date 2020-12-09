import pickle
import socket


def set_client_connection(host, port):
    server_socket = socket.socket()
    server_socket.connect((host, port))
    print('Client has been set up!')
    return server_socket


def get_data(socket_connection, batch_idx):
    print('fetching data')
    data = []
    while True:
        packet = socket_connection.recv(4096)
        if not packet:
            break
        data.append(packet)

    data_arr = pickle.loads(b"".join(data))
    print('Sending ack from client for batch {}'.format(batch_idx))
    ack_text = "This is ACK for batch " + str(batch_idx)
    socket_connection.send(ack_text.encode())

    return data_arr
