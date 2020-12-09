import pickle
import socket


def set_client_connection(host, port):
    server_socket = socket.socket()
    server_socket.connect((host, port))
    return server_socket


def get_data(socket_connection):
    data = []
    while True:
        packet = socket_connection.recv(4096)
        if not packet:
            break
        data.append(packet)
    data_arr = pickle.loads(b"".join(data))
    return data_arr
