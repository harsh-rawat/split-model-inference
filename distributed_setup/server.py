import pickle
import socket


def set_server_connection(port):
    server_socket = socket.socket()
    server_self_hostname = socket.gethostname()
    server_socket.bind((server_self_hostname, port))
    server_socket.listen(5)
    return server_socket


def send_data(s, data):
    data = pickle.dumps(data)
    s.send(data)
