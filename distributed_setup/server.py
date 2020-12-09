import pickle
import socket
import time


def set_server_connection(port):
    server_socket = socket.socket()
    server_self_hostname = socket.gethostname()
    server_socket.bind((server_self_hostname, port))
    server_socket.listen(5)
    print('The server has started!')
    return server_socket


def send_data(s, data):
    print('Sending data')
    data = pickle.dumps(data)

    s.send(data)
    end_tag = "END"
    print("End: {}".format(end_tag))
    s.send(end_tag.encode())
    print('Will start waiting for ack')
    # wait until ACK received from node1
    recv_data = str(s.recv(1024))
    while not recv_data:
        recv_data = str(s.recv(1024))
    print('Acknowledged received: {}'.format(recv_data))
