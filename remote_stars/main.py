import numpy as np
import matplotlib.pyplot as plt
import socket
from skimage.measure import label

host = "84.237.21.36"
port = 5152

def dist(center1, center2):
    return ((center1[1] - center2[1]) ** 2 + (center1[0] - center2[0]) ** 2) ** 0.5

def get_center(image, labeled, label):
    center = np.unravel_index(np.argmax(image * (labeled == label)), image.shape)
    return center

def recvall(sock, nbytes):
    data = bytearray()
    while len(data) < nbytes:
        packet = sock.recv(nbytes - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def get_res(im) -> float:
    labeled = label(im > 0)
    center1 = get_center(im, labeled, 1)
    center2 = get_center(im, labeled, 2)
    return dist(center1, center2)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((host, port))
        sock.send(b"124ras1")
        print(sock.recv(10))

        beat = b"nope"
        while beat != b"yep":
            sock.send(b"get")
            bts: None | bytearray = recvall(sock, 40002)

            im = np.frombuffer(bts[2:40002], dtype="uint8").reshape(bts[0], bts[1])
            answer: float = round(get_res(im), 1)

            sock.send(str(answer).encode())
            print(sock.recv(10))
            sock.send(b"beat")
            beat: bytes = sock.recv(10)


main()