from crypten import mpc

import crypten
import torch

import numpy

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

numpy.random.seed(0)


@mpc.run_multiprocess(world_size=2)
def main():
    crypten.init()
    numpy.random.seed(0)
    rank = crypten.communicator.get().get_rank()
    print(rank)

    if rank == 0:
        tensor = torch.tensor([1, 1, 1, 1, 1])
        shape = torch.tensor([tensor.shape])
    if rank == 1:
        tensor = torch.tensor([20, 20, 20, 20])
        shape = torch.tensor([tensor.shape])

    tensor_shape_0 = crypten.cryptensor(shape, src=0)
    tensor_shape_1 = crypten.cryptensor(shape, src=1)
    tensor_shapes_enc = crypten.cat([tensor_shape_0, tensor_shape_1])

    tensor_shapes = [int(x) for x in tensor_shapes_enc.get_plain_text()]
    print(tensor_shapes)

    if rank == 0:
        tensor_enc_0 = crypten.cryptensor(tensor, src=0)
    else:
        tensor_enc_0 = crypten.cryptensor(torch.empty(tensor_shapes[0]))

    if rank == 1:
        tensor_enc_1 = crypten.cryptensor(tensor, src=1)
    else:
        tensor_enc_1 = crypten.cryptensor(torch.empty(tensor_shapes[1]))

    # # print(rank, "tensor_enc_0.shape", tensor_enc_0.shape)
    # # print(rank, "tensor_enc_1.shape", tensor_enc_1.shape)
    # print(tensor_enc_0.get_plain_text())

    i = crypten.cat([tensor_enc_0, tensor_enc_1])
    print(rank, i.shape, i.get_plain_text())

    j = i.get_plain_text()

    if rank == 0:
        for x in j:
            print(x.item())

    # print(rank, tensor)
    # print(rank, tensor_enc_0.get_plain_text())
    # print(rank, tensor_enc_1.get_plain_text())
    return


'''
Dataset Augmentation must has same size?
If not same size, then create dummy data?
'''


@mpc.run_multiprocess(world_size=2)
def main_for():
    crypten.init()

    rank = crypten.communicator.get().get_rank()
    if rank == 0:
        num = 10
    if rank == 1:
        num = 11

    for i in range(num):
        print(rank, i)
        a = torch.tensor([rank])
        s0 = crypten.cryptensor(a, src=0)
        s1 = crypten.cryptensor(a, src=1)
        b = crypten.cat([s0, s1])
        print(b.get_plain_text())

    return


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def run():
    crypten.init()

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7])
    y = torch.tensor([7, 6, 5, 4, 3, 2, 1])

    x_enc = crypten.cryptensor(x, src=0)
    y_enc = crypten.cryptensor(y, src=0)

    a = CustomDataset(x_enc, y_enc)
    dl = DataLoader(a)

    for d in dl:
        print(d)


if __name__ == "__main__":
    # main()
    # main_for()
    run()
