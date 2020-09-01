import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

