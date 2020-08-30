#!/usr/local/bin/python3
import mxnet as mx;
from mxnet import nd;
from mxnet.gluon import nn;

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs);
        self.hidden = nn.Dense(256, activation = "relu");
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
w = net(X)
