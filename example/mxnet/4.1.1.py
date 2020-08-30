#!/usr/local/bin/python3
from mxnet import nd;
from mxnet.gluon import nn;

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs);
        self.hidden = nn.Dense(256, activation = "relu");
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()

X = nd.random.uniform(shape=(2, 20))
Y = net(X)
print(net[0].params)
