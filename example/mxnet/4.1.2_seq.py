from mxnet import nd;
from mxnet.gluon import nn;
class MySeq(nn.Block):
    def __init__(self, **kwargs):
        super(MySeq, self).__init__(**kwargs);

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)

        return x

X = nd.random.uniform(shape=(2, 20))
net = MySeq()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
