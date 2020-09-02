from mxnet import nd, init;
from mxnet.gluon import nn;


class MyInit(init.Initializer):
    def __init_weight(self, name, data):
        print("Init", name, data.shape)

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs);
    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
#layer(nd.array([1, 2, 3, 4, 5]))

net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer());


net.initialize();
y = net(nd.random.uniform(shape=(4, 8)));
print(y.mean().asscalar())
