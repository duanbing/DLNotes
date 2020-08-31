# lr 适合连续样本分类
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs));

labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
print(labels.shape)
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 准备数据
from mxnet.gluon import data as gdata;
batch_size = 10;
dataset = gdata.ArrayDataset(features, labels);
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True);
for X, y in data_iter:
    print(X, y)
    break


# 定义模型
from mxnet.gluon import nn
from mxnet import init

net = nn.Sequential();
net.add(nn.Dense(1));
net.initialize(init.Normal(sigma = 0.01));

# 定义损失函数

from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()

# 定义优化算法
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})


# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)  # 前向计算，计算损失函数
        l.backward()  # 后向计算损失函数的梯度
        trainer.step(batch_size)
    l = loss(net(features), labels) ## 评估
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))


