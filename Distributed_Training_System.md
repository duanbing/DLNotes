# 分布式深度学习框架

[toc]

## 背景

​	在[上一篇](./Deep_Learning_Basics.pdf)的结尾简单介绍了分布式深度学习框架的对比，本节主要针对分布式学习过程中的挑战与解决方案进行调研。

### 主要挑战

#### 梯度计算算法

​	常用的SGD算法串行更新参数，处理

#### 节点之间通信量

#### 节点容错

#### 异构设备支持



## 既有实现

### 主要思路

 	1. GPU并行模式
      	1. AlexNET等基于双GPU的并行训练；
 	2. 多节点分批并行执行，取梯度平均值<sup>[4]</sup><sup>[5]</sup>；
 	3. 图计算模式，结合数据并行思路，分析依赖子图，然后将子图分发到多节点执行<sup>[5]</sup>；
 	4. Pipeline<sup>[5]</sup><sup>[6]</sup>,将训练过程分解为不同的stages，然后利用线程池等技术并行执行训练，可以有效的解决资源瓶颈的问题，
 	5. 平衡内存和IO使用，将稀疏网络转换为稠密网络<sup>[6]</sup>；

#### DistBelief/Tensorflow<sup>[1]</sup><sup>[5]</sup>

​	[DistBelief](https://en.wikipedia.org/wiki/TensorFlow#DistBelief)是TensorFlow的前身，通过并行、同步以及通信的优化，支持深度神经学习可以在节点内和节点间进行并行训练，支持超大规模参数的模型的训练。 实验显示，借助于自适应学习率调整和足够的计算资源，其在非凸问题上也有很好的表现。

​	DistBelief提出了2种并行化的思路。

*  Downpour SGD

  DSGD是一种[在线算法](https://en.wikipedia.org/wiki/Online_algorithm),  首先将数据分片，分配到各个机器，然后每个机器在本地借助AdaGrad等自适应学习率优化方法进行模型训练，根据实现选定的nFetch和nPush参数，定期的拉取和推送本地更新的参数。  这种方式一方面提高了标准同步SGD算法的鲁棒性（节点失效导致更新夯住），更好的随机性。当模型较大的嘶吼，加速会随着机器数量的增加而提高，最好的情况通过128机器实现12倍的加速。在解决非凸问题上虽然理论支撑不足，但是实际效果还不错。

* Sandblaster L-BFGS

  Sandblaster是一种批处理算法，

#### Hogwild!

​	Hogwild!提供了一种”无锁方式并行运行SGD“。



## 主要思路

### 主要算法

#### 模型并行 Model parallelism

​	模型并行指一个模型分布式在多个节点进行训练。最早在ResNet出现，将网络运行在2个GPU上进行模型训练。DistBelief是典型的代表， 其思路类似于Spark的RDD的并行思路，把网络的一层或者多层作为作为一行，然后将一行按照机器数进行划分列，从而分离出多个小的划分，每个划分交给一个机器进行处理。 每个划分内部还可以在一个机器的多个core上进行类似的划分执行。

​	

#### 数据并行



#### Pipeline



#### 异步梯度下降

##### n-softsync protocal[3]





## 参考

[1] Jeffrey Dean, et.al.  Large Scale Distributed Deep Networks, 2012

[2] Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent

[3] Wei Zhang, et.al.  Staleness-aware Async-SGD for Distributed Deep Learning

[4] L. Deng, et.al. Scalable stacking and learning for building deep architecure, In ICASSP, 2012

[5] Mart ́ın, et.al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems, 2015

[6] Biye Jiang, et.al XDL: An Industrial Deep Learning Framework for High-dimensional Sparse Data, 2019

 



