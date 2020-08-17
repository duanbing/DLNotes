# Deep Learning Basics

[toc]

## 基本模型

### 经典回归模型

假设输入是符合特征为$x \in R^d$的样本，偏差项为$b$，$W$是权重（矩阵）参数，$\hat{y}$是对$y$的预估，经典回归模型可以表达如下：
$$
\hat{y} = Wx + b
$$
有时候bias $b$可以作为$W$的一列进行表示. 也被写作$\theta$, $W$和$b$都是张量。

### 损失函数(cost function)

定义为：
$$
Loss (W, b) = ||\hat{y} - y||_p
$$
p 一般取1、2或者$\infin$. 

学习的目标：
$$
W, b = \mathop{\arg \min}_{W, b} Loss(W, b)
$$

### 深度神经网络

​	包含如下4个主要对象：

* 层，各层链接形成model或者网络
* 输入输出和对应的目标
* 损失函数
* 优化器

基本原理：

1) 正向传播
$$
z = W^{(1)} x \\  
h = \phi(z) \\
o = W^{(2)}h \\
L = Loss(o, y)  \\
s = \frac{\lambda}{2}(||W^{(1)}||_F^2 + ||W^{(2)}||_F^2) \\
J = L + s
$$
目标函数$J(W, b, \lambda)$为带正则项的损失函数,其中  $W^{(1)} \in R^{h \times d}$是隐藏层权重参数, $z \in R^{h}$是隐藏层变量, $o \in R^{q}$是输出层变量，$W^{(2)} \in R^{q \times h}$是输出层权重，假设损失函数为Loss, 样本标签是y. 给定超参为$\lambda$, 计算正则项 $s$，

2)  反向传播

​	相对正向传播计算出代价函数，反向传播目标是通过求梯度，计算(W, b)。首先计算目标函数$J$有关损失项$L$和正则项$s$. 需要借助链式法则。

​	推导过程如下：
$$
\frac{\partial J}{\partial L} = 1, \frac{\partial J}{\partial s} = 1 \\
	\frac{\partial J}{\partial o} = \frac{\partial J}{\partial L} \cdot \frac{\partial L}{\partial o} = \frac{\partial L}{\partial o}\\
	\frac{\partial s}{\partial W^{(1)}} = \lambda W^{(1)},\frac{\partial s}{\partial W^{(2)}} = \lambda W^{(2)} \\
	\frac{\partial J}{\partial W^{(2)}} =\frac{\partial J}{\partial o} \cdot \frac{\partial o}{\partial W^{(2)}}  +  \frac{\partial J}{\partial s} \cdot \frac{\partial s}{\partial W^{(2)}} = \frac{\partial J}{\partial o} h^{T} + \lambda W^{(2)} \\
	
	\frac{\partial J}{\partial h} = \frac{\partial J}{\partial o} \cdot \frac{\partial o}{\partial h} = {W^{(2)}}^T\frac{\partial J}{\partial o}  \\
	\frac{\partial J}{\partial z} = \frac{\partial J}{\partial h} \cdot \frac{\partial h}{\partial z} = \frac{\partial h}{\partial z} \cdot {\phi(z)}^{'} \\
	\frac{\partial J}{\partial W^{(1)}} = \frac{\partial J}{\partial o} \cdot \frac{\partial o}{\partial W^{(1)}}  +  \frac{\partial J}{\partial s} \cdot \frac{\partial s}{\partial W^{(1)}} = \frac{\partial J}{\partial z} x^{T} + \lambda W^{(1)}
$$
​	其中注意$\cdot$表示向量的内积，可能需要对向量调整(专职或者互换输入位置等)。 以上推导过程参考[Book DIDL].

基本过程如下：

<img src=".\image-20200817124701562.png" alt="image-20200817124701562" style="zoom:50%;" />

<center> 图1： 深度学习计算过程，来自[Book DLP]</center>

## 优化方法

​	优化难点：

 * 非凸优化、平坦局部最小值以及鞍点

 * 梯度消失和爆炸

   由bp算法推动的最后2个步骤得知： $\phi(z)^{'}$ 就是对激活函数求导，如果其求导结果大于1，经过多层计算之后，$\frac{\partial J}{\partial W^{(1)}}$ 就会以指数形式增加，出现梯度爆炸，否则以指数形式递减，形成梯度消失。

* 泛化问题(generalization error):   过拟合问题

### 优化算法

#### 学习率调整（含梯度估计修正）

​	对于正常的梯度下降，设定学习率$\eta \gt 0$,  f为连续可导函数，根据泰勒展开，权重矩阵的训练过程如下：
$$
W = W - \eta \Delta f(x)
$$

1. MBGD: 小批量梯度下降法(mini-batch gradient descent)，从大的训练集里面，分批并且取样进行训练，朝着当前所在位置的坡度最大的方向前进, 在时间步t，
   $$
   g_t = \frac{1}{|B|}  \sum_{i \in B_t} {\Delta f(x)} \\
   W = W - \eta_t g_t
   $$

2. Momentum: 参照小球在碗中滚动的物理规则进行移动,  $\alpha \in (0,1)$, 借助指数加权移动平均使得自变量的更新方向更加一致，从而 可以在开始选择较大的学习率，加速收敛。
   $$
   v_t = \alpha v_{t-1} + \eta_t g_t \\
   W = W - v_t
   $$

3. AdaGrad: Ada来自英文单词Adaptive，即“适当的”的意思；AdaGrad会为参数的每个元素适当地调整更新步伐(学习率)，即学习率衰减，随着学习的进行，使学习率逐渐减小，一开始“多”学，然后逐渐“少”学。$\cdot$表示内积。
   $$
   h_t = h_{t-1} + g_t \cdot g_t  \\
   W = W -  \frac{\eta}{\sqrt{h_t}} g_t
   $$

4. RMSProp:  在AdaGrad的基础上,应用指数加权平均.
   $$
   s_t = \alpha s_{t-1} + (1-\alpha) g_t \cdot g_t \\
   W = W -  \frac{\eta}{\sqrt{h_t}} g_t
   $$

#### 批量大小

​	批大小会印象随机梯度的方差, 跟泛化能力也相关. 批量大小的选择不宜过大或过小，需根据实际需求做出选择，较大的批量可以更准确地估计梯度，而较小的批量可以获得更快的收敛速度.

#### 参数初始化

* 预训/随机/固定值 初始化
* 基于固定方差的初始化： 例如使用高斯分布$N(0, \sigma^2)$.
* 基于方差缩放的参数初始化, 例如Xavier初始化
* 正交初始化： W初始化为正交矩阵， 满足$W^{(l)} (W^{(l)})^T = I$, 使得误差项$\delta^{(l-1)} = ((W^{(l)})^{T}) \delta^{(l)}$满足**范数保持性**，即$\left\|\delta^{(l-1)}\right\|^2 = \left\|\delta^{(l)}\right\|^2 = \left\|((W^{(l)})^{T}) \delta^{(l)}\right\|^2$.

#### 数据预处理[Book NNDL]

* 最小最大值归一化
* 标准化
* 合并

#### 超参数优化[JRYB2011]

​	常见的超参数有:

1. 网络结构,包括神经元之间的连接关系、层数、每层的神经元数量、激活函数的类型. 
2. 批量大小、学习率以及梯度评估方法等
3. 正则化系数等。

常使用随机搜索等方法进行最优组合寻找。

### 网络正则化

#### L1/l2正则

$$
\theta^* = \mathop{\arg \min}_{\theta}\frac{1}{N}\sum_{i}{Loss(y^{(n)}, f(x^{(n)}; \theta))} + \lambda l_p (\theta)
$$

$l_p$为范数。 通过添加范数减少过拟合。

#### 权重衰减

​	 引入衰减系数$\beta$, 在时间步骤t:
$$
\theta_t = (1-\beta)\theta_{t-1} - \alpha g_t
$$

#### 丢弃法

​	是指在训练一个深度神经网络时，我们可以随机丢弃一部分神经元来避免过拟合，每次选择丢弃的神经元是随机的。



## 库和框架

### 深度学习库对比(Keras, Pytouch) 

Keras

### 深度学习框架对比(TensorFlow、Theano和CNTK)



参考：

[JRYB2011] [Algorithms for Hyper-Parameter Optimization](http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)

[Book DLP] Deep Learning with Python

[Book DIDL] Diving Into Deep Learning

[Book NNDL] https://nndl.github.io/ , 总结： https://zhuanlan.zhihu.com/p/162943650

