# MXNet源码分析

[toc]



## 基本结构

![System Overview](https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/mxnet/system/overview.png)

* Runtime Dependency Engine: Schedules and executes the operations according to their read/write dependency.

- Storage Allocator: Efficiently allocates and recycles memory blocks on host (CPU) and devices (GPUs).
- Resource Manager: Manages global resources, such as the random number generator and temporal space.
- NDArray: Dynamic, asynchronous n-dimensional arrays, which provide flexible imperative programs for MXNet.
- Symbolic Execution: Static symbolic graph executor, which provides efficient symbolic graph execution and optimization.
- Operator: Operators that define static forward and gradient calculation (backprop).
- SimpleOp: Operators that extend NDArray operators and symbolic operators in a unified fashion.
- Symbol Construction: Symbolic construction, which provides a way to construct a computation graph (net configuration).
- KVStore: Key-value store interface for efficient parameter synchronization.
- Data Loading(IO): Efficient distributed data loading and augmentation.

### Context

​	Context维护每个任务的上下文，构造的时候需要传递设备信息。包含如下四种主要类型的Context:

* cpu:  运行于普通CPU对应的RAM分配
* pinned_cpu[1] ： GPU模式下，主机分配CPU Pinned Memory[2]；
* gpu： GPU模式下，主机分配可分页；
* cpushared:  Android Shared Memory分配；

### Storage Allocator

​	Allocator是内存分配器，其提供的方法如下：

![Collaboration graph](https://mxnet.apache.org/versions/1.0.0/doxygen/classmxnet_1_1Storage__coll__graph.png)

​	`storage.cc/StorageImpl` 在实现Storage接口的时候，需要首先加载对应的内存管理器(`storage_manager.h/StorageManager`)，然后由管理器实现的实际的`Alloc/Free/DirectFree`操作。

​	内存管理器根据不同设备类型，实现不同实现了7种不同的内存管理器。主要实现为：

* NaiveStorageManager
  * CPUDeviceStorage :  [地址对齐](https://man7.org/linux/man-pages/man3/posix_memalign.3.html)的内存分配. 对应Context::kCPU. 
  * GPUDeviceStorage:    调用`cudaMalloc/cudaFree`进行内存分配和释放； 对应Context::kGPU. 
  * PinnedMemoryStorage:  调用`cudaAllocHost/cudaFreeHost`进行主机锁页分配和释放； 对应Context::kCPUPinned. 
* CPUSharedStorageManager:  对应Context::kCPUShared. 
* PooledStorageManager： 使用内存池进行内存分配和管理。根据Round方式不同, 采用不同的Container索引不同状态的内存。

### Resource Manager

​	ResourceManager主要针对随机数生成器资源以及临时空间资源管理进行分配。其提供的方法如下：

![Collaboration graph](https://mxnet.apache.org/versions/0.12.1/doxygen/classmxnet_1_1ResourceManager__coll__graph.png)

默认实现是`resource::ResourceManagerImpl`。

​	Request()根据用户需要的资源类型返回对应的资源，主要资源包括：

> * kRandom :  对应mshadow::Random<xpu> object。 就是在不同设备上提供的随机数生成器的封装。
>   * CPU:  C++ [mt19937](http://www.cplusplus.com/reference/random/mt19937/)
>   * GPU: [cuRADN](https://docs.nvidia.com/cuda/curand/host-api-overview.html)
> * kTempSpace： 任意长度的动态内存分配。使用前面提到的Storage Allocator分配；
> * kParallelRandom：异步随机数分配，参考依赖引擎里面的解释。
> * kCuDNNDropoutDesc：参考 [3.1.1.3. cudnnDropoutDescriptor_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDropoutDescriptor_t)。提供dropout操作。

### Runtime Dependency Engine

​	[运行时依赖引擎](https://mxnet.apache.org/versions/1.6/api/architecture/note_engine)主要负责计算图的全生命周期管理，目标是最大程度的实现并行计算。计算图也就是训练任务/通用计算任务的数据流图。

​	

### NDArray

​	

### NNVM

### Runtime Dependency Engine



### 运算库[mshadow](https://github.com/dmlc/mshadow/tree/master/guide)

* 延迟计算： 在“=”操作符上执行真正的计算。默认将计算定向到MKL或者BLAS
* 复合模板和递归计算： 通过模板(Unary/Binary)表达式支持多种类型(scalar/vector/matrix等tensor)的运算。TBlob是一种shape可动态改变的数据结构。
* 支持在异构硬件(xpu)上计算/随机数生成等

## 其他基础库

### dmlc-core

https://dmlc-core.readthedocs.io/en/latest/parameter.html

### 



## 参考

1. https://lwn.net/Articles/600502/
2. https://blog.csdn.net/chenxiuli0810/article/details/90899014

