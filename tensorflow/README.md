
<!--<img src="https://www.tensorflow.org/images/tf_logo_transp.png" width = "100%" height = "400" div align=center />-->


> #### 关于TensorFlow
TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。
TensorFlow可被用于语音识别或图像识别等多项机器学习和深度学习领域，对2011年开发的深度学习基础架构DistBelief进行了各方面的改进，它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow将完全开源，任何人都可以用。

  

> #### 核心概念

- ##### 数据流图
数据流图用“结点”（nodes）和“线”(edges)的有向图来描述数学计算。
“节点” 一般用来表示施加的数学操作，但也可以表示数据输入（feed in）的起点/输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。
“线”表示“节点”之间的输入/输出关系。这些数据“线”可以输运“size可动态调整”的多维数据数组，即“张量”（tensor）。
张量从图中流过的直观图像是这个工具取名为“Tensorflow”的原因。一旦输入端的所有张量准备好，节点将被分配到各种计算设备完成异步并行地执行运算。

![数据流图](http://tensorfly.cn/images/tensors_flowing.gif)

TensorFlow主要是由计算图、张量以及模型会话三个部分组成。

- ##### 计算图

  在编写程序时，我们都是一步一步计算的，每计算完一步就可以得到一个执行结果。
在TensorFlow中，首先需要构建一个计算图，然后按照计算图启动一个会话，在会话中完成变量赋值，计算，得到最终结果等操作。
因此，可以说TensorFlow是一个按照计算图设计的逻辑进行计算的编程系统。
TensorFlow的计算图可以分为两个部分：

  - 构造部分，包含计算流图；
  - 执行部分，通过session执行图中的计算。

  构造部分又分为两部分：
  - 创建源节点；
  - 源节点输出传递给其他节点做运算。

&emsp;&emsp;TensorFlow默认图：TensorFlow python库中有一个默认图(default graph)。节点构造器(op构造器)可以增加节点。

- ##### 张量

  在TensorFlow中，张量是对运算结果的引用，运算结果多以数组的形式存储，与numpy中数组不同的是张量还包含三个重要属性名字、维度、类型。
  张量的名字，是张量的唯一标识符，通过名字可以发现张量是如何计算出来的。比如“add:0”代表的是计算节点"add"的第一个输出结果。维度和类型与数组类似。
  
- ##### OP
  
    OP 是一个计算图的节点，用来运行在 tensor 上执行的计算操作，它接受零或多个 tensor 作为输入，产生零或多个 Tensor 作为输出，简单的使用任意一个关联到 OP 创建器的操作都会成功的创建 OP 的对象实例

 
 > #### 系统架构概述
TensorFlow的系统结构以C API为界，将整个系统分为「前端」和「后端」两个子系统：

- 前端系统：提供编程模型，负责构造计算图
- 后端系统：提供运行时环境，负责执行计算图。


![TensorFlow架构图](https://upload-images.jianshu.io/upload_images/2254249-bf86142555d23538.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/629)
 
 
如上图所示，重点关注系统中如下4个基本组件，它们是系统分布式运行机制的核心

1. **Client**：
Client是前端系统的主要组成部分，它是一个支持多语言的编程环境。它提供基于计算图的编程模型，方便用户构造各种复杂的计算图，实现各种形式的模型设计。
Client通过Session为桥梁，连接TensorFlow后端的「运行时」，并启动计算图的执行过程。

2. **Distributed Master**：
在分布式的运行时环境中，Distributed Master根据Session.run的Fetching参数，从计算图中反向遍历，找到所依赖的「最小子图」。
然后，Distributed Master负责将该「子图」再次分裂为多个「子图片段」，以便在不同的进程和设备上运行这些「子图片段」。
最后，Distributed Master将这些「子图片段」派发给Work Service；随后Work Service启动「子图片段」的执行过程。

3. **Worker Service**：
对于每个任务，TensorFlow都将启动一个Worker Service。Worker Service将按照计算图中节点之间的依赖关系，根据当前的可用的硬件环境(GPU/CPU)，调用OP的Kernel实现完成OP的运算(一种典型的多态实现技术)。
另外，Worker Service还要负责将OP运算的结果发送到其他的Work Service；或者接受来自其他Worker Service发送给它的OP运算的结果。

4. **Kernel Implements**：
Kernel是OP在某种硬件设备的特定实现，它负责执行OP的运算。


> #### 扩展功能

在tensorflow中比较重要的拓展功能有，自动求导，子图执行，计算图控制流以及队列/容器
求导是机器学习中计算损失函数常用的运算，TensorFlow原生支持自动求导运算，它是通过计算图中的拓展节点实现。
子图执行是通过控制张量的流向实现。
计算图控制流：是指控制计算图的节点极其运行的设备管理，它提供了快速执行计算和满足设备施加的各种约束。比如限制内存总量为了执行它的图子集而在设备上所需的节点。
队列是一个有用的功能，它们允许图的不同部分异步执行，对数据进行入队和出队操作。
容器是用来存放变量，默认的容器是持久的，直到进程终止才会清空，同时容器中的变量也可以共享给其他计算图使用。


> #### TensorFlow 的特点
与其他深度学习框架相比， TensorFlow
- 原生支持支持大规模的分布式模训练，为开发生产模型提供了强大后盾。
- 部署简单高效。有专门部署的tensorflow的框架TensorFlow Serving，并支持移动以及嵌入式部署。
- 自带的Tensooboard提供了强大的可视化功能，为模型训练带来极大的便利。
- 良好的社区支持和齐全的文档，以及众多的形式多样的学习资源。


  
------------------------
 
#### MNIST手写数字识别

通过上面的描述，可见TensorFlow是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。以下是我们构建一个TensorFlow模型的基本步骤，并将通过这些步骤为MNIST构建一个深度卷积神经网络。

- ##### **项目结构**

```
├── mnist_demo
│   ├── cnn_model   # 卷积神经网络模型 以下也是集于此介绍tensorflow
│   │   ├── input_data.py
│   │   └── mnist_deep.py   # 主入口
│   ├── __init__.py
│   ├── linear_model   # 线性模型
│   │   ├── fully_connected_feed.py   # 主入口
│   │   ├── input_data.py
│   │   ├── load_model.py
│   │   └── mnist.py
│   └── MNIST_data   # MNIST手写数字数据
└── README.md
```
- ##### **加载数据**
input_data.py 脚本来自动下载和导入MNIST数据集。它会自动创建一个'MNIST_data'的目录来存储数据。

```
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
这里，mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch，后面我们将会用到。
(由于国内网络的关系， 这里直接将下载好的MNIST_data放在项目里)


- ##### **构建模型**

针对此问题决定建立一个拥有多层卷积网络的softmax回归模型。

- ##### **卷积和池化**

TensorFlow在卷积和池化上有很强的灵活性。在这里，我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。

```
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

&emsp;&emsp;**第一层卷积**

现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。

```
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```
为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数

```
x_image = tf.reshape(x, [-1,28,28,1])
```
我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。

```
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```
&emsp;&emsp;**第二层卷积**

为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。

```
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```



- ##### **密集连接层**

现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。

```
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```


- ##### **Dropout**
为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。

```
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

- ##### **输出层**

最后，我们添加一个softmax层，计算每个分类的softmax概率值，用于实现我们的回归模型。
```
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```


- ##### **训练和评估模型**


可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。然后用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。然后每100次迭代输出一次日志。

```
# 选用交叉熵作为损失函数，我们计算的交叉熵是指整个minibatch的
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# 选用ADAM优化器来做梯度最速下降
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
```

以上代码，在最终测试集上的准确率大概是99.2%。