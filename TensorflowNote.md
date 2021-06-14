# Tensorflow笔记
## 一、Tensor的概念
1、0阶张量称为标量scalar，s = 1 2 3  
2、1阶张量称为向量vector，v = [1,2,3]  
3、2阶张量称为矩阵matrix，m = [[1,2,3],[4,5,6],[7,8,9]]  
4、n阶张量的形式为，t = [[[[[[....n  
## 二、Tensorflow中的数据类型和张量的操作
###  1、数据类型  

```
tf.int, tf,float, tf.int32, tf.float32, tf.float64  
tf.bool  
tf.constant([True, False])  
tf.string  
tf.constant("Hello, World!")
```
### 2、张量的创建  

```python
import tensorflow as tf
# 创建张量
a = tf.constant([1, 5], dtype=tf.int64)
print(a) 
print(a.dtype)
print(a.shape)
# 结果
<tf.Tensof([1, 5],shape(2,),dtype=int64)>
<dtype:‘int64’>
(2,)

# 将numpy格式的数据转成tensor
tf.convert_to_tensor(数据名，dtype=数据类型(可选))

import tensorflow as tf
import numpy as np
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print(a)
print(b)
# 结果
[0 1 2 3 4]
tf.Tensor([0 1 2 3 4],shape(5,),dtype=int64)

# 创建特殊张量
tf.zeros(维度) # 全0张量
tf.ones(维度) # 全1张量
tf.fill(维度，指定值) #创建全为指定值的张量

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2,2], 9)
print(a)
print(b)
print(c)
# 结果
tf.Tensor([[0. 0. 0.] [0. 0. 0.]],shape=(2,3),dtype=float)  
tf.Tensor([1. 1. 1. 1.],shape=(4,),dtype=float) 
tf.Tensor([[9 9] [9 9]],shape=(4,),dtype=int32)  


# 创建正态分布的随机数，默认均值为0，标准差为1
tf.random.normal(维度，mean=均值，stddev=标准差)
# 生成截断式正态分布的随机数（均值-2*标准差， 均值+2*标准差）
tf.random.truncated_normal(维度，mean=均值, stddev=标准差)
# 生成均匀分布随机数[minval, maxval)
tf.random.uniform(维度，minval=最小值， maxval=最大值)


```
### 3、常用函数

```python
# (1) 强制tensor转换为该数据类型
	tf.cast(张量名, dtype=数据类型)

# (2) 计算张量维度上元素最小值
	tf.reduce_min(张量名)

# (3) 计算张量维度上元素最大值
	tf.reduce_max(张量名)

# (4) axis表示张量中的轴，axis=0表示一列数据，axis=1表示一行数据
# 计算张量沿某个指定的维度的平均
	tf.reduce_mean(张量名, axis=操作轴)
# 计算张量沿着某个指定的维度的和
	tf.reduce_sum(张量名, axis=操作轴)

# (5) tf.Variable该函数可以把变量标记为“可训练”，被标记的函数会在反向传播中记录梯度信息。神经网络中常用其标记待标记的训练参数。
	tf.Variable(初始值)

# （6）加减乘除函数：tf.add,tf.subtract,tf.multiply,tf.divide 
# 平方、次方、开方：tf.square, tf.pow, tf.sqrt
# 矩阵乘法：tf.matmul

# （7）配对数据与标签的函数（对numpy和tensor都是有效的）
    data = tf.data.Dataset.from_tensor_slicec((输入特征，标签))

    features = tf.constant([12, 23, 10, 17])
    labels = tf.constant([0, 1, 1 ,0])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    print(dataset)
    for element in dataset:
        print(element)

# 结果  
    <TensorSliceDataset shapes: ((), ()), types: (tf.int32, tf.int32)>
    (<tf.Tensor: shape=(), dtype=int32, numpy=12>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=23>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=10>, <tf.Tensor: shape=(), dtype=int32, numpy=1>)
    (<tf.Tensor: shape=(), dtype=int32, numpy=17>, <tf.Tensor: shape=(), dtype=int32, numpy=0>)

# （8）求导函数 tf.GradientTape()
# 使用with结构记录计算过程，gradient求出张量梯度
    with tf.GradientTape() as tape:
        grad = tape.gradient(函数，对谁求导)

    with tf.GradientTape() as tape:
        w = tf.Variable(tf.constant(3.0))
        loss = tf.pow(w, 2)
    grad = tape(loss, w)
    print(grad)  

# （9）遍历列表元组函数  
    seq = ["one", "two", 'three']
    for i, element in enumerate(seq):
        print(i, element)
# 结果
    0 one
    1 two
    2 three

# （10）独热标签（ont-hot encoding）
# 一般在分类问题中经常使用独热标签，1表示是，0表示否
# 创建独热标签 tf.one_hot(待转换数据， depth=分类数)
    classes = 3
    labels = tf.constant([1,0,2])
    output = tf.one_hot(labels, depth=classes)
    print(output)
# 结果
    tf.Tensor(
    [[0. 1. 0.]
     [1. 0. 0.]
     [0. 0. 1.]], shape=(3, 3), dtype=float32)


# （11）Softmax 映射函数
# Softmax 函数可以将一组数据映射到0-1之间并且符合概率分布
	tf.nn.softmax(x)

# （12）赋值更新操作(减法) assign_sub 
# 使用assign_sub之前，需要用tf.Variable定义变量w为可训练(可自更新)。
w = tf.Variable(4)
w.assign_sub(1) 
print(w)
# 结果
	<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=3>

# （13）返回张量沿指定轴的最大值索引下标
	tf.argmax(张量名, axis=操作轴)
```

### 4、数据读入方法（鸢尾花数据集Iris的读入）

```python
# klearn包datasets读入数据集
from sklearn datasets 
# 返回iris数据集的所有输入特征
x_data = datasets.load_iris().data
# 返回iris数据集的所有标签
y_data = datasets.load_iris().target;
```

### 5、基于鸢尾花数据集的分类案例

具体流程如下：

1、准备数据：

​	读入数据集

​	打乱数据集

​	生成训练集和测试集

​	配成（输入特征，标签）对，每次读入一小撮（batch）

2、搭建网络

​	定义网络结构以及其中的参数

3、参数优化

​	嵌套循环迭代，with结构更新参数，显示loss

4、测试效果

​	计算当前参数前向传播后的准确率，显示当前acc

acc / loss 可视化

```
# -*- coding: UTF-8 -*-
# 利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线

# 导入所需模块
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# 导入数据，分别为输入特征和标签
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

# 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# from_tensor_slices函数使输入特征和标签值一一对应。（把数据集分批次，每个批次batch组数据）
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 生成神经网络的参数，4个输入特征故，输入层为4个输入节点；因为3分类，故输出层为3个神经元
# 用tf.Variable()标记参数可训练
# 使用seed使每次生成的随机数相同（方便教学，使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1  # 学习率为0.1
train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
epoch = 500  # 循环500轮
loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

# 训练部分
for epoch in range(epoch):  #数据集级别的循环，每个epoch循环一次数据集
    for step, (x_train, y_train) in enumerate(train_db):  #batch级别的循环 ，每个step循环一个batch
        with tf.GradientTape() as tape:  # with结构记录梯度信息

            y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
            y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
            y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
            loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
            loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
        # 计算loss对各个参数的梯度
        grads = tape.gradient(loss, [w1, b1])

        # 实现梯度更新 w1 = w1 - lr * w1_grad    b = b - lr * b_grad
        w1.assign_sub(lr * grads[0])  # 参数w1自更新
        b1.assign_sub(lr * grads[1])  # 参数b自更新

    # 每个epoch，打印loss信息
    print("Epoch {}, loss: {}".format(epoch, loss_all/4))
    train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
    loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

    # 测试部分
    # total_correct为预测对的样本个数, total_number为测试的总样本数，将这两个变量都初始化为0
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        # 使用更新后的参数进行预测
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
        # 将pred转换为y_test的数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct数加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
        total_number += x_test.shape[0]
    # 总的准确率等于total_correct/total_number
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("--------------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Loss')  # y轴变量名称
plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()  # 画出曲线图标
plt.show()  # 画出图像

# 绘制 Accuracy 曲线
plt.title('Acc Curve')  # 图片标题
plt.xlabel('Epoch')  # x轴变量名称
plt.ylabel('Acc')  # y轴变量名称
plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend()
plt.show()

```

## 三、神经网络的优化

### 1、预备知识（常用API）

```python
# 条件选择语句
tf.where(条件语句, 真返回A, 假返回B)

a = tf.constant([1,2,3,1,1])
b = tf.constant([0,1,3,4,5])
c = tf.where(tf.greater(a,b), a, b)
print("c:",c)
# 结果
c: tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
    
# 返回[0,1)之间的随机数
np.random.RandomState.rand(维度)

# 将2个数组按垂直方向叠加
np.vstack(数组1，数组2)
a = np.array([1,2])
b = np.array([3,4])
c = np.vstack((a,b))
print(c)
# 结果
[[1 2]
 [3 4]]

# 生成网格坐标点
np.mgrid[起始值:结束值:步长，起始值:结束值:步长,...]
# 将x变为一维数组
x.ravel()
# 将返回的间隔数组点配对
np.c_[数组1,数组2]

x,y = np.mgrid(1:3:1,2:4:0.5)
grid = np.c_[x.ravel(), y.ravel()]
print("x:",x)
print("y:",y)
print("grid:\n",grid)
# 结果
x: [[1. 1. 1. 1.]
 [2. 2. 2. 2.]]
y: [[2.  2.5 3.  3.5]
 [2.  2.5 3.  3.5]]
grid:
 [[1.  2. ]
 [1.  2.5]
 [1.  3. ]
 [1.  3.5]
 [2.  2. ]
 [2.  2.5]
 [2.  3. ]
 [2.  3.5]]
```

### 2、指数衰减学习率

在神经网络的训练中，学习率的选择决定着训练的快慢，较大的学习率在梯度下降的时候可能每次移动的较长，但是很容易错过局部最优点，但是如果使用较小的学习率的话可能会导致训练的比较慢，所以我们可以在训练初期使用较大的学习率，根据训练的轮数逐步减小学习率以达到局部最优点。
$$
learning\_rate = init\_learning\_rate * decay\_rate^\frac{当前轮数}{多少论衰减一次}
$$

```
lr_init = 0.2
decay_rate = 0.99
epoch_num = 100
step = 1
lr_rate = []

for epoch in range(epoch_num):
    lr_rate.append(lr_init * decay_rate ** (epoch / step))

plt.plot(range(epoch_num), lr_rate)
plt.legend()
plt.show()
# 上述代码所产生的曲线呈指数形式下降
```



### 3、激活函数

激活函数可以使得神经网络更具有表达能力，所以选择一个好的激活函数尤为重要。

优秀的激活函数有如下的特点：

非线性：激活函数非线性时，多层神经网络可以逼近所有函数

可微性：优化器大多使用梯度下降更新参数

单调性：当激活函数是单调的，能保证单层网络的损失函数是凸函数

近似恒等性：f(x)约等于x当参数初始化时为随机最小值时，神经网络更稳定

激活函数输出值的范围：

激活函数输出为有限值时，基于梯度的优化方法更稳定

激活函数输出为无限值时，建议调小学习率

#### 3.1、常见激活函数

**Sigmoid函数** **（tf.nn.sigmoid(x)）**
$$
f(x) = \frac{1}{1+e^{-x}}
$$
![sigmoid](.\Img\Sigmoid.JPG)

**特点：**

​	（1）容易造成梯度消失，因为深层神经网络需要逐层求导每层都用sigmoid激活，由于sigmoid导数的值在（0，0.25）之间，多乘几次就变得很小了。

​	（2）输出非0均值，收敛慢

​	（3）幂运算复杂，训练时间长



**Tanh函数（tf.math.tanh(x)）**
$$
f(x)=\frac{1-e^{-2x}}{1+e^{-2x}}
$$
![](./Img/Tanh.jpg)

**特点：**

​	（1）输出是0均值

​    （2）易造成梯度消失

​    （3）幂运算复杂，训练时间长

**ReLu函数（tf.nn.relu(x)）**
$$
f(x)=max(x,0)
$$
![](./Img/ReLu.jpg)

优点：

​	（1）解决了梯度消失问题（在正区间）

​	（2）只需要判断输入是否大于0，计算速度快

​	（3）收敛速度远远快于sigmoid和tanh

缺点：

   （1）输出非0均值，收敛慢

  （2）Dead ReLu问题：某些神经元可能永远也不会被激活，导致相应的参数永远不能被更新

**Leaky Relu函数（tf.nn.leaky_relu(x)）**
$$
f(x)=max(ax,x)
$$
![](./Img/LeakyReLu.jpg)

**特点：**

该函数主要就是用于解决上述ReLu函数会导致DeadReLu的情况，这里引入了一个固定斜率来避免，但是很多实际操作中Leaky Relu也不总是好于Relu的

#### 3.2、激活函数的选用

对于初学者而言有如下几条建议：

​	（1）首选ReLu函数

​	（2）设置较小的学习率

​	（3）输入特征标准化，即让输入特征满足以0为均值，1为标准差的正态分布

​	（4）初始参数中心化，即让随机生成的参数满足以0为均值，（2 / 当前层的输入特征个数）^ (1/2) 为标准差的正态分布
