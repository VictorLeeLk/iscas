# Python基本函数

## Python类中方法介绍

python类中方法：

```
class Test(object):
    def __init__(self, title):  #可定义多个参数
        self.title = title
    def get_title(self):   #定义了实例方法
        return self.title
    @classmethod
    def get_time(cls):  #定义了类方法
        print("On July 2")
    @staticmethod
    def get_grade():      #定义了静态方法
        print("89")
```

对三种方法的归纳总结：

| 方法     | 调用情况                 | 访问权限                                       |
| -------- | ------------------------ | ---------------------------------------------- |
| 普通方法 | 可以通过实例来调用       | 可访问实例属性，无法访问类属性                 |
| 类方法   | 可以通过类名和实例来调用 | 可访问类属性，无法访问实例属性                 |
| 静态方法 | 可以通过类名和实例来调用 | 无法访问类属性及实例属性（仅可通过传值的方式） |



## python3 zip()函数

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。

> zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
>
> 如果需要了解 Pyhton3 的应用，可以参考 [Python3 zip()](http://www.runoob.com/python3/python3-func-zip.html)。

```
>>>a = [1,2,3]
>>> b = [4,5,6]
>>> c = [4,5,6,7,8]
>>> zipped = zip(a,b)     # 返回一个对象
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() 转换为列表
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]
 
>>> a1, a2 = zip(*zip(a,b))          # 与 zip 相反，*zip 可理解为解压，返回二维矩阵式
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
>>>
```

python zip(*a) 

> 解压

a为已经经过压缩的序列，比如[(1,2,3)],(4,5,6)],这个序列是由3个序列压缩得到的，分别为（1,4）,（2,5）,（3,6）。

经过zip([(1,2,3),(4,5,6)])得到3个元组

```
for i in zip(*a):
	print(i)
输出：
（1,4）
（2,5）
（3,6）
```

- 6) np.zeros()

```
    Examples
    --------
np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])    
np.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])
np.zeros((2, 1))
    array([[ 0.],
           [ 0.]]) 
s = (2,2)
np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])
```

- 7)np.ones()

```
    Examples
    --------
np.ones(5)
    array([ 1.,  1.,  1.,  1.,  1.])
    
np.ones((5,), dtype=int)
    array([1, 1, 1, 1, 1])
    
np.ones((2, 1))
    array([[ 1.],
           [ 1.]])
    
s = (2,2)
np.ones(s)
    array([[ 1.,  1.],
           [ 1.,  1.]])
```

## np.stack()

```
stack(arrays, axis=0, out=None)
    Join a sequence of arrays along a new axis.
    
    The `axis` parameter specifies the index of the new axis in the dimensions
    of the result. For example, if ``axis=0`` it will be the first dimension
    and if ``axis=-1`` it will be the last dimension.
    
    .. versionadded:: 1.10.0
    
    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.
    
    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.
    
    See Also
    --------
    concatenate : Join a sequence of arrays along an existing axis.
    split : Split array into a list of multiple sub-arrays of equal size.
    block : Assemble arrays from blocks.
    
    Examples
    --------
arrays = [np.random.randn(3, 4) for _ in range(10)]
np.stack(arrays, axis=0).shape
    (10, 3, 4)
    
np.stack(arrays, axis=1).shape
    (3, 10, 4)
    
np.stack(arrays, axis=2).shape
    (3, 4, 10)
    
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.stack((a, b))
    array([[1, 2, 3],
           [2, 3, 4]])
    
np.stack((a, b), axis=-1)
    array([[1, 2],
           [2, 3],
           [3, 4]])
```

- **9) os.path.join()**

## os.makedirs()

os.mkdirs()-os.makedirs()-os.rmdir()-os.removedirs()

> 说明：os.makedirs()创建多级目录，包含子目录。其中os.mkdir()创建单级目录。
>
> >  os.rmdir()删除单级目录，os.removedirs()删除多级目录

```
  import os
  if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)
  if not os.path.exists(monitor_path):
          os.makedirs(monitor_path)
```

## numpy.append() 里的axis的用法

```
def append(arr, values, axis=None):
    """
    Append values to the end of an array.
    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.
    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

```

numpy.append(arr, values, axis=None):

简答来说，就是arr和values会重新组合成一个新的数组，做为返回值。而axis是一个可选的值

1. 当axis无定义时，是横向加成，返回总是为一维数组！

```
    Examples
    --------
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

```

  2.当axis有定义的时候，分别为0和1的时候。（注意加载的时候，数组要设置好，行数或者列数要相同。不然会有error：all the input array dimensions except for the concatenation axis must match exactly）

```
当axis为0时，数组是加在下面（列数要相同）：

import numpy as np
aa= np.zeros((1,8))
bb=np.ones((3,8))
c = np.append(aa,bb,axis = 0)
print(c)
[[ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.]
 [ 1.  1.  1.  1.  1.  1.  1.  1.]]

当axis为1时，数组是加在右边（行数要相同）：

import numpy as np
aa= np.zeros((3,8))
bb=np.ones((3,1))
c = np.append(aa,bb,axis = 1)
print(c)
[[ 0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  1.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.  1.]]
```



## python random.seed()函数

描述：**seed()** 方法改变随机数生成器的种子，可以在调用其他随机模块函数之前调用此函数。。

以下是 seed() 方法的语法:

```
import random
random.seed ( [x] )
```

**注意：**seed(()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。

- x -- 改变随机数生成器的种子seed。如果你不了解其原理，你不必特别去设定seed，Python会帮你选择seed。

返回值：本函数没有返回值。

以下展示了使用 seed(() 方法的实例：

```
#!/usr/bin/python
import random

random.seed( 10 )
print "Random number with seed 10 : ", random.random()
# 生成同一个随机数
random.seed( 10 )
print "Random number with seed 10 : ", random.random()
# 生成同一个随机数
random.seed( 10 )
print "Random number with seed 10 : ", random.random()
```

以上实例运行后输出结果为：

```
Random number with seed 10 :  0.57140259469
Random number with seed 10 :  0.57140259469
Random number with seed 10 :  0.57140259469
```

## lambda匿名函数

python 使用 lambda 来创建匿名函数。

- lambda只是一个表达式，函数体比def简单很多。
- lambda的主体是一个表达式，而不是一个代码块。仅仅能在lambda表达式中封装有限的逻辑进去。
- lambda函数拥有自己的命名空间，且不能访问自有参数列表之外或全局命名空间里的参数。
- 虽然lambda函数看起来只能写一行，却不等同于C或C++的内联函数，后者的目的是调用小函数时不占用栈内存从而增加运行效率。

语法

lambda函数的语法只包含一个语句，如下：

```
lambda [arg1 [,arg2,.....argn]]:expression
```

如下实例：

实例(Python 2.0+)

```
#!/usr/bin/python # -*- coding: UTF-8 -*-   
# 可写函数说明 
sum = lambda arg1, arg2: arg1 + arg2;   
# 调用sum函数 print "相加后的值为 : ", sum( 10, 20 ) 
print "相加后的值为 : ", sum( 20, 20 )
```

以上实例输出结果：

```
相加后的值为 :  30
相加后的值为 :  40
```

## python  collections()

collections是Python内建的一个集合模块，提供了许多有用的集合类。

- namedtuple

我们知道`tuple`可以表示不变集合，例如，一个点的二维坐标就可以表示成：

```
>>> p = (1, 2)
```

但是，看到`(1, 2)`，很难看出这个`tuple`是用来表示一个坐标的。

定义一个class又小题大做了，这时，`namedtuple`就派上了用场：

```
>>> from collections import namedtuple
>>> Point = namedtuple('Point', ['x', 'y'])
>>> p = Point(1, 2)
>>> p.x
1
>>> p.y
2
```

`namedtuple`是一个函数，它用来创建一个自定义的`tuple`对象，并且规定了`tuple`元素的个数，并可以用属性而不是索引来引用`tuple`的某个元素。

这样一来，我们用`namedtuple`可以很方便地定义一种数据类型，它具备tuple的不变性，又可以根据属性来引用，使用十分方便。

可以验证创建的`Point`对象是`tuple`的一种子类：

```
>>> isinstance(p, Point)
True
>>> isinstance(p, tuple)
True
```

类似的，如果要用坐标和半径表示一个圆，也可以用`namedtuple`定义：

```
# namedtuple('名称', [属性list]):
Circle = namedtuple('Circle', ['x', 'y', 'r'])
```

-  collection.namedtuple

 namedtuple是继承自tuple的子类。namedtuple和tuple比，有更多更酷的特性。namedtuple创建一个和tuple类似的对象，而且对象拥有可以访问的属性。这对象更像带有数据属性的类，不过数据属性是只读的。

```
>>> from collections import namedtuple
>>> TPoint = namedtuple('TPoint', ['x', 'y'])
>>> p = TPoint(x=10, y=10)
>>> p
TPoint(x=10, y=10)
>>> p.x
10
>>> p.y
10
>>> p[0]
10
>>> type(p)
<class '__main__.TPoint'>
>>> for i in p:
	print(i)	
10
10
>>> 
import collections
MyTupleClass = collections.namedtuple('MyTupleClass',['name', 'age', 'job'])
obj = MyTupleClass("Tomsom",12,'Cooker')
print(obj.name)
print(obj.age)
print(obj.job)

输出：
Tomsom
12
Cooker
```



- deque

使用`list`存储数据时，按索引访问元素很快，但是插入和删除元素就很慢了，因为`list`是线性存储，数据量大的时候，插入和删除效率很低。

deque是为了高效实现插入和删除操作的双向列表，适合用于队列和栈：

```
>>> from collections import deque
>>> q = deque(['a', 'b', 'c'])
>>> q.append('x')
>>> q.appendleft('y')
>>> q
deque(['y', 'a', 'b', 'c', 'x'])
```

`deque`除了实现list的`append()`和`pop()`外，还支持`appendleft()`和`popleft()`，这样就可以非常高效地往头部添加或删除元素。



# TensorFlow源码阅读-函数说明

>  说明：1)主要记录平时遇到的tf函数，并且对函数的功能进行简单说明，举出相应的示例理解。
>
>              2)numpy函数以及相关python3相关函数说明

## tf.ConfigProto()

`tf.ConfigProto`一般用在创建`session`的时候。用来对`session`进行参数配置

```python
with tf.Session(config = tf.ConfigProto(...),...)
#tf.ConfigProto()的参数
log_device_placement=True : 是否打印设备分配日志
allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)1234
```

- 控制GPU资源使用率

```python
#allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片123456
# per_process_gpu_memory_fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config, ...)
#设置每个GPU应该拿出多少容量给进程使用，0.4代表 40%12345
```

- 控制使用哪块GPU

```python
~/ CUDA_VISIBLE_DEVICES=0  python your.py#使用GPU0
~/ CUDA_VISIBLE_DEVICES=0,1 python your.py#使用GPU0,1
#注意单词不要打错

#或者在 程序开头
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
```

##  tf.placeholder()

```
tf.placeholder 函数
placeholder(
    dtype,
    shape=None,
    name=None
)
```

定义在：[tensorflow/python/ops/array_ops.py](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-9gip2cze.html)

请参阅指南：[输入和读取器>占位符](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-s7pv28uf.html)

插入一个张量的占位符，这个张量将一直被提供。 

注意：如果计算，该张量将产生一个错误，其值必须使用 feed_dict 可选参数来进行 session . run()、Tensor.eval() 或 oper.run()。

例如：

```
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
  print(sess.run(y))  # ERROR: will fail because x was not fed.

  rand_array = np.random.rand(1024, 1024)
  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

参数：

- dtype：要输入的张量中元素的类型。
- shape：要输入的张量的形状（可选）。如果未指定形状，则可以输入任何形状的张量。
- name：操作的名称（可选）。

返回：

一个可能被用作提供一个值的句柄的张量，但不直接计算。

该函数意思为占位符，类似于函数的形参，在运行时需要传递参数。

```
# --coding: utf-8 --

"""
Createdon Mon Aug  6 17:43:32 2018
 @author:27485

""" 
测试tensorflow里面placeholder函数的作用
占位符，运行时候需要传入参数，有点类似于函数的参数。
"""
import tensorflow as tf
import numpy as np

# 定义placeholder
input1= tf.placeholder(tf.float32)
input2= tf.placeholder(tf.float32) 
# 定义乘法运算
output= tf.multiply(input1, input2)
# 通过session执行乘法运行
with tf.Session() as sess:
# 执行时要传入placeholder的值
     print(sess.run(output, feed_dict = {input1:[7.],input2: [2.]}))
```

## tf.contrib层

由 Carrie 创建， 最后一次修改 2017-08-22

> 包含用于构建神经网络层，正则化，摘要等的操作。

**建立神经网络层的更高层次的操作**

此包提供了一些操作, 它们负责在内部创建以一致方式使用的变量, 并为许多常见的机器学习算法提供构建块。

- [tf.contrib.layers.avg_pool2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/avg_pool2d)
- [tf.contrib.layers.batch_norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm)
- [tf.contrib.layers.convolution2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d)
- [tf.contrib.layers.conv2d_in_plane](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_in_plane)
- [tf.contrib.layers.convolution2d_in_plane](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_in_plane)
- [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose)
- [tf.contrib.layers.convolution2d_transpose](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/conv2d_transpose)
- [tf.nn.dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)
- [tf.contrib.layers.flatten](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten)
- [tf.contrib.layers.fully_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)
- [tf.contrib.layers.layer_norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm)
- [tf.contrib.layers.linear](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/linear)
- [tf.contrib.layers.max_pool2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/max_pool2d)
- [tf.contrib.layers.one_hot_encoding](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/one_hot_encoding)
- [tf.nn.relu](https://www.tensorflow.org/api_docs/python/tf/nn/relu)
- [tf.nn.relu6](https://www.tensorflow.org/api_docs/python/tf/nn/relu6)
- [tf.contrib.layers.repeat](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/repeat)
- [tf.contrib.layers.safe_embedding_lookup_sparse](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/safe_embedding_lookup_sparse)
- [tf.nn.separable_conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
- [tf.contrib.layers.separable_convolution2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/separable_conv2d)
- [tf.nn.softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)
- [tf.stack](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-36hu2mm9.html)
- [tf.contrib.layers.unit_norm](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/unit_norm)
- [tf.contrib.layers.embed_sequence](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)

设置默认激活功能的 fully_connected 的别名可用：relu，relu6 和 linear。

stack 操作也可用，它通过重复应用层来构建一叠层。

**正则化**

正则化可以帮助防止过度配合。这些都有签名 fn(权重)。损失通常被添加到 tf.GraphKeys.REGULARIZATION_LOSSES。

- [tf.contrib.layers.apply_regularization](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/apply_regularization)
- [tf.contrib.layers.l1_regularizer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/l1_regularizer)
- [tf.contrib.layers.l2_regularizer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/l2_regularizer)
- [tf.contrib.layers.sum_regularizer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sum_regularizer)

**初始化**

用于初始化具有明确值的变量，给出其大小，数据类型和目的。

- [tf.contrib.layers.xavier_initializer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
- [tf.contrib.layers.xavier_initializer_conv2d](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer)
- [tf.contrib.layers.variance_scaling_initializer](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer)

**优化**

由于损失而优化权重。

- [tf.contrib.layers.optimize_loss](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss)

**摘要**

帮助函数来汇总特定变量或操作。

- [tf.contrib.layers.summarize_activation](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/summarize_activation)
- [tf.contrib.layers.summarize_tensor](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/summarize_tensor)
- [tf.contrib.layers.summarize_tensors](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/summarize_tensors)
- [tf.contrib.layers.summarize_collection](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/summarize_collection)

层模块定义方便的函数 summarize_variables，summarize_weights 和 summarize_biases，分别将 summarize_collection 集合参数设置为变量、权重和偏差。

- [tf.contrib.layers.summarize_activations](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/summarize_activations)

**功能列**

功能列提供了将数据映射到模型的机制。

- [tf.contrib.layers.bucketized_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/bucketized_column)
- [tf.contrib.layers.check_feature_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/check_feature_columns)
- [tf.contrib.layers.create_feature_spec_for_parsing](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/create_feature_spec_for_parsing)
- [tf.contrib.layers.crossed_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/crossed_column)
- [tf.contrib.layers.embedding_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embedding_column)
- [tf.contrib.layers.scattered_embedding_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/scattered_embedding_column)
- [tf.contrib.layers.input_from_feature_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/input_from_feature_columns)
- [tf.contrib.layers.joint_weighted_sum_from_feature_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/joint_weighted_sum_from_feature_columns)
- [tf.contrib.layers.make_place_holder_tensors_for_base_features](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/make_place_holder_tensors_for_base_features)
- [tf.contrib.layers.multi_class_target](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/multi_class_target)
- [tf.contrib.layers.one_hot_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/one_hot_column)
- [tf.contrib.layers.parse_feature_columns_from_examples](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/parse_feature_columns_from_examples)
- [tf.contrib.layers.parse_feature_columns_from_sequence_examples](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/parse_feature_columns_from_sequence_examples)
- [tf.contrib.layers.real_valued_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/real_valued_column)
- [tf.contrib.layers.shared_embedding_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/shared_embedding_columns)
- [tf.contrib.layers.sparse_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_hash_bucket)
- [tf.contrib.layers.sparse_column_with_integerized_feature](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_integerized_feature)
- [tf.contrib.layers.sparse_column_with_keys](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sparse_column_with_keys)
- [tf.contrib.layers.weighted_sparse_column](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/weighted_sparse_column)
- [tf.contrib.layers.weighted_sum_from_feature_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/weighted_sum_from_feature_columns)
- [tf.contrib.layers.infer_real_valued_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/infer_real_valued_columns)
- [tf.contrib.layers.sequence_input_from_feature_columns](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/sequence_input_from_feature_columns)

##  tf.contrib.layers.conv2d()

说明：定义卷积层

```
defconvolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None):
```

常用的参数说明如下：

·        inputs:形状为[batch_size, height, width, channels]的输入。

·        num_outputs：代表输出几个channel。这里不需要再指定输入的channel了，因为函数会自动根据inpus的shpe去判断。

·        kernel_size：卷积核大小，不需要带上batch和channel，只需要输入尺寸即可。[5,5]就代表5x5的卷积核，如果长和宽都一样，也可以只写一个数5.

·        stride：步长，默认是长宽都相等的步长。卷积时，一般都用1，所以默认值也是1.如果长和宽都不相等，也可以用一个数组[1,2]。

·        padding：填充方式，'SAME'或者'VALID'。

·        activation_fn：激活函数。默认是ReLU。也可以设置为None

·        weights_initializer：权重的初始化，默认为initializers.xavier_initializer()函数。

·        weights_regularizer：权重正则化项，可以加入正则函数。biases_initializer：偏置的初始化，默认为init_ops.zeros_initializer()函数。

·        biases_regularizer：偏置正则化项，可以加入正则函数。

·        **trainable****：**是否可训练，如作为训练节点，必须设置为True，默认即可。如果我们是微调网络，有时候需要冻结某一层的参数，则设置为False。

 

- 3)   tf.contirb.layers.flatten()

说明：定义平铺层

 

## tf.contrib.layers.fully_connected()

说明：定义全连接层

```
def fully_connected(inputs,
                    num_outputs,
                    activation_fn=nn.relu,
                    normalizer_fn=None,
                    normalizer_params=None,
                    weights_initializer=initializers.xavier_initializer(),
                    weights_regularizer=None,
                   biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None):
```



[![复制代码](file:///C:\Users\27485\AppData\Local\Temp\msohtmlclip1\01\clip_image001.gif)](javascript:void(0);)

·        inputs: A tensor of at least rank 2 and static value forthe last dimension; i.e. `[batch_size, depth]`, `[None, None, None, channels]`.

·        num_outputs: Integer or long, the number of output unitsin the layer.

·        activation_fn: Activation function. The default value isa ReLU function.Explicitly set it to None to skip it and maintain a linearactivation.

·        normalizer_fn: Normalization function to use instead of`biases`. If `normalizer_fn` is provided then `biases_initializer` and

·        `biases_regularizer` are ignored and `biases` are notcreated nor added.default set to None for no normalizer function

·        normalizer_params: Normalization function parameters.

·        weights_initializer: An initializer for the weights.

·        weights_regularizer: Optional regularizer for theweights.

·        biases_initializer: An initializer for the biases. IfNone skip biases.

·        biases_regularizer: Optional regularizer for the biases.

·        reuse: Whether or not the layer and its variables shouldbe reused. To be able to reuse the layer scope must be given.

·        variables_collections: Optional list of collections forall the variables or a dictionary containing a different list of collectionsper variable.

·        outputs_collections: Collection to add the outputs.

·        **trainable:** If `True`also add variables to the graph collection `GraphKeys.TRAINABLE_VARIABLES` (seetf.Variable).如果我们是微调网络，有时候需要冻结某一层的参数，则设置为False。

·        scope: Optional scope for variable_scope.

 

 ```
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 12:29:16 2018

@author: zy
"""

'''
建立一个带有全连接层的卷积神经网络  并对CIFAR-10数据集进行分类
1.使用2个卷积层的同卷积操作，滤波器大小为5x5，每个卷积层后面都会跟一个步长为2x2的池化层，滤波器大小为2x2
2.对输出的64个feature map进行全局平均池化，得到64个特征
3.加入一个全连接层，使用softmax激活函数，得到分类
'''

import cifar10_input
import tensorflow as tf
import numpy as np

def print_op_shape(t):
    '''
    输出一个操作op节点的形状
    '''
    print(t.op.name,'',t.get_shape().as_list())

'''
一 引入数据集
'''
batch_size = 128
learning_rate = 1e-4
training_step = 15000
display_step = 200
#数据集目录
data_dir = './cifar10_data/cifar-10-batches-bin'
print('begin')
#获取训练集数据
images_train,labels_train = cifar10_input.inputs(eval_data=False,data_dir = data_dir,batch_size=batch_size)
print('begin data')


'''
二 定义网络结构
'''

#定义占位符
input_x = tf.placeholder(dtype=tf.float32,shape=[None,24,24,3])   #图像大小24x24x
input_y = tf.placeholder(dtype=tf.float32,shape=[None,10])        #0-9类别 

x_image = tf.reshape(input_x,[batch_size,24,24,3])

#1.卷积层 ->池化层

h_conv1 = tf.contrib.layers.conv2d(inputs=x_image,num_outputs=64,kernel_size=5,stride=1,padding='SAME', activation_fn=tf.nn.relu)    #输出为[-1,24,24,64]
print_op_shape(h_conv1)
h_pool1 = tf.contrib.layers.max_pool2d(inputs=h_conv1,kernel_size=2,stride=2,padding='SAME')         #输出为[-1,12,12,64]
print_op_shape(h_pool1)


#2.卷积层 ->池化层

h_conv2 =tf.contrib.layers.conv2d(inputs=h_pool1,num_outputs=64,kernel_size=[5,5],stride=[1,1],padding='SAME', activation_fn=tf.nn.relu)    #输出为[-1,12,12,64]
print_op_shape(h_conv2)
h_pool2 =  tf.contrib.layers.max_pool2d(inputs=h_conv2,kernel_size=[2,2],stride=[2,2],padding='SAME')   #输出为[-1,6,6,64]
print_op_shape(h_pool2)



#3全连接层

nt_hpool2 = tf.contrib.layers.avg_pool2d(inputs=h_pool2,kernel_size=6,stride=6,padding='SAME')          #输出为[-1,1,1,64]
print_op_shape(nt_hpool2)
nt_hpool2_flat = tf.reshape(nt_hpool2,[-1,64])            
y_conv = tf.contrib.layers.fully_connected(inputs=nt_hpool2_flat,num_outputs=10,activation_fn=tf.nn.softmax)
print_op_shape(y_conv)

'''
三 定义求解器
'''

#softmax交叉熵代价函数
cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(y_conv),axis=1))

#求解器
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#返回一个准确度的数据
correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(input_y,1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

'''
四 开始训练
'''
sess = tf.Session();
sess.run(tf.global_variables_initializer())
# 启动计算图中所有的队列线程 调用tf.train.start_queue_runners来将文件名填充到队列，否则read操作会被阻塞到文件名队列中有值为止。
tf.train.start_queue_runners(sess=sess)

for step in range(training_step):
    #获取batch_size大小数据集
    image_batch,label_batch = sess.run([images_train,labels_train])
    
    #one hot编码
    label_b = np.eye(10,dtype=np.float32)[label_batch]
    
    #开始训练
    train.run(feed_dict={input_x:image_batch,input_y:label_b},session=sess)
    
    if step % display_step == 0:
        train_accuracy = accuracy.eval(feed_dict={input_x:image_batch,input_y:label_b},session=sess)
        print('Step {0} tranining accuracy {1}'.format(step,train_accuracy))

 ```



## tf.variable_scope()官方定义

定义在：[tensorflow/python/ops/variable_scope.py](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-hbaz2o9y.html)

@tf_export("variable_scope")  # pylint: disable=invalid-name
class variable_scope(object):
  """A context manager for defining ops that creates variables (layers).
  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.
  If `name_or_scope` is not None, it is used as is. If `scope` is None, then
  `default_name` is used.  In that case, if the same name has been previously
  used in the same scope, it will be made unique by appending `_N` to it.
  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the @{$variables$Variable Scope How To}, here we present only a few basic
  examples.
  Simple example of how to create a new variable:

  ```python
  with tf.variable_scope("foo"):
      with tf.variable_scope("bar"):
          v = tf.get_variable("v", [1])
          assert v.name == "foo/bar/v:0"
  ```
  Basic example of sharing a variable AUTO_REUSE:
  ```python
  def foo():
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
      v = tf.get_variable("v", [1])
    return v
  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```
  Basic example of sharing a variable with reuse=True:
  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
  with tf.variable_scope("foo", reuse=True):
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```
  Sharing a variable by capturing a scope and setting reuse:
  ```python
  with tf.variable_scope("foo") as scope:
      v = tf.get_variable("v", [1])
      scope.reuse_variables()
      v1 = tf.get_variable("v", [1])
  assert v1 == v
  ```
  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.
  ```python
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
      v1 = tf.get_variable("v", [1])
      #  Raises ValueError("... v already exists ...").
  ```
  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.
  ```python
  with tf.variable_scope("foo", reuse=True):
      v = tf.get_variable("v", [1])
      #  Raises ValueError("... v does not exists ...").
  ```
  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.
  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)
  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.
  """

## tf.variable_scope()-1

**tensorflow 为了更好的管理变量,提供了variable scope机制** 
**官方解释:** 
Variable scope object to carry defaults to provide to get_variable.

Many of the arguments we need for get_variable in a variable store are most easily handled with a context. This object is used for the defaults.

Attributes:

- name: name of the current scope, used as prefix in get_variable.
- initializer: 传给get_variable的默认initializer.如果get_variable的时候指定了initializer,那么将覆盖这个默认的initializer.
- regularizer: 传给get_variable的默认regulizer.
- reuse: Boolean or None, setting the reuse in get_variable.
- caching_device: string, callable, or None: the caching device passed to get_variable.
- partitioner: callable or None: the partitioner passed to get_variable.
- custom_getter: default custom getter passed to get_variable.
- name_scope: The name passed to tf.name_scope.
- dtype: default type passed to get_variable (defaults to DT_FLOAT).

`regularizer`参数的作用是给在本`variable_scope`下创建的`weights`加上正则项.这样我们就可以不同`variable_scope`下的参数加不同的正则项了.

**可以看出,用variable scope管理get_varibale是很方便的**

- 如何确定 get_variable 的 prefixed name

首先, variable scope是可以嵌套的:

```python
with variable_scope.variable_scope("tet1"):
    var3 = tf.get_variable("var3",shape=[2],dtype=tf.float32)
    print var3.name
    with variable_scope.variable_scope("tet2"):
        var4 = tf.get_variable("var4",shape=[2],dtype=tf.float32)
        print var4.name
#输出为****************
#tet1/var3:0
#tet1/tet2/var4:0
#*********************12345678910
```

get_varibale.name 以创建变量的 `scope` 作为名字的prefix

```python
def te2():
    with variable_scope.variable_scope("te2"):
        var2 = tf.get_variable("var2",shape=[2], dtype=tf.float32)
        print var2.name
        def te1():
            with variable_scope.variable_scope("te1"):
                var1 = tf.get_variable("var1", shape=[2], dtype=tf.float32)
            return var1
        return te1() #在scope te2 内调用的
res = te2()
print res.name
#输出*********************
#te2/var2:0
#te2/te1/var1:0
#************************123456789101112131415
```

观察和上个程序的不同

```python
def te2():
    with variable_scope.variable_scope("te2"):
        var2 = tf.get_variable("var2",shape=[2], dtype=tf.float32)
        print var2.name
        def te1():
            with variable_scope.variable_scope("te1"):
                var1 = tf.get_variable("var1", shape=[2], dtype=tf.float32)
            return var1
    return te1()  #在scope te2外面调用的
res = te2()
print res.name
#输出*********************
#te2/var2:0
#te1/var1:0
#************************123456789101112131415
```

**还有需要注意一点的是tf.variable_scope("name") 与 tf.variable_scope(scope)的区别，看下面代码**

代码1

```python
import tensorflow as tf
with tf.variable_scope("scope"):
    tf.get_variable("w",shape=[1])#这个变量的name是 scope/w
    with tf.variable_scope("scope"):
        tf.get_variable("w", shape=[1]) #这个变量的name是 scope/scope/w
# 这两个变量的名字是不一样的，所以不会产生冲突123456
```

代码2

```python
import tensorflow as tf
with tf.variable_scope("yin"):
    tf.get_variable("w",shape=[1])
    scope = tf.get_variable_scope()#这个变量的name是 scope/w
    with tf.variable_scope(scope):#这种方式设置的scope，是用的外部的scope
        tf.get_variable("w", shape=[1])#这个变量的name也是 scope/w
# 两个变量的名字一样，会报错1234567
```

- 共享变量

共享变量的前提是，变量的名字是一样的，变量的名字是由`变量名`和其`scope`前缀一起构成， `tf.get_variable_scope().reuse_variables()` 是允许共享当前`scope`下的所有变量。`reused variables`可以看同一个节点

```python
with tf.variable_scope("level1"):
    tf.get_variable("w",shape=[1])
    scope = tf.get_variable_scope()
    with tf.variable_scope("level2"):
        tf.get_variable("w", shape=[1])

with tf.variable_scope("level1", reuse=True): #即使嵌套的variable_scope也会被reuse
    tf.get_variable("w",shape=[1])
    scope = tf.get_variable_scope()
    with tf.variable_scope("level2"):
        tf.get_variable("w", shape=[1])1234567891011
```

- 其它

`tf.get_variable_scope()` :获取当前scope 
`tf.get_variable_scope().reuse_variables()` 共享变量

## tf.variable_scope()-2

variable_scope类

定义在：[tensorflow/python/ops/variable_scope.py](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-hbaz2o9y.html)

请参阅指南：[变量>共享变量](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-rv8u2c6g.html)

用于定义创建变量（层）的操作的上下文管理器。 

此上下文管理器验证（可选）values是否来自同一图形，确保图形是默认的图形，并推送名称范围和变量范围。

如果name_or_scope不是None，则使用as is。如果scope是None，则使用default_name。在这种情况下，如果以前在同一范围内使用过相同的名称，则通过添加_N来使其具有唯一性。

变量范围允许您创建新变量并共享已创建的变量，同时提供检查以防止意外创建或共享。在本文中我们提供了几个基本示例。

**示例1-如何创建一个新变量：**

```
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
```

**示例2-共享变量AUTO_REUSE：**

```
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

**示例3-使用reuse=True共享变量：**

```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

**示例4-通过捕获范围并设置重用来共享变量：**

```
with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
    scope.reuse_variables()
    v1 = tf.get_variable("v", [1])
assert v1 == v
```

为了防止意外共享变量，我们在获取非重用范围中的现有变量时引发异常。

```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...")
```

同样，我们在尝试获取重用模式中不存在的变量时引发异常。

```
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...")
```

请注意，reuse（重用）标志是有继承性的：如果我们打开一个重用范围，那么它的所有子范围也会重用。 

关于名称范围的说明：设置reuse不会影响其他操作（如多重）的命名。

请注意，1.0版本开始（包含）允许（虽然明确劝阻）将False传递给重用参数，从而产生与None无关的未记录行为。从1.1.0版本开始传递None和False作为重用具有完全相同的效果。

方法

__init__

```
__init__(
    name_or_scope,
    default_name=None,
    values=None,
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    reuse=None,
    dtype=None,
    use_resource=None,
    constraint=None,
    auxiliary_name_scope=True
)
```

用于初始化上下文管理器。

参数：

- name_or_scope：string或者VariableScope表示打开的范围。
- default_name：如果name_or_scope参数为None，则使用默认的名称，该名称将是唯一的；如果提供了name_or_scope，它将不会被使用，因此它不是必需的，并且可以是None。
- values：传递给操作函数的Tensor参数列表。
- initializer：此范围内变量的默认初始值设定项。
- regularizer：此范围内变量的默认正规化器。
- caching_device：此范围内变量的默认缓存设备。
- partitioner：此范围内变量的默认分区程序。
- custom_getter：此范围内的变量的默认自定义吸气。
- reuse：可以是True、None或tf.AUTO_REUSE；如果是True，则我们进入此范围的重用模式以及所有子范围；如果是tf.AUTO_REUSE，则我们创建变量（如果它们不存在），否则返回它们；如果是None，则我们继承父范围的重用标志。当启用紧急执行时，该参数总是被强制为tf.AUTO_REUSE。
- dtype：在此范围中创建的变量类型（默认为传入范围中的类型，或从父范围继承）。
- use_resource：如果为false，则所有变量都将是常规变量；如果为true，则将使用具有明确定义的语义的实验性 ResourceVariables。默认为false（稍后将更改为true）。当启用紧急执行时，该参数总是被强制为true。
- constraint：一个可选的投影函数，在被Optimizer（例如用于实现层权重的范数约束或值约束）更新之后应用于该变量。该函数必须将代表变量值的未投影张量作为输入，并返回投影值的张量（它必须具有相同的形状）。进行异步分布式培训时，约束条件的使用是不安全的。
- auxiliary_name_scope：如果为True，则我们用范围创建一个辅助名称范围；如果为False，则我们不接触名称范围。

返回值：

返回可以捕获和重用的范围。

可能引发的异常：

- ValueError：在创建范围内尝试重用时，或在重用范围内创建时。
- TypeError：某些参数的类型不合适时。

__enter__

```
__enter__()
```

__exit__

```
__exit__(
    type_arg,
    value_arg,
    traceback_arg
)
```

## tf.get_variable()

由 Carrie 创建， 最后一次修改 2017-10-24

```
函数：tf.get_variable
get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None
)
```

定义在：[tensorflow/python/ops/variable_scope.py](https://www.github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/variable_scope.py)

参见指南：[变量>共享变量](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-b4zy2evf.html)

**获取具有这些参数的现有变量或创建一个新变量。 **
此函数将名称与当前变量范围进行前缀，并执行重用检查。有关重用如何工作的详细说明，请参见变量范围。下面是一个基本示例:

```
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v")  # The same as v above.
```

如果初始化器为 None（默认），则将使用在变量范围内传递的默认初始化器。如果另一个也是 None，那么一个 glorot_uniform_initializer 将被使用。初始化器也可以是张量，在这种情况下，变量被初始化为该值和形状。 

类似地，如果正则化器是 None（默认），则将使用在变量范围内传递的默认正则符号（如果另一个也是 None，则默认情况下不执行正则化）。

如果提供了分区，则返回 PartitionedVariable。作为张量访问此对象将返回沿分区轴连接的碎片。 
一些有用的分区可用。例如：variable_axis_size_partitioner 和 min_max_variable_partitioner。

参数：

- name：新变量或现有变量的名称。
- shape：新变量或现有变量的形状。
- dtype：新变量或现有变量的类型（默认为 DT_FLOAT）。
- initializer：创建变量的初始化器。
- regularizer：一个函数（张量 - >张量或无）；将其应用于新创建的变量的结果将被添加到集合 tf.GraphKeys.REGULARIZATION_LOSSES 中，并可用于正则化。
- trainable：如果为 True，还将变量添加到图形集合：GraphKeys.TRAINABLE_VARIABLES。
- collections：要将变量添加到其中的图形集合键的列表。默认为 [GraphKeys.LOCAL_VARIABLES]。
- caching_device：可选的设备字符串或函数，描述变量应该被缓存以读取的位置。默认为变量的设备，如果不是 None，则在其他设备上进行缓存。典型的用法的在使用该变量的操作所在的设备上进行缓存，通过 Switch 和其他条件语句来复制重复数据删除。 
- partitioner：（可选）可调用性，它接受要创建的变量的完全定义的 TensorShape 和 dtype，并且返回每个坐标轴的分区列表（当前只能对一个坐标轴进行分区）。
- validate_shape：如果为假，则允许使用未知形状的值初始化变量。如果为真，则默认情况下，initial_value 的形状必须是已知的。
- use_resource：如果为假，则创建一个常规变量。如果为真，则创建一个实验性的 ResourceVariable，而不是具有明确定义的语义。默认为假（稍后将更改为真）。
- custom_getter：可调用的，将第一个参数作为真正的 getter，并允许覆盖内部的 get_variable 方法。custom_getter 的签名应该符合这种方法，但最经得起未来考验的版本将允许更改：def custom_getter(getter, *args, **kwargs)。还允许直接访问所有 get_variable 参数：def custom_getter(getter, name, *args, **kwargs)。创建具有修改的名称的变量的简单标识自定义 getter 是：python def custom_getter(getter, name, *args, **kwargs): return getter(name + '_suffix', *args, **kwargs) 

返回值：

创建或存在Variable（或者PartitionedVariable，如果使用分区器）。

可能引发的异常：

- ValueError：当创建新的变量和形状时，在变量创建时违反重用，或当 initializer 的 dtype 和 dtype 不匹配时。在 variable_scope 中设置重用。

## tf.reshape函数重塑张量

由 Carrie 创建， 最后一次修改 2017-12-23

TensorFlow - tf.reshape 函数

```
reshape(
    tensor,
    shape,
    name=None
)
```

参见指南：[张量变换>形状和形状](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-85v22c69.html)

重塑张量。 

给定tensor，这个操作返回一个张量，它与带有形状shape的tensor具有相同的值。

如果shape的一个分量是特殊值-1，则计算该维度的大小，以使总大小保持不变。特别地情况为，一个[-1]维的shape变平成1维。至多能有一个shape的分量可以是-1。

如果shape是1-D或更高，则操作返回形状为shape的张量，其填充为tensor的值。在这种情况下，隐含的shape元素数量必须与tensor元素数量相同。

例如：

```
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
reshape(t, [3, 3]) ==> [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]

# tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                        [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]

# tensor 't' is [7]
# shape `[]` reshapes to a scalar
reshape(t, []) ==> 7
```

参数：

- tensor：一个Tensor。
- shape：一个Tensor；必须是以下类型之一：int32，int64；用于定义输出张量的形状。
- name：操作的名称（可选）。

返回值：

该操作返回一个Tensor。与tensor具有相同的类型。

## tf.train.ExponentialMovingAverage

Some training algorithms, such as GradientDescent and Momentum often benefit from maintaining a moving average of variables during optimization. Using the moving averages for evaluations often improve results significantly. 
`tensorflow` 官网上对于这个方法功能的介绍。`GradientDescent` 和 `Momentum` 方式的训练 都能够从 `ExponentialMovingAverage` 方法中获益。

**什么是MovingAverage?** 
假设我们与一串时间序列
$$
\{a_1, a_2, a_3, ..., a_{t-1}, a_t, ...\}
$$
那么，这串时间序列的 MovingAverage就是： 

$$
mv_{t}=decay∗mv_{t−1}+(1−decay)∗a_{t}
$$


这是一个递归表达式。 

如何理解这个式子呢？他就像一个滑动窗口，
$$
mv_{t}
$$
 的值只和这个窗口内的 

$$
a_{i}
$$
 有关， 为什么这么说呢？将递归式拆开 : 

$$
mv_{t}=(1−decay)∗a_{t}+decay∗mv_{t−1}
$$

$$
mv_{t−1}=(1−decay)∗a_{t−1}+decay∗mv_{t−2}
$$

$$
mv_{t−2}=(1−decay)∗a_{t−2}+decay∗mv_{t−3}
$$



得到： 

$$
mv_{t}=∑_{i=1}^tdecay^{t−i}∗(1−decay)∗a_{i}
$$




当 t−i>C，C 为某足够大的数时

$$
decay^{t−i}∗(1−decay)∗a_{i}≈0
$$
, 所以: 

$$
mv_{t}≈∑_{i=t−C}^{t}decay^{t−i}∗(1−decay)∗a_i
$$
。即， 
$$
mv_t
$$
 的值只和 {at−C,...,at} 有关。

**tensorflow 中的 ExponentialMovingAverage**

这时，再看官方文档中的公式: 



shadowVariable=decay∗shadowVariable+(1−decay)∗variable

,就知道各代表什么意思了。 

`shadow variables are created with trainable=False`。用其来存放 ema 的值

```python
import tensorflow as tf
w = tf.Variable(1.0)
ema = tf.train.ExponentialMovingAverage(0.9)
update = tf.assign_add(w, 1.0)

with tf.control_dependencies([update]):
    #返回一个op,这个op用来更新moving_average,i.e. shadow value
    ema_op = ema.apply([w])#这句和下面那句不能调换顺序
# 以 w 当作 key， 获取 shadow value 的值
ema_val = ema.average(w)#参数不能是list，有点蛋疼

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3):
        sess.run(ema_op)
        print(sess.run(ema_val))
# 创建一个时间序列 1 2 3 4
#输出：
#1.1      =0.9*1 + 0.1*2
#1.29     =0.9*1.1+0.1*3
#1.561    =0.9*1.29+0.1*4123456789101112131415161718192021
```

你可能会奇怪，明明 只执行三次循环， 为什么产生了 4 个数？ 
这是因为，当程序执行到 `ema_op = ema.apply([w])` 的时候，如果 `w` 是 `Variable`， 那么将会用 `w` 的初始值初始化 `ema` 中关于 `w` 的 `ema_value`，所以 emaVal0=1.0emaVal0=1.0。如果 `w` 是 `Tensor`的话，将会用 `0.0` 初始化。

官网中的示例：

```python
# Create variables.
var0 = tf.Variable(...)
var1 = tf.Variable(...)
# ... use the variables to build a training model...
...
# Create an op that applies the optimizer.  This is what we usually
# would use as a training op.
opt_op = opt.minimize(my_loss, [var0, var1])

# Create an ExponentialMovingAverage object
ema = tf.train.ExponentialMovingAverage(decay=0.9999)

# Create the shadow variables, and add ops to maintain moving averages
# of var0 and var1.
maintain_averages_op = ema.apply([var0, var1])

# Create an op that will update the moving averages after each training
# step.  This is what we will use in place of the usual training op.
with tf.control_dependencies([opt_op]):
    training_op = tf.group(maintain_averages_op)
    # run这个op获取当前时刻 ema_value
    get_var0_average_op = ema.average(var0)12345678910111213141516171819202122
```

**使用 ExponentialMovingAveraged parameters**

假设我们使用了`ExponentialMovingAverage`方法训练了神经网络， 在`test`阶段，如何使用 `ExponentialMovingAveraged parameters`呢？ 官网也给出了答案 
**方法一：**

```python
# Create a Saver that loads variables from their saved shadow values.
shadow_var0_name = ema.average_name(var0)
shadow_var1_name = ema.average_name(var1)
saver = tf.train.Saver({shadow_var0_name: var0, shadow_var1_name: var1})
saver.restore(...checkpoint filename...)
# var0 and var1 now hold the moving average values123456
```

**方法二：**

```python
#Returns a map of names to Variables to restore.
variables_to_restore = ema.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
...
saver.restore(...checkpoint filename...)12345
```

这里要注意的一个问题是，用于保存的`saver`可不能这么写，参考 <http://blog.csdn.net/u012436149/article/details/56665612>

参考资料

<https://www.tensorflow.org/versions/master/api_docs/python/train/moving_averages>

## tf.get_collection()

```
函数：tf.get_collection
get_collection(
    key,
    scope=None
)
```

定义在：[tensorflow/python/framework/ops.py](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-s62t2d4y.html)。

参见指南：[构建图>图形集合](https://www.tensorflow.org/api_guides/python/framework#Graph_collections)

使用默认图形来包装 Graph.get_collection()。

参数：

- key：收集的关键。例如，GraphKeys 类包含许多集合的标准名称。
- scope：（可选）如果提供，则筛选结果列表为仅包含 name 属性匹配 re.match 使用的项目。如果一个范围是提供的，并且选择或 re. match 意味着没有特殊的令牌过滤器的范围，则不会返回没有名称属性的项。

返回值：

集合中具有给定 name 的值的列表，或者如果没有值已添加到该集合中，则为空列表。该列表包含按其收集顺序排列的值。

```
函数：tf.get_collection_ref
get_collection_ref(key)
```

定义在：[tensorflow/python/framework/ops.py](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-s62t2d4y.html)。

参见指南：[构建图>图形集合](https://www.tensorflow.org/api_guides/python/framework#Graph_collections)

使用默认图表来包装 Graph.get_collection_ref()。

参数：

- key：收集的关键。例如，GraphKeys 类包含许多标准的集合名称。

返回值：

集合中具有给定 name 的值的列表，或者如果没有值已添加到该集合中，则为空列表。请注意，这将返回集合列表本身，可以修改该列表来更改集合。

- 其他

在一个计算图中，可以通过集合(collection)来管理不同类别的资源。比如通过`tf.add_to_collection`函数可以将资源加入一个 或多个集合中，**然后通过`tf.get_collection`获取一个集合里面的所有资源(如张量，变量，或者运行TensorFlow程序所需的队列资源等等)**

**TensorFlow中维护的集合列表**

| 集合名称                                | 集合内容                             | 使用场景                     |
| --------------------------------------- | ------------------------------------ | ---------------------------- |
| `tf.GraphKeys.VARIABLES`                | 所有变量                             | 持久化TensorFlow模型         |
| `tf.GraphKeys.TRAINABLE_VARIABLES`      | 可学习的变量(一般指神经网络中的参数) | 模型训练、生成模型可视化内容 |
| `tf.GraphKeys.SUMMARIES`                | 日志生成相关的张量                   | TensorFlow计算可视化         |
| `tf.GraphKeys.QUEUE_RUNNERS`            | 处理输入的QueueRunner                | 输入处理                     |
| `tf.GraphKeys.MOVING_AVERAGE_VARIABLES` | 所有计算了滑动平均值的变量           | 计算变量的滑动平均值         |

## Variables: 创建、初始化、保存和加载

- 引言

当你训练一个模型的时候，你使用变量去保存和更新参数。在Tensorflow中变量是内存缓冲区中保存的张量（tensor）。它们必须被显示的初始化，可以在训练完成之后保存到磁盘上。之后，你可以重新加载这些值用于测试和模型分析。 
本篇文档引用了如下的Tensorflow类。以下的链接指向它们更加详细的API：

- [tf.Variable ](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#Variable)类。
- [tf.train.Saver ](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#Variable)类。

---- ---



- 创建 

当你创建一个变量时，你传递一个tensor数据作为它的初始值给Variable()构造器。Tensorflow提供了一堆操作从常量或者随机值中产生tensor数据用于初始化。 
**注意**这些操作要求你指定tensor数据的形状。这个形状自动的成为变量的形状。变量的形状是固定的。不过，Tensorflow提供了一些高级机制用于改变变量的形状。

```python
# 创建两个变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")123
```

调用tf.Variable()会在计算图上添加这些节点：

- 一个变量节点，用于保存变量的值
- 一个初始化操作节点，用于将变量设置为初始值。它实际上是一个tf.assign节点。
- 初始值节点，例如例子中的zeros节点也会被加入到计算图中。

-------

tf.Variable()返回值是Python类tf.Variable的一个实例。

- 设备配置

一个变量在创建时可以被塞进制定的设备，通过使用[ with tf.device(…):](https://www.tensorflow.org/versions/r0.12/api_docs/python/framework.html#device):

```python
# 将变量塞进CPU里
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# 将变量塞进GPU里
with tf.device("/gpu:0"):
  v = tf.Variable(...)

# 将变量塞进指定的参数服务任务里
with tf.device("/job:ps/task:7"):
  v = tf.Variable(...)1234567891011
```

**注意**一些改变变量的操作，例如v.assign()和在tf.train.Optimizer中变量的更新操作，必须与与变量创建时运行在同一设备上。创建这些操作是，不兼容的设备配置将会忽略。

- 初始化

变量的初始化必须找模型的其他操作之前，而且必须显示的运行。最简单的方式是添加一个节点用于初始化所有的变量，然后在使用模型之前运行这个节点。 
或者你可以选择从checkpoint文件中加载变量，之后将会介绍。 
使用tf.global_variables_initializer()添加节点用于初始化所有的变量。在你构建完整个模型并在会话中加载模型后，运行这个节点。

```python
# 创建两个变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")
...
# 添加用于初始化变量的节点
init_op = tf.global_variables_initializer()

# 然后，在加载模型的时候
with tf.Session() as sess:
  # 运行初始化操作
  sess.run(init_op)
  ...
  # 使用模型
  ...123456789101112131415
```

- 从别的变量中初始化

有时候你需要利用另一个变量来初始化当前变量。由于tf.global_variables_initializer()添加的节点适用于并行的初始化所有变量，所有如果你有这个需求，你得小心谨慎。 
为了从另一个变量中初始化一个新的变量，使用变量的另一个方法initialized_value()。你可以直接将旧变量的初始值作为新变量的初始值，或者你可以将旧变量的初始值进行一些运算后再作为新变量的初始值。

```python
# 使用随机数创建一个变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
# 创建另一个变量，它与weights拥有相同的初始值
w2 = tf.Variable(weights.initialized_value(), name="w2")
# 创建另一个变量，它的初始值是weights的两倍
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")1234567
```

- 自定义初始化

tf.global_variables_initializer()能够将所有的变量一步到位的初始化，非常的方便。你也可以将指定的列表传递给它，只初始化列表中的变量。 更多的选项请查看[Variables Documentation](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html)，包括检查变量是否初始化。

- 保存和加载

最简单的保存和加载模型的方法是使用tf.train.Saver 对象。它的构造器将在计算图上添加save和restore节点，针对图上所有或者指定的变量。saver对象提供了运行这些节点的方法，只要指定用于读写的checkpoint的文件。

- checkpoint文件

变量以二进制文件的形式保存在checkpoint文件中，粗略地来说就是变量名与tensor数值的一个映射 
当你创建一个Saver对象是，你可以选择变量在checkpoint文件中名字。默认情况下，它会使用Variable.name作为变量名。 
为了理解什么变量在checkpoint文件中，你可以使用[inspect_checkpoint](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py)库，更加详细地，使用print_tensors_in_checkpoint_file函数。

- 保存变量

使用tf.train.Saver()创建一个Saver对象，然后用它来管理模型中的所有变量。

```python
# 创建一些变量
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加用于初始化变量的节点
init_op = tf.global_variables_initializer()

# 添加用于保存和加载所有变量的节点
saver = tf.train.Saver()

# 然后，加载模型，初始化所有变量，完成一些操作后，把变量保存到磁盘上
with tf.Session() as sess:
  sess.run(init_op)
  # 进行一些操作
  ..
  # 将变量保存到磁盘上
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)123456789101112131415161718
```

- 加载变量

Saver对象还可以用于加载变量。**注意**当你从文件中加载变量是，你不用实现初始化它们。

```python
# 创建两个变量
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加用于保存和加载所有变量的节点
saver = tf.train.Saver()

# 然后，加载模型，使用saver对象从磁盘上加载变量，之后再使用模型进行一些操作
with tf.Session() as sess:
  # 从磁盘上加载对象
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # 使用模型进行一些操作
  ...1234567891011121314
```

- 选择变量进行保存和加载

如果你不传递任何参数给tf.train.Saver()，Saver对象将处理图中的所有变量。每一个变量使用创建时传递给它的名字保存在磁盘上。 
有时候，我们需要显示地指定变量保存在checkpoint文件中的名字。例如，你可能使用名为“weights”的变量训练模型；在保存的时候，你希望用“params”为名字保存。 
有时候，我们只保存和加载模型的部分参数。例如，你已经训练了一个5层的神经网络；现在你想训练一个新的神经网络，它有6层。加载旧模型的参数作为新神经网络前5层的参数。 
通过传递给tf.train.Saver()一个Python字典，你可以简单地指定名字和想要保存的变量。字典的keys是保存在磁盘上的名字，values是变量的值。 
**注意:** 
如果你需要保存和加载不同子集的变量，你可以随心所欲地创建任意多的saver对象。同一个变量可以被多个saver对象保存。它的值仅仅在restore()方法运行之后发生改变。 
如果在会话开始之初，你仅加载了部分变量，你还需要为其他变量运行初始化操作。参见[tf.initialize_variables() ](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops.html#initialize_variables)查询更多的信息。

```python
# 创建一些对象
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# 添加一个节点用于保存和加载变量v2，使用名字“my_v2”
saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.
...12345678
```





#  **DQN源码阅读**

> 2018/8/20

## dqn.py源码如下：

```
import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]

class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def deep_q_learning(sess,
                    env,
                    q_estimator,
                    target_estimator,
                    state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing 
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the 
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state

    # Record videos
    # Use the gym env Monitor wrapper
    env = Monitor(env,
                  directory=monitor_path,
                  resume=True,
                  video_callable=lambda count: count % record_video_every ==0)

    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_t += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward", tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length", tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, total_t)
        q_estimator.summary_writer.flush()

        yield total_t, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode+1],
            episode_rewards=stats.episode_rewards[:i_episode+1])

    env.monitor.close()
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))


```



### 状态处理

输入：state,也就是Atari RGB State(图像)

输出：给定尺寸大小灰度图像

```
class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })
```

### Q-值评估神经网络，包括Q值网络以及目标网络

```
1）  tf.summary()的各类方法：能够保存训练过程以及参数分布图并在tensorboard显示。

        tf.summary.FileWritter(path,sess.graph)

        指定一个文件用来保存图。

        可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
```
```
class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])


    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
```



### gather函数

```
import tensorflow as tf

temp = tf.range(0,10)*10 + tf.constant(1,shape=[10])
temp2 = tf.gather(temp,[1,5,9])
with tf.Session() as sess:
    print(sess.run(temp))
    print(sess.run(temp2))
```

输出：

```
[ 1 11 21 31 41 51 61 71 81 91]
[11 51 91]
```

### tf.train.Saver(模型保存和读取)

深度学习平台：TensorFlow    

目标：训练网络后想保存训练好的模型，以及在程序中读取以保存的训练好的模型。

**简介**

首先，保存和恢复都需要实例化一个 tf.train.Saver。

- saver = tf.train.Saver()

然后，在训练循环中，定期调用 saver.save() 方法，向文件夹中写入包含了当前模型中所有可训练变量的 checkpoint 文件。

- saver.save(sess, FLAGS.train_dir, global_step=step)

之后，就可以使用 saver.restore() 方法，重载模型的参数，继续训练或用于测试数据。

- saver.restore(sess, FLAGS.train_dir)

一次 saver.save() 后可以在文件夹中看到新增的四个文件，

![img](https://img-blog.csdn.net/20170704144050301)

实际上每调用一次保存操作会创建后3个数据文件并创建一个检查点（checkpoint）文件，简单理解就是权重等参数被保存到 .ckpt.data 文件中，以字典的形式；图和元数据被保存到 .ckpt.meta 文件中，可以被 tf.train.import_meta_graph 加载到当前默认的图。

**示例**

下面代码是简单的保存和读取模型：（不包括加载图数据)

```
import tensorflow as tf
import numpy as np
import os
 
#用numpy产生数据
x_data = np.linspace(-1,1,300)[:, np.newaxis] #转置
noise = np.random.normal(0,0.05, x_data.shape)
y_data = np.square(x_data)-0.5+noise
 
#输入层
x_ph = tf.placeholder(tf.float32, [None, 1])
y_ph = tf.placeholder(tf.float32, [None, 1])
 
#隐藏层
w1 = tf.Variable(tf.random_normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10])+0.1)
wx_plus_b1 = tf.matmul(x_ph, w1) + b1
hidden = tf.nn.relu(wx_plus_b1)
 
#输出层
w2 = tf.Variable(tf.random_normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1])+0.1)
wx_plus_b2 = tf.matmul(hidden, w2) + b2
y = wx_plus_b2
 
#损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ph-y),reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
#保存模型对象saver
saver = tf.train.Saver()
 
#判断模型保存路径是否存在，不存在就创建
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')
 
#初始化
with tf.Session() as sess:
    if os.path.exists('tmp/checkpoint'):         #判断模型是否存在
        saver.restore(sess, 'tmp/model.ckpt')    #存在就从模型中恢复变量
    else:
        init = tf.global_variables_initializer() #不存在就初始化变量
        sess.run(init)
 
    for i in range(1000):
        _,loss_value = sess.run([train_op,loss], feed_dict={x_ph:x_data, y_ph:y_data})
        if(i%50==0):
            save_path = saver.save(sess, 'tmp/model.ckpt')
            print("迭代次数：%d , 训练损失：%s"%(i, loss_value))

```

注：

1. saver 的操作必须在 sess 建立后进行。
2. model.ckpt 必须存在给定文件夹中，'tmp/model.ckpt' 这里至少要有一层文件夹，否则无法保存。
3. 恢复模型时同保存时一样，是 ‘tmp/model.ckpt’，和那3个文件名都不一样。



#### 



