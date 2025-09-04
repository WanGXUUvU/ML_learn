## Numpy Ndarray对象
ndarray 内部由以下内容组成：
- 一个指向数据（内存或内存映射文件中的一块数据）的指针。
- 数据类型或 dtype，描述在数组中的固定大小值的格子。
- 一个表示数组形状（shape）的元组，表示各维度大小的元组。
- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。
- ![](assets/17568917352662.png)
## dtype数据类型对象
数据类型对象（numpy.dtype 类的实例）用来描述与数组对应的内存区域是如何使用，它描述了数据的以下几个方面：
- 数据的类型（整数，浮点数或者 Python 对象）
- 数据的大小（例如， 整数使用多少个字节存储）
- 数据的字节顺序（小端法或大端法）
- 在结构化类型的情况下，字段的名称、每个字段的数据类型和每个字段所取的内存块的部分
- 如果数据类型是子数组，那么它的形状和数据类型是什么。
字节顺序是通过对数据类型预先设定 < 或 > 来决定的。 < 意味着小端法(最小值存储在最小的地址，即低位组放在最前面)。> 意味着大端法(最重要的字节存储在最小的地址，即高位组放在最前面)。
- object	数组或嵌套的数列
- dtype	数组元素的数据类型，可选
- copy	对象是否需要复制，可选
- order	创建数组的样式，C为行方向，F为列方向，A为任意方向（默认）
- subok	默认返回一个与基类类型一致的数组
- ndmin	指定生成数组的最小维度
## 数组属性
### 秩
数组的维度称为秩。一维数组的秩为1。
每一个线性的数组称为一个轴（axis）。二维数组相当于两个一维数组。
axis=0表示沿着第0轴及每一列的操作
axis=1表示沿着第1轴及每一行的操作

shape 是用于查看数组当前的形状。
reshape 是用于创建一个新的数组，改变其形状。

## 从已经有的数组创建数组
### numpy.asarray 
类似 numpy.array，但 numpy.asarray 参数只有三个，比 numpy.array 少两个。
- a	任意形式的输入参数，可以是，列表, 列表的元组, 元组, 元组的元组, 元组的列表，多维数组
- dtype	数据类型，可选
- order	可选，有"C"和"F"两个选项,分别代表，行优先和列优先，在计算机内存中的存储元素的顺序。
- numpy.frombuffer 用于实现动态数组。
### numpy.frombuffer 
接受 buffer 输入参数，以流的形式读入转化成 ndarray 对象。
`numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)`
- buffer	可以是任意对象，会以流的形式读入。
- dtype	返回数组的数据类型，可选
- count	读取的数据数量，默认为-1，读取所有数据。
- offset	读取的起始位置，默认为0。
### numpy.fromiter
numpy.fromiter 方法从可迭代对象中建立 ndarray 对象，返回一维数组。
`numpy.fromiter(iterable, dtype, count=-1)`
- iterable	可迭代对象
- dtype	返回数组的数据类型
- count	读取的数据数量，默认为-1，读取所有数据

## Numpy从数值范围创建数组
### numpy.arange
`numpy.arange(start, stop, step, dtype)`
- start	起始值，默认为0
- stop	终止值（不包含）
- step	步长，默认为1
- dtype	返回ndarray的数据类型，如果没有提供，则会使用输入数据的类型。

### numpy.linspace
numpy.linspace 函数用于创建一个一维数组，数组是一个等差数列构成的，格式如下：
`np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)`
- start	序列的起始值
- stop	序列的终止值，如果endpoint为true，该值包含于数列中
- num	要生成的等步长的样本数量，默认为50
- endpoint	该值为 true 时，数列中包含stop值，反之不包含，默认是True。
- retstep	如果为 True 时，生成的数组中会显示间距，反之不显示。
- dtype	ndarray 的数据类型

### numpy.logspace
numpy.logspace 函数用于创建一个于等比数列。格式如下：
`np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None)`
- start	序列的起始值为：base ** start
- stop	序列的终止值为：base ** stop。如果endpoint为true，该值包含于数列中
- num	要生成的等步长的样本数量，默认为50
- endpoint	该值为 true 时，数列中中包含stop值，反之不包含，默认是True。
- base	对数 log 的底数。
- dtype	ndarray 的数据类型

## Numpy切片和索引
ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。
ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

## 高级索引
NumPy 比一般的 Python 序列提供更多的索引方式。
除了之前看到的用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引。
### 整数数组索引
整数数组索引是指使用一个数组来访问另一个数组的元素。这个数组中的每个元素都是目标数组中某个维度上的索引值。
### 布尔索引
尔索引通过布尔运算（如：比较运算符）来获取符合指定条件的元素的数组。
## 花式索引
花式索引根据索引数组的值作为目标数组的某个轴的下标来取值。
对于使用一维整型数组作为索引，如果目标是一维数组，那么索引的结果就是对应位置的元素，如果目标是二维数组，那么就是对应下标的行。
花式索引跟切片不一样，它总是将数据复制到新数组中。