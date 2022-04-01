## Bert TensorRT加速推断

**XuJing**

### 1.环境配置

+ python package:
```
pip install -r rquirements.txt
```
+ tensorrt 8.2
+ pycuda
+ 下载tensorflow版bert的中文预训练模型！这里以tensorflow中文预训练模型为例！

### 2.修改代码

+ build.py中的下游任务的结点去掉（squad)。
+ 如果你的Bert是基于下游任务fine-tune的，请将下游任务的节点也通过trt api实现，可以参考squad任务的实现方式，比较简单。


### 3.TensorRT FP32

```
python builder.py -m models/chinese_L-12_H-768_A-12/bert_model.ckpt -o engines/bert_base_128_zh.engine -b 1 -s 128 -c models/chinese_L-12_H-768_A-12
```

### 4.TensorRT FP16

```
python builder.py -m models/chinese_L-12_H-768_A-12/bert_model.ckpt -o engines/bert_base_128_zh.engine -b 1 -s 128 -c models/chinese_L-12_H-768_A-12 --fp16

```

### 5.TensorRT INT8量化

```
python builder.py -m models/chinese_L-12_H-768_A-12/bert_model.ckpt -o engines/bert_base_128_zh.engine -b 1 -s 128 -c models/chinese_L-12_H-768_A-12 --int8
```

+ 注意修改 calibrator.py int8量化数据的加载默认是squad.


### 6.运行TensorRT加速的模型

```
python bert_tensorrt.py

```