# ANA-SL-ELMO

## 1. NewModelNetworkKBP

- 主要用来实现数据集TAC40的模型

### 1.1 dataElmoKBP

- KBP的数据预处理：
- 1.process the KBP dataset
- 2.concate the "glove embedding" and "Elmo embedding" and "Pos embedding"

## 2. ANASLELMO_in_Conll

- 用来实现ANA-SL-ELMO模型
- 执行顺序：

（1）splitTrainTest.py：要先分割train data和test data

（2）dataProcessConll和gloveGenerate

### 2.1 dataProcessConll

- (1)process the Conll04 dataset
- (2)concate the "glove embedding" and "Elmo embedding" and "Pos embedding"

### 2.2 gloveGenerate.py
- 生成glove词向量

### 2.3 splitTrainTest.py
- 把数据分割成train文件和test文件

### 2.4 prExperiment.py
- 计算RP值和其曲线

## 3. OtherModel
- 用来实现其他模型的，包括：CNN，RNN，BLSTM，ATT-BLSTM，Att-Pooling-CNN

## 4.result
- experiment_PR (copy)：文件夹存放了原始的用来做实验的pr图
- experiment_PR：新存放的

## 5.ComparedLossFun
- 实现了SIL loss, SDL
- 实现了各种不同的loss对比
- test
