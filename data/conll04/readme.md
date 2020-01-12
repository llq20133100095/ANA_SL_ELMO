# 1.数据集描述
1.1 conll04_relation:关系分类数据集
https://cogcomp.seas.upenn.edu/page/resource_view/43下载的；
共有三个类别，kill，born_in和relation

1.2 conll04st_name recognise：命名实体识别

1.3 conll04_elmo: elmo

1.4 conll04_relation_train_test: 分割了主要的train和test文件

# 2.做法
- 先分别得到no_other和other的sen文件
- 手动合no_other和other文件，得到sen_all和label_all
- 运行splitTrainTest.py文件得到train和test文件
- 生成elmo和glove文件
- 转到java程序中，生成sdp文件
