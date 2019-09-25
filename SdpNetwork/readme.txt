1.SDP的平均长度为5.607625，最大长度为31（train data）
SDP的平均长度为5.6157526683842471，最大长度为26（test data）

2.自定义层：CustomLayers.py

3.数据预处理：sdpProcessData.py

4.自定义loss：CustomLoss.py

5.在整个句子长度下，实现Arc-One model：networkArcOneBiGRU.py

6.在SDP数据中，实现Arc-One model：sdpArcOneCNNGRU.py

7.最基本的SDP数据，在CNN和GRU上实现：sdpCNNGRU.py

8.在SDP数据中，实现new loss：sdpNlCNNGRU.py

9.在利用全部数据进行测试：
（1）最基本：networkSlBiGRU.py（stimulative loss）
（2）使用stimulative-loss、keywords-attention：networkSlKABiGRU.py
（3）使用keywords-attention：networkKeyAttSoftmaxBiGRU.py
（4）使用了entity-GRU attention：networkNlAttBiGRU.py
（5）使用了stimulative-loss、一开始提出的self-attention、ensemble-learning：
networkNlSelfAttEnsBiGRU.py+networkNlSelfAttEnsBiGRU2.py


###############SemEval-2010 Task 8#################
1.networkSlKAMutualBiGRU7.py:最终的论文实验代码。
（1）mutual learning+KAS+stimulation
（2）SSL:stimulative=1+coding_dist_true2inf-true_pre
         loss=stimulative*T.log(1+y_pre2true+y_pre2false)
（3）用3个KL

2.networkSlKAMutualBiGRU6.py：
（1）mutual learning+KAS+stimulation
（2）SSL:stimulative=1+coding_dist_true2inf-true_pre
         loss=stimulative*T.log(1+y_pre2true+y_pre2false)
（3）用2个KL

3.networkSlKABiGRU5Mutual.py
（1）mutual learning+KAS+stimulation
（2）SSL：
        stimulative=T.exp(2+coding_dist_true2inf-true_pre)
        loss=0.5*T.nnet.sigmoid(y_pre2true)*T.nnet.sigmoid(y_pre2false)*T.log(1+y_pre2true+y_pre2false)

4.networkNlSelfAttEnsBiGRU.py + networkNlSelfAttEnsBiGRU2.py ：使用了集成学习

5.networkSlKABiGRU2.py：SSL+KAS

