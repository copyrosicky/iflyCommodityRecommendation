#                              讯飞商品推荐预测比赛复盘总结

代码执行顺序：

preprocessing.py >>  

```
preprocessing:
	数据预处理 整理手机品牌的字符串格式 整理targid格式 timestamp排序 targid根据对应的timestamp排序 类别特征数字化 
	
eda ：
     不同的feature下label的分布      # feature值   对应取值的feature数量 其中label为1的比率
	 targid-len的pdf和箱线图

stat feature ：
	 timestamp的时间特征处理： 对时间序列做一阶差分处理 统计差分序列的最大值、最小值、中位数、均值、方差和时间序列的长度
	 targid的统计特征处理： targid的个数和去重的targid个数
	 					 str(targid)后统计targid的tfidf向量： max_feature = 200 norm = l2 sommth_idf = true 
	 					只考虑按照词频排序的前200个word 得出的tfidf向量除以向量的模长标准化 idf(t)=log( (1+训                                                                                      练集文本总数) / (1+包含词t的文本数) )+1
     对类别特征做one-hot-encoding 
     对所有特征储存为csr矩阵处理

embedding feature : 
	CBOW skip-gram  对于每个targid构造对应的embedding 将一个序列的targid加总之后求平均（除以targid的数目） 得到了整个序列对应的embedding向量 （CBOW 和 skip-gram 分别都要构造对应的序列embedding）
	tfidf sentences2vec 对应的kwag！！！

itemPropertyFeature ：
	对item序列的一些基本property进行累计
	

    
```

代码继续改进方向？？