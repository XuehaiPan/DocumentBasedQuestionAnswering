# 基于文档的自动问答评测
<center>潘学海 1500011317<br>杨纪翔 1500011342<br>杜尚宸<br>曾皓 1600012955</center>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [基于文档的自动问答评测](#基于文档的自动问答评测)
	* [问题概述](#问题概述)
	* [模型实现](#模型实现)
	* [训练结果](#训练结果)
	* [验证结果](#验证结果)
	* [附录](#附录)
		* [编译/运行环境](#编译运行环境)
		* [分工情况](#分工情况)
		* [参考文献](#参考文献)

<!-- /code_chunk_output -->
## 问题概述
我们试图建立一个模型，判断给定的一段话是不是一个问题的答案。可供使用的训练集和验证集的每条数据均由两个句子和一个标签组成。样例如下：
```
内迪尔科参加欧洲冠军联赛的赛况如何？	代表国家队：出场0次，进0球	0
内迪尔科参加欧洲冠军联赛的赛况如何？	欧洲三大杯：出场0次，进0球	0
内迪尔科参加欧洲冠军联赛的赛况如何？	欧洲冠军联赛：出场0次，进0球	1
```
在进行建模之前，我们对数据进行了预处理，检查了问题和答案的长度分布，如下图所示。
![]()

## 模型实现

## 训练结果

## 验证结果

## 附录
### 编译/运行环境
+ 全部代码在`Python 3.6`环境下编译，使用到的主要的开源库及其版本如下：
	+ Tensorflow 1.12.0
	+ Keras 2.2.4
	+ jieba 0.39
+ 模型的训练环境是【？】
+ 验证集及测试集的评估在【？】下完成
### 分工情况

### 参考文献
+ Yang L, Ai Q, Guo J, et al. aNMM: Ranking short answer texts with attention-based neural matching model[C]//Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 2016: 287-296.