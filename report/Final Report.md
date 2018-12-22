# 基于文档的自动问答评测
<center>潘学海 1500011317<br>杨纪翔 1500011342<br>杜尚宸<br>曾 皓 1600012955</center>

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [基于文档的自动问答评测](#基于文档的自动问答评测)
	* [问题概述](#问题概述)
	* [模型概述](#模型概述)
		* [问题-回答匹配矩阵(QA Matching Matrix)](#问题-回答匹配矩阵qa-matching-matrix)
		* [价值共享的权重(Value-shared Weight)](#价值共享的权重value-shared-weight)
		* [问题注意力网络(Question Attention Network)](#问题注意力网络question-attention-network)
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
在进行建模之前，我们对数据进行了预处理，检查了问题和答案的长度分布，如下图所示（注：test集的数据是最后更新的，建模时未作参考，但不影响结果）。绝大多数的问题的长度在15个词以下，回答的长度在60词以下。为平衡训练的质量和效率，我们最后选择了问题的截断长度为40词，答案的截断长度为160词。此外，大多数问题有30条回答，我们猜测这是数据发布者人为选择的结果。
![](../figures/data_dist.png)

## 模型概述
对于Question Answering(QA)问题，除了传统的feature engineering，也可以使用深度学习方法进行实现。但使用卷积神经网络(CNN)或长短期记忆(LSTM)方法构建模型时，往往需要使用额外的特征进行辅助，否则效果会很不好。因此，在本课题中，我们参考相关文献[1]，实现了一个aNMM (attention-based Nerual Matching Model)模型，对问答结果进行分析。以下作简要介绍。
![](../figures/aNMM-F1.png)
<center>图1: aNMM模型网络结构[1]</center>

### 问题-回答匹配矩阵(QA Matching Matrix)

文字处理任务中，将词语转化为词向量是很常规的一步(Word Embedding)。在这之后，我们面对的问题是如何将一个问题和一个答案的两组词向量联系起来。这里采用的方法是构建问题-回答匹配矩阵(QA Matching Matrix)，即将问题中的每一个词和回答中的每一个词的一一比较，矩阵元即为其相似度。相似度函数为`cos`函数，若问题中的词$q_j$和回答中的词$a_k$完全相同则相似度为1。即
$$x_{jk}=\cos (<\vec q_j,\vec x_k>)=\frac{\vec q_j\cdot \vec x_k}{|\vec q_j|\cdot|\vec x_k|}$$

### 价值共享的权重(Value-shared Weight)
传统神经网络如CNN本身是为图像处理而设计的，因此它的主要特点是，每一个filter总是处理临近的若干数据点，因此原数据集中相邻位置的点会共享相同的网络权重(position-shared weight)。
而在自然语言中，由于语法句式的复杂多样，重要信息可以出现在句子的每一个位置，基于位置的filter很难提取特征。因此我们给价值相似的词赋予相同的权重连入网络(value-shared weight)，会有更好的结果。这里的价值就是在上面定义的问题词和答案词之间的相似度。即$$h_j=\delta(\sum_{k=0}^{K}w_k\cdot x_{jk})$$这里$h_j$表示第$j$个节点的输出，采用`sigmoid`激活函数，$w_k$表示第$k$个价值区间对应的权重。每一个价值区间，如下图中的$[0,0.5)$、$[0.5,1)$和${1}$，分别称为一个容器(bin)。
![](../figures/aNMM-F2.png)
<center>图2: 位置共享权重(CNN)和价值共享权重(aNMM)的比较[1]</center>


### 问题注意力网络(Question Attention Network)
传统方法的另一问题是，对于问题中的哪个词比较重要这件事并没有特别的关注。看以下这个例子：
> ==内迪尔科==参加==欧洲冠军联赛==的==赛况==如何？

显然，句子中有一些词比另一些词更加关键，因此应当在结果中更加关注这些词对应的输出。在本模型中，我们在输出层之前加入一层网络，问题中的每个词对应一个`softmax`门函数来控制对应的网络输出，网络需要训练参数使得门函数挑出那些比较重要的词。具体来说如下式所示
$$y=\sum_{j=1}^M{g_j\cdot h_j}=\sum_{j=1}^M{\frac{\exp(\bold{v}\cdot\bold{q_j})}{\sum_{l=1}^L\exp(\bold{v}\cdot\bold{q_l})}}\cdot h_j$$
式中的$\bold{v}$是一个模型参量，通过训练达到最优化。

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
1. Yang L, Ai Q, Guo J, et al. aNMM: Ranking short answer texts with attention-based neural matching model[C]//Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 2016: 287-296.