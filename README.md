
* 论文标题：Listwise Reward Estimation for Offline Preference\-based Reinforcement Learning，ICML 2024。
* arxiv：[https://arxiv.org/abs/2408\.04190](https://github.com)
* pdf：[https://arxiv.org/pdf/2408\.04190](https://github.com)
* html：[https://ar5iv.org/html/2408\.04190](https://github.com)
* GitHub：[https://github.com/chwoong/LiRE](https://github.com)
* （感觉关于 构造 A\>B\>C 的 RLT 列表，得到更多 preference 数据，[SeqRank](https://github.com) 也是这个 idea。）


## 0 abstract


In Reinforcement Learning (RL), designing precise reward functions remains to be a challenge, particularly when aligning with human intent. Preference\-based RL (PbRL) was introduced to address this problem by learning reward models from human feedback. However, existing PbRL methods have limitations as they often overlook the second\-order preference that indicates the relative strength of preference. In this paper, we propose Listwise Reward Estimation (LiRE), a novel approach for offline PbRL that leverages secondorder preference information by constructing a Ranked List of Trajectories (RLT), which can be efficiently built by using the same ternary feedback type as traditional methods. To validate the effectiveness of LiRE, we propose a new offline PbRL dataset that objectively reflects the effect of the estimated rewards. Our extensive experiments on the dataset demonstrate the superiority of LiRE, i.e., outperforming state\-of\-the\-art baselines even with modest feedback budgets and enjoying robustness with respect to the number of feedbacks and feedback noise. Our code is available at [https://github.com/chwoong/LiRE](https://github.com)


* background \& gap：
	+ 在强化学习 （RL） 中，设计精确的、与人类意图保持一致的奖励函数，具有挑战性。Preference\-based RL（PbRL）从人类反馈中学习奖励模型，可以解决这个问题。
	+ 然而，现有的 PbRL 方法存在局限性：它们只能应对“A 比 B 好”“B 比 A 好”这种 0 1 的情况，而忽略了表示偏好相对强度的二阶（second\-order）偏好。
* method：
	+ 在本文中，我们提出了 Listwise Reward Estimation （LiRE），一种新颖的 offline PbRL 方法，它通过构建轨迹排名列表（Ranked List of Trajectories，RLT）来利用二阶偏好信息。
	+ 构建 RLT：使用与传统 PbRL 相同的三元组 feedback (σ0,σ1,p) 。对于新给出的 segment，用插入排序的方式将其放到目前的 RLT 里。
* experiment：
	+ 这篇文章提出了一个新的 offline PbRL dataset，用于评价 reward model 的学习效果。因为 d4rl 环境太简单，还会有 survival instinct（生存本能）现象，[不适用于 reward 学习](https://github.com) 。
	+ 实验证明，LiRE 在反馈预算适中的情况下 outperform baselines，并且在 feedback 数量和 noisy feedback 方面更加稳健。


## 2 related work


* offline PbRL：
	+ Reward Learning from Human Preferences and Demonstrations in Atari. [arxiv](https://github.com):[飞数机场](https://ze16.com) 这篇是 2018 年的文章，先对 expert demo 做模仿学习，然后 rollout 得到一些 segment，拿这些 segment 去打 preference，最后 PbRL 微调。
	+ Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback. [open review](https://github.com) 这篇是 SeqRank，是 2023 neurips 的文章。SeqRank 把新得到的 segment 和先前收集的 segment（最近收集的 / 先前所有 segment 里最好的）拿去比较。如果能比出 σ(t0)\<σ(t1)\<σ(t2)\<σ(t3) 的结果，就能得到 3×2×1 \= 6 \= (n\-1\)! 个 preference，但是我们其实只比了三次；SeqRank 就通过这种思想来对 preference 做数据增强。
	+ lire 讲 offline PbRL 的思路是，最近的工作专注于直接优化策略、省掉 reward model（比如 DPO），但是选择 informative query 也很重要。OPRL 是一种 active query selection 方法，选择 disagreement 最大的 query，但它没有使用二阶偏好。
* Second\-order Preference Feedback：
	+ 有些方法直接获得一个相对 preference 数值（明显更好 / 略好），或每个轨迹的绝对 rating（非常好 好 一般 中 差），但它们获取 feedback 的成本较高。
		- Llama 2: Open Foundation and Fine\-Tuned Chat Models.
		- Weak Human Preference Supervision For Deep Reinforcement Learning. TNNLS（Transactions on Neural Networks and Learning Systems）2021 年的文章。[arxiv](https://github.com) 首先让 p 从 {0, 1} 变成 \[0, 1]，直接优化交叉熵损失，做了一些神秘归一化，然后搞了一个 preference predictor 做数据增强；没有仔细看。
		- Rating\-based Reinforcement Learning. AAAI 2024，[arxiv](https://github.com)。这篇文章的名字是 RbRL；人类直接给一个 segment 一个 {0, ..., n\-2, n\-1} 的 rating，然后我们学一个 rating model，试图判断一个 segment 被分到了哪个 rating。这个 rating model 通过判断 Σr hat(σ) 和定义的 n 个 rating 类别的奖励值边界的大小关系，来判断 segment 被分到了哪个 rating。rating 类别的奖励值边界会动态更新。
	+ 有很多 Learning\-to\-Rank 的工作，它们已经拿到了二阶偏好，试图去学一个 ranking 的评分。
	+ 还有一些工作，它们可以从多个轨迹的全序列表（比如 A
	+ 然后又提了提 SeqRank。


## 4 method


* 首先对 preference 做了一些假设：
	+ 完备性：假设拿到 σ0 σ1，要不是 σ0≻σ1, σ0≺σ1，要不就认为一样好 σ0\=σ1，认为没有比不出来的情况。
	+ 传递性：假设如果有 σ0≻σ1, σ1≻σ2，那么有 σ0≻σ2。


### 4\.1 构建 RLT（Ranked List of Trajectories）


* 我们的目标是得到形式为 L\=\[g1≺g2≺⋯≺gs] 的 RLT，其中 gi\={σi1,⋯,σik} 是一组具有相同优先级的 segment。（有点像 帕累托前沿 分层 之类）
* 具体构建方法：我们每拿到一个新 segment，就把它拿去跟目前的 RLT 插入排序比较，找到一个放新 segment 的位置。直到 feedback 预算用完。
* 表 1 计算了 LiRE 的 feedback efficiency 和 sample diversity，并且与 vanilla 方法、SeqRank 进行了比较。feedback efficiency 定义为 \[我们获得的 feedback 数量] / \[我们进行比较的数量] 。sample diversity 定义为 \[我们获得的 feedback 数量] / \[所用到的 segment 数量] 。
* 我们并没有构建一个很长的 RLT，而是去构建多个相对短的 RLT，为每个 RLT 都设置 feedback 预算。


### 4\.2 从 RLT 里学 reward model


从 RLT 里推导出 (σ0,σ1,p) 的 preference 数据，其中 p∈{0,0\.5,1} 。


然后优化 PbRL 的 cross\-entropy loss：（我也不知道包含 p \= 0\.5 的 cross\-entropy loss 是不是这么写）


(1\)L\=−E(σ0,σ1,p)∼D\[p(0)log⁡Pψ\[σ0≻σ1]\+p(1)log⁡Pψ\[σ0≺σ1]\+ p(0\.5)log⁡Pψ\[σ0\=σ1]]Pψ\[σ0≻σ1]\=exp⁡∑tr^ψ(st0,at0)∑i∈{0,1}exp⁡∑tr^ψ(sti,ati)不知道为什么，LiRE 把 reward model 建模成了线性形式（而非指数形式）：


(2\)Pψ\[σ0≻σ1]\=∑tr^ψ(st0,at0)∑i∈{0,1}∑tr^ψ(sti,ati)LiRE 声称这样可以放大 learned reward model 的奖励值的差异，拉高比较好的 (s,a) 的奖励。这个线性 reward model 的最后一层 activator 也是 tanh，为确保概率（公式 2）是正的，reward model 的输出是 1 \+ tanh() 。


也可以使用 listwise loss，在 Appendix A.3，有点复杂 先不看了（）


## 5 experiment


### 5\.1 settings


* LiRE 的自定义 dataset：


	+ d4rl 存在问题，即使使用错误的 reward，训练出来结果也很好，因此 [不适用于 reward 学习](https://github.com) 。
	+ 因此，LiRE 对 metaworld 和 dmcontrol 收集了新的 medium\-replay 数据集，并使用了 IPL 的一部分 medium\-expert 数据集，细节见 Appendix C.2。
		- medium\-replay：用三个 seed 训 ground truth reward 下的 SAC，当 success rate 达到大概 50% 的时候，就停止训练，把 replay buffer 作为 offline dataset。
		- 然后，对每个数据集，他们验证了使用 0 reward、负 reward 训出来策略的性能不佳，因此适合评测 reward learning。
	+ 先前工作也有一些自定义数据集，但它们在这个数据集上的实验只使用了 100 个或更少的 feedback，而 LiRE 要使用 500 1000 这个数量级的 feedback。
* baselines：


	+ 马尔可夫奖励（MR）、[Preference Transformer](https://github.com)（PT），[OPRL](https://github.com)，Inverse Preference Learning（IPL）、Direct Preference\-based Policy Optimization（DPPO）、SeqRank。
	+ MR 是 PT 的 baseline 之一。PT 的主要贡献是把 reward model 换成 transformer，但是故事很合理。OPRL 的主要贡献是用类似 pebble 的方法选 disagreement 最大的 query。IPL 和 DPPO 没有 reward model，直接优化 policy。
* LiRE 的实现细节：


	+ 对于 LiRE，我们使用线性 reward model，并设置为每个 RLT 的 feedback 预算 Q 为 100：如果反馈的总数为 500，则将构造 5 个 RLT。所有的 offline RL 部分都使用 [IQL](https://github.com)。Appendix C.4 有超参数之类的具体细节（表 18）。
	+ preference 的 segment length \= 25。因为 metaworld 的 ground truth reward 在 \[0, 10] 之间，因此，LiRE 标记 segment reward 之和差异小于 12\.5 的 query 为 p\=0\.5。


### 5\.2 实验结果


* 实验主要在 LiRE 自己收集的 MetaWorld medium\-replay 上做。Meta\-World medium\-expert 在 Appendix A。
* LiRE 声称 PT 跟 MR 差不多；OPRL 因为最小化了（？）reward model 之间的 disagreement，所以性能会有提升；IPL 和 DPPO 基本比不上 MR；但 LiRE 结果很好。


### 5\.3 \& 5\.4 ablation


* LiRE 声称自己的性能提升主要因为 1\. 线性 reward model，2\. RLT。
	+ 表 3 显示，线性 reward model 可以有效提高性能（到底是为什么呢……）RLT 也能提高性能。
	+ 图 2 可视化了 reward model 预测的奖励和 ground truth 奖励的关系，发现无论是 MR 还是 LiRE，线性 reward model 都能得到更与 ground truth reward 线性相关的 reward 预测值，因此认为是更好的（怎么会有这么神奇的事情呢）。
	+ LiRE 推测，使用线性 reward model 和 RLT 可以更清楚地区分估计奖励的最佳和次优部分，还是在讲二阶偏好的故事。
	+ Appendix A.5 有线性 reward model 的更多实验。表 12 显示，MR 和 OPRL 换上线性 reward model 后性能都有提升，PT DPPO 性能则有所下降。图 7 声称 online PbRL 中线性 reward model 也可以表现更好。
* 图 3 做了不同 feedback 数量的 ablation。表 4 做了不同 Q（单个 RTL feedback 预算数量）的 ablation。
* 图 4 做了 noisy feedback，随机 filp preference 结果。表 5 6 比了 SeqRank。
* 图 5 改了给 p\=0\.5 的 reward 阈值。
* 图 6 把 LiRE 跟 OPRL 和 PT 相结合，发现性能有升有降。
	+ OPRL 性能下降是因为，基于 disagreement 的 query selection 可能会对相似的 segment pair 过度采样，这些片段可能很难区分。
	+ PT 的 motivation 是捕获奖励建模中的时间依赖关系，因此它似乎难以从 RLT 中准确捕获二阶偏好信息，可能因为对过去 segment 的过拟合。


### 5\.5 human 实验


* 表 7 在 button\-press\-topdown 用了作者打的 200 个 feedback，发现 LiRE 比 MR 和 SeqRank 好。


## 6 \& 7 conclusion


* LiRE 的 limitations：
	+ 一个 RLT 可能无法并行化地构建。
	+ LiRE 的 RLT 依赖于对 preference 的完备性 \+ 传递性假设。


