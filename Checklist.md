# IFT3395/6390 Final Exam Review Cheat Sheet

> **Note:** 包含必考大题、易错概念、伪代码、以及考前增补的非参数/决策树模块。

-----

## 第一部分：必考大题与推导 (The "Must-Haves")

> **策略：** 这部分是拿分的硬通货，必须死记硬背推导步骤。

### 1\. 证明 MLE 等价于 MSE (高斯噪声假设) ⭐⭐⭐⭐⭐

  * **题目类型：** 证明题 / 简答题 (Midterm Q4 变种)
  * **问题：** 证明在线性回归中，假设噪声服从高斯分布，最大化似然 (MLE) 等价于最小化均方误差 (MSE)。

**证明步骤 (直接抄写)：**

1.  **假设 (Assumption)：**
    $$y^{(i)} = w^T x^{(i)} + \epsilon^{(i)}$$
    其中噪声 $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$。
    这意味着：
    $$P(y^{(i)}|x^{(i)}; w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2}\right)$$

2.  **写出似然函数 (Likelihood)：**
    $$L(w) = \prod_{i=1}^N P(y^{(i)}|x^{(i)}; w)$$

3.  **取对数似然 (Log-Likelihood)：**
    $$\ell(w) = \sum_{i=1}^N \log P(y^{(i)}|x^{(i)}; w)$$
    $$= \sum_{i=1}^N \left[ \log(\frac{1}{\sqrt{2\pi\sigma^2}}) - \frac{(y^{(i)} - w^T x^{(i)})^2}{2\sigma^2} \right]$$

4.  **最大化与最小化转化 (Optimization)：**
    Maximize $\ell(w)$ 等价于 Maximize $\sum - (y^{(i)} - w^T x^{(i)})^2$ (因为第一项是常数，第二项分母也是常数)。
    最大化负值 $\iff$ 最小化正值。
    $$\iff \text{Minimize } \sum_{i=1}^N (y^{(i)} - w^T x^{(i)})^2$$

5.  **结论：** 上式即为 MSE (均方误差)。证毕。

-----

### 2\. 神经网络反向传播 (Backpropagation) ⭐⭐⭐⭐⭐

  * **题目类型：** 计算题 / 推导题 (Devoir 4 / Final 2018 Q5)
  * **核心难点：** 矩阵维度的匹配（$W$ 何时转置）。

**符号定义：**

  * $L$: 损失函数
  * $W^{(l)}$: 第 $l$ 层的权重
  * $h^{(l-1)}$: 第 $l$ 层的输入 (即上一层的输出)
  * $z^{(l)} = W^{(l)} h^{(l-1)}$: 线性激活前
  * $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$: 第 $l$ 层的误差项 (Error term)

**推导流程 (背诵版)：**

1.  **输出层误差 ($\delta_{out}$):**
    假设 Softmax + CrossEntropy 或 Linear + MSE。
    $$\delta^{(L)} = \hat{y} - y \quad (\text{预测值} - \text{真实值})$$

2.  **倒传误差到隐层 ($\delta_{hidden}$):**
    **公式：** 后层误差 乘 后层权重转置 乘 激活导数
    $$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot g'(z^{(l)})$$
    *注：$\odot$ 是逐元素乘积 (Element-wise product)。*

3.  **计算权重梯度 ($\nabla_W J$):**
    **公式：** 本层误差 乘 上层输入转置
    $$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (h^{(l-1)})^T$$

4.  **计算偏置梯度 ($\nabla_b J$):**
    $$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

-----

### 3\. SGD 与 矩阵梯度推导 ⭐⭐⭐⭐⭐

  * **题目类型：** 伪代码 / 计算 (Midterm Q3)

#### A. 矩阵梯度公式 (必背)

  * **Loss:** $J(w) = \|Xw - y\|^2$
  * **Gradient:**
    $$\nabla_w J = 2X^T (Xw - y)$$
    **记忆法：** 看到 $X$ 在前面，求导时把 $X^T$ 提出来放在最左边。
  * **带正则化 (Ridge):** $J(w) = \|Xw - y\|^2 + \lambda \|w\|^2$
  * **Gradient:**
    $$2X^T(Xw - y) + 2\lambda w$$

#### B. SGD 伪代码 (Python Logic)

*(Midterm Q3 失分点修复)*

```python
# 参数: X (数据), y (标签), w (权重), eta (学习率), epochs
# N = 样本总数

For epoch = 1 to epochs:
    Shuffle(X, y)  # 1. 必须打乱!
    
    For i = 1 to N:  # 2. 逐个样本遍历 (Stochastic)
        xi = X[i]  # 形状 (d, 1)
        yi = y[i]  # 标量
        
        # 3. 计算预测和误差
        prediction = dot(w.T, xi)
        error = prediction - yi
        
        # 4. 计算梯度 (注意: 这里的梯度是针对单个样本的)
        # Loss = (w^T x - y)^2  =>  Grad = 2 * error * xi
        grad = 2 * error * xi 
        
        # 5. 更新权重
        w = w - eta * grad
```

-----

## 第二部分：易错概念辨析 (Concept Clarity)

> **策略：** 这部分主要应对简答题、判断题 (True/False) 和画图题。

### 1\. Bias (偏差) vs Variance (方差) ⭐⭐⭐⭐⭐

**必背口诀：**

  * 模型太简单 (学不会) $\to$ **High Bias** $\to$ Underfitting (欠拟合)
  * 模型太复杂 (死记硬背) $\to$ **High Variance** $\to$ Overfitting (过拟合)

**参数对 Bias/Variance 的影响表 (必考):**

| 算法 | 参数变化 | 对 Bias 的影响 | 对 Variance 的影响 | 模型复杂度 |
| :--- | :--- | :--- | :--- | :--- |
| **K-NN** | $k$ 变大 (1 $\to$ 100) | 升高 $\uparrow$ (欠拟合) | 降低 $\downarrow$ (更稳) | 变简单 |
| **K-NN** | $k$ 变小 (100 $\to$ 1) | 降低 $\downarrow$ | 升高 $\uparrow$ (过拟合) | 变复杂 |
| **决策树** | 深度 (Depth) 变大 | 降低 $\downarrow$ | 升高 $\uparrow$ | 变复杂 |
| **正则化** | $\lambda$ 变大 (强正则) | 升高 $\uparrow$ (欠拟合) | 降低 $\downarrow$ | 变简单 |
| **正则化** | $\lambda$ 变小 (弱正则) | 降低 $\downarrow$ | 升高 $\uparrow$ (过拟合) | 变复杂 |
| **神经网络** | 层数/节点 增加 | 降低 $\downarrow$ | 升高 $\uparrow$ | 变复杂 |

### 2\. L1 (Lasso) vs L2 (Ridge) 正则化 ⭐⭐⭐⭐

  * **L1 ($\lambda |w|$):**
      * **几何：** 等高线碰到菱形 (Diamond) 的尖角。
      * **效果：** 稀疏解 (Sparsity)。很多权重变成纯 0。
      * **用途：** 特征选择 (Feature Selection)。
  * **L2 ($\lambda \|w\|^2$):**
      * **几何：** 等高线碰到圆形 (Circle) 的边。
      * **效果：** 权重衰减 (Weight Decay)。权重变小，但不为 0。
      * **用途：** 防止过拟合，处理共线性。

### 3\. 生成式 (Generative) vs 判别式 (Discriminative) ⭐⭐⭐

  * **生成式:** 建模联合概率 $P(x, y)$。先学“在这个类里数据长什么样”。
      * *例子:* Naive Bayes, GMM, GANs.
  * **判别式:** 建模条件概率 $P(y|x)$。直接学“怎么区分这两个类”。
      * *例子:* Logistic Regression, SVM, Neural Networks, KNN.

-----

## 第三部分：计算公式与参数维度 (Arithmetic & Dimensions)

> **策略：** 应对填空题和计算小题。

### 1\. CNN 尺寸与参数计算 ⭐⭐⭐⭐

假设输入为 $W \times H$，卷积核 $K \times K$，填充 $P$，步长 $S$，输入通道 $C_{in}$，输出通道 $C_{out}$ (即核的数量)。

  * **输出尺寸 (Output Size):**
    $$O = \frac{W - K + 2P}{S} + 1$$
    *(必须向下取整 floor)*
  * **参数数量 (Number of Parameters):**
    $$\text{Params} = (K \times K \times C_{in} + 1) \times C_{out}$$
    *(注意：那个 $+1$ 是 Bias，每个 Filter 有一个 Bias。)*

### 2\. 模型参数维度速查表 ⭐⭐⭐

假设数据维度 $D$ (特征数)，类别数 $K$。

| 模型 | 参数构成 | 总参数量 (标量个数) |
| :--- | :--- | :--- |
| **线性回归** | 权重 $w$ ($D \times 1$) + 偏置 $b$ | $D + 1$ |
| **Logistic (2类)** | 同上 | $D + 1$ |
| **Softmax (K类)** | 权重 $W$ ($D \times K$) + 偏置 $b$ ($K$) | $(D + 1) \times K$ |
| **MLP (1隐层 H)** | 层1: $(D+1)\times H$, 层2: $(H+1)\times K$ | $(D+1)H + (H+1)K$ |

-----

## 第四部分：核心伪代码集锦 (Code Logic)

### 1\. Mini-batch SGD (带 L2 正则) ⭐⭐⭐⭐

```python
# Hyperparameters: eta (learning rate), lambda (regularization), batch_size B
Initialize w randomly

For epoch = 1 to E:
    Shuffle data (X, y)
    For each batch (X_b, y_b) of size B:
        # 1. Compute Gradient (Data term + Regularization term)
        # Gradient of MSE part: 2 * X^T * (Xw - y) / B
        prediction = dot(X_b, w)
        diff = prediction - y_b
        grad_data = 2 * dot(X_b.T, diff) / B
        
        # Gradient of Regularization part: 2 * lambda * w
        grad_reg = 2 * lambda * w
        
        total_grad = grad_data + grad_reg
        
        # 2. Update
        w = w - eta * total_grad
```

### 2\. K-Means 聚类 (Unsupervised) ⭐⭐⭐

```python
Initialize k centroids randomly
Repeat until convergence:
    # Step 1: Assignment (E-step)
    For each data point x:
        Find the nearest centroid c_j
        Assign x to cluster j
        
    # Step 2: Update (M-step)
    For each cluster j:
        Update centroid c_j = Mean of all points assigned to cluster j
```

-----

## 第五部分：考前最后 1 小时检查清单 (Safety Checklist)

  * **MLE 推导:** 看到 $\prod$ 能马上写出 $\sum \log$ 吗？常数能直接扔掉吗？
  * **矩阵转置:** 计算梯度 $2X^T(Xw-y)$ 时，记得 $X^T$ 在前面吗？反向传播时 $W^T$ 在后面吗？
  * **U 型曲线:** 会画 Train/Validation Error 随 Epochs 变化的图吗？知道哪里是 Overfitting 吗？
  * **Hinge Loss:** 知道 SVM 的 Loss 是 $\max(0, 1-y \cdot f(x))$ 吗？
  * **CNN 算术:** 给你 input=32, kernel=5, stride=1, padding=0，能算出 output=28 吗？
      * 计算：$(32 - 5 + 0)/1 + 1 = 28$。
  * **祝你考试顺利！这份 Cheat Sheet 覆盖了你 85% 以上的考点。**

-----

-----

# 考点预测 (Prediction Modules)

## 模块一：模型参数维度速查表 (Dimension & Parameter Count)

  * **预测题型：** 填空题 / 选择题
  * **问题模板：** 给定输入特征维度 $D$，输出类别 $K$，隐藏层单元 $H$，请问模型有多少个可学习参数？

| 模型 (Model) | 权重形状 (Weights) | 偏置形状 (Bias) | 总参数量公式 (Total Params) | 备注 (Note) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | $w \in \mathbb{R}^{D}$ | $b \in \mathbb{R}^1$ | $D + 1$ | 输出是标量 |
| **Logistic Regression** | $w \in \mathbb{R}^{D}$ | $b \in \mathbb{R}^1$ | $D + 1$ | 同上 |
| **Softmax Regression** | $W \in \mathbb{R}^{D \times K}$ | $b \in \mathbb{R}^K$ | $(D + 1) \times K$ | 每个类有一组 $w$ 和 $b$ |
| **Perceptron** | $w \in \mathbb{R}^{D}$ | $b \in \mathbb{R}^1$ | $D + 1$ | 同 Linear |
| **SVM (Primal Form)** | $w \in \mathbb{R}^{D}$ | $b \in \mathbb{R}^1$ | $D + 1$ | 原始形式参数量固定 |
| **Naive Bayes** | $\mu, \sigma^2 \in \mathbb{R}^{D \times K}$ | Prior $\in \mathbb{R}^K$ | $2DK + K$ | 每个类的每个特征都有均值方差 |
| **MLP (1 Hidden Layer)** | $W_1: D \times H$ | $b_1: H, b_2: K$ | $(D+1)H + (H+1)K$ | 必考计算 |

## 模块二：激活函数 (Activations) —— 必背公式与导数

  * **预测题型：** 推导题的一部分 / 简答题 (为什么用 ReLU？)

| 函数 | 公式 g(z) | 导数 g′(z) | 核心考点 (简答/判断) |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $g(z)(1-g(z))$ | 缺点：梯度消失 (Vanishing Gradient)，输出不以0为中心。 |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - (g(z))^2$ | 优点：输出以0为中心 (Zero-centered)。缺点：仍有梯度消失。 |
| **ReLU** | $\max(0, z)$ | $1 (z>0), 0 (z\le 0)$ | 优点：计算快，无梯度消失。缺点：Dead ReLU。 |
| **Softmax** | $\frac{e^{z_i}}{\sum e^{z_j}}$ | - | 用途：多分类输出层，概率和为1。 |

## 模块三：经典算法 T/F 扫雷 (True/False Trap)

  * **预测题型：** 判断对错，并给出理由。

<!-- end list -->

1.  **SVM (支持向量机)**

      * Q: SVM 的决策边界只由支持向量 (Support Vectors) 决定？
      * **True。** 移除任何非支持向量的数据点，边界不会变。
      * Q: 引入松弛变量 $\xi$ (Slack Variable) 是为了解决非线性问题？
      * **False。** 是为了解决线性不可分（允许一点点错误，Soft Margin），非线性是通过 Kernel Trick 解决的。
      * Q: 高斯核 (RBF Kernel) 把数据映射到了无穷维空间？
      * **True。** 这是 Kernel Trick 的威力。

2.  **Decision Tree & Random Forest**

      * Q: 决策树越深，Bias 越高？
      * **False。** 树越深，模型越复杂，Bias 越低，Variance 越高 (过拟合)。
      * Q: Random Forest 通过 Bagging 主要是为了降低 Bias？
      * **False。** Bagging (Bootstrap Aggregating) 对多个高方差模型取平均，主要目的是降低 Variance (防过拟合)。
      * Q: 决策树的决策边界是平滑曲线？
      * **False。** 是轴平行 (Axis-aligned) 的阶梯状边界。

3.  **Linear & Logistic & Perceptron**

      * Q: Logistic Regression 的损失函数是凸函数 (Convex)？
      * **True。** 这就是为什么我们可以放心地用梯度下降找全局最优。
      * Q: Perceptron 算法在数据线性不可分时也能收敛？
      * **False。** 如果不可分，Perceptron 会一直震荡，永远停不下来。
      * Q: 线性回归有解析解 (Closed-form solution)？
      * **True。** $w = (X^T X)^{-1} X^T y$。

4.  **Naive Bayes**

      * Q: 朴素贝叶斯假设所有特征完全独立，这在现实中通常是真的？
      * **False。** 现实中很少成立，但该算法依然很有效 (Works surprisingly well)。
      * Q: 朴素贝叶斯是判别式模型？
      * **False。** 它是生成式模型 (Generative)，因为它对 $P(x|y)$ 和 $P(y)$ 建模。

## 模块四：简答题必背话术 (Short Answer Cheat Sheet)

如果考试问你以下“对比”或“为什么”，请直接套用这些话术：

  * **为什么需要非线性激活函数 (Non-linearity)？**
      * Ans: Without activation functions, a multi-layer Neural Network is mathematically equivalent to a single linear layer (Linear Regression), limiting its power to learn complex patterns.
      * *(没有非线性，多少层神经网络都等价于一层。)*
  * **Soft Margin SVM (软间隔) vs Hard Margin SVM (硬间隔)？**
      * Ans: Hard Margin requires data to be perfectly linearly separable (no errors allowed), which causes overfitting. Soft Margin allows some misclassification (using slack variables $\xi$) to improve generalization.
  * **Random Forest 为什么要随机采样特征 (Random feature selection)？**
      * Ans: To decorrelate the trees. If every tree uses the best feature, they will all look the same. Random features make trees different, making the average (voting) more robust.
      * *(为了让树与树之间不一样，这样取平均才有意义。)*
  * **L1 Regularization 为什么会产生稀疏解 (Sparsity)？**
      * Ans: Because the L1 penalty term ($\|w\|_1$) has sharp corners (diamond shape) at the axes. The optimization contours are likely to touch these corners, setting some weights exactly to zero.
      * *(关键词：Sharp corners, Diamond shape, Zero weights)*

**总结：如何把这些放进 Cheat Sheet？**
你的 Side B (概念面) 现在应该包含三个板块：右上角（Model Parameters 表格），中间（激活函数公式表），下方（易错概念对比）。加上之前整理的 Side A (SGD/Backprop 推导)，你的复习资料就非常完整了！

-----

# 终极补漏 (The "Gap Fillers" - Post-Analysis)

> **来源：** 这是一个关键的补漏！回顾了 Midterm Exam (期中考 Q2(a), Q5) 和 Homework 1 (Exo\_bonus\_1)，补充非参数方法和信息论。

## 模块六：非参数方法 (Histograms & Parzen Windows) ⭐⭐⭐⭐

  * **来源：** Midterm Q2(a), Q5; HW1 Q2, Q3
  * **考点：** 直方图的桶宽、Parzen 窗口的大小对 Bias/Variance 的影响（必考概念）。

| 算法 | 核心参数 | 过拟合 (Overfitting) | 欠拟合 (Underfitting) | 核心公式/概念 |
| :--- | :--- | :--- | :--- | :--- |
| **Histogram** | 桶宽 $h$ (Bin Width) | $h$ 太小 (太多桶，太噪) | $h$ 太大 (桶太少，太粗糙) | **维度灾难:** 维度 $d$ 增加，桶的数量需指数级增加 ($m^d$)，导致大部分桶是空的 (Empty Bins)。 |
| **Hard Parzen** | 窗口边长 $h$ | $h$ 太小 | $h$ 太大 | 像一个滑动的立方体，数里面有几个点。 |
| **Soft Parzen** | 核带宽 $\sigma$ (Sigma) | $\sigma$ 太小 (High Variance) | $\sigma$ 太大 (High Bias) | 使用高斯核 $K(x) \propto \exp(-\frac{\|x-x_i\|^2}{2\sigma^2})$ 进行平滑。 |
| **K-NN** | 邻居数 $k$ | $k$ 太小 ($k=1$) | $k$ 太大 ($k=N$) | 距离度量: L1 (Sum of abs), L2 (Euclidean). |

  * **简答题必背 (Curse of Dimensionality):**
      * **Q:** Why histograms fail in high dimensions?
      * **Answer:** Because the number of bins grows exponentially with dimension $d$. Data becomes extremely sparse, leaving most bins empty (density $\approx 0$).

## 模块七：决策树与信息论 (Decision Trees) ⭐⭐⭐⭐

  * **来源：** Exos\_bonus\_5 Q4
  * **考点：** 手算熵 (Entropy) 和 信息增益 (Information Gain)。

**1. 熵 (Entropy) 公式 (必背计算)**
衡量数据的混乱程度。对于二分类 ($p_+$ 正例比例, $p_-$ 负例比例)：
$$H(S) = - p_+ \log_2(p_+) - p_- \log_2(p_-)$$
*注：$0 \log 0 = 0$。如果全是一类，熵为 0（最纯）；如果各占一半，熵为 1（最乱）。*

**2. 信息增益 (Information Gain)**
衡量“切这一刀”能让数据变纯多少。
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

  * **口语记忆：** (切分前的熵) - (切分后各子节点熵的加权平均)。
  * **决策规则：** 选 IG 最大的特征来切分。

## 模块八：贝叶斯与分类器理论 (Bayes & Theory) ⭐⭐⭐

  * **来源：** Exo\_bonus\_2 Q4; Midterm Q1
  * **考点：** Bayes Error, Naive Bayes 假设。

**1. 贝叶斯最优分类器 (Bayes Optimal Classifier)**

  * **定义：** 如果我们知道真实的概率分布 $P(Y|X)$，那么预测 $\hat{y} = \text{argmax}_y P(y|x)$ 就是理论上最好的分类器。
  * **贝叶斯误差 (Bayes Error)：** 即使是最优分类器也有误差（由数据本身的噪声决定），这是 **不可约误差 (Irreducible Error)**。
  * **考点：** $R(f^*) \le R(g)$ 对任何分类器 $g$ 都成立。

**2. 朴素贝叶斯 (Naive Bayes)**

  * **核心假设：** 给定类别 $y$，所有特征 $x_1, \dots, x_d$ **相互独立**。
  * **公式：** $P(y|x_1\dots x_d) \propto P(y) \prod_{j=1}^d P(x_j|y)$。
  * **注意：** 如果是文本分类 (Bag of Words)，$P(x_j|y)$ 就是单词 $j$ 在类别 $y$ 中出现的频率。

## 模块九：集成学习 (Ensemble Methods) ⭐⭐⭐

  * **来源：** Exos\_bonus\_5 Q3 (Boosting)
  * **考点：** Bagging vs Boosting 的区别。

| 方法 | 代表算法 | 核心思想 | 解决什么问题？ |
| :--- | :--- | :--- | :--- |
| **Bagging** | Random Forest | **并行**训练很多强模型 (过拟合的树)，然后取平均/投票。 | 降低 **Variance** (防过拟合)。 |
| **Boosting** | AdaBoost / GBDT | **串行**训练很多弱模型 (欠拟合的树)，每个模型修正前一个的错误。 | 降低 **Bias** (提升精度)。 |

-----

### Cheat Sheet 空间管理建议

你的 Cheat Sheet 应该已经很满了。如果空间不够，请按以下策略删减：

1.  **删掉：** 具体的 K-Means 伪代码（逻辑简单，容易现场推）。
2.  **删掉：** SVM 的软间隔具体公式（记住 Hinge Loss 即可）。
3.  **保留：** Histogram/Parzen 的 Bias/Variance 表（因为你在 Midterm Q2(a) 丢分了，这里是陷阱）。
4.  **保留：** Entropy 计算公式（HW4 考了，计算题容易拿满分）。
5.  **保留：** 维度灾难（必考简答）。

现在这份 Cheat Sheet 真正涵盖了从期中前的基础理论到期中后的深度学习，无死角了！加油！
