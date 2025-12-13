没问题！这是为你整理好的 **Markdown 格式复习表格**。你可以直接复制到 Notion、Obsidian 或 Typora 中，或者直接打印出来作为复习清单。

我把它分成了 **两部分**：

1.  **复习大纲表**（按模块分类，含 PPT 和习题指引）。
2.  **作弊纸专用表**（含公式速查和概念对比，适合考前突击）。

-----

### **第一部分：IFT3395/6390 终极复习清单 (Study Checklist)**

#### **模块一：神经网络核心引擎 (必考大题 ⭐⭐⭐⭐⭐)**

> **目标**：拿下反向传播推导题、SGD 更新题。

| 核心考点 | 详细子概念 (复习关键词) | 对应 PPT | 必做习题 (重点) |
| :--- | :--- | :--- | :--- |
| **反向传播推导**<br>(Backpropagation) | 1. **链式法则**: 复合求导<br>2. **矩阵维度**: $W^T$ 的位置<br>3. **误差传递**: $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot g'$ | **PPT 14**<br>(Slide 25-32) | **2015 Final Q3** (Autoencoder推导)<br>**2018 Final Q5** (MLP推导) |
| **激活函数**<br>(Activations) | 1. **Sigmoid**: $\frac{1}{1+e^{-z}}$, 导数 $y(1-y)$<br>2. **ReLU**: $\max(0,z)$, 导数 $1(z>0)$<br>3. **Softmax**: 多分类概率归一化 | **PPT 13**<br>(Slide 14-18) | **Lab 4** (Sigmoid实现)<br>**2018 Final Q5** |
| **损失函数**<br>(Loss Functions) | 1. **MSE**: 回归用，对应高斯噪声<br>2. **Cross-Entropy**: 分类用，对应 Softmax | **PPT 13**<br>(Slide 22-25) | **Midterm Q3** (MSE推导)<br>**Lab 4** |
| **SGD 梯度下降**<br>(Optimization) | 1. **更新公式**: $w \leftarrow w - \eta \nabla J$<br>2. **Mini-batch**: 伪代码逻辑<br>3. **Epoch**: 每一轮完整训练 | **PPT 15**<br>(Slide 5-10) | **Lab 4** (SGD手写代码)<br>**2018 Final Q3** (伪代码) |

#### **模块二：CNN 卷积神经网络 (送分计算 ⭐⭐⭐⭐)**

> **目标**：计算输出尺寸、参数量。

| 核心考点 | 详细子概念 (复习关键词) | 对应 PPT | 必做习题 (重点) |
| :--- | :--- | :--- | :--- |
| **CNN 算术**<br>(Arithmetic) | 1. **输出尺寸**: $O = (I - K + 2P)/S + 1$<br>2. **参数数量**: $(K^2 \times C_{in} + 1) \times C_{out}$<br>3. **组件**: Conv, Pooling, Stride | **PPT 16**<br>(Slide 20-30) | **2015 Final Q4** (参数计算)<br>**2018 Final Q4** (维度计算) |

#### **模块三：正则化与评估 (画图/简答 ⭐⭐⭐⭐)**

> **目标**：画 U 型曲线，解释过拟合。

| 核心考点 | 详细子概念 (复习关键词) | 对应 PPT | 必做习题 (重点) |
| :--- | :--- | :--- | :--- |
| **过拟合/欠拟合**<br>(Generalization) | 1. **Bias-Variance**: 偏差与方差权衡<br>2. **Learning Curves**: 画 U 型图 (Train/Val)<br>3. **Capacity**: 模型容量与复杂度 | **PPT 15**<br>(Slide 28-35) | **2015 Final Q2** (画图题)<br>**2018 Final Q2** (判断题) |
| **正则化方法**<br>(Regularization) | 1. **L2 (Ridge)**: Weight Decay (权重变小)<br>2. **L1 (Lasso)**: Sparsity (稀疏/特征选择)<br>3. **Early Stopping**: 验证集最低点停止<br>4. **Dropout**: 随机失活 | **PPT 15**<br>(Slide 40-50) | **Midterm Q3** (带L2的梯度)<br>**2018 Final Q1** (简答) |

#### **模块四：线性模型 & SVM (概念/推导 ⭐⭐⭐)**

> **目标**：理解 Margin，Hinge Loss。

| 核心考点 | 详细子概念 (复习关键词) | 对应 PPT | 必做习题 (重点) |
| :--- | :--- | :--- | :--- |
| **SVM & 核技巧**<br>(Kernel Methods) | 1. **Margin**: 最大化边界距离<br>2. **Hinge Loss**: $\max(0, 1-yf(x))$<br>3. **RBF Kernel**: 映射高维，高斯公式 | **PPT 12** | **Lab 5** (Hinge Loss代码) |

#### **模块五：概率与无监督 (扫盲 ⭐⭐)**

> **目标**：MLE=MSE 证明，K-Means 逻辑。

| 核心考点 | 详细子概念 (复习关键词) | 对应 PPT | 必做习题 (重点) |
| :--- | :--- | :--- | :--- |
| **概率与无监督**<br>(Prob & Unsupervised) | 1. **MLE**: 高斯假设下等价于 MSE (必背)<br>2. **K-Means**: 聚类步骤<br>3. **PCA**: 降维 (最大方差) | **PPT 11**<br>**Lab 3** | **Midterm Q4** (MLE推导)<br>**2018 Final Q12** (采样伪代码) |

-----

### **第二部分：作弊纸专用表 (Cheat Sheet Tables)**

建议将这两张表直接抄在你的 A4 纸正反面。

#### **表 A：核心数学公式速查 (Side A - Math)**

| 场景 | 公式 (Formula) | 备注 (Note) |
| :--- | :--- | :--- |
| **线性回归梯度** | $\nabla J = 2 X^T (Xw - y)$ | 看见 $X$ 记得加 $T$ 放前面 |
| **SGD 更新规则** | $w_{new} \leftarrow w_{old} - \eta \cdot \nabla J$ | $\eta$ 是学习率 |
| **反向传播 (误差)** | $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot g'(a^{(l)})$ | 核心公式，$\odot$ 是逐元素乘 |
| **反向传播 (梯度)** | $\nabla_W J = \delta^{(l)} \cdot (h^{(l-1)})^T$ | 误差 $\times$ 上一层输出的转置 |
| **CNN 输出尺寸** | $O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$ | $I$:输入, $K$:核, $P$:填充, $S$:步长 |
| **CNN 参数量** | $\text{Params} = (K^2 \cdot C_{in} + 1) \cdot C_{out}$ | 别忘了 $+1$ (Bias) |
| **Sigmoid 导数** | $g'(z) = g(z)(1 - g(z))$ | 必背 |
| **Softmax + CE Loss** | $\nabla_z J = \hat{y} - y$ | 预测概率 - 真实标签 |

#### **表 B：概念对比与关键词速查 (Side B - Concepts)**

| 对比项 | 概念 A | 概念 B | 关键区别 (Key Difference) |
| :--- | :--- | :--- | :--- |
| **MLE vs MSE** | **MLE** (概率最大化) | **MSE** (误差最小化) | 高斯噪声假设下，两者等价 |
| **L1 vs L2** | **L1 (Lasso)** | **L2 (Ridge)** | L1 产生稀疏解 (0); L2 让权重变小 |
| **过拟合 (Overfit)** | High Variance | Low Bias | Train好, Val差; 模型太复杂 |
| **欠拟合 (Underfit)** | Low Variance | High Bias | Train差, Val差; 模型太简单 |
| **Generative** | Naive Bayes | - | 建模联合概率 $P(x,y)$ |
| **Discriminative** | Logistic Reg, SVM | - | 建模条件概率 $P(y\|x)$ |
| **K-Means** | 无监督聚类 | - | 需指定 K; 对初始值敏感 |
| **SVM** | Hinge Loss | Max Margin | 只关心支持向量 (边界附近的点) |
