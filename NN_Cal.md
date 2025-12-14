# Softmax + Cross-Entropy 
这是一份整理好的 Markdown 文档。我已经优化了 LaTeX 数学公式的排版，使用了代码块和层级标题，使其看起来更加清晰、专业。
## 0. 场景设定 (Setup)


### 网络结构
* **输入层 ($x$)**：2 个神经元
* **隐藏层 ($h$)**：2 个神经元 (激活函数使用 ReLU)
* **输出层 ($z$)**：2 个神经元 (对应 2 分类，使用 Softmax)
* **Batch Size**：1 (单样本 SGD)

### 输入数据
* 输入：$x = [1, 2]^T$
* 真实标签：$y = [1, 0]^T$ (Class 1 是真的)

### 初始权重 (随机初始化)
偏置 $b$ 暂时设为 0 以简化计算。

* **$W^{(1)}$ (输入 $\to$ 隐藏):**
    $$W^{(1)} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.1 \end{bmatrix}$$

* **$W^{(2)}$ (隐藏 $\to$ 输出):**
    $$W^{(2)} = \begin{bmatrix} 0.6 & 0.3 \\ 0.4 & 0.5 \end{bmatrix}$$

---

## 阶段一：前向传播 (Forward Pass)

**目标**：算出预测值 $a$ 和 损失 $L$。

### 1. 第一层 (Hidden Layer)
**线性计算**: $z^{(1)} = W^{(1)}x$
$$
z^{(1)} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 0.5(1) + 0.2(2) \\ 0.1(1) + 0.1(2) \end{bmatrix} = \begin{bmatrix} 0.9 \\ 0.3 \end{bmatrix}
$$

**激活 (ReLU)**: $h = \max(0, z^{(1)})$
由于 $z^{(1)}$ 均为正数，激活后数值不变：
$$
h = \begin{bmatrix} 0.9 \\ 0.3 \end{bmatrix}
$$

### 2. 第二层 (Output Layer)
**线性计算**: $z^{(2)} = W^{(2)}h$
$$
z^{(2)} = \begin{bmatrix} 0.6 & 0.3 \\ 0.4 & 0.5 \end{bmatrix} \begin{bmatrix} 0.9 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 0.6(0.9) + 0.3(0.3) \\ 0.4(0.9) + 0.5(0.3) \end{bmatrix} = \begin{bmatrix} 0.63 \\ 0.51 \end{bmatrix}
$$
得到的 Logits 是 $z^{(2)} = [0.63, 0.51]^T$。

**Softmax 激活**:
* $e^{0.63} \approx 1.878$
* $e^{0.51} \approx 1.665$
* Sum $= 1.878 + 1.665 = 3.543$

计算概率 $a$:
* $a_1 = 1.878 / 3.543 \approx \mathbf{0.53}$
* $a_2 = 1.665 / 3.543 \approx \mathbf{0.47}$

最终预测:
$$a^{(2)} = \begin{bmatrix} 0.53 \\ 0.47 \end{bmatrix}$$

### 3. 计算 Loss (Cross-Entropy)
$$
L = -\sum y \log a = - (1 \cdot \log(0.53) + 0 \cdot \log(0.47)) \approx \mathbf{0.635}
$$

---

## 阶段二：反向传播 (Backpropagation)

**目标**：算出 $W^{(2)}$ 和 $W^{(1)}$ 的梯度。
**核心思想**：计算误差项 (Delta $\delta$)，然后像接力棒一样往回传。



### 1. 输出层的误差 ($\delta^{(2)}$)
对于 Softmax + Cross-Entropy 组合，导数有极简结论：
$$
\delta^{(2)} = a^{(2)} - y
$$
$$
\delta^{(2)} = \begin{bmatrix} 0.53 \\ 0.47 \end{bmatrix} - \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \mathbf{-0.47} \\ \mathbf{0.47} \end{bmatrix}
$$
*(直觉：第一个神经元预测少了，误差为负；第二个预测多了，误差为正)*

### 2. 计算 $W^{(2)}$ 的梯度
**公式**：$\nabla W^{(2)} = \delta^{(2)} \cdot h^T$ (本层误差 $\times$ 上层输入)

$$
\nabla W^{(2)} = \begin{bmatrix} -0.47 \\ 0.47 \end{bmatrix} \cdot \begin{bmatrix} 0.9 & 0.3 \end{bmatrix} = \begin{bmatrix} -0.423 & -0.141 \\ 0.423 & 0.141 \end{bmatrix}
$$

### 3. 误差传回隐藏层 ($\delta^{(1)}$)
这是有隐藏层时最关键的一步。
**公式**：$\delta^{(1)} = (W^{(2)})^T \delta^{(2)} \odot g'(z^{(1)})$

* **第一部分：加权反传** $(W^{(2)})^T \delta^{(2)}$
    $$
    \begin{bmatrix} 0.6 & 0.4 \\ 0.3 & 0.5 \end{bmatrix} \begin{bmatrix} -0.47 \\ 0.47 \end{bmatrix} = \begin{bmatrix} 0.6(-0.47) + 0.4(0.47) \\ 0.3(-0.47) + 0.5(0.47) \end{bmatrix} = \begin{bmatrix} -0.094 \\ 0.094 \end{bmatrix}
    $$

* **第二部分：激活函数导数** $g'(z^{(1)})$
    我们用的是 ReLU。因为 $z^{(1)} = [0.9, 0.3]$ 都是正数，导数均为 1。

* **结果**：
    $$
    \delta^{(1)} = \begin{bmatrix} -0.094 \\ 0.094 \end{bmatrix} \odot \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.094 \\ 0.094 \end{bmatrix}
    $$

### 4. 计算 $W^{(1)}$ 的梯度
**公式**：$\nabla W^{(1)} = \delta^{(1)} \cdot x^T$ (本层误差 $\times$ 上层输入)

$$
\nabla W^{(1)} = \begin{bmatrix} -0.094 \\ 0.094 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} -0.094 & -0.188 \\ 0.094 & 0.188 \end{bmatrix}
$$

---

## 阶段三：参数更新 (Update)

假设学习率 $\eta = 0.1$。
更新公式：$W_{new} = W_{old} - \eta \cdot \nabla W$

### 更新 $W^{(2)}$
$$
W^{(2)}_{new} = \begin{bmatrix} 0.6 & 0.3 \\ 0.4 & 0.5 \end{bmatrix} - 0.1 \times \begin{bmatrix} -0.423 & -0.141 \\ 0.423 & 0.141 \end{bmatrix} = \begin{bmatrix} \mathbf{0.64} & \mathbf{0.31} \\ \mathbf{0.36} & \mathbf{0.49} \end{bmatrix}
$$

### 更新 $W^{(1)}$
$$
W^{(1)}_{new} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.1 \end{bmatrix} - 0.1 \times \begin{bmatrix} -0.094 & -0.188 \\ 0.094 & 0.188 \end{bmatrix} = \begin{bmatrix} \mathbf{0.51} & \mathbf{0.22} \\ \mathbf{0.09} & \mathbf{0.08} \end{bmatrix}
$$

# Sigmoid + Cross-Entropy 

这是一份整理好的 Markdown 文档。重点突出“非标准搭配”下的求导细节和梯度消失问题，适合作为考试复习笔记。

***

# 2. MSE 搭配 Sigmoid (二分类) 的计算实例

既然作业涉及到了这个组合，我们就以 **Sigmoid + MSE** 为例。这是考试中最可能出现的“非标准搭配”，也是考察你是否真正理解反向传播链式法则的经典陷阱。

## 0. 场景设定 (Setup)

* **模型**：单层神经元（逻辑回归结构），但强行用 MSE 做 Loss。
* **输入**：$x = 2$
* **真实标签**：$y = 1$
* **参数**：$w = 0.5, b = 0$
* **激活函数**：Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$
* **损失函数**：MSE $L = \frac{1}{2}(a - y)^2$
    *(注意：为了求导方便，MSE 前面通常乘 1/2)*

---

## 阶段一：前向传播 (Forward)

### 1. 线性计算 ($z$)
$$
z = w \cdot x + b = 0.5 \times 2 + 0 = 1.0
$$

### 2. 激活输出 ($a$)
$$
a = \sigma(1.0) = \frac{1}{1+e^{-1}} \approx \mathbf{0.731}
$$

### 3. 计算 Loss
$$
L = \frac{1}{2}(0.731 - 1)^2 = \frac{1}{2}(-0.269)^2 \approx \mathbf{0.036}
$$

---

## 阶段二：反向传播 (Backward) —— 关键考点 

我们要算 $\frac{\partial L}{\partial w}$。根据链式法则：
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

### Step 1: Loss 对 $a$ 求导
$$
L = \frac{1}{2}(a - y)^2 \implies \frac{\partial L}{\partial a} = (a - y)
$$
**数值**：
$$0.731 - 1 = \mathbf{-0.269}$$

### Step 2: 激活函数 $\sigma$ 对 $z$ 求导 (致命差异点)
不同于 Cross-Entropy 里的导数抵消（直接得到 $a-y$），这里 **必须保留 Sigmoid 的导数**！
$$
\frac{\partial a}{\partial z} = \sigma'(z) = a(1 - a)
$$
**数值**：
$$0.731 \times (1 - 0.731) = 0.731 \times 0.269 \approx \mathbf{0.197}$$

> **Tutor 笔记 (重点)**：
> 看！这就是问题所在。如果 $a$ 接近 0 或 1（预测非常有信心），这项 $a(1-a)$ 会趋近于 0，导致 **梯度消失 (Gradient Vanishing)**。即使模型预测完全错误，梯度也会非常小，导致学不动。而 Cross-Entropy 正好消掉了这一项。

### Step 3: 计算误差项 $\delta$
$$
\delta = \frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = (a - y) \cdot a(1 - a)
$$
**数值**：
$$(-0.269) \cdot 0.197 \approx \mathbf{-0.053}$$

### Step 4: 计算 $w$ 的梯度
$$
\nabla w = \delta \cdot x
$$
**数值**：
$$-0.053 \times 2 = \mathbf{-0.106}$$

---

## 总结对比表 (Cheat Sheet 素材)

| 特性 | MSE + Sigmoid | Cross-Entropy + Sigmoid |
| :--- | :--- | :--- |
| **Loss 公式** | $\frac{1}{2}(a-y)^2$ | $-[y\ln a + (1-y)\ln(1-a)]$ |
| **误差项 $\delta$** | $(a-y) \cdot a(1-a)$ | $(a-y)$ |
| **计算复杂度** | 繁琐 (需额外算 sigmoid 导数) | 简单 (预测值直接减真实值) |
| **梯度消失问题** | **有** (当预测值接近 0 或 1 时，梯度趋近 0) | **无** (误差越大，梯度越大，修正越快) |
| **适用场景** | 回归问题 (Regression) | 分类问题 (Classification) |

# SVM Hinge Loss + L2 Regularisation
这也是一份整理好的 Markdown 文档。重点在于清晰展示 SVM 的**梯度更新逻辑**（即 Pegasos 算法），这是考试计算题的高频考点。

-----

# SVM 的黄金搭档：L2 正则化 + Hinge Loss

## 为什么是这两个？

  * **Hinge Loss ($\max(0, 1 - y f(x))$)**：负责 **“推”**。
    它只关心那些在边界附近或者分错的点（**支持向量**）。对于分得很好的点（即 $y f(x) \ge 1$），它的 Loss 为 0，梯度也为 0。

  * **L2 正则化 ($\frac{1}{2}\|w\|^2$)**：负责 **“扩”**。
    数学上可以证明，最小化 $\|w\|^2$ 等价于 **最大化几何间隔 (Geometric Margin)**。

### SVM 的优化目标函数 (Objective Function)

标准形式如下：

$$
J(w,b) = \underbrace{\frac{\lambda}{2} \|w\|^2}_{\text{最大化间隔}} + \underbrace{\frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i(w^T x_i + b))}_{\text{最小化分类错误}}
$$

-----

## 计算实例 (SVM Gradient Descent)

**考试提示**：考试中不太可能让你手算 SVM 的对偶问题（Dual Problem）或二次规划（QP），但非常可能考你 SVM 的梯度下降（即 **Pegasos 算法**）。

### 0\. 场景设定 (Setup)

  * **任务**：二分类 ($y \in \{+1, -1\}$)
  * **输入数据**：
      * 样本 1 (正类): $x_1 = [1, 2]^T, y_1 = +1$
      * 样本 2 (负类): $x_2 = [2, 1]^T, y_2 = -1$
  * **当前权重**：$w = [0.4, 0.4]^T$ (忽略偏置 $b$ 以简化计算)
  * **正则化系数**：$\lambda = 1$
  * **损失函数**：$J = \frac{1}{2} \|w\|^2 + \sum \max(0, 1 - y_i w^T x_i)$

### 阶段一：前向计算 Loss (Forward)

我们要检查每个点是否“违反”了边界（即是否进入了 Margin 内，或分错了）。
**判断标准**：计算 $y_i (w^T x_i)$。如果 $<1$，就有 Loss。

**1. 检查样本 1:**
$$w^T x_1 = 0.4(1) + 0.4(2) = 1.2$$
$$y_1 (w^T x_1) = 1 \times 1.2 = 1.2$$

  * **判定**：$1.2 \ge 1$。这个点分得很对，而且在间隔之外。**安全！**
  * **Hinge Loss**：$\max(0, 1 - 1.2) = 0$

**2. 检查样本 2:**
$$w^T x_2 = 0.4(2) + 0.4(1) = 1.2$$
$$y_2 (w^T x_2) = -1 \times 1.2 = -1.2$$

  * **判定**：$-1.2 < 1$。这简直是大错特错（预测是正的，标签是负的）。
  * **Hinge Loss**：$\max(0, 1 - (-1.2)) = 1 + 1.2 = 2.2$

**3. 计算总目标函数 $J$:**

  * 正则项：$\frac{1}{2} \|w\|^2 = \frac{1}{2} (0.4^2 + 0.4^2) = 0.16$
  * 总 Loss：$J = 0.16 + (0 + 2.2) = \mathbf{2.36}$

### 阶段二：反向传播求梯度 (Backward)

这是考试的核心点：**Hinge Loss 的导数是分段的（Sub-gradient）。**

  * 如果点是安全的 ($y w^T x \ge 1$)：数据项梯度为 **0**。
  * 如果点是违反的 ($y w^T x < 1$)：数据项梯度为 **$-y \cdot x$**。
  * 正则项 $\frac{\lambda}{2} \|w\|^2$ 的梯度：**$\lambda w$**。

**计算本例的梯度 $\nabla J(w)$:**

1.  **正则项贡献** ($\lambda = 1$):
    $$1 \cdot [0.4, 0.4] = [0.4, 0.4]$$

2.  **数据项贡献**:

      * 样本 1 (安全)：贡献 $[0, 0]$
      * 样本 2 (违反)：贡献 $-y_2 \cdot x_2 = -(-1) \cdot [2, 1] = [2, 1]$

3.  **总梯度**:
    $$\nabla w = [0.4, 0.4] + [2, 1] = \mathbf{[2.4, 1.4]}$$

### 阶段三：参数更新 (Update)

假设学习率 $\eta = 0.1$。

$$
w_{new} = w - \eta \nabla w
$$

$$
w_{new} = \begin{bmatrix} 0.4 \\ 0.4 \end{bmatrix} - 0.1 \times \begin{bmatrix} 2.4 \\ 1.4 \end{bmatrix} = \begin{bmatrix} 0.4 - 0.24 \\ 0.4 - 0.14 \end{bmatrix} = \begin{bmatrix} \mathbf{0.16} \\ \mathbf{0.26} \end{bmatrix}
$$

> **直觉解释**：
> 样本 2 分错了，所以梯度里有一个巨大的 `[2, 1]`（样本 2 的方向）。这会强行把权重 $w$ 往样本 2 的反方向拽，试图修正错误。

-----

## Cheat Sheet 必抄：SVM 梯度伪代码

考试如果让你写 SVM 的 SGD 算法（即 Pegasos 算法），请直接抄这个：

```python
# SVM Stochastic Gradient Descent (Pegasos)
# 参数: X, y, w, lambda (正则系数), eta (学习率)

For each sample (x_i, y_i):
    # 1. 计算 Margin (函数间隔)
    condition = y_i * dot(w, x_i)
    
    # 2. 判断是否违反边界 (Loss > 0)
    # 注意是 < 1，不仅仅是 < 0
    If condition < 1:
        # 梯度 = 正则项梯度 + 数据项梯度
        # 数据项梯度是 -y * x
        grad = lambda * w - y_i * x_i
    Else:
        # 如果分类正确且在间隔外，只需要更新正则项
        grad = lambda * w
        
    # 3. 更新权重
    w = w - eta * grad
```

-----

## 总结：黄金搭档对比表

| 特性 | Neural Net (MLP) | SVM |
| :--- | :--- | :--- |
| **黄金搭档** | **Softmax + Cross-Entropy** | **L2 Reg + Hinge Loss** |
| **Loss 公式** | $-\sum y \log a$ | $\max(0, 1 - y f(x))$ |
| **梯度 (Error)** | $a - y$ <br>(所有点都贡献梯度) | $-y x$ <br>(只有分错的点和边界内的点贡献梯度) |
| **稀疏性** | **梯度稠密 (Dense)** | **梯度稀疏 (Sparse)** <br>(安全点不贡献梯度) |
| **核心思想** | 概率最大化 (MLE) | 间隔最大化 (Max Margin) |

# Batch size > 1 == Batch 中含有多个样本
这是一份整理好的 Markdown 文档。对于 IFT3395/6390 这类课程，**维度检查 (Shape Check)** 是做对矩阵求导题的唯一救命稻草。

-----

# Batch Size N=100 的详细计算过程 (Matrix Style)

## 0\. 维度设定 (The Dimensions)

为了不晕头转向，我们必须先死死盯住形状 (Shape)。

  * **Batch Size ($N$)**: 100
  * **输入特征 ($D_{in}$)**: 2 (例如 $x=[x_1, x_2]$)
  * **输出类别 ($D_{out}$)**: 3 (例如 3分类)

### 数据矩阵化

  * **输入矩阵 $X$**: 形状 **(100, 2)**。每一行是一个样本。
  * **真实标签 $Y$**: 形状 **(100, 3)**。每一行是一个 One-hot 向量。
  * **权重矩阵 $W$**: 形状 **(2, 3)**。(输入 2 $\to$ 输出 3)。
    *(注意：有些教材把 $W$ 写成 $(3, 2)$，那样公式就是 $Wx$；如果写成 $(2, 3)$，公式就是 $xW$。这里我们按 PyTorch 习惯 $X \cdot W$)*
  * **偏置向量 $b$**: 形状 **(1, 3)**。

-----

## Phase 1: 前向传播 (Forward - Matrix Style)

我们要一次性算出 100 个样本的预测结果。

### Step 1: 线性输出 (Logits)

**公式**：$Z = X \cdot W + b$

  * $X$: $(100, 2)$
  * $W$: $(2, 3)$
  * **矩阵乘法**: $(100, 2) \times (2, 3) \to \mathbf{(100, 3)}$
  * **加偏置**: $b$ 会利用 **广播机制 (Broadcasting)** 加到每一行上。
  * **结果 $Z$**: **(100, 3)**。每一行对应一个样本的 Logits。

### Step 2: Softmax 激活

**公式**：$A = \text{Softmax}(Z, \text{axis}=1)$

  * 对 $Z$ 的每一行做 Softmax。
  * **结果 $A$**: **(100, 3)**。每一行是该样本的预测概率分布。

### Step 3: 计算 Loss

**公式**：
$$J = -\frac{1}{N} \sum_{i=1}^{100} \sum_{k=1}^{3} Y_{ik} \log(A_{ik})$$

这就是算出所有 100 个样本的 Cross-Entropy，加起来，再 **除以 100**。得到一个**标量 (Scalar)** 数字。

-----

## Phase 2: 反向传播 (Backward - The Aggregation)

这里是核心！我们要算出 $W$ 的梯度。$W$ 只有 $2 \times 3 = 6$ 个参数，但我们有 100 个样本在给它提意见。我们需要把这 100 个意见汇总。

### Step 1: 计算误差矩阵 ($\Delta$)

回想单样本结论：$\delta = a - y$。在矩阵形式下，这依然成立！

**公式**：$\Delta = A - Y$

  * $A$: $(100, 3)$
  * $Y$: $(100, 3)$
  * **结果 $\Delta$**: **(100, 3)**。
      * 第 1 行是第 1 个样本的误差向量。
      * ...
      * 第 100 行是第 100 个样本的误差向量。

### Step 2: 计算权重的梯度 ($\nabla W$)

对于单个样本，梯度是 $\delta^T \cdot x$ (或 $x^T \cdot \delta$)。对于 Batch，我们使用矩阵乘法自动完成“加权求和”。

**公式**：

$$
\nabla W = \frac{1}{N} (X^T \cdot \Delta)
$$

**让我们看看维度发生了什么**：

1.  $X^T$: 形状变为了 **(2, 100)**。
2.  $\Delta$: 形状是 **(100, 3)**。
3.  **矩阵乘法**: $(2, 100) \times (100, 3) \to \mathbf{(2, 3)}$。
    *看！形状变回了 (2, 3)，这正是 $W$ 的形状！*
4.  **除以 $N$**: $\frac{1}{100}$。因为 Loss 是平均值 (Mean)，所以梯度也是平均梯度。

> **物理意义**：
> 这个矩阵乘法 $X^T \cdot \Delta$ 实际上执行了这样的操作：
> $$\sum_{i=1}^{100} (\text{样本}_i \text{的特征} \times \text{样本}_i \text{的误差})$$
> 它自动把 100 个样本对权重的“修改意见”加在了一起。

### Step 3: 计算偏置的梯度 ($\nabla b$)

对于偏置，我们需要把 100 个样本的误差直接加起来。

**公式**：

$$
\nabla b = \frac{1}{N} \sum_{i=1}^{100} \Delta_i
$$

  * $\Delta$ 是 $(100, 3)$。
  * 沿着 `axis=0` (列) 求和。
  * 结果是 **(1, 3)**。
  * 同样不要忘记除以 $N$。

-----

## 总结：作弊单上的“矩阵版”公式

如果考试问你 Batch 计算，或者写伪代码，请写这个版本，它比 `for` 循环高级得多：

```python
# Matrix/Batch Implementation

# 1. Forward
Z = X @ W + b               # Shape: (N, D_out)
A = Softmax(Z)              # Shape: (N, D_out)

# 2. Backward Error
Delta = A - Y               # Shape: (N, D_out)

# 3. Gradients (Don't forget 1/N !)
Grad_W = (1/N) * (X.T @ Delta)    # Shape: (D_in, D_out)
Grad_b = (1/N) * Sum(Delta, axis=0)
```

> **一句话总结 Batch 过程**：
> 算出 100 个 $a-y$，把它们和对应的 $x$ 乘起来，最后取平均。
