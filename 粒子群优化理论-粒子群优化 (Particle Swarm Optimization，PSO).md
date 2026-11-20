## 粒子群优化理论 (Particle Swarm Optimization，PSO)

### 1.1 粒子群优化算法概述与基本原理

#### 1.1.1 优化问题的形式化描述
在工程控制与科学计算中，大量实际问题可归结为连续空间的参数寻优问题。考虑如下连续非线性极小化优化问题：
$$
\min_{\boldsymbol{x} \in \mathcal{S}} f(\boldsymbol{x})
\tag{p-1}
$$
式中：
- $\boldsymbol{x} = [x_1, x_2, \dots, x_d]^\mathrm{T} \in \mathbb{R}^d$ 为 $d$ 维待优化的决策变量向量（候选解）；
- $\mathcal{S} \subset \mathbb{R}^d$ 为可行搜索空间，通常定义为超矩形有界闭区域：
$$
\mathcal{S} = \prod_{j=1}^d [x_{j,\min}, x_{j,\max}] = \{ \boldsymbol{x} \in \mathbb{R}^d \mid x_{j,\min} \leqslant x_j \leqslant x_{j,\max}, \, j = 1, \dots, d \}
\tag{p-2}
$$
- $f: \mathcal{S} \to \mathbb{R}$ 为适应度评价函数（目标函数）。
![[粒子群优化理论-粒子群优化 (Particle Swarm Optimization，PSO)-2.png]]
对于复杂非凸、多峰值、不可导或黑盒系统，传统的梯度下降法容易陷入局部极小值，甚至因导数不存在而失效。在此背景下，基于群体智能的元启发式全局优化方法受到广泛关注。

#### 1.1.2 仿生学背景与群体智能哲学
粒子群优化（Particle Swarm Optimization, PSO）算法由 James Kennedy 和 Russell Eberhart 于 1995 年提出，其灵感源于对鸟群、鱼群等生物群体协作觅食与信息交互行为的仿真模拟。
>算法命名中，“群（Swarm）”的概念源于 M. M. Millonas 在研究人工生命（Artificial Life）模型时所确立的群体智能五项基本原则（邻近、品质、多样性反应、稳定性与适应性）；而“粒子（Particle）”则是对质点力学模型的抽象与折中：一方面将群体成员抽象为无质量、无体积的几何点以降低算法复杂度，另一方面赋予其速度与加速度等动力学属性以驱动解空间的轨迹演化。

在鸟群觅食过程中，单个个体（粒子）并不预先知晓全局最优解的确切位置，但通过以下两类信息的协同利用，能够高效汇聚至最优区域：
-  **个体认知经验（Cognitive Part）**：个体记忆并保留自身历史搜索过程中所历经的最优位置；
-  **社会群体经验（Social Part）**：群体内部保持分布式信息交互与共享，每个个体实时获取群体所探知到的全局最优位置。
![[粒子群优化理论-粒子群优化 (Particle Swarm Optimization，PSO)-3.png]]
通过个体认知与社会经验的动态权衡，每个粒子自主调整飞行速度与轨迹演化方向，从而驱动整个群体在解空间中向高适应度区域渐近收敛。

---

### 1.2 标准粒子群算法的数学模型与更新方程

#### 1.2.1 粒子的状态向量表示
设粒子群包含 $N$ 个粒子，群体索引集合为 $\mathcal{I} = \{1, 2, \dots, N\}$。在第 $k$ 次离散迭代步中，任意第 $i$ 个粒子（$i \in \mathcal{I}$）的状态由以下向量完全表征：
- **当前位置向量**：
$$
\boldsymbol{x}_{i,[k]} = [x_{i,1,[k]}, x_{i,2,[k]}, \dots, x_{i,d,[k]}]^\mathrm{T} \in \mathcal{S}
\tag{p-3}
$$
表示粒子在搜索空间中所处的坐标，即问题的一个候选解。

- **当前速度向量**：
$$
\boldsymbol{v}_{i,[k]} = [v_{i,1,[k]}, v_{i,2,[k]}, \dots, v_{i,d,[k]}]^\mathrm{T} \in \mathbb{R}^d
\tag{p-4}
$$
表示粒子单步位移的大小与运动方向。

- **个体历史最优位置（Personal Best, pbest）**：
$$
\boldsymbol{p}_{i,[k]} = [p_{i,1,[k]}, p_{i,2,[k]}, \dots, p_{i,d,[k]}]^\mathrm{T} \in \mathcal{S}
\tag{p-5}
$$
记录粒子 $i$ 自初始时刻至第 $k$ 步所遍历过的适应度最优位置。其状态更新逻辑为：
$$
\boldsymbol{p}_{i,[k]} = \begin{cases} 
\boldsymbol{x}_{i,[k]}, & \text{若 } f(\boldsymbol{x}_{i,[k]}) < f(\boldsymbol{p}_{i,[k-1]}) \\ 
\boldsymbol{p}_{i,[k-1]}, & \text{若 } f(\boldsymbol{x}_{i,[k]}) \geqslant f(\boldsymbol{p}_{i,[k-1]}) 
\end{cases}
\tag{p-6}
$$
初始条件设为 $\boldsymbol{p}_{i,[0]} = \boldsymbol{x}_{i,[0]}$。

- **群体全局最优位置（Global Best, gbest）**：
$$
\boldsymbol{g}_{[k]} = [g_{1,[k]}, g_{2,[k]}, \dots, g_{d,[k]}]^\mathrm{T} \in \mathcal{S}
\tag{p-7}
$$
在第 $k$ 次离散迭代中，从全体粒子所维护的个体历史最优解集合中，选取目标函数值最小（适应度最优）的位置作为当前群体的全局最优解：
$$
\boldsymbol{g}_{[k]} = \arg\min_{\displaystyle{\boldsymbol{p} \in \{ \boldsymbol{p}_{1,[k]}, \boldsymbol{p}_{2,[k]}, \dots, \boldsymbol{p}_{N,[k]} \}}} f(\boldsymbol{p})
\tag{p-8}
$$

#### 1.2.2 经典速度与位置更新方程
![[粒子群优化理论-粒子群优化 (Particle Swarm Optimization，PSO)-4.png]]
在标准惯性权重粒子群优化算法（Shi & Eberhart, 1998）中，第 $i$ 个粒子的速度与位置按如下代数方程递推更新：
- **速度更新方程**：
$$
\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-9}
$$
- **位置更新方程**：
$$
\boldsymbol{x}_{i,[k+1]} = \boldsymbol{x}_{i,[k]} + \boldsymbol{v}_{i,[k+1]}
\tag{p-10}
$$
式 (p-9) 与式 (p-10) 中各参数的物理意义与功能为：
**惯性项** $\omega \boldsymbol{v}_{i,[k]}$：
   - $\omega > 0$ 称为**惯性权重（Inertia Weight）**；
   - 表征粒子保持先前运动状态的惯性趋势。较小的 $\omega$ 增强算法在局部邻域的精细搜索能力（开发，Exploitation）；较大的 $\omega$ 则有助于跳出局部极值陷阱，扩展解空间的全局探索范围（探索，Exploration）。
   
**认知项**（个体记忆）$c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]})$：
   - $c_1 > 0$ 称为**认知学习因子（Cognitive Acceleration Constant）**；
   - 驱动粒子向个体历史最佳位置逼近，表征个体的自我探索机制与认知记忆。

 **社会项**（群体协同）$c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})$：
   - $c_2 > 0$ 称为**社会学习因子（Social Acceleration Constant）**；
   - 驱动粒子向群体全局历史最佳位置汇聚，表征个体间的信息共享与协同交互机制。

**随机扰动标量** $r_{1,[k]}, r_{2,[k]}$：
   - 在各步迭代中独立采样的标准均匀分布随机变量：$r_{1,[k]}, r_{2,[k]} \sim \mathcal{U}(0, 1)$；
   - 为粒子动力学引入随机扰动，打破规则确定性震荡，增强解空间的遍历度。

#### 1.2.3 边界处理与速度限幅
为保证算法运行的数值稳定性并确保解落在可行域内，需设置限幅约束。
**速度限幅（Velocity Clamping）**：用于抑制粒子因步长发散而逃逸出可行解空间，将各分量速度严格限制在区间 $[-v_{j,\max}, v_{j,\max}]$ 内：
$$
v_{i, j,[k+1]} = \begin{cases}
v_{j,\max}, & \text{若 } v_{i, j,[k+1]} > v_{j,\max} \\
-v_{j,\max}, & \text{若 } v_{i, j,[k+1]} < -v_{j,\max} \\
v_{i, j,[k+1]}, & \text{其他}
\end{cases}
\tag{p-11}
$$
在工程实现中，通常依据对应维度的搜索跨度按比例设定**限幅阈值**：
$$
v_{j,\max} = \alpha (x_{j,\max} - x_{j,\min})
\tag{p-12}
$$
经验比例系数常取 $\alpha \in [0.1, 0.2]$。

**位置边界约束处理（Boundary Handling）**：当粒子更新位置越出可行域 $\mathcal{S}$ 时，常用边界约束处理策略包括：
- **截断吸收策略（Absorbing Boundary）**：将超界坐标分量直接投影重置于边界极值处：
$$
x_{i, j,[k+1]} = \max \big( x_{j,\min}, \, \min(x_{i, j,[k+1]}, x_{j,\max}) \big)
\tag{p-13}
$$
- **镜像反射策略（Reflecting Boundary）**：粒子到达边界时发生镜像反弹，反向折回可行域内部并反转对应维度的速度方向。

---

### 1.3 粒子动力系统建模与收敛性分析
为揭示粒子在搜索过程中的轨迹运动规律，可将粒子的微观迭代过程抽象为一个**二阶离散时间线性动态系统**。

#### 1.3.1 一维二阶差分方程推导
在各维度完全解耦的条件下，考察任意**一维标量动态**（省略粒子与维度下标）。令 $p$ 为个体最优， $g$ 为全局最优。由速度更新方程 (p-9) 与位置更新方程 (p-10)，将速度方程代入位置方程：
$$
x_{[k+1]} = x_{[k]} + \omega v_{[k]} + c_1 r_1 (p - x_{[k]}) + c_2 r_2 (g - x_{[k]})
\tag{p-14}
$$
由于 $v_{[k]} = x_{[k]} - x_{[k-1]}$，将前一时刻位移差代入式 (p-14)，整理合并同类项，可得：
$$
x_{[k+1]} - (1 + \omega - \phi) x_{[k]} + \omega x_{[k-1]} = \phi p^*
\tag{p-15}
$$
式中：
$$
\phi = c_1 r_1 + c_2 r_2
\tag{p-16}
$$
$$
p^* = \frac{c_1 r_1 p + c_2 r_2 g}{c_1 r_1 + c_2 r_2}
\tag{p-17}
$$
式 (p-15) 表明：粒子的运动本质上是一个以 $p^*$ 为平衡点的**二阶受迫阻尼谐振子离散差分动力学系统**。

#### 1.3.2 状态空间转移矩阵与特征方程
为进一步分析粒子动力系统的极点分布与收敛边界，需将标量差分方程转化为控制理论中的标准状态空间形式。由于经典 PSO 算法遵循“先基于旧状态更新速度 $v_{[k+1]}$，再基于新速度累加得到新位置 $x_{[k+1]}$”的离散递推时序，选取粒子的位移与速度构成二维状态向量：
$$\boldsymbol{z}_{[k]} \triangleq \begin{bmatrix} x_{[k]} \\ v_{[k]} \end{bmatrix} \in \mathbb{R}^2$$
在分析其稳定性时，考虑局部最优尚未更新的名义定常工况（$p^* = \text{const}$），并通过对随机扰动项取数学期望：
$$
\bar{\phi} = \mathbb{E}[\phi] = \mathbb{E}[c_1 r_1 + c_2 r_2] = c_1 \mathbb{E}[r_1] + c_2 \mathbb{E}[r_2] = \frac{c_1 + c_2}{2}
\tag{p-18}
$$
构建**确定性名义系统**（剥离了随机噪声、扰动和不确定性之后的理想基准模型）。


##### 名义速度更新方程
在粒子群优化（PSO）算法中，任意一维解耦标量动力学在离散迭代步 $k$ 到 $k+1$ 的原始随机速度更新方程表述为：
$$
v_{[k+1]} = \omega v_{[k]} + c_1 r_{1,[k]} (p_{[k]} - x_{[k]}) + c_2 r_{2,[k]} (g_{[k]} - x_{[k]})
\tag{p-19a}
$$
式中，$x_{[k]} \in \mathbb{R}$ 与 $v_{[k]} \in \mathbb{R}$ 分别为粒子在第 $k$ 步的位置与速度状态变量；$p_{[k]}$ 与 $g_{[k]}$ 分别为个体历史最优位置与群体全局最优位置；$\omega > 0$ 为**惯性权重**；$c_1, c_2 > 0$ 为**加速常数**；$r_{1,[k]}, r_{2,[k]} \sim \mathcal{U}(0, 1)$ 为相互独立的标准均匀分布随机扰动序列。

展开右端含状态变量 $x_{[k]}$ 的括号项并按因式重新聚合，得：
$$
v_{[k+1]} = \omega v_{[k]} - (c_1 r_{1,[k]} + c_2 r_{2,[k]}) x_{[k]} + (c_1 r_{1,[k]} p_{[k]} + c_2 r_{2,[k]} g_{[k]})
\tag{p-19b}
$$
引入综合随机加速度参数 $\phi_{[k]}$ 与瞬时综合吸引子中心 $p^*_{[k]}$ 的代数定义：
$$
\begin{aligned}
\phi_{[k]} &\triangleq c_1 r_{1,[k]} + c_2 r_{2,[k]} \\
p^*_{[k]} &\triangleq \frac{c_1 r_{1,[k]} p_{[k]} + c_2 r_{2,[k]} g_{[k]}}{c_1 r_{1,[k]} + c_2 r_{2,[k]}} = \frac{c_1 r_{1,[k]} p_{[k]} + c_2 r_{2,[k]} g_{[k]}}{\phi_{[k]}}
\end{aligned}
\tag{p-19c}
$$
由此可得恒等代数关系：$c_1 r_{1,[k]} p_{[k]} + c_2 r_{2,[k]} g_{[k]} = \phi_{[k]} p^*_{[k]}$。将上述定义代回速度方程，随机速度更新方程可重写为：
$$
v_{[k+1]} = \omega v_{[k]} + \phi_{[k]} (p^*_{[k]} - x_{[k]})
\tag{p-19d}
$$
假定在算法局部收敛阶段的连续若干迭代步内，个体历史最优与群体历史最优尚未发生跳变，综合平衡点稳定于某一常数工作点，即：
$$
p^*_{[k]} \equiv p^* = \text{const}
\tag{p-19e}
$$
在第 $k$ 步离散演化中，粒子的当前位置 $x_{[k]}$ 与速度 $v_{[k]}$ 已由前序历史决定，属于确定性状态量；而随机序列 $r_{1,[k]}, r_{2,[k]} \sim \mathcal{U}(0, 1)$ 为当前迭代步独立生成的伪随机采样，因而综合随机加速度 $\phi_{[k]}$ 与系统当前状态变量 $x_{[k]}, v_{[k]}$ 具备统计独立性：
$$
\phi_{[k]} \text{ 独立于 } x_{[k]}, v_{[k]}
\tag{p-19f}
$$
在已知当前状态的条件下，对速度更新方程 (p-19d) 两端施加数学期望算子 $\mathbb{E}[\,\cdot\,]$，考察状态变量的均值动力学响应：
$$
\mathbb{E}[v_{[k+1]}] = \mathbb{E}\big[\omega v_{[k]} + \phi_{[k]} (p^* - x_{[k]})\big]
\tag{p-19g}
$$
根据数学期望算子的线性性质，将方程右端拆解展开：
$$
\mathbb{E}[v_{[k+1]}] = \omega \mathbb{E}[v_{[k]}] + \mathbb{E}[\phi_{[k]} p^*] - \mathbb{E}[\phi_{[k]} x_{[k]}]
\tag{p-19h}
$$
其中，各项分别表示：
- **惯性项**：当前速度 $v_{[k]}$ 为当前计算步已知的确定性状态，故 $\mathbb{E}[v_{[k]}] = v_{[k]}$；
- **外生吸引项**：常数标量 $p^*$ 依据期望算子的线性齐次性可提至算子外部，故 $\mathbb{E}[\phi_{[k]} p^*] = p^* \mathbb{E}[\phi_{[k]}] = \bar{\phi} p^*$；
- **状态反馈项**：由于当前位置 $x_{[k]}$ 为已知量，且随机变量 $\phi_{[k]}$ 与 $x_{[k]}$ 相互独立，依据独立随机变量乘积的期望性质有 $\mathbb{E}[\phi_{[k]} x_{[k]}] = \mathbb{E}[\phi_{[k]}] x_{[k]} = \bar{\phi} x_{[k]}$。
其中，期望系数 $\bar{\phi}$ 由标准均匀分布的期望值直接确定（即式 p-18）：
$$
\bar{\phi} \triangleq \mathbb{E}[\phi_{[k]}] = c_1 \mathbb{E}[r_{1,[k]}] + c_2 \mathbb{E}[r_{2,[k]}] = \frac{c_1 + c_2}{2}
\tag{p-19i}
$$
合并同类项，整理可得状态均值的演化方程为：
$$
\mathbb{E}[v_{[k+1]}] = -\bar{\phi} x_{[k]} + \omega v_{[k]} + \bar{\phi} p^*
\tag{p-19j}
$$
在动力系统稳定性分析中，名义确定性系统（Nominal System）直接采用状态变量的期望演化轨迹来表征其名义行为，略去外层期望算子符号，即可得到标量形式的**名义速度更新方程**：
$$
v_{[k+1]} = -\bar{\phi} x_{[k]} + \omega v_{[k]} + \bar{\phi} p^*
\tag{p-19}
$$
写成状态向量分量的线性组合与行向量（Row Vector）乘积形式：
$$
v_{[k+1]} = \begin{bmatrix} -\bar{\phi} & \omega \end{bmatrix} \begin{bmatrix} x_{[k]} \\ v_{[k]} \end{bmatrix} + \bar{\phi} p^*
\tag{p-19k}
$$

##### 名义位置更新方程
根据标准 PSO 位置更新方程 (p-10)，位置方程可表示为：
$$
x_{[k+1]} = x_{[k]} + v_{[k+1]}
\tag{p-20a}
$$
式中，位置更新所累加的是当前步已计算完成的后验速度 $v_{[k+1]}$，而非上一时刻的历史先验速度 $v_{[k]}$。将式 (p-19) 代入式 (p-20a)：
$$
x_{[k+1]} = x_{[k]} + \Big( -\bar{\phi} x_{[k]} + \omega v_{[k]} + \bar{\phi} p^* \Big)
\tag{p-20b}
$$
提取状态变量 $x_{[k]}$ 与 $v_{[k]}$ 并合并同类项，可得标量形式的**名义位置更新方程**：
$$
x_{[k+1]} = (1 - \bar{\phi}) x_{[k]} + \omega v_{[k]} + \bar{\phi} p^*
\tag{p-20}
$$
对应写成行向量乘积形式：
$$
x_{[k+1]} = \begin{bmatrix} 1 - \bar{\phi} & \omega \end{bmatrix} \begin{bmatrix} x_{[k]} \\ v_{[k]} \end{bmatrix} + \bar{\phi} p^*
\tag{p-20c}
$$
##### 状态空间模型方程
将式 (p-20) 与式 (p-19) 上下堆叠，则一阶离散状态空间形式 $\boldsymbol{z}_{[k+1]} = \boldsymbol{T} \boldsymbol{z}_{[k]} + \boldsymbol{B} u$ 为：
$$
\begin{bmatrix} x_{[k+1]} \\ v_{[k+1]} \end{bmatrix} = \begin{bmatrix} 1 - \bar{\phi} & \omega \\ -\bar{\phi} & \omega \end{bmatrix} \begin{bmatrix} x_{[k]} \\ v_{[k]} \end{bmatrix} + \begin{bmatrix} \bar{\phi} p^* \\ \bar{\phi} p^* \end{bmatrix}
\tag{p-21}
$$
其中系统状态转移矩阵（系统阵，亦常记作 $\boldsymbol{A}$）为：
$$
\boldsymbol{T} = \begin{bmatrix} 1 - \bar{\phi} & \omega \\ -\bar{\phi} & \omega \end{bmatrix}
\tag{p-22}
$$

##### 特征方程与极点模态分析
系统的稳定性完全取决于矩阵 $\boldsymbol{T}$ 的特征值。由 $\det(\lambda \boldsymbol{I}_2 - \boldsymbol{T}) = 0$ 可得特征多项式方程：
$$
\begin{aligned}
\det(\lambda \boldsymbol{I}_2 - \boldsymbol{T}) &= \det \begin{bmatrix} \lambda - (1 - \bar{\phi}) & -\omega \\ \bar{\phi} & \lambda - \omega \end{bmatrix} \\
&= [\lambda - (1 - \bar{\phi})](\lambda - \omega) - (-\omega)(\bar{\phi}) \\
&= \lambda^2 - (1 + \omega - \bar{\phi})\lambda + \omega = 0
\end{aligned}
\tag{p-23}
$$
该特征方程与直接对二阶差分方程 (p-15) 施加 $\mathcal{Z}$ 变换（或设系统模态通解形式为 $x_{[k]} = \lambda^k$）所导出的特征方程完全一致。求解该二次特征多项式方程，可得特征根（闭环极点）解析解为：
$$
\lambda_{1, 2} = \frac{(1 + \omega - \bar{\phi}) \pm \sqrt{(1 + \omega - \bar{\phi})^2 - 4\omega}}{2}
\tag{p-24}
$$
式中判别式定义为：
$$
\Delta \triangleq (1 + \omega - \bar{\phi})^2 - 4\omega
\tag{p-24a}
$$
依据特征方程 (p-23) 的韦达定理（Vieta's Formulas），两特征根之积由方程常数项决定：
$$
\lambda_1 \lambda_2 = \omega
\tag{p-24b}
$$
当判别式 $\Delta < 0$ 时，两根为一对共轭复数 $\lambda_{1,2} = a \pm b\mathrm{i}$，其实部与虚部分别为：
$$
a = \frac{1 + \omega - \bar{\phi}}{2}, \quad b = \frac{\sqrt{4\omega - (1 + \omega - \bar{\phi})^2}}{2}
\tag{p-24c}
$$
由复数模长平方定义：
$$
|\lambda|^2 = a^2 + b^2 = \left(\frac{1 + \omega - \bar{\phi}}{2}\right)^2 + \frac{4\omega - (1 + \omega - \bar{\phi})^2}{4} = \frac{4\omega}{4} = \omega = \lambda_1 \lambda_2
\tag{p-24d}
$$
在惯性权重的控制与物理设定下，自然满足 $\omega > 0$，故复特征根的模长严格为：
$$
|\lambda| = \sqrt{\lambda_1 \lambda_2} = \sqrt{\omega}
\tag{p-24e}
$$
根据离散动态系统李雅普诺夫稳定性理论，系统渐近稳定收敛的充要条件为所有特征根严格位于复平面单位圆内部（$|\lambda| < 1$），由此可导出**振荡收敛模态**下惯性权重的必然取值区间为 $0 < \omega < 1$。

判别式 $\Delta$ 决定了系统的运动模态响应特征：
- 欠阻尼振荡收敛模态（$\Delta < 0$）：特征根为一对共轭复数，在 $0 < \omega < 1$ 时模长严格满足 $|\lambda| = \sqrt{\omega} < 1$。粒子绕最优平衡点 $p^*$ 呈**衰减谐振荡收敛**，在空间中呈现多方向螺旋盘旋遍历，具有优秀的全局探索（Exploration）性能；
- 过阻尼指数收敛模态（$\Delta > 0$）：特征根为互异实数，粒子呈**单调非振荡指数收敛**，迅速向平衡点 $p^*$ 滑移（Exploitation）；
- 临界阻尼收敛模态（$\Delta = 0$）：系统具备无振荡下的最快几何衰减速率。


#### 1.3.3 Clerc-Kennedy 收缩因子定理
在标准 PSO 的早期探索中，若参数选取不当，粒子速度易出现几何级数发散的“群体爆炸”（Swarm Explosion）现象。Maurice Clerc 与 James Kennedy (2002) 通过引入**收缩因子（Constriction Factor）$\chi$**，为算法的收敛性建立了严格的动力系统解析框架。

标准速度更新方程为 (p-9)：
$$
\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-9}
$$
带收缩因子的速度更新方程表述为：
$$
\boldsymbol{v}_{i,[k+1]} = \chi \left[ \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]}) \right]
\tag{p-25}
$$
由式 (p-16) 与 (p-18) ， $\phi = c_1 r_1 + c_2 r_2$ 为随机变量，$\bar{\phi} = \dfrac{c_1 + c_2}{2}$ 为其期望值。而在 Clerc-Kennedy 解析框架中，定义**确定性总加速度增益**为两学习常数的代数和：
$$
\phi_{\text{total}} \triangleq c_1 + c_2
\tag{p-25a}
$$
式中，$\phi_{\text{total}}$ 为**确定的代数标量参数**，不包含随机变量亦不引入期望折半。

将式 (p-25) 展开并与标准速度方程 (p-9) 进行对比：
$$
\boldsymbol{v}_{i,[k+1]} = (\chi) \boldsymbol{v}_{i,[k]} + (\chi c_1) r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + (\chi c_2) r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-25b}
$$
这表明收缩因子法在动力学结构上等价于引入如下等效参数：
$$
\omega_{\text{eq}} \triangleq \chi, \quad c_1' \triangleq \chi c_1, \quad c_2' \triangleq \chi c_2
\tag{p-25c}
$$
从而在动力系统分析上将收缩因子模型与前述状态空间转移矩阵 $\boldsymbol{T}$ 建立了严格的代数对应与理论统一。


**定理 1.1 (Clerc-Kennedy 收敛定理)**  
当确定性总增益满足 $\phi_{\text{total}} = c_1 + c_2 > 4$ 时，若收缩因子 $\chi$ 按如下解析式进行配置，则系统状态转移矩阵的特征根严格位于复平面单位圆内部（即谱半径 $\rho(\boldsymbol{T}) < 1$），算法呈现严格渐近收敛：
$$
\chi = \frac{2}{\left| 2 - \phi_{\text{total}} - \sqrt{\phi_{\text{total}}^2 - 4\phi_{\text{total}}} \right|} = \frac{2}{\phi_{\text{total}} - 2 + \sqrt{\phi_{\text{total}}^2 - 4\phi_{\text{total}}}}
\tag{p-26}
$$

**经典标称参数设置**：通常取 $c_1 = c_2 = 2.05$，则 $\phi_{\text{total}} = 4.1 > 4$。代入式 (p-26) 计算：
$$
\chi = \frac{2}{4.1 - 2 + \sqrt{4.1^2 - 4 \times 4.1}} = \frac{2}{2.1 + \sqrt{0.41}} \approx 0.7298
\tag{p-27}
$$
对应的等效动力学参数为：
$$
\omega_{\text{eq}} \approx 0.7298, \quad c_1' = c_2' = \chi \times 2.05 \approx 0.7298 \times 2.05 \approx 1.4962
\tag{p-28}
$$
在此组标称参数下，算法具备内生阻尼收缩特性，理论上无需引入人为设定的速度硬限幅 $v_{\max}$ 即可严格保证粒子运动轨迹的渐近稳定性与收敛性。

#### 1.3.4 Shi-Eberhart 惯性权重收敛性区域准则
对于使用惯性权重 $\omega$ 的标准 PSO：
$$
v_{[k+1]} = \omega v_{[k]} + c_1 r_{1,[k]} (p_{[k]} - x_{[k]}) + c_2 r_{2,[k]} (g_{[k]} - x_{[k]})
\tag{p-19a}
$$
依据离散线性时不变控制系统的 **Jury 稳定性判据**，可严格解析求解使特征根严格位于复平面单位圆内部（$|\lambda| < 1$）的充要参数稳定域。

将系统特征多项式（式 p-23）按幂次重写为标准二阶形式：
$$
P(\lambda) = \lambda^2 - (1 + \omega - \bar{\phi})\lambda + \omega = a_2 \lambda^2 + a_1 \lambda + a_0 = 0
\tag{p-28a}
$$
式中各项系数分别为 $a_2 = 1$，$a_1 = -(1 + \omega - \bar{\phi})$，$a_0 = \omega$。

依据 Jury 代数判据，二阶多项式极点严格位于单位圆内的充分必要条件为：
$$
\begin{cases}
P(1) > 0 \\
P(-1) > 0 \\
|a_0| < a_2
\end{cases}
\tag{p-29a}
$$
逐项代入多项式系数求解：
-  **一阶和条件**：$P(1) = 1^2 - (1 + \omega - \bar{\phi})\cdot 1 + \omega = \bar{\phi} > 0$；
- **交错和条件**：$P(-1) = (-1)^2 - (1 + \omega - \bar{\phi})\cdot (-1) + \omega = 2(1 + \omega) - \bar{\phi} > 0 \implies \bar{\phi} < 2(1 + \omega)$；
- **常数模长条件**：$|a_0| < a_2 \implies |\omega| < 1 \implies -1 < \omega < 1$。结合惯性权重的物理意义与阻尼要求，限定 $0 < \omega < 1$。

联立上述三式，得到对名义均值期望参数 $\bar{\phi}$ 的原生收敛域约束：
$$
\begin{cases}
0 < \omega < 1 \\
0 < \bar{\phi} < 2(1 + \omega)
\end{cases}
\tag{p-29b}
$$
将名义系统均值期望定义 $\bar{\phi} = \dfrac{c_1 + c_2}{2}$（式 p-18）代入式 (p-29b)，两端同乘以 $2$：
$$
0 < \frac{c_1 + c_2}{2} < 2(1 + \omega) \implies 0 < c_1 + c_2 < 4(1 + \omega)
\tag{p-29c}
$$
可导出标准 PSO 加速度常数的 **Shi-Eberhart 收敛性区域准则**：
$$
\begin{cases}
0 < \omega < 1 \\
0 < c_1 + c_2 < 4(1 + \omega)
\end{cases}
\tag{p-29}
$$
Jury 判据原生推导针对一阶矩期望 $\bar{\phi}$ 的边界为 $2(1+\omega)$；由于随机数采样服从均匀分布 $\mathcal{U}(0, 1)$，其均值期望产生折半系数（$\mathbb{E}[r_1] = \mathbb{E}[r_2] = \dfrac{1}{2}$），还原为算法原始设定参数 $c_1 + c_2$ 时产生系数 $2$ 的倍增，从而严谨揭示了边界常数 $4(1+\omega)$ 的代数本源。

在经典常数惯性权重配置下，通常推荐设定为 $\omega \in [0.4, 0.9]$ 且 $c_1 = c_2 = 2.0$（此时满足 $c_1 + c_2 = 4 < 4(1+\omega)$，确保在均值意义下的收敛稳定域内）；若采用 Clerc 收缩因子法，则对应等效动力学参数 $\omega_{\text{eq}} \approx 0.7298$ 且 $c_1' = c_2' \approx 1.4962$。

---

### 1.4 群体拓扑结构与信息传递机制
在标准速度更新方程 (p-9) 中：
$$
\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-9}
$$
所有粒子均直接向全局最优解 $\boldsymbol{g}_{[k]}$ 学习，这种信息流动拓扑称为**全局拓扑（Global Best / gbest 拓扑）**。在实际算法设计中，拓扑结构直接决定了群体的信息交互模式与探索-开发权衡。

#### 1.4.1 常见群体拓扑结构
1. **星型拓扑（Star Topology，即 gbest 模型）**：
   - 拓扑特点：所有粒子均与全局最优粒子互联通信；
   - 动力学特性：信息传播速率最快，收敛速度极高；
   - 局限性：群体空间多样性衰减迅速，在复杂多峰地形下极易出现早熟收敛并陷入局部极值。

1. **环形拓扑（Ring Topology，即 lbest 模型）**：
   - 拓扑特点：每个粒子仅与环形排列中左右相邻的粒子通信；
   - 动力学特性：信息在环形网络中缓慢扩散，形成多个具有异质性的局部子群体；
   - 拓扑优势：能长时间维持群体的空间多样性，具备优异的跳出局部极值能力，但其全局信息协同效率与收敛速率相对平缓。

1. **冯·诺依曼拓扑（von Neumann Topology）**：
   - 拓扑特点：粒子排列为二维网格，与上下左右相邻的 4 个粒子互联；
   - 动力学特性：网络连接度与直径介于星型与环形之间，综合表现出优良的探索与收敛平衡性能。

---


### 1.5 经典改进策略与前沿变体

#### 1.5.1 线性递减惯性权重（LDW-PSO）
在算法优化初期，需要保持较强的全局探索能力以定位高潜力解区域；在优化后期，则需要增强局部精细开采能力以提升解的精度。为此，Shi 与 Eberhart 提出了**线性递减惯性权重**（Linearly Decreasing Weight, LDW）自适应调节机制：
$$
\omega_{[k]} = \omega_{\max} - \frac{k}{K_{\max}} (\omega_{\max} - \omega_{\min})
\tag{p-30}
$$
式中：
- $k$ 为当前迭代步数，$K_{\max}$ 为最大允许迭代代数；
- 经典工程经验参数通常取：$\omega_{\max} = 0.9, \omega_{\min} = 0.4$。

#### 1.5.2 时变加速度学习因子（TVAC-PSO）
Ratnaweera 等人提出了**时变加速度学习因子**（Time-Varying Acceleration Coefficients, TVAC）策略：
- 迭代初期配置较大的认知因子 $c_1$ 与较小的社会因子 $c_2$，引导粒子在全解空间开展自主广泛探索；
- 迭代后期逐步减小 $c_1$ 并增大 $c_2$，强化群体社会信息共享，引导粒子向全局最优快速聚集：
$$
c_{1,[k]} = c_{1,\max} - \frac{k}{K_{\max}} (c_{1,\max} - c_{1,\min})
\tag{p-31}
$$
$$
c_{2,[k]} = c_{2,\min} + \frac{k}{K_{\max}} (c_{2,\max} - c_{2,\min})
\tag{p-32}
$$
通常取 $c_{1,\max} = 2.5, c_{1,\min} = 0.5$，$c_{2,\min} = 0.5, c_{2,\max} = 2.5$。

#### 1.5.3 变异操作与抗早熟机制
针对粒子群在复杂解空间中易出现的早熟停滞（Premature Stagnation）与群体退化问题，变异机制是恢复群体多样性的关键手段：
**高斯变异 / 柯西变异（Cauchy Mutation）**：当群体空间分布多样性测度低于预设门限时，对当前全局最优位置 $\boldsymbol{g}_{[k]}$ 施加随机摄动：
$$
\boldsymbol{g}_{[k]}^* = \boldsymbol{g}_{[k]} + \eta \odot \mathcal{C}(0, 1)
\tag{p-33}
$$
式中 $\eta$ 为变异步长因子，$\mathcal{C}(0, 1)$ 为标准柯西分布随机变量。鉴于柯西分布具有显著的厚尾重尾特性，其产生大步长跳跃扰动的概率远高于高斯分布，可有效助力粒子摆脱局部极小吸引子阱的束缚。


---

### 1.6 标准粒子群算法实施流程

标准粒子群优化算法的完整计算步骤如下：

**初始化阶段**：
   - 设定群体规模 $N$、最大迭代代数 $K_{\max}$、惯性权重 $\omega$、学习因子 $c_1, c_2$；
   -  依式 (p-12) 计算速度限幅：
$$
v_{j,\max} = \alpha (x_{j,\max} - x_{j,\min})
\tag{p-12}
$$
   - 在可行域 $[x_{j,\min}, x_{j,\max}]$ 内均匀随机采样生成初始位置 $\boldsymbol{x}_{i,[0]}$ 与速度 $\boldsymbol{v}_{i,[0]}$；
   - 评估各粒子初始适应度值，初始化个体历史最优位置 $\boldsymbol{p}_{i,[0]} = \boldsymbol{x}_{i,[0]}$，并从中筛选出初始全局最优位置 $\boldsymbol{g}_{[0]}$。

**循环迭代阶段**（针对每一步 $k = 0, 1, \dots, K_{\max}-1$）：
   - 对每一个粒子 $i \in \{1, \dots, N\}$：
     - 生成均匀随机数 $r_1, r_2 \sim \mathcal{U}(0, 1)$；
     - 依式 (p-9) 更新速度 $\boldsymbol{v}_{i,[k+1]}$：
$$
\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-9}
$$
     - 按式 (p-11) 实施速度限幅：
$$
v_{i, j,[k+1]} = \begin{cases}
v_{j,\max}, & \text{若 } v_{i, j,[k+1]} > v_{j,\max} \\
-v_{j,\max}, & \text{若 } v_{i, j,[k+1]} < -v_{j,\max} \\
v_{i, j,[k+1]}, & \text{其他}
\end{cases}
\tag{p-11}
$$
     - 依式 (p-10) 更新位置 $\boldsymbol{x}_{i,[k+1]}$：
$$
\boldsymbol{x}_{i,[k+1]} = \boldsymbol{x}_{i,[k]} + \boldsymbol{v}_{i,[k+1]}
\tag{p-10}
$$
     - 并按式 (p-13) 实施边界约束处理：
$$
x_{i, j,[k+1]} = \max \big( x_{j,\min}, \, \min(x_{i, j,[k+1]}, x_{j,\max}) \big)
\tag{p-13}
$$
     - 评估新位置的适应度值 $f(\boldsymbol{x}_{i,[k+1]})$；
     - 依式 (p-6) 更新个体历史最优位置 $\boldsymbol{p}_{i,[k+1]}$：
$$
\boldsymbol{p}_{i,[k+1]} = \begin{cases} 
\boldsymbol{x}_{i,[k+1]}, & \text{若 } f(\boldsymbol{x}_{i,[k+1]}) < f(\boldsymbol{p}_{i,[k]}) \\ 
\boldsymbol{p}_{i,[k]}, & \text{若 } f(\boldsymbol{x}_{i,[k+1]}) \geqslant f(\boldsymbol{p}_{i,[k]}) 
\end{cases}
\tag{p-6}
$$
   - 遍历所有粒子，依式 (p-8) 更新全局最优位置 $\boldsymbol{g}_{[k+1]}$：
$$
\boldsymbol{g}_{[k+1]} = \arg\min_{\displaystyle{\boldsymbol{p} \in \{ \boldsymbol{p}_{1,[k+1]}, \boldsymbol{p}_{2,[k+1]}, \dots, \boldsymbol{p}_{N,[k+1]} \}}} f(\boldsymbol{p})
\tag{p-8}
$$
   - 依式 (p-30) **动态调节惯性权重** $\omega$：
$$
\omega_{[k]} = \omega_{\max} - \frac{k}{K_{\max}} (\omega_{\max} - \omega_{\min})
\tag{p-30}
$$

**终止准则**：
   - 若达到最大迭代代数 $K_{\max}$，或全局最优适应度变化量小于预设容差 $\epsilon$，则停止迭代，输出最优解 $\boldsymbol{g}^*$ 与对应最优适应度值 $f(\boldsymbol{g}^*)$。

#### 1.6.2 核心更新方程对照表

|    计算环节    |                                                                                      数学计算公式                                                                                      |
| :--------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **限幅阈值计算** |                                                       $v_{j,\max} = \alpha (x_{j,\max} - x_{j,\min})$                                                        |
|  **速度更新**  | $\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_1 (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_2 (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})$ |
|  **速度限幅**  |                                              $v_{i,j,[k+1]} = \operatorname{clip}(v_{i,j,[k+1]}, \, -v_{j,\max}, \, v_{j,\max})$                                               |
|  **位置更新**  |                                                  $\boldsymbol{x}_{i,[k+1]} = \boldsymbol{x}_{i,[k]} + \boldsymbol{v}_{i,[k+1]}$                                                  |
|  **边界处理**  |                                                    $x_{i,j,[k+1]} = \max(x_{j,\min}, \, \min(x_{i,j,[k+1]}, \, x_{j,\max}))$                                                     |
| **个体最优更新** |                                                        $f(\boldsymbol{x}_{i,[k+1]}) < f(\boldsymbol{p}_{i,[k]})$ 时执行状态更新                                                         |
| **全局最优更新** |                                               $\boldsymbol{g}_{[k+1]} = \arg\min_{i \in \mathcal{I}} f(\boldsymbol{p}_{i,[k+1]})$                                                |
| **惯性权重调节** |                                       $\omega_{[k]} = \omega_{\max} - \frac{k}{K_{\max}} (\omega_{\max} - \omega_{\min})$                                        |

---

### 1.7 算法特征

#### 1.7.1 算法优势
- **无导数依赖与强鲁棒性**：优化机制不依赖目标泛函的一阶导数或二阶海森（Hessian）矩阵信息，广泛适用于非凸、不连续、高阶非线性及黑盒工程系统的复杂优化问题；

- **超参数架构精炼**：核心配置参数主要为群体规模 $N$、惯性权重 $\omega$ 与学习因子 $c_1, c_2$，超参数调优维度相对精简；

- **天然契合高并发并行计算**：各粒子的状态更新与适应度评估在单步迭代内相互解耦，天然适配 GPU 硬件加速与多核多节点分布式并行计算架构。

#### 1.7.2 局限性与工程注意事项
- **高维维数灾难（Curse of Dimensionality）**：当决策变量维度 $d \gg 100$ 时，搜索空间体积随维度呈指数级膨胀，有限规模的粒子群在超高维稀疏空间中的采样密度急剧衰减，导致全局搜索效能显著下降；

- **早熟收敛与群体多样性停滞**：在迭代后期，粒子速度模长易迅速衰减并向局部极值吸引子塌缩，导致群体丧失解空间遍历与跳出局部陷阱的能力；

- **工程参数配置准则**：
	- 群体规模 $N$：常规中低维连续参数寻优问题，通常设定 $N \in [20, 50]$ 即可在收敛性能与计算开销之间取得良好均衡；
	- 惯性权重 $\omega$：推荐采用时变线性递减策略（如 $\omega_{\max}=0.9 \to \omega_{\min}=0.4$），实现前期全局探索与后期局部精细收敛的动态自适应；
	- 渐近收敛性保证：建议优先采纳 Clerc 收缩因子法（$\chi \approx 0.7298, c_1 = c_2 \approx 1.4962$），以先验数学稳定性消除发散风险并规避对速度限幅经验参数的依赖。


---

### 1.8 粒子群优化算法的 Python 实现
#### 1.8.1 通用向量化 PSO 优化器类实现
以下代码采用面向对象设计与 NumPy 向量化并行计算，消除了低效的多层循环，支持任意维度与超矩形边界约束，并提供线性递减惯性权重（LDW）和 Clerc 收缩因子双模式支持：
```python
# -*- coding: utf-8 -*-
"""
================================================================================
粒子群优化算法 (Particle Swarm Optimization, PSO) 核心实现与工程应用
================================================================================

本模块包含：
1. ParticleSwarmOptimizer: 通用向量化 PSO 优化器类（支持线性递减惯性权重 LDW 与 Clerc 收缩因子双模式）
2. rastrigin / run_rastrigin_example: 10 维 Rastrigin 高度非凸多峰基准测试函数全局寻优
"""

import sys  # 导入系统接口模块
import numpy as np  # 导入科学计算与矩阵运算基础库
from typing import Callable, Tuple, List, Optional  # 导入类型提示注解模块，增强代码可读性与健壮性

# 强制重设标准输出流编码为 UTF-8，彻底解决 Windows 控制台/IDE 终端中文乱码问题
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


class ParticleSwarmOptimizer:
    """
    通用向量化粒子群优化器 (Particle Swarm Optimizer, PSO)
    
    参数说明:
        fitness_func : 目标适应度函数，输入为一维决策向量 (d,)，返回标量适应度值 (float)
        dim          : 优化决策变量的维度 d (int)
        bounds       : 搜索空间超矩形上下界元组 (lb, ub)，形式为 ([x1_min, ...], [x1_max, ...])
        n_particles  : 粒子群落规模 N (默认 40)
        max_iter     : 最大允许迭代代数 K_max (默认 150)
        w_mode       : 惯性权重调节模式，可选 'ldw' (线性递减), 'const' (常数), 'constriction' (收缩因子)
        w            : 基础惯性权重或收缩因子 (默认 0.7298，来源于 Clerc-Kennedy 收缩因子解析解)
        w_min, w_max : 线性递减权重上下限 (默认 0.4 与 0.9)
        c1, c2       : 认知学习因子与社会学习因子 (默认 1.4962，与收缩因子 0.7298 对应匹配)
        v_ratio      : 速度限幅占搜索空间跨度的比例 (默认 0.15，即单步最大跨度不超过可行域跨度的 15%)
        seed         : 随机数发生器种子 (便于实验结果与测试基准的可复现性)
    """
    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],  # 待优化的目标函数句柄
        dim: int,                                     # 决策空间搜索维度
        bounds: Tuple[List[float], List[float]],      # 各维度坐标取值上下界元组
        n_particles: int = 40,                        # 粒子种群规模数量
        max_iter: int = 150,                          # 最大迭代演化代数
        w_mode: str = "ldw",                          # 惯性权重自适应演化模式
        w: float = 0.7298,                            # 标称惯性权重基准值（Clerc 收缩因子）
        w_min: float = 0.4,                           # 线性递减模式下的惯性权重下界
        w_max: float = 0.9,                           # 线性递减模式下的惯性权重上界
        c1: float = 1.4962,                           # 个体认知加速常数
        c2: float = 1.4962,                           # 群体社会协同加速常数
        v_ratio: float = 0.15,                        # 速度限幅相对于搜索空间跨度的比例系数
        seed: Optional[int] = None                    # 随机数种子（可选）
    ):
        if seed is not None:
            np.random.seed(seed)                      # 若指定种子，则固定 NumPy 全局伪随机数序列以保证结果可复现
            
        self.fitness_func = fitness_func              # 缓存目标适应度函数句柄
        self.dim = dim                                # 记录优化决策变量的维度 d
        self.lb = np.array(bounds[0], dtype=np.float64)  # 将各维搜索空间下界转换为双精度 1D 数组 (d,)
        self.ub = np.array(bounds[1], dtype=np.float64)  # 将各维搜索空间上界转换为双精度 1D 数组 (d,)
        self.n_particles = n_particles                # 记录群体中的粒子总数 N
        self.max_iter = max_iter                      # 记录最大允许循环迭代步数 K_max
        self.w_mode = w_mode                          # 记录惯性权重更新机制类型 ('ldw' / 'constriction' / 'const')
        self.w = w                                    # 记录基准惯性权重常数
        self.w_min = w_min                            # 记录线性递减惯性权重的终态最小值
        self.w_max = w_max                            # 记录线性递减惯性权重的初始最大值
        self.c1 = c1                                  # 记录认知学习因子常数
        self.c2 = c2                                  # 记录社会学习因子常数
        
        # 依据式 (p-12) 工程经验比例公式 v_{j,max} = alpha * (x_{j,max} - x_{j,min}) 确立各维速度限幅阈值
        self.v_max = v_ratio * (self.ub - self.lb)    # 按搜索区间跨度比例计算各维度最大速度上限向量 v_max (v_ratio 即经验比例因子 alpha)
        self.v_min = -self.v_max                      # 速度对称反向下界向量 [-v_max, v_max] (d,)
        
        # 依据式 (p-3) 与式 (p-4) 初始化群体位置与速度矩阵: 形状均为 (N, d)
        self.X = np.random.uniform(self.lb, self.ub, size=(n_particles, dim))  # 在搜索空间 [lb, ub] 内均匀采样生成初始位置矩阵
        self.V = np.random.uniform(self.v_min, self.v_max, size=(n_particles, dim))  # 在限幅区间 [-v_max, v_max] 内均匀采样生成初始速度矩阵
        
        # 依据式 (p-5) 初始化各个粒子的个体历史最优位置 (pbest)
        self.pbest_X = self.X.copy()                   # 初始步的个体历史最优坐标即当前初始坐标 (深度拷贝矩阵，避免引用别名污染)
        self.pbest_fit = np.array([self.fitness_func(x) for x in self.X])  # 计算并缓存所有粒子初始位置的适应度值 (N,)
        
        # 依据式 (p-8) 初始化群体全局最优位置 (gbest)
        gbest_idx = np.argmin(self.pbest_fit)          # 找出所有粒子个体最优适应度中“最小值所在的索引下标”（整数）
        self.gbest_X = self.pbest_X[gbest_idx].copy()  # 提取该全局最优粒子对应的坐标向量；使用 .copy() 进行深拷贝，防止后续位置更新时因内存引用被意外篡改
        self.gbest_fit = self.pbest_fit[gbest_idx]     # 记录当前全局最优粒子所对应的目标函数适应度数值（标量浮点数）
        
        # 初始化收敛曲线记录列表
        self.convergence_curve = [self.gbest_fit]     # 将初始第 0 步的全局最优适应度值推入收敛曲线跟踪列表

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行粒子群优化算法的主循环寻优流程
        
        返回:
            gbest_X           : 全局最优解坐标向量，形状为 (d,)
            gbest_fit         : 全局最优适应度数值 (float)
            convergence_curve : 每一代全局最优适应度值的演化历史列表 (长度为 max_iter + 1)
        """
        for k in range(self.max_iter):                # 进入离散步主迭代循环，迭代代数 k 从 0 循环到 max_iter - 1
            # ------------------------------------------------------------------
            # 1. 依据式 (p-30) 动态计算当前迭代步的惯性权重 w
            # ------------------------------------------------------------------
            if self.w_mode == "ldw":                  # 若采用线性递减惯性权重策略 (LDW)
                # 前期权重较大有利于全空间探索 (Exploration)，后期权重较小有利于局部精细寻优 (Exploitation)
                current_w = self.w_max - (self.w_max - self.w_min) * (k / self.max_iter)
            elif self.w_mode == "constriction":       # 若采用 Clerc 收缩因子确定性模式
                current_w = self.w                    # 保持理论收缩因子常数 (约 0.7298)
            else:                                     # 其它固定常数权重模式 ('const')
                current_w = self.w                    # 使用标称基准值

            # ------------------------------------------------------------------
            # 2. 生成均匀分布随机扰动矩阵 r1, r2 ~ U(0, 1)
            # ------------------------------------------------------------------
            r1 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.dim))  # 为每个粒子各维度独立采样认知项随机扰动因子，形状 (N, d)
            r2 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.dim))  # 为每个粒子各维度独立采样社会项随机扰动因子，形状 (N, d)

            # ------------------------------------------------------------------
            # 3. 依据式 (p-9) 与式 (p-11) 执行向量化速度更新与限幅截断
            # ------------------------------------------------------------------
            cognitive = self.c1 * r1 * (self.pbest_X - self.X)  # 计算个体认知吸引项：驱动粒子向自身历史最佳位置靠近，形状 (N, d)
            social = self.c2 * r2 * (self.gbest_X - self.X)     # 计算群体社会协同项：驱动粒子向全群全局历史最优解汇聚，广播为 (N, d)
            self.V = current_w * self.V + cognitive + social    # 依据式 (p-9) 向量化累加动量项、认知项与社会项，合成新速度矩阵
            self.V = np.clip(self.V, self.v_min, self.v_max)    # 依据式 (p-11) 实施速度限幅，防止步长发散导致飞出可行域

            # ------------------------------------------------------------------
            # 4. 依据式 (p-10) 与式 (p-13) 执行向量化位置外推与可行域边界截断
            # ------------------------------------------------------------------
            # 依据式 (p-10) 更新位置: X[k+1] = X[k] + V[k+1]
            # 依据式 (p-13) 采用截断吸收边界策略 (Absorbing Boundary)，将超出 [lb, ub] 的坐标投影截断回边界上
            self.X = np.clip(self.X + self.V, self.lb, self.ub)

            # ------------------------------------------------------------------
            # 5. 评估所有粒子在新位置处的适应度值
            # ------------------------------------------------------------------
            current_fit = np.array([self.fitness_func(x) for x in self.X])  # 遍历群体中每个粒子当前坐标，计算对应适应度数值，形成一维数组 (N,)

            # ------------------------------------------------------------------
            # 6. 依据式 (p-6) 向量化更新个体历史最优位置 (pbest)
            # ------------------------------------------------------------------
            improved_mask = current_fit < self.pbest_fit        # 找出当前步适应度优于历史最佳适应度的粒子布尔掩码向量 (N,)
            self.pbest_X[improved_mask] = self.X[improved_mask] # 仅对适应度获得改善的粒子，覆盖更新其历史最佳坐标 pbest_X
            self.pbest_fit[improved_mask] = current_fit[improved_mask]  # 同步更新对应的历史最佳适应度记录值 pbest_fit

            # ------------------------------------------------------------------
            # 7. 依据式 (p-8) 更新群体全局最优位置 (gbest)
            # ------------------------------------------------------------------
            min_fit_idx = np.argmin(self.pbest_fit)             # 获取当前代所有粒子个体历史最优中“最小值所在的索引下标”（整数）
            if self.pbest_fit[min_fit_idx] < self.gbest_fit:    # 若当前代发现的新解优于历史全局最优解（目标函数值更小）
                self.gbest_fit = self.pbest_fit[min_fit_idx]    # 刷新群体全局最优的适应度标量数值
                self.gbest_X = self.pbest_X[min_fit_idx].copy() # 使用 .copy() 进行深拷贝提取该粒子的坐标向量，覆盖更新全局最优解向量 gbest_X

            self.convergence_curve.append(self.gbest_fit)       # 记录当前迭代代数的全局最优适应度，用于收敛性轨迹分析

        return self.gbest_X, self.gbest_fit, self.convergence_curve  # 返回优化完成后的全局最优坐标、最优函数值与完整收敛曲线


```



#### 1.8.2 应用示例 ：经典多峰测试函数全局寻优（Rastrigin 函数）
Rastrigin 函数是数值优化领域中最具代表性的高度非凸多峰基准测试函数之一。其二维与高维图景包含大量密集的局部极小陷阱，传统基于梯度的算法极易陷入局部最优：
$$
f(\boldsymbol{x}) = 10d + \sum_{j=1}^d \left( x_j^2 - 10 \cos(2\pi x_j) \right), \quad x_j \in [-5.12, 5.12]
\tag{p-34}
$$
其全局最优解位于坐标原点 $\boldsymbol{x}^* = [0, 0, \dots, 0]^\mathrm{T}$，对应理论最优值为 $f(\boldsymbol{x}^*) = 0$。

```python
# ==============================================================================
# 应用示例：经典多峰测试函数全局寻优（Rastrigin 函数）
# ==============================================================================

def rastrigin(x: np.ndarray) -> float:
    """
    Rastrigin 经典高度非凸多峰基准测试函数
    数学定义: f(x) = 10*d + sum(x_j^2 - 10*cos(2*pi*x_j))
    理论全局最优解在原点 x* = [0, 0, ..., 0]，理论全局最小值 f(x*) = 0.0
    """
    d = len(x)                                                       # 获取输入向量的维度 d
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))  # 计算 Rastrigin 多峰函数的代数标量值


def run_rastrigin_example():
    """执行 10 维 Rastrigin 函数测试案例的求解与结果展示"""
    print("\n" + "=" * 60)                                    # 打印格式分隔线
    print("应用示例 1: 10 维 Rastrigin 多峰基准函数全局寻优")   # 打印示例标题
    print("=" * 60)                                           # 打印格式分隔线
    
    dim_ras = 10                                              # 设置优化问题的决策维度为 10 维
    bounds_ras = ([-5.12] * dim_ras, [5.12] * dim_ras)        # 设置 Rastrigin 函数的标准超矩形搜索区间 [-5.12, 5.12]^10

    # 实例化粒子群优化器对象
    optimizer = ParticleSwarmOptimizer(
        fitness_func=rastrigin,                               # 传入待最小化的目标函数
        dim=dim_ras,                                          # 决策变量维度
        bounds=bounds_ras,                                    # 搜索可行域边界
        n_particles=50,                                       # 群体粒子数量设为 50 个
        max_iter=200,                                         # 最大迭代 200 代
        w_mode="ldw",                                         # 采用线性递减惯性权重策略
        seed=42                                               # 设置随机数种子以保证可重复性
    )

    best_x, best_f, curve = optimizer.optimize()              # 执行优化寻优主过程，解包返回最优坐标、最优适应度与收敛曲线

    # 输出寻优结果与理论基准对比
    print(f"理论全局最小值: 0.0 (位于原点)")                     # 输出理论已知全局最小值
    print(f"PSO 寻得最佳适应度: {best_f:.6e}")                 # 输出算法寻找到的最优适应度数值（科学计数法）
    print(f"最优解前 3 维坐标: {np.round(best_x[:3], 5)}")      # 打印前 3 维坐标的数值验证（保留 5 位小数）
    print(f"迭代代数: {len(curve) - 1}, 初始适应度: {curve[0]:.4f} -> 最终适应度: {curve[-1]:.6e}")  # 打印收敛前后对比指标


if __name__ == "__main__":
    run_rastrigin_example()                                   

```



