# 粒子群优化器逐行深度代码解析

本文档针对**粒子群优化器**的完整 Python 源码，开展严谨的代数更新逻辑、NumPy 连续内存向量化机理及离散动力系统收敛特性的逐行剖析。全文划分为**运行环境与依赖导入**、**类定义与参数签名**、**优化器状态初始化 (`__init__`)**、**优化主循环 (`optimize()`)**、**高维多峰基准函数寻优示例** 以及 **完整可执行代码** 六大核心板块。

---

## 目录
- [一、 库导入与运行环境配置](#一-库导入与运行环境配置)
  - [1.1 模块导入及类型提示库 typing](#11-模块导入及类型提示库-typing)
  - [1.2 Windows 跨平台标准输出流 UTF-8 重设](#12-windows-跨平台标准输出流-utf-8-重设)
- [二、 类定义与构造函数参数解析](#二-类定义与构造函数参数解析)
  - [2.1 类声明与设计规范](#21-类声明与设计规范)
  - [2.2 构造函数参数签名及类型注解](#22-构造函数参数签名及类型注解)
- [三、 优化器状态初始化 (`__init__`)](#三-优化器状态初始化-__init__)
  - [3.1 随机种子设定与标量超参数绑定](#31-随机种子设定与标量超参数绑定)
  - [3.2 搜索空间边界数组转换与速度限幅计算](#32-搜索空间边界数组转换与速度限幅计算)
  - [3.3 状态矩阵与空间采样机制](#33-状态矩阵与空间采样机制)
  - [3.4 个体历史最优 (pbest) 初始化与切片展开机制](#34-个体历史最优-pbest-初始化与切片展开机制)
  - [3.5 群体全局最优 (gbest) 初始化与内存隔离保护](#35-群体全局最优-gbest-初始化与内存隔离保护)
  - [3.6 收敛演化轨迹跟踪机制](#36-收敛演化轨迹跟踪机制)
- [四、 优化主循环 `optimize()`](#四-优化主循环-optimize)
  - [4.1 函数签名与主迭代循环框架](#41-函数签名与主迭代循环框架)
  - [4.2 步骤 1：动态调节惯性权重 $\omega$ (LDW 与收缩因子模式)](#42-步骤-1动态调节惯性权重-omega-ldw-与收缩因子模式)
  - [4.3 步骤 2：多维解耦独立均匀随机扰动矩阵 $r_1, r_2$ 采样](#43-步骤-2多维解耦独立均匀随机扰动矩阵-r_1-r_2-采样)
  - [4.4 步骤 3：向量化速度更新与限幅截断](#44-步骤-3向量化速度更新与限幅截断)
  - [4.5 步骤 4：向量化位置外推与吸收边界截断](#45-步骤-4向量化位置外推与吸收边界截断)
  - [4.6 步骤 5：新位置适应度批量评估](#46-步骤-5新位置适应度批量评估)
  - [4.7 步骤 6：基于布尔掩码的高级索引更新个体历史最优 (pbest)](#47-步骤-6基于布尔掩码的高级索引更新个体历史最优-pbest)
  - [4.8 步骤 7：群体全局最优 (gbest) 贪婪更新与收敛轨迹记录](#48-步骤-7群体全局最优-gbest-贪婪更新与收敛轨迹记录)
- [五、 应用示例：Rastrigin 多峰测试函数全局寻优](#五-应用示例rastrigin-多峰测试函数全局寻优)
  - [5.1 Rastrigin 经典高度非凸多峰基准函数实现](#51-rastrigin-经典高度非凸多峰基准函数实现)
  - [5.2 测试用例执行函数 `run_rastrigin_example()`](#52-测试用例执行函数-run_rastrigin_example)
  - [5.3 独立执行入口保护守护 `__main__`](#53-独立执行入口保护守护-__main__)
- [六、 完整可执行代码](#六-完整可执行代码)

---

## 一、 库导入与运行环境配置
```python
import sys  # 导入系统接口模块
import numpy as np  # 导入高性能科学计算与矩阵运算基础库
from typing import Callable, Tuple, List, Optional  # 导入静态类型提示模块，增强接口规范性与健壮性

# 重设标准输出与标准错误流编码为 UTF-8，消除 Windows 控制台与 IDE 终端编码冲突
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
```

### 1.1 模块导入及类型提示库 `typing`

- `import sys`：
  - 导入 Python 内置的系统级底层交互接口模块，用于提取宿主操作系统平台标识（`sys.platform`）并对标准输入输出流对象（`sys.stdout`, `sys.stderr`）进行属性配置与重定向。
- `import numpy as np`：
  - 导入高性能科学计算与矩阵运算基础库 NumPy，并依学术规范赋予别名 `np`。粒子群算法中群体状态的并行演化、速度与位置更新、超矩形域投影截断及极值统计，均深度依托 NumPy 底层的 C 语言连续内存数组（`ndarray`）与广播机制（Broadcasting）实现，彻底消除了 Python 解释执行层面的逐粒子显式循环所导致的性能开销。
- `from typing import Callable, Tuple, List, Optional`：
  - `typing` 是 Python 标准库提供的静态类型注解模块。引入类型提示可在保持动态语言灵活性的同时，为 IDE 智能提示、静态分析工具（如 `mypy`）提供完备的类型契约，显著提升大型科学计算工程代码的可维护性与稳健性：

| 类型注解工具 | 形式化语义 | 在本算法工程中的具体映射与作用 |
| :---: | :---: | :---: |
| **`Callable`** | **可调用对象（函数或方法句柄）** | `Callable[[np.ndarray], float]` 约束输入为一维 NumPy 数组、输出为标量适应度值的目标评估函数 |
| **`Tuple`** | **固定长度与异构类型的不可变元组** | `Tuple[List[float], List[float]]` 约束超矩形搜索空间的边界定义为 `(下界列表, 上界列表)` |
| **`List`** | **动态变长列表容器** | `List[float]` 用于记录历代全局最优适应度标量所构成的离散收敛时序轨迹 |
| **`Optional`** | **可选可空类型（联合类型 `Union[T, None]`）** | `Optional[int]` 允许显式传入整型伪随机数种子，或缺省置为 `None` |

`Callable` 的语法规范：
```python
Callable[[参数1类型, 参数2类型, ...], 返回值类型]
```
方括号内第一项列出该可调用对象接受的**形参类型列表**，第二项为该对象执行后返回的数据类型。例如 `Callable[[np.ndarray], float]` 严格契合连续空间单目标黑箱优化问题的数学映射：$f: \mathcal{S} \subset \mathbb{R}^d \to \mathbb{R}$。

---

### 1.2 Windows 跨平台标准输出流 UTF-8 重设
```python
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
```
- **问题机理与工程背景**：
  - 在 Windows 操作系统下，简体中文语言区域的系统控制台默认活动代码页（Active Code Page）通常为 GBK/CP936，而现代化集成开发环境（如 VS Code 终端、Obsidian 运行环境）及跨平台运行容器普遍采用 UTF-8 统一编码规范；
  - 若 Python 解释器向标准 I/O 文本流写入 GBK 编码字节序列，终端按 UTF-8 规则解码时将产生不可逆的乱码异常（例如中文字符串“应用示例”被误解释为西里尔乱码 `Ӧʾ`）。
- **执行逻辑与反射机制**：
  - `sys.platform == "win32"`：宿主操作系统平台判定。在 Windows 环境（包含 32 位与 64 位架构）中，`sys.platform` 的返回值均固定为 `"win32"`；
  - `hasattr(sys.stdout, "reconfigure")`：基于反射（Reflection）的防御性特性探测。检查标准输出流对象（通常为 `io.TextIOWrapper` 实例）是否包含属性方法 `reconfigure`（该接口自 Python 3.7 正式引入），以保障代码在不同运行时环境下的向下兼容性；
  - `sys.stdout.reconfigure(encoding="utf-8")` 与 `sys.stderr.reconfigure(encoding="utf-8")`：重设标准输出流与标准错误流底层的文本编码格式为严格的 UTF-8，从输入输出流源头消除跨平台环境下的编码冲突与乱码缺陷。

---

## 二、 类定义与构造函数参数解析
```python
class ParticleSwarmOptimizer:
    """
    通用向量化粒子群优化器 (Particle Swarm Optimizer, PSO)
    
    参数说明:
        fitness_func : 目标适应度函数，输入为一维决策向量 (d,)，返回标量适应度值 (float)
        dim          : 优化决策变量的维度 d (int)
        bounds       : 搜索空间超矩形上下界元组 (lb, ub)，形式为 ([x1_min, ...], [x1_max, ...])
        n_particles  : 粒子群规模 N (默认 40)
        max_iter     : 最大允许迭代代数 K_max (默认 150)
        w_mode       : 惯性权重调节模式，可选 'ldw' (线性递减), 'const' (常数), 'constriction' (收缩因子)
        w            : 基础惯性权重或收缩因子 (默认 0.7298，来源于 Clerc-Kennedy 收缩因子解析解)
        w_min, w_max : 线性递减权重上下限 (默认 0.4 与 0.9)
        c1, c2       : 认知学习因子与社会学习因子 (默认 1.4962，与收缩因子 0.7298 对应匹配)
        v_ratio      : 速度限幅占搜索空间跨度的比例 (默认 0.15，即单步最大位移不超过可行域跨度的 15%)
        seed         : 伪随机数发生器种子 (用于基准测试与数值实验的可复现性验证)
    """
    def __init__(
        self,                                         # 类实例自身引用
        fitness_func: Callable[[np.ndarray], float],  # 待优化的目标函数句柄
        dim: int,                                     # 决策空间搜索维度 d
        bounds: Tuple[List[float], List[float]],      # 决策变量各维度的超矩形取值上下界元组
        n_particles: int = 40,                        # 粒子种群规模 N
        max_iter: int = 150,                          # 最大迭代演化代数 K_max
        w_mode: str = "ldw",                          # 惯性权重演化策略 ('ldw' / 'constriction' / 'const')
        w: float = 0.7298,                            # 标称惯性权重基准值（Clerc-Kennedy 收缩因子）
        w_min: float = 0.4,                           # 线性递减模式下的惯性权重下界
        w_max: float = 0.9,                           # 线性递减模式下的惯性权重上界
        c1: float = 1.4962,                           # 个体认知加速常数
        c2: float = 1.4962,                           # 群体社会协同加速常数
        v_ratio: float = 0.15,                        # 速度限幅相对于搜索空间跨度的比例系数 alpha
        seed: Optional[int] = None                    # 伪随机数发生器种子（可选）
    ):
```

### 2.1 类声明与设计规范
- `class ParticleSwarmOptimizer:`：
  - 声明粒子群优化算法的核心封装类。基于面向对象架构（OOP），将优化器的核心内部状态（包括粒子空间位置矩阵、速度矩阵、个体历史极值缓存、全局最优解及收敛演化轨迹）与动力学行为（状态初始化、动态速度演化、空间边界投影及极值刷新）进行模块化封装。
- 类级文档说明（Docstring）：
  - 严格遵循 Google/NumPy 规范，完整界定类的功能定位、参数维度、数学物理意义以及缺省超参数取值。

### 2.2 构造函数参数签名及类型注解
构造函数为类实例注入待求解问题特性与控制超参数：
1. **优化问题先验参数**：
   - `fitness_func: Callable[[np.ndarray], float]`：待最小化的目标函数 $f(\boldsymbol{x})$。
   - `dim: int`：决策变量的特征维度 $d$。
   - `bounds: Tuple[List[float], List[float]]`：定义紧致有界的超矩形搜索空间 $\displaystyle{\mathcal{S} = \prod_{j=1}^d [x_{j,\min}, x_{j,\max}]}$。`bounds[0]` 为下界向量，`bounds[1]` 为上界向量。
2. **群体与代数超参数**：
   - `n_particles: int = 40`：群体规模 $N$。对于中低维连续空间优化问题，经验上取群体规模 $N \in [20, 50]$ 即可在全局空间遍历性与计算时间开销之间实现良好折衷。
   - `max_iter: int = 150`：算法演化代数上限 $K_{\max}$。
3. **动力系统控制参数**：
   - `w_mode: str = "ldw"`：惯性权重调节模式，内置三种策略：
     - `"ldw"`：线性递减惯性权重策略（Linear Decreasing Weight, LDW），动态权衡全局广域探索（Exploration）与局部精细开发（Exploitation）；
     - `"constriction"`：Clerc-Kennedy 速度收缩因子确定性模式；
     - `"const"`：常数惯性权重模式。
   - `w: float = 0.7298`：标称基准惯性权重常数，对应 Clerc-Kennedy 收缩因子解析解 $\chi \approx 0.7298$。
   - `w_min: float = 0.4, w_max: float = 0.9`：线性递减模式的动态阻尼范围 $[\omega_{\min}, \omega_{\max}]$。
   - `c1: float = 1.4962, c2: float = 1.4962`：认知学习因子与社会协同因子。该默认参数与收缩因子 $\chi = 0.7298$ 严格匹配（满足 $c_1' = c_2' \approx 0.7298 \times 2.05 \approx 1.4962$）。
   - `v_ratio: float = 0.15`：对应速度限幅经验比例因子 $\alpha = 0.15 \in [0.1, 0.2]$。
   - `seed: Optional[int] = None`：全局伪随机数种子，用于科研对比与工程基准测试的可复现性保障。

---

## 三、 优化器状态初始化 (`__init__`)
```python
if seed is not None:
    np.random.seed(seed)                      # 若指定种子，则置位 NumPy 全局伪随机数生成器以保障实验可复现

self.fitness_func = fitness_func              # 缓存目标适应度函数句柄
self.dim = dim                                # 记录优化决策变量的维度 d
self.lb = np.array(bounds[0], dtype=np.float64)  # 将各维搜索空间下界转换为连续内存双精度 1D 数组 (d,)
self.ub = np.array(bounds[1], dtype=np.float64)  # 将各维搜索空间上界转换为连续内存双精度 1D 数组 (d,)
self.n_particles = n_particles                # 记录群体中的粒子总数 N
self.max_iter = max_iter                      # 记录最大允许循环迭代步数 K_max
self.w_mode = w_mode                          # 记录惯性权重更新机制类型 ('ldw' / 'constriction' / 'const')
self.w = w                                    # 记录基准惯性权重常数
self.w_min = w_min                            # 记录线性递减惯性权重的终态最小值
self.w_max = w_max                            # 记录线性递减惯性权重的初始最大值
self.c1 = c1                                  # 记录认知学习因子常数
self.c2 = c2                                  # 记录社会学习因子常数
```

### 3.1 随机种子设定与标量超参数绑定
- `np.random.seed(seed)`：
  - 置位 NumPy 伪随机数生成器（PRNG）的初始种子。显式指定整型种子时，粒子群的初始空间采样、初始速度分布以及后续主循环各代随机扰动矩阵序列均具备严格的确定性，为算法基准对比与实验复现提供保障。
- 实例属性状态绑定：
  - 将输入超参数统一绑定为优化器实例属性（`self.*`），实现算法状态在全生命周期内的持久化与各子步骤间的高效共享。

---

### 3.2 搜索空间边界数组转换与速度限幅计算
依据速度限幅经验公式：
$$
v_{j,\max} = \alpha (x_{j,\max} - x_{j,\min})
\tag{p-12}
$$

```python
# 依据式 (p-12) 工程经验比例公式 v_{j,max} = alpha * (x_{j,max} - x_{j,min}) 确立各维速度限幅阈值
self.v_max = v_ratio * (self.ub - self.lb)    # 按搜索区间跨度比例计算各维度最大速度上限向量 v_max (v_ratio 即经验比例因子 alpha)
self.v_min = -self.v_max                      # 速度对称反向下界向量 [-v_max, v_max] (d,)
```

- `self.lb` 与 `self.ub`：
  - 将用户输入的边界列表统一转换为连续物理内存布局的双精度浮点（`float64`）一维数组，形状均为 `(d,)`；
- `self.v_max = v_ratio * (self.ub - self.lb)`：
  - 计算各决策维度的空间跨度 $\Delta x_j = x_{j,\max} - x_{j,\min}$，并加权经验比例因子 $\alpha$（`v_ratio`，默认值取 $0.15$）；
  - 生成的最大速度向量 `self.v_max` 形状为 `(d,)`，约束各决策轴允许的最大单步运动位移；
- `self.v_min = -self.v_max`：
  - 构建原点对称的速度反向下界向量 $[-v_{j,\max}]$，形成对称的速度限幅超矩形区间 $[-v_{j,\max}, v_{j,\max}]$。

---

### 3.3 状态矩阵与空间采样机制
依据连续状态空间形式化定义：
$$
\boldsymbol{x}_{i,[k]} = [x_{i,1,[k]}, x_{i,2,[k]}, \dots, x_{i,d,[k]}]^\mathrm{T} \in \mathcal{S}
\tag{p-3}
$$
$$
\boldsymbol{v}_{i,[k]} = [v_{i,1,[k]}, v_{i,2,[k]}, \dots, v_{i,d,[k]}]^\mathrm{T} \in \mathbb{R}^d
\tag{p-4}
$$

```python
# 依据式 (p-3) 与式 (p-4) 初始化群体位置与速度矩阵: 形状均为 (N, d)
self.X = np.random.uniform(self.lb, self.ub, size=(n_particles, dim))  # 在搜索空间 [lb, ub] 内均匀采样生成初始位置矩阵
self.V = np.random.uniform(self.v_min, self.v_max, size=(n_particles, dim))  # 在限幅区间 [-v_max, v_max] 内均匀采样生成初始速度矩阵
```

- `np.random.uniform(self.lb, self.ub, size=(n_particles, dim))`：
  - **连续空间均匀采样与广播机理**：`self.lb` 与 `self.ub` 均为维度为 `(d,)` 的一维向量。当指定生成尺寸 `size=(n_particles, dim)` 时，NumPy 沿粒子轴（第 0 轴）自动执行边界向量广播：
    - 对群体中每个粒子 $i$ 的各个决策维度 $j$，均独立从均匀分布 $\mathcal{U}(x_{j,\min}, x_{j,\max})$ 中采样生成初始坐标分量；
    - 最终构造形状为 $(N, d)$ 的初始位置状态矩阵 $\boldsymbol{X}$，每一行表征单个粒子在决策空间中的候选解向量。
- `self.V`：
  - 采用相同机制在对称限幅区间 $[-v_{j,\max}, v_{j,\max}]$ 内独立均匀采样，生成初始速度矩阵 $\boldsymbol{V} \in \mathbb{R}^{N \times d}$。

---

### 3.4 个体历史最优 (pbest) 初始化与切片展开机制
依据个体历史最优状态定义：
$$
\boldsymbol{p}_{i,[k]} = [p_{i,1,[k]}, p_{i,2,[k]}, \dots, p_{i,d,[k]}]^\mathrm{T} \in \mathcal{S}
\tag{p-5}
$$
记录粒子 $i$ 自初始代至第 $k$ 步所遍历过的适应度最优位置。其状态更新方程为：
$$
\boldsymbol{p}_{i,[k]} = \begin{cases} 
\boldsymbol{x}_{i,[k]}, & \text{若 } f(\boldsymbol{x}_{i,[k]}) < f(\boldsymbol{p}_{i,[k-1]}) \\ 
\boldsymbol{p}_{i,[k-1]}, & \text{若 } f(\boldsymbol{x}_{i,[k]}) \geqslant f(\boldsymbol{p}_{i,[k-1]}) 
\end{cases}
\tag{p-6}
$$
初始边界条件设定为 $\boldsymbol{p}_{i,[0]} = \boldsymbol{x}_{i,[0]}$。

```python
# 依据式 (p-5) 初始化各个粒子的个体历史最优位置 (pbest)
self.pbest_X = self.X.copy()                   # 初始步的个体历史最优坐标即当前初始坐标 (深度拷贝矩阵，避免引用别名污染)
self.pbest_fit = np.array([self.fitness_func(x) for x in self.X])  # 串行评估并缓存全群粒子初始位置的适应度值 (N,)
```

- `self.pbest_X = self.X.copy()`：
  - **深拷贝内存隔离机制**：在初始阶段，粒子的个体历史最优位置即初始坐标。显式调用 `.copy()` 开辟独立连续物理内存，切断 `pbest_X` 与 `X` 之间的浅拷贝视图（View）与内存别名（Aliasing）绑定，杜绝后续优化迭代更新粒子位置时发生隐式原地篡改。
- `for x in self.X` 的切片展开机制：
  - 在 Python 环境下遍历二维 NumPy 数组时，解释器默认沿着主轴（第 0 轴，即粒子索引维度）执行一维行切片提取：
    - 第 0 步循环：`x = self.X[0, :]` $\to$ 提取粒子 0 的位置向量，形状为 `(d,)`；
    - 第 1 步循环：`x = self.X[1, :]` $\to$ 提取粒子 1 的位置向量，形状为 `(d,)`；
    - ... 
    - 第 $N-1$ 步循环：`x = self.X[N-1, :]` $\to$ 提取粒子 $N-1$ 的位置向量，形状为 `(d,)`；
- `np.array([...])`：
  - 将列表推导式求得的 $N$ 个标量适应度重构成形状为 `(N,)` 的一维连续双精度浮点数组 `self.pbest_fit`。

---

### 3.5 群体全局最优 (gbest) 初始化与内存隔离保护
依据群体全局最优解形式化定义：
$$
\boldsymbol{g}_{[0]} = \arg\min_{\displaystyle{\boldsymbol{p} \in \{ \boldsymbol{p}_{1,[0]}, \dots, \boldsymbol{p}_{N,[0]} \}}} f(\boldsymbol{p})
\tag{p-8}
$$

```python
# 依据式 (p-8) 初始化群体全局最优位置 (gbest)
gbest_idx = np.argmin(self.pbest_fit)          # 检索个体历史最优适应度数组中的全局极小值索引下标 (整数)
self.gbest_X = self.pbest_X[gbest_idx].copy()  # 深拷贝提取全局最优粒子的坐标向量，规避后续位置更新引发的内存别名污染
self.gbest_fit = self.pbest_fit[gbest_idx]     # 记录当前全局最优粒子对应的适应度标量 (float)
```

- `gbest_idx = np.argmin(self.pbest_fit)`：
  - 在一维适应度数组 `self.pbest_fit` 上执行线性复杂度 $\mathcal{O}(N)$ 的极小值索引定位，返回当前群体全局最优粒子的整数下标。
- `self.gbest_X = self.pbest_X[gbest_idx].copy()`：
  - 切片提取全局最优粒子的位置向量，并通过 `.copy()` 重新分配独立内存块，防止后续粒子位移演化对全局最优解造成指针重叠与内存污染。
- `self.gbest_fit = self.pbest_fit[gbest_idx]`：
  - 固化记录初始代全局最优适应度标量。

---

### 3.6 收敛演化轨迹跟踪机制
将初始代（第 0 步）的最优适应度存入原生列表作为基准点。在后续演化过程中，每代追加记录全局最优适应度标量，用于量化分析算法的收敛速率、诊断群体是否发生早熟收敛（Premature Convergence）与停滞状态（Stagnation），并为绘制收敛演化曲线提供完整的时序数据支撑。

```python
# 初始化收敛曲线记录列表
self.convergence_curve = [self.gbest_fit]     # 将初始第 0 步的全局最优适应度值存入收敛轨迹跟踪列表
```

---

## 四、 优化主循环 `optimize()`
### 4.1 函数签名与主迭代循环框架
```python
def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
    """
    执行粒子群优化算法的主循环寻优流程
    
    返回:
        gbest_X           : 全局最优解坐标向量，形状为 (d,)
        gbest_fit         : 全局最优适应度数值 (float)
        convergence_curve : 每一代全局最优适应度值的演化历史列表 (长度为 max_iter + 1)
    """
    for k in range(self.max_iter):                # 启动离散时间步迭代演化主循环，代数 k in [0, max_iter - 1]
```

- **类型提示与返回值规范**：
  - `-> Tuple[np.ndarray, float, List[float]]`：借助类型注解明确函数执行完毕后返回的三元组数据结构：
    - `gbest_X`：优化过程终止时获得的全局最优解向量，形状为 `(d,)`；
    - `gbest_fit`：全局最优解所对应的标量目标函数值（适应度极值）；
    - `convergence_curve`：长度为 $K_{\max} + 1$ 的迭代代数收敛轨迹演化序列。
  - `for k in range(self.max_iter):`：
    - 构建离散时间步演化主循环，代数计数器 $k \in [0, K_{\max}-1]$。每轮迭代均严格执行全群状态更新、边界截断、适应度评估与极值竞争刷新。

---

### 4.2 步骤 1：动态调节惯性权重 $\omega$ (LDW 与收缩因子模式)
依据线性递减惯性权重演化公式：
$$
\omega_{[k]} = \omega_{\max} - \frac{k}{K_{\max}} (\omega_{\max} - \omega_{\min})
\tag{p-30}
$$

```python
# ------------------------------------------------------------------
# 1. 依据式 (p-30) 动态计算当前迭代步的惯性权重 w
# ------------------------------------------------------------------
if self.w_mode == "ldw":                  # 若采用线性递减惯性权重策略 (LDW)
    # 前期大权重强化全局探索 (Exploration)，后期小权重聚焦局部开发 (Exploitation)
    current_w = self.w_max - (self.w_max - self.w_min) * (k / self.max_iter)
elif self.w_mode == "constriction":       # 若采用 Clerc 收缩因子确定性模式
    current_w = self.w                    # 保持理论收缩因子常数 (约 0.7298)
else:                                     # 其它固定常数权重模式 ('const')
    current_w = self.w                    # 使用标称基准值
```

- **线性递减惯性权重策略（LDW）动力学分析**：
  - 演化初始阶段（$k=0$），$\omega = \omega_{\max} = 0.9$。较高的惯性权重赋予粒子充足的动量，使其在全解空间内保持广域漫游，强化全局探索能力（Exploration）以定位潜在的优质吸引盆；
  - 演化收敛阶段（$k \to K_{\max}$），$\omega$ 线性衰减至 $\omega_{\min} = 0.4$。较低的惯性权重削弱了粒子的高频动能，促使其运动步长平稳收缩，聚焦于已定位最优解邻域进行高精度的局部精细开发（Exploitation）。
- **收缩因子模式**：
  - 严格保持 Clerc 解析常数 $\chi \approx 0.7298$（与学习常数 $c_1=c_2=2.05$ 配套），从离散二阶动力系统特征根配置角度确保算法在无速度限幅条件下的收敛稳定性。

---

### 4.3 步骤 2：多维解耦独立均匀随机扰动矩阵 $r_1, r_2$ 采样

```python
# ------------------------------------------------------------------
# 2. 生成均匀分布随机扰动矩阵 r1, r2 ~ U(0, 1)
# ------------------------------------------------------------------
r1 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.dim))  # 为每个粒子各维度独立采样认知项随机扰动因子，形状 (N, d)
r2 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.dim))  # 为每个粒子各维度独立采样社会项随机扰动因子，形状 (N, d)
```

- **空间解耦独立随机扰动准则**：
  - 在标准粒子群优化理论中，认知项与社会项的随机扰动因子必须对每个粒子在各个决策维度上进行独立采样，即构建形状为 $(N, d)$ 的随机矩阵，矩阵中各个独立标量满足 $r_{1,ij}, r_{2,ij} \sim \mathcal{U}(0, 1)$；
  - 维间独立解耦采样打破了多维位移更新中的空间各向同性（Isotropy），促使粒子在多维空间中形成各向异性的随机超矩形探索范围，显著提升群体跳脱复杂多峰局部极小值陷阱的能力。

---

### 4.4 步骤 3：向量化速度更新与限幅截断
依据速度演化与限幅方程：
$$
\boldsymbol{v}_{i,[k+1]} = \omega \boldsymbol{v}_{i,[k]} + c_1 r_{1,[k]} (\boldsymbol{p}_{i,[k]} - \boldsymbol{x}_{i,[k]}) + c_2 r_{2,[k]} (\boldsymbol{g}_{[k]} - \boldsymbol{x}_{i,[k]})
\tag{p-9}
$$
$$
v_{i, j,[k+1]} = \operatorname{clip}(v_{i, j,[k+1]}, \, -v_{j,\max}, \, v_{j,\max})
\tag{p-11}
$$

```python
# ------------------------------------------------------------------
# 3. 依据式 (p-9) 与式 (p-11) 执行向量化速度更新与限幅截断
# ------------------------------------------------------------------
cognitive = self.c1 * r1 * (self.pbest_X - self.X)  # 计算个体认知吸引项：驱动粒子向自身历史最佳位置靠近，形状 (N, d)
social = self.c2 * r2 * (self.gbest_X - self.X)     # 计算群体社会协同项：驱动粒子向全群全局历史最优解汇聚，广播为 (N, d)
self.V = current_w * self.V + cognitive + social    # 依据式 (p-9) 向量化累加动量项、认知项与社会项，合成新速度矩阵
self.V = np.clip(self.V, self.v_min, self.v_max)    # 依据式 (p-11) 执行速度限幅截断，防止单步位移发散导致粒子逃逸出可行域
```

- `cognitive = self.c1 * r1 * (self.pbest_X - self.X)`：
  - **个体认知项（Cognitive Component）**：表征粒子向个体历史最佳位置逼近的向心吸引力，由各粒子当前位置与自身历史极值的偏差量、随机扰动因子与认知加速常数 $c_1$ 共同决定，运算结果为 $(N, d)$ 矩阵。
- `social = self.c2 * r2 * (self.gbest_X - self.X)`：
  - **群体社会协同项（Social Component）与广播机制**：`self.gbest_X` 为 `(d,)` 向量，`self.X` 为 `(N, d)` 矩阵。NumPy 自动将 `gbest_X` 沿粒子轴广播扩展 $N$ 次，计算全群全局最优解对各个粒子的协同牵引力。
- `self.V = current_w * self.V + cognitive + social`：
  - **速度合成动力学更新**：将动量惯性项、个体认知记忆项与群体社会协同项进行矢量线性叠加，全过程由底层 C 级连续内存向量化算子执行，彻底消除解释型逐粒子循环的性能开销。
- `self.V = np.clip(self.V, self.v_min, self.v_max)`：
  - **速度限幅截断（Velocity Clamping）**：将粒子在各坐标轴上的速度分量严格约束在区间 $[-v_{j,\max}, v_{j,\max}]$ 内，避免因粒子加速度过大引发速度发散，从而防止粒子群脱离可行搜索空间。

---

### 4.5 步骤 4：向量化位置外推与吸收边界截断
依据位置更新与吸收边界截断方程：
$$
\boldsymbol{x}_{i,[k+1]} = \boldsymbol{x}_{i,[k]} + \boldsymbol{v}_{i,[k+1]}
\tag{p-10}
$$
$$
x_{i, j,[k+1]} = \operatorname{clip}(x_{i, j,[k+1]}, \, x_{j,\min}, \, x_{j,\max})
\tag{p-13}
$$

```python
# ------------------------------------------------------------------
# 4. 依据式 (p-10) 与式 (p-13) 执行向量化位置外推与可行域边界截断
# ------------------------------------------------------------------
# 依据式 (p-10) 更新位置: X[k+1] = X[k] + V[k+1]
# 依据式 (p-13) 实施吸收边界策略 (Absorbing Boundary)，将超出超矩形域的坐标正交投影截断至可行域边界
self.X = np.clip(self.X + self.V, self.lb, self.ub)
```

- `self.X + self.V`：
  - 显式一阶位移积分外推，维度运算满足 $(N, d) + (N, d) \to (N, d)$。
- `np.clip(..., self.lb, self.ub)`：
  - **吸收边界截断策略（Absorbing Boundary）**：当粒子运动坐标超出预设的超矩形搜索空间 $[x_{j,\min}, x_{j,\max}]$ 时，强制将其正交投影截断至对应边界上，确保算法生成的全部候选解均严格位于可行域内部。

---

### 4.6 步骤 5：新位置适应度批量评估
```python
# ------------------------------------------------------------------
# 5. 评估所有粒子在新位置处的适应度值
# ------------------------------------------------------------------
current_fit = np.array([self.fitness_func(x) for x in self.X])  # 批量评估当前代粒子群在新坐标下的适应度数值，构建一维数组 (N,)
```

- 通过列表推导式依次提取群体中各粒子的解向量（切片维度为 `(d,)`）并调用目标函数句柄进行评估，最后重新组装为形状为 `(N,)` 的一维连续双精度浮点数组。

---

### 4.7 步骤 6：基于布尔掩码的高级索引更新个体历史最优 (pbest)
依据个体历史最优状态递推定义：
$$
\boldsymbol{p}_{i,[k+1]} = \begin{cases}
\boldsymbol{x}_{i,[k+1]}, & \text{若 } f(\boldsymbol{x}_{i,[k+1]}) < f(\boldsymbol{p}_{i,[k]}) \\
\boldsymbol{p}_{i,[k]}, & \text{若 } f(\boldsymbol{x}_{i,[k+1]}) \geqslant f(\boldsymbol{p}_{i,[k]})
\end{cases}
\tag{p-6}
$$

```python
# ------------------------------------------------------------------
# 6. 依据式 (p-6) 向量化更新个体历史最优位置 (pbest)
# ------------------------------------------------------------------
improved_mask = current_fit < self.pbest_fit        # 构建当前步适应度优于个体历史最优的布尔掩码 (N,)
self.pbest_X[improved_mask] = self.X[improved_mask] # 依据掩码向量化就地更新解质量获得提升的粒子历史最佳坐标 pbest_X
self.pbest_fit[improved_mask] = current_fit[improved_mask]  # 同步刷新获得解改善粒子的历史最优适应度记录值 pbest_fit
```

- `improved_mask = current_fit < self.pbest_fit`：
  - 逐元素比对当前代适应度与个体历史最佳记录，生成长度为 $N$ 的布尔条件掩码向量（Boolean Mask），如 `[True, False, True, ...]`。
- **基于布尔掩码的高级索引向量化更新**：
  - 仅对掩码判定为 `True`（即解质量实现严格提升）的粒子子集，将其当前空间坐标与适应度覆盖写入历史极值缓冲区 `pbest_X` 与 `pbest_fit`；未获得提升的粒子状态保持不变，避免无谓的内存写操作。

---

### 4.8 步骤 7：群体全局最优 (gbest) 贪婪更新与收敛轨迹记录
依据全局极值更新准则：
$$
\boldsymbol{g}_{[k+1]} = \arg\min_{\displaystyle{\boldsymbol{p} \in \{ \boldsymbol{p}_{1,[k+1]}, \dots, \boldsymbol{p}_{N,[k+1]} \}}} f(\boldsymbol{p})
\tag{p-8}
$$

```python
# ------------------------------------------------------------------
# 7. 依据式 (p-8) 更新群体全局最优位置 (gbest)
# ------------------------------------------------------------------
min_fit_idx = np.argmin(self.pbest_fit)             # 检索当前代个体历史最优集合中最小适应度的索引下标
if self.pbest_fit[min_fit_idx] < self.gbest_fit:    # 若当前代涌现出的最优解严格优于历史全局极值
    self.gbest_fit = self.pbest_fit[min_fit_idx]    # 刷新群体全局最优的适应度标量数值
    self.gbest_X = self.pbest_X[min_fit_idx].copy() # 深拷贝刷新全局最优解向量 gbest_X，阻断数组别名污染

self.convergence_curve.append(self.gbest_fit)       # 追加当前迭代代数的全局最优适应度标量，用于收敛轨迹分析

return self.gbest_X, self.gbest_fit, self.convergence_curve  # 返回全局最优解向量、最优适应度标量与完整收敛演化轨迹
```

- **全局极值贪婪更新策略**：仅当当前代群体涌现出的最小适应度严格优于历史极值 `self.gbest_fit` 时，才执行全局最优解的状态更新，在数学上严格保证收敛曲线的**单调非增性**（Monotonic Non-increasing Property）。
- **内存深拷贝防护**：通过 `self.pbest_X[min_fit_idx].copy()` 独立开辟内存空间，隔绝全局最优解向量与粒子群状态矩阵的指针关联，避免后续主循环位置演化对全局极值造成潜在的内存别名污染。
- 退出代数演化循环后，解包返回全局最优解向量、全局最小适应度与完整收敛曲线三元组。

---

## 五、 应用示例：Rastrigin 多峰测试函数全局寻优
### 5.1 Rastrigin 经典高度非凸多峰基准函数实现
依据 Rastrigin 测试函数数学定义：
$$
f(\boldsymbol{x}) = 10d + \sum_{j=1}^{d} \left( x_j^2 - 10\cos(2\pi x_j) \right), \quad x_j \in [-5.12, 5.12]
\tag{p-34}
$$

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
    d = len(x)                                                       # 获取输入向量的决策维度 d
    return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))  # 计算 Rastrigin 多峰函数的代数标量值
```

- **目标函数地形特征分析**：
  - 二次项 $x_j^2$ 构建了指向原点的宏观抛物面全局势能漏斗；
  - 余弦调制项 $-10\cos(2\pi x_j)$ 在全局势能面上叠加了高频周期性波动，构造出极具挑战性的局部极值陷阱。在 10 维超矩形决策空间内，包含多达 $11^{10} \approx \mathbf{2.59 \times 10^{10}}$（约 259 亿个）局部极小点；
  - 理论全局最优驻点位于原点 $\boldsymbol{x}^* = [0, 0, \dots, 0]^\mathrm{T}$，对应的理论全局最优最小适应度为 $f(\boldsymbol{x}^*) = 0.0$。

---

### 5.2 测试用例执行函数 `run_rastrigin_example()`

```python
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
```

- 设定超矩形搜索空间边界，配置种群规模为 50 个粒子、最大迭代代数为 200 代、采用线性递减惯性权重策略（LDW）并指定随机数种子 `seed=42`。
- 格式化输出最终寻优结果与理论全局极值的比对指标，验证向量化粒子群算法针对复杂多峰函数的高维寻优与收敛精度。

---

### 5.3 独立执行入口保护守护 `__main__`
```python
if __name__ == "__main__":
    run_rastrigin_example()                                   # 运行多峰函数测试案例 1
```
- **模块执行守护规范**：
  - 确保该源文件作为独立脚本执行时，自动触发 Rastrigin 基准测试案例的运行与结果输出；
  - 当该文件作为优化器工具模块被外部系统（如卡尔曼滤波超参数优化、动态系统参数辨识等工程应用）通过 `import ParticleSwarmOptimizer` 导入时，有效抑制测试逻辑的主动执行，保证模块的独立性与高复用度。

## 六、 完整可执行代码
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
        self,                                         # 类实例自身引用
        fitness_func: Callable[[np.ndarray], float],  # 待优化的目标函数句柄
        dim: int,                                     # 决策空间搜索维度 d
        bounds: Tuple[List[float], List[float]],      # 决策变量各维度的超矩形取值上下界元组
        n_particles: int = 40,                        # 粒子种群规模 N
        max_iter: int = 150,                          # 最大迭代演化代数 K_max
        w_mode: str = "ldw",                          # 惯性权重演化策略 ('ldw' / 'constriction' / 'const')
        w: float = 0.7298,                            # 标称惯性权重基准值（Clerc-Kennedy 收缩因子）
        w_min: float = 0.4,                           # 线性递减模式下的惯性权重下界
        w_max: float = 0.9,                           # 线性递减模式下的惯性权重上界
        c1: float = 1.4962,                           # 个体认知加速常数
        c2: float = 1.4962,                           # 群体社会协同加速常数
        v_ratio: float = 0.15,                        # 速度限幅相对于搜索空间跨度的比例系数 alpha
        seed: Optional[int] = None                    # 伪随机数发生器种子（可选）
    ):
        if seed is not None:
            np.random.seed(seed)                      # 若指定种子，则置位 NumPy 全局伪随机数生成器以保障实验可复现
            
        self.fitness_func = fitness_func              # 缓存目标适应度函数句柄
        self.dim = dim                                # 记录优化决策变量的维度 d
        self.lb = np.array(bounds[0], dtype=np.float64)  # 将各维搜索空间下界转换为连续内存双精度 1D 数组 (d,)
        self.ub = np.array(bounds[1], dtype=np.float64)  # 将各维搜索空间上界转换为连续内存双精度 1D 数组 (d,)
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
        self.pbest_fit = np.array([self.fitness_func(x) for x in self.X])  # 串行评估并缓存全群粒子初始位置的适应度值 (N,)
        
        # 依据式 (p-8) 初始化群体全局最优位置 (gbest)
        gbest_idx = np.argmin(self.pbest_fit)          # 检索个体历史最优适应度数组中的全局极小值索引下标 (整数)
        self.gbest_X = self.pbest_X[gbest_idx].copy()  # 深拷贝提取全局最优粒子的坐标向量，规避后续位置更新引发的内存别名污染
        self.gbest_fit = self.pbest_fit[gbest_idx]     # 记录当前全局最优粒子对应的适应度标量 (float)
        
        # 初始化收敛曲线记录列表
        self.convergence_curve = [self.gbest_fit]     # 将初始第 0 步的全局最优适应度值存入收敛轨迹跟踪列表

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        执行粒子群优化算法的主循环寻优流程
        
        返回:
            gbest_X           : 全局最优解坐标向量，形状为 (d,)
            gbest_fit         : 全局最优适应度数值 (float)
            convergence_curve : 每一代全局最优适应度值的演化历史列表 (长度为 max_iter + 1)
        """
        for k in range(self.max_iter):                # 启动离散时间步迭代演化主循环，代数 k in [0, max_iter - 1]
            # ------------------------------------------------------------------
            # 1. 依据式 (p-30) 动态计算当前迭代步的惯性权重 w
            # ------------------------------------------------------------------
            if self.w_mode == "ldw":                  # 若采用线性递减惯性权重策略 (LDW)
                # 前期大权重强化全局探索 (Exploration)，后期小权重聚焦局部开发 (Exploitation)
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
            self.V = np.clip(self.V, self.v_min, self.v_max)    # 依据式 (p-11) 执行速度限幅截断，防止单步位移发散导致粒子逃逸出可行域

            # ------------------------------------------------------------------
            # 4. 依据式 (p-10) 与式 (p-13) 执行向量化位置外推与可行域边界截断
            # ------------------------------------------------------------------
            # 依据式 (p-10) 更新位置: X[k+1] = X[k] + V[k+1]
            # 依据式 (p-13) 实施吸收边界策略 (Absorbing Boundary)，将超出超矩形域的坐标正交投影截断至可行域边界
            self.X = np.clip(self.X + self.V, self.lb, self.ub)

            # ------------------------------------------------------------------
            # 5. 评估所有粒子在新位置处的适应度值
            # ------------------------------------------------------------------
            current_fit = np.array([self.fitness_func(x) for x in self.X])  # 批量评估当前代粒子群在新坐标下的适应度数值，构建一维数组 (N,)

            # ------------------------------------------------------------------
            # 6. 依据式 (p-6) 向量化更新个体历史最优位置 (pbest)
            # ------------------------------------------------------------------
            improved_mask = current_fit < self.pbest_fit        # 构建当前步适应度优于个体历史最优的布尔掩码 (N,)
            self.pbest_X[improved_mask] = self.X[improved_mask] # 依据掩码向量化就地更新解质量获得提升的粒子历史最佳坐标 pbest_X
            self.pbest_fit[improved_mask] = current_fit[improved_mask]  # 同步刷新获得解改善粒子的历史最优适应度记录值 pbest_fit

            # ------------------------------------------------------------------
            # 7. 依据式 (p-8) 更新群体全局最优位置 (gbest)
            # ------------------------------------------------------------------
            min_fit_idx = np.argmin(self.pbest_fit)             # 检索当前代个体历史最优集合中最小适应度的索引下标
            if self.pbest_fit[min_fit_idx] < self.gbest_fit:    # 若当前代涌现出的最优解严格优于历史全局极值
                self.gbest_fit = self.pbest_fit[min_fit_idx]    # 刷新群体全局最优的适应度标量数值
                self.gbest_X = self.pbest_X[min_fit_idx].copy() # 深拷贝刷新全局最优解向量 gbest_X，阻断数组别名污染

            self.convergence_curve.append(self.gbest_fit)       # 追加当前迭代代数的全局最优适应度标量，用于收敛轨迹分析

        return self.gbest_X, self.gbest_fit, self.convergence_curve  # 返回全局最优解向量、最优适应度标量与完整收敛演化轨迹


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
    run_rastrigin_example()                                   # 运行多峰函数测试案例 1

```