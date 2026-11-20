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
    print(f"最优解前 3 维坐标: {np.round(best_x[:], 5)}")      # 打印前 3 维坐标的数值验证（保留 5 位小数）
    print(f"迭代代数: {len(curve) - 1}, 初始适应度: {curve[0]:.4f} -> 最终适应度: {curve[-1]:.6e}")  # 打印收敛前后对比指标


if __name__ == "__main__":
    run_rastrigin_example()                                   # 运行多峰函数测试案例 1
