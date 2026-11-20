# -*- coding: utf-8 -*-
"""
================================================================================
多进程并行粒子群优化器 (Parallel Particle Swarm Optimizer, Parallel-PSO)
================================================================================
本模块在基础向量化 PSO 的基础上，引入多进程并行计算架构 (Multiprocessing Pool)，
为计算密集型/耗时较长的目标函数（如卡尔曼滤波超参数调优、动力学系统仿真、
控制系统参数辨识）提供接近线性的多核加速能力。
- 运动学演化 (速度、位置、边界限幅、布尔掩码更新): 基于 NumPy C 底层 SIMD 向量化并行；
- 适应度评估 (耗时瓶颈 fitness_func): 基于多进程进程池的多核 CPU 硬件级并行。
================================================================================
"""

import sys
import os
import time
import multiprocessing
import numpy as np
from typing import Callable, Tuple, List, Optional, Union


# ------------------------------------------------------------------------------
# Windows 控制台标准流 UTF-8 重设（防止乱码）
# ------------------------------------------------------------------------------
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


# ------------------------------------------------------------------------------
# 多进程 Worker 全局缓存（避免每代重复 pickle 序列化目标函数）
# ------------------------------------------------------------------------------
_WORKER_FITNESS_FUNC: Optional[Callable[[np.ndarray], float]] = None


def _init_worker(fitness_func: Callable[[np.ndarray], float]) -> None:
    global _WORKER_FITNESS_FUNC
    _WORKER_FITNESS_FUNC = fitness_func


def _eval_worker(x: np.ndarray) -> float:
    return _WORKER_FITNESS_FUNC(x)


class ParallelParticleSwarmOptimizer:
    """
    多进程并行粒子群优化器 (Parallel Particle Swarm Optimizer)
    
    参数说明:
        fitness_func : 目标适应度函数，签名 Callable[[np.ndarray], float]，输入为 (d,)，输出标量适应度值
        dim          : 优化决策变量的维度 d (int)
        bounds       : 搜索空间超矩形上下界元组 (lb, ub)，形式如 ([-5.0, ...], [5.0, ...])
        n_particles  : 粒子群落规模数量 N (默认 40)
        max_iter     : 最大迭代演化代数 K_max (默认 150)
        w_mode       : 惯性权重模式，可选 'ldw' (线性递减), 'const' (常数), 'constriction' (收缩因子)
        w            : 基础惯性权重常数或收缩因子 (默认 0.7298)
        w_min, w_max : 线性递减权重上下限 (默认 0.4 与 0.9)
        c1, c2       : 认知学习因子与社会学习因子 (默认 1.4962)
        v_ratio      : 速度限幅占可行域跨度的比例 (默认 0.15)
        n_workers    : 并行工作进程数。-1 或 None 表示自动使用系统全部可用核心；1 表示单核串行
        seed         : 随机数发生器种子，保证可复现性 (可选)
        verbose      : 是否在优化过程中打印当前迭代进度 (默认 True)
        print_step   : 迭代进度打印间隔代数 (默认 1，即每代均打印)
        init_x       : 可选先验初始解集合，形状为 (m, d) 或 (d,)，注入前 m 个粒子
        tol          : 早停收敛容差 (默认 0.0)
        patience     : 连续多少代改进不超过 tol 时触发早停 (None 表示关闭早停)
    """

    def __init__(
        self,
        fitness_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: Tuple[List[float], List[float]],
        n_particles: int = 40,
        max_iter: int = 150,
        w_mode: str = "ldw",
        w: float = 0.7298,
        w_min: float = 0.4,
        w_max: float = 0.9,
        c1: float = 1.4962,
        c2: float = 1.4962,
        v_ratio: float = 0.15,
        n_workers: Optional[int] = -1,
        seed: Optional[int] = None,
        verbose: bool = True,
        print_step: int = 1,
        init_x: Optional[Union[np.ndarray, List[List[float]], List[float]]] = None,
        tol: float = 0.0,
        patience: Optional[int] = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if n_particles < 1:
            raise ValueError(f"n_particles must be >= 1, got {n_particles}")
        if max_iter < 0:
            raise ValueError(f"max_iter must be >= 0, got {max_iter}")

        self.rng = np.random.default_rng(seed)
        self.fitness_func = fitness_func
        self.dim = dim
        self.lb = np.array(bounds[0], dtype=np.float64)
        self.ub = np.array(bounds[1], dtype=np.float64)
        if self.lb.shape != (dim,) or self.ub.shape != (dim,):
            raise ValueError(f"bounds shape must match dim={dim}, got lb={self.lb.shape}, ub={self.ub.shape}")
        if not np.all(self.lb < self.ub):
            raise ValueError("All lower bounds must be strictly less than upper bounds.")

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_mode = w_mode
        self.w = w
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.print_step = max(1, int(print_step))
        self.tol = float(tol)
        self.patience = int(patience) if patience is not None else None

        # 确立并行工作进程数量
        cpu_total = os.cpu_count() or 1
        if n_workers is None or n_workers == -1:
            # 自动探测核心数，但不超过粒子数
            self.n_workers = min(cpu_total, self.n_particles)
        elif n_workers <= 0:
            self.n_workers = 1
        else:
            self.n_workers = min(n_workers, self.n_particles)

        # 依据式 (p-12) 计算各维速度限幅阈值
        self.v_max = v_ratio * (self.ub - self.lb)
        self.v_min = -self.v_max

        # 依据式 (p-3) 与式 (p-4) 初始化粒子群位置矩阵与速度矩阵: (N, d)
        self.X = self.rng.uniform(self.lb, self.ub, size=(n_particles, dim))
        self.V = self.rng.uniform(self.v_min, self.v_max, size=(n_particles, dim))

        # 若提供先验初值 init_x，注入前若干个粒子
        if init_x is not None:
            init_arr = np.atleast_2d(np.asarray(init_x, dtype=np.float64))
            if init_arr.shape[1] != dim:
                raise ValueError(f"init_x must have {dim} columns, got {init_arr.shape}")
            n_inject = min(len(init_arr), self.n_particles)
            self.X[:n_inject] = np.clip(init_arr[:n_inject], self.lb, self.ub)

        # 个体历史最优 (pbest) 与全群全局最优 (gbest) 初始化占位符
        self.pbest_X = self.X.copy()
        self.pbest_fit = np.full(self.n_particles, np.inf)
        self.gbest_X = np.zeros(self.dim)
        self.gbest_fit = np.inf
        self.convergence_curve: List[float] = []
     
    def _eval_batch(self, X_matrix: np.ndarray, pool: Optional[multiprocessing.Pool] = None) -> np.ndarray:
        """
        批量评估粒子群当前位置矩阵的适应度值
        
        参数:
            X_matrix : 形状为 (N, d) 的粒子坐标二维矩阵
            pool     : 多进程工作池实例，为 None 时退化为单核串行计算
            
        返回:
            一维适应度数值数组 (N,)
        """
        # 将矩阵拆分为单粒子一维向量列表 [x_0, x_1, ..., x_{N-1}]，确保连续内存与 pickle 兼容
        particles_list = [np.ascontiguousarray(X_matrix[i]) for i in range(len(X_matrix))]

        if pool is not None and self.n_workers > 1:
            # 多进程并行映射求解（worker 已通过 initializer 缓存 fitness_func，仅传输坐标向量）
            chunksize = max(1, len(particles_list) // (self.n_workers * 2))
            fitness_values = pool.map(_eval_worker, particles_list, chunksize=chunksize)
            return np.array(fitness_values, dtype=np.float64)
        else:
            # 纯串行列表推导式求解
            return np.array([self.fitness_func(x) for x in particles_list], dtype=np.float64)

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        启动粒子群优化主循环寻优引擎
        
        返回:
            gbest_X           : 全局最优解坐标向量，形状为 (d,)
            gbest_fit         : 全局最优适应度数值 (float)
            convergence_curve : 每一代全局最优适应度演化历史列表 (长度为 max_iter + 1)
        """
        # 1. 管理多进程生命周期: 在整个优化寻优过程中仅创建一次进程池，并通过 initializer 缓存目标函数
        pool = None
        if self.n_workers > 1:
            pool = multiprocessing.Pool(
                processes=self.n_workers,
                initializer=_init_worker,
                initargs=(self.fitness_func,),
            )

        try:
            # ------------------------------------------------------------------
            # 初始适应度评估与极值初始化 (第 0 代)
            # ------------------------------------------------------------------
            t_opt_start = time.perf_counter()
            initial_fit = self._eval_batch(self.X, pool=pool)
            self.pbest_fit = initial_fit.copy()
            self.pbest_X = self.X.copy()

            # 依据式 (p-8) 初始化全局最优
            gbest_idx = int(np.argmin(self.pbest_fit))
            self.gbest_X = self.pbest_X[gbest_idx].copy()
            self.gbest_fit = float(self.pbest_fit[gbest_idx])
            self.convergence_curve = [self.gbest_fit]

            if self.verbose:
                elapsed_0 = time.perf_counter() - t_opt_start
                print(
                    f"  [PSO 迭代   0/{self.max_iter:3d}] "
                    f"当前全局最优适应度 gbest_fit = {self.gbest_fit:.6e} | "
                    f"累计耗时 = {elapsed_0:.2f} s",
                    flush=True
                )

            # ------------------------------------------------------------------
            # 主代数迭代演化循环 (k = 0 到 max_iter - 1)
            # ------------------------------------------------------------------
            no_improve_count = 0
            for k in range(self.max_iter):
                # 步骤 1: 动态调节惯性权重 w / 收缩因子 chi
                if self.w_mode == "ldw":
                    current_w = self.w_max - (self.w_max - self.w_min) * (k / max(1, self.max_iter))
                elif self.w_mode == "constriction":
                    phi = self.c1 + self.c2
                    if phi > 4.0:
                        current_w = 2.0 / abs(2.0 - phi - np.sqrt(phi ** 2 - 4.0 * phi))
                    else:
                        current_w = self.w
                else:
                    current_w = self.w

                # 步骤 2: 独立采样均匀随机扰动矩阵 r1, r2 ~ U(0, 1)
                r1 = self.rng.uniform(0.0, 1.0, size=(self.n_particles, self.dim))
                r2 = self.rng.uniform(0.0, 1.0, size=(self.n_particles, self.dim))

                # 步骤 3: 向量化速度更新与限幅截断
                cognitive = self.c1 * r1 * (self.pbest_X - self.X)
                social = self.c2 * r2 * (self.gbest_X - self.X)
                if self.w_mode == "constriction":
                    self.V = current_w * (self.V + cognitive + social)
                else:
                    self.V = current_w * self.V + cognitive + social
                self.V = np.clip(self.V, self.v_min, self.v_max)

                # 步骤 4: 向量化位置外推、吸收边界截断及越界速度清零
                x_next = self.X + self.V
                out_mask = (x_next < self.lb) | (x_next > self.ub)
                self.X = np.clip(x_next, self.lb, self.ub)
                self.V[out_mask] = 0.0

                # 步骤 5: 多进程多核并行评估群体新位置适应度
                current_fit = self._eval_batch(self.X, pool=pool)

                # 步骤 6: 依据式 (p-6) 向量化更新个体历史最优位置 (pbest)
                improved_mask = current_fit < self.pbest_fit
                self.pbest_X[improved_mask] = self.X[improved_mask]
                self.pbest_fit[improved_mask] = current_fit[improved_mask]

                # 步骤 7: 依据式 (p-8) 刷新全群全局历史最优位置 (gbest)
                prev_gbest = self.gbest_fit
                min_idx = int(np.argmin(self.pbest_fit))
                if self.pbest_fit[min_idx] < self.gbest_fit:
                    self.gbest_fit = float(self.pbest_fit[min_idx])
                    self.gbest_X = self.pbest_X[min_idx].copy()

                if prev_gbest - self.gbest_fit <= self.tol:
                    no_improve_count += 1
                else:
                    no_improve_count = 0

                self.convergence_curve.append(self.gbest_fit)

                iter_num = k + 1
                if self.verbose and (iter_num % self.print_step == 0 or iter_num == self.max_iter):
                    elapsed_k = time.perf_counter() - t_opt_start
                    print(
                        f"  [PSO 迭代 {iter_num:3d}/{self.max_iter:3d}] "
                        f"w = {current_w:.4f} | "
                        f"当前全局最优适应度 gbest_fit = {self.gbest_fit:.6e} | "
                        f"累计耗时 = {elapsed_k:.2f} s",
                        flush=True
                    )

                if self.patience is not None and self.patience > 0 and no_improve_count >= self.patience:
                    if self.verbose:
                        print(f"  [PSO 早停] 连续 {self.patience} 代改进 <= {self.tol:.1e}，提前终止于第 {iter_num} 代。", flush=True)
                    break

        finally:
            # 确保在出现异常或优化正常结束时，子进程被安全关闭并回收资源
            if pool is not None:
                pool.close()
                pool.join()

        return self.gbest_X, self.gbest_fit, self.convergence_curve


# ==============================================================================
# 基准测试与应用示例
# ==============================================================================

def rastrigin(x: np.ndarray) -> float:
    """标准 10 维 Rastrigin 高度非凸多峰测试函数"""
    d = len(x)
    return 10.0 * d + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


def expensive_simulation_task(x: np.ndarray) -> float:
    """
    模拟工程仿真/卡尔曼滤波超参数辨识的计算密集型目标函数
    包含模拟的计算耗时 (约 2 毫秒/次)，以体现多核并行的真实加速倍率
    """
    # 模拟真实工程中卡尔曼滤波迭代或矩阵求逆开销
    time.sleep(0.002)
    # 底层依然采用多峰优化曲面计算适应度
    return rastrigin(x)


def run_parallel_benchmark():
    """执行串行 vs 多进程并行 PSO 性能基准测试与加速比对比"""
    dim = 10
    bounds = ([-5.12] * dim, [5.12] * dim)
    n_particles = 48  # 粒子数设为 48，便于在 4/8/16 核上均匀分配
    max_iter = 50     # 迭代 50 代，总评估次数为 48 * 51 = 2448 次

    print("\n" + "=" * 72)
    print("      多进程并行粒子群优化器 (Parallel-PSO) 性能基准测试")
    print(f"      硬件可用 CPU 核心总数: {os.cpu_count()} 核心")
    print(f"      测试环境: 维度={dim}, 粒子数={n_particles}, 迭代代数={max_iter}")
    print(f"      单次评估模拟耗时: ~2 ms (模拟卡尔曼滤波/系统辨识计算开销)")
    print("=" * 72)

    # --------------------------------------------------------------------------
    # 1. 单核串行测试 (n_workers = 1)
    # --------------------------------------------------------------------------
    print("\n>>> [1/2] 正在运行单核串行优化 (Serial PSO)...")
    pso_serial = ParallelParticleSwarmOptimizer(
        fitness_func=expensive_simulation_task,
        dim=dim,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        n_workers=1,
        seed=42
    )
    t0 = time.perf_counter()
    best_x_s, best_f_s, curve_s = pso_serial.optimize()
    serial_time = time.perf_counter() - t0
    print(f"    - 单核串行总耗时: {serial_time:.3f} 秒")
    print(f"    - 寻优最佳适应度: {best_f_s:.6e}")

    # --------------------------------------------------------------------------
    # 2. 多核并行测试 (n_workers = 自动探测全部核心)
    # --------------------------------------------------------------------------
    workers_count = os.cpu_count() or 4
    print(f"\n>>> [2/2] 正在运行多核并行优化 (Parallel PSO, 启用 {workers_count} 核心)...")
    pso_parallel = ParallelParticleSwarmOptimizer(
        fitness_func=expensive_simulation_task,
        dim=dim,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        n_workers=workers_count,
        seed=42
    )
    t0 = time.perf_counter()
    best_x_p, best_f_p, curve_p = pso_parallel.optimize()
    parallel_time = time.perf_counter() - t0
    speedup = serial_time / max(parallel_time, 1e-6)

    print(f"    - 多核并行总耗时: {parallel_time:.3f} 秒")
    print(f"    - 寻优最佳适应度: {best_f_p:.6e}")
    print(f"    - 最优解前 3 维坐标: {np.round(best_x_p[:3], 5)}")

    # --------------------------------------------------------------------------
    # 3. 加速效果对比总结
    # --------------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("                     性能与加速比总结")
    print("-" * 72)
    print(f" 串行模式 (1 核心)    : {serial_time:7.3f} 秒 | 最佳值: {best_f_s:.4e}")
    print(f" 并行模式 ({workers_count:2d} 核心)   : {parallel_time:7.3f} 秒 | 最佳值: {best_f_p:.4e}")
    print(f" 实际计算加速比 (Speedup) : {speedup:.2f} 倍加速 (耗时缩短 {(1.0 - parallel_time / serial_time) * 100:.1f}%)")
    print("=" * 72)


# Windows 平台多进程启动入口守护 (必须放置于 if __name__ == '__main__' 下)
if __name__ == "__main__":
    # 防止程序在使用 PyInstaller、cx_Freeze 等工具打包成 Windows 下的 .exe 可执行文件后，
    # 多进程启动时无限递归创建新进程（即“进程炸弹”导致系统卡死崩溃）。
    multiprocessing.freeze_support() 
    run_parallel_benchmark()
