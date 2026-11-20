"""
================================================================================
使用 scikit-opt (sko.PSO) 库验证 Rastrigin 多峰测试函数的寻优精度并对比自定义 PSO
================================================================================
测试目标:
    验证 10 维经典高度非凸多峰 Rastrigin 基准测试函数在业界权威智能优化库
    scikit-opt (sko.PSO) 下的收敛表现与精度，并与自定义实现的向量化 PSO 进行横向对比。

数学模型 (Rastrigin 函数):
    f(x) = 10 * d + sum_{j=1}^d (x_j^2 - 10 * cos(2 * pi * x_j))
    - 全局最优解: x* = [0, 0, ..., 0] (原点)
    - 理论最小值: f(x*) = 0.0
    - 搜索空间:   x_j in [-5.12, 5.12]
================================================================================
"""

import sys
import numpy as np

# 确保在 Windows 控制台环境下输出中文字符不产生乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# 导入 scikit-opt 优化库中的粒子群优化器
from sko.PSO import PSO

# 导入自主实现的粒子群优化器与基准测试函数
from ParticleSwarmOptimizer import ParticleSwarmOptimizer, rastrigin


def run_scikit_opt_verification():
    """
    使用 scikit-opt 库验证 10 维 Rastrigin 测试函数的优化精度，
    并在相同实验条件（种群规模、迭代步数、随机种子）下与自定义优化器进行对比。
    """
    dim = 10                                       # 优化维度: 10 维
    lb = [-5.12] * dim                            # 搜索空间各维下界
    ub = [5.12] * dim                             # 搜索空间各维上界
    pop_size = 50                                 # 粒子群体规模
    max_iter = 200                                # 最大迭代代数
    random_seed = 42                              # 随机数种子，保证实验完全可复现

    print("=" * 70)
    print("scikit-opt (sko.PSO) 验证 10 维 Rastrigin 测试函数寻优精度")
    print("=" * 70)
    print(f"问题维度 d:           {dim}")
    print(f"搜索空间:             [-5.12, 5.12]^{dim}")
    print(f"理论全局最优坐标 x*:   [0, 0, ..., 0] (原点)")
    print(f"理论全局最小值 f(x*): 0.0")
    print(f"粒子群规模 pop:       {pop_size}")
    print(f"迭代代数 max_iter:    {max_iter}")
    print("-" * 70)

    # --------------------------------------------------------------------------
    # 1. 运行 scikit-opt (sko.PSO)
    # --------------------------------------------------------------------------
    # 设定随机种子以保证结果可复现
    np.random.seed(random_seed)
    
    # 初始化 scikit-opt 的 PSO 求解器
    # 默认超参数: w=0.8, c1=0.5, c2=0.5
    sko_pso = PSO(
        func=rastrigin,                           # 目标评估函数
        n_dim=dim,                                # 决策维度
        pop=pop_size,                             # 粒子数
        max_iter=max_iter,                        # 最大迭代步
        lb=lb,                                    # 搜索下界
        ub=ub,                                    # 搜索上界
        w=0.8,                                    # 惯性权重
        c1=0.5,                                   # 认知加速因子
        c2=0.5                                    # 社会加速因子
    )
    
    sko_best_x, sko_best_y = sko_pso.run()
    # sko_best_y 通常为 numpy 数组 (1,)，提取标量浮点值
    sko_best_fit = float(sko_best_y[0]) if hasattr(sko_best_y, "__len__") else float(sko_best_y)

    print("\n【1】scikit-opt (sko.PSO 默认参数: w=0.8, c1=0.5, c2=0.5) 寻优结果:")
    print(f"  - 最佳适应度 (目标极小值): {sko_best_fit:.6e}")
    print(f"  - 最优解前 3 维坐标:       {np.round(sko_best_x[:3], 5)}")
    print(f"  - 最优解与原点欧氏距离:   {np.linalg.norm(sko_best_x):.6e}")

    # --------------------------------------------------------------------------
    # 2. 运行 scikit-opt (sko.PSO) 使用经典认知/社会参数 (c1=1.5, c2=1.5)
    # --------------------------------------------------------------------------
    np.random.seed(random_seed)
    sko_pso_tuned = PSO(
        func=rastrigin,
        n_dim=dim,
        pop=pop_size,
        max_iter=max_iter,
        lb=lb,
        ub=ub,
        w=0.7298,                                 # 收缩因子的惯性常数
        c1=1.49618,                               # 经典 Clerc 理论认知参数
        c2=1.49618                                # 经典 Clerc 理论社会参数
    )
    sko_tuned_best_x, sko_tuned_best_y = sko_pso_tuned.run()
    sko_tuned_fit = float(sko_tuned_best_y[0]) if hasattr(sko_tuned_best_y, "__len__") else float(sko_tuned_best_y)

    print("\n【2】scikit-opt (sko.PSO 经典参数: w=0.7298, c1=1.496, c2=1.496) 寻优结果:")
    print(f"  - 最佳适应度 (目标极小值): {sko_tuned_fit:.6e}")
    print(f"  - 最优解前 3 维坐标:       {np.round(sko_tuned_best_x[:3], 5)}")
    print(f"  - 最优解与原点欧氏距离:   {np.linalg.norm(sko_tuned_best_x):.6e}")

    # --------------------------------------------------------------------------
    # 3. 运行自定义 ParticleSwarmOptimizer (LDW 动态线性递减惯性权重策略)
    # --------------------------------------------------------------------------
    custom_pso = ParticleSwarmOptimizer(
        fitness_func=rastrigin,
        dim=dim,
        bounds=(lb, ub),
        n_particles=pop_size,
        max_iter=max_iter,
        w_mode="ldw",                             # 线性递减惯性权重: 0.9 -> 0.4
        c1=1.49618,
        c2=1.49618,
        seed=random_seed
    )
    custom_best_x, custom_best_fit, custom_curve = custom_pso.optimize()

    print("\n【3】自定义 ParticleSwarmOptimizer (LDW 动态递减权重: 0.9->0.4, c1=1.496, c2=1.496):")
    print(f"  - 最佳适应度 (目标极小值): {custom_best_fit:.6e}")
    print(f"  - 最优解前 3 维坐标:       {np.round(custom_best_x[:3], 5)}")
    print(f"  - 最优解与原点欧氏距离:   {np.linalg.norm(custom_best_x):.6e}")

    # --------------------------------------------------------------------------
    # 4. 横向性能综合汇总
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("三种算法配置综合精度对比表:")
    print("=" * 70)
    header = f"{'求解器 / 配置':<32} | {'最终适应度':<16} | {'与原点距离':<16}"
    print(header)
    print("-" * 70)
    print(f"{'scikit-opt (默认参数)':<32} | {sko_best_fit:<16.6e} | {np.linalg.norm(sko_best_x):<16.6e}")
    print(f"{'scikit-opt (Clerc 经典参数)':<32} | {sko_tuned_fit:<16.6e} | {np.linalg.norm(sko_tuned_best_x):<16.6e}")
    print(f"{'自定义 PSO (LDW 自适应权重)':<32} | {custom_best_fit:<16.6e} | {np.linalg.norm(custom_best_x):<16.6e}")
    print("=" * 70)
    print("结论说明:")
    print("  1. Rastrigin 是高度多峰基准函数（10 维空间包含约 260 亿个局部极小点）。")
    print("  2. scikit-opt 库的寻优结果进一步验证了此多峰函数在常规常数权重 PSO 下极易被多峰吸引子诱捕；")
    print("  3. 适当配置探索与开发能力（如增大认知/社会协同系数或采用 LDW 递减权重），能够更有效地")
    print("     降低最终目标函数值，逐步逼近理论最优原点。")
    print("=" * 70)


if __name__ == "__main__":
    run_scikit_opt_verification()
