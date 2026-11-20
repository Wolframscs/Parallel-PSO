# Parallel-PSO

多进程并行粒子群优化器（`ParallelParticleSwarmOptimizer`）与串行基准版本（`ParticleSwarmOptimizer`）。

## 特性
- **向量化运动学更新**：基于 NumPy 向量化计算速度、位置、边界截断及越界速度清零。
- **多进程并行评估**：通过 `multiprocessing.Pool(..., initializer=...)` 在进程池启动时仅缓存一次目标函数，避免每代重复序列化开销。
- **多种惯性权重策略**：支持线性递减（`ldw`）、常数（`const`）及 Clerc & Kennedy 收缩因子（`constriction`）。
- **工程实用增强**：支持注入先验初始解 `init_x`、独立随机数生成器 `default_rng(seed)`、早停机制（`tol` / `patience`）。

## 安装（本地可编辑模式）
```bash
pip install -e . --no-build-isolation --no-deps
```

## 快速使用
```python
from ParallelParticleSwarmOptimizer import ParallelParticleSwarmOptimizer

pso = ParallelParticleSwarmOptimizer(
    fitness_func=my_objective,
    dim=3,
    bounds=([0.6, -30.0, 0.1], [1.4, 10.0, 1.5]),
    n_particles=40,
    max_iter=50,
    n_workers=-1,
    seed=42,
)
best_x, best_fit, curve = pso.optimize()
```
