# main.py
import numpy as np
from problem import DLB_UCS_Problem
from algorithm import CI_DMOEA
from config import NUM_ENVIRONMENTS

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 1. 初始化问题模型 (20个任务，30%不确定组件)
    print("Initializing DLB-UCS Problem...")
    problem = DLB_UCS_Problem(num_tasks=20, uncertain_ratio=0.3)

    # 2. 初始化 CI-DMOEA 算法
    optimizer = CI_DMOEA(problem)

    # 3. 运行动态优化过程
    optimizer.run(num_envs=NUM_ENVIRONMENTS)

    print("\nOptimization Complete.")