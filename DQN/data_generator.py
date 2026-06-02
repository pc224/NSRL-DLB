import numpy as np


def generate_dataset(set_size, n, CT, seed=None):

    if seed is not None:
        np.random.seed(seed)

    dataset = []
    for _ in range(set_size):
        A = list(range(n))
        TP = np.zeros((n, n), dtype=int)

        # 随机生成无环的优先关系图 (DAG)
        for num in range(n):
            a = np.random.choice(A)
            A.remove(a)
            # 前驱任务数量在 [0, len(A)] 之间
            pre_count = np.random.randint(0, len(A) + 1)
            if pre_count > 0:
                pre_tasks = np.random.choice(A, pre_count, replace=False)
                for p in pre_tasks:
                    TP[p, a] = 1

        # 生成作业时间，范围在 [1, CT]
        task_times = np.random.randint(1, CT + 1, size=n)
        dataset.append((TP, task_times))

    return dataset


def get_fixed_test_case(n, CT, seed):
    """生成一个固定的测试案例用于验证"""
    return generate_dataset(1, n, CT, seed)[0]