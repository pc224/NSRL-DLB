# problem.py
import numpy as np
import networkx as nx


class DLB_UCS_Problem:
    """
    DLB-UCS 问题模型定义
    包含任务优先关系图(DAG)、基础拆卸时间、不确定状态模拟
    """

    def __init__(self, num_tasks=20, uncertain_ratio=0.3):
        self.num_tasks = num_tasks
        self.uncertain_tasks = []
        self.base_times = np.random.uniform(2.0, 8.0, num_tasks)
        self.current_times = self.base_times.copy()

        # 生成随机DAG作为优先关系约束 (TAOG的简化通用表达)
        self.G = nx.gnp_random_graph(num_tasks, 0.15, directed=True)
        self.G = nx.DiGraph([(u, v) for u, v in self.G.edges() if u < v])  # 保证无环

        # 标记不确定组件 (Uncertain Component States)
        num_uncertain = int(num_tasks * uncertain_ratio)
        self.uncertain_tasks = np.random.choice(num_tasks, num_uncertain, replace=False)

        self.executed_tasks = []
        self.ongoing_task = None

    def reset_environment(self):
        """重置拆卸状态"""
        self.current_times = self.base_times.copy()
        self.executed_tasks = []
        self.ongoing_task = None

    def trigger_dynamic_change(self, progress_ratio=0.3):
        """
        模拟拆卸过程中的动态环境变化 (组件状态暴露)
        当拆卸进度达到 progress_ratio 时，随机一个未执行的不确定组件被发现为"损坏"
        """
        # 模拟已执行任务
        topo_sort = list(nx.topological_sort(self.G))
        split_idx = int(len(topo_sort) * progress_ratio)
        self.executed_tasks = topo_sort[:split_idx]
        self.ongoing_task = topo_sort[split_idx] if split_idx < len(topo_sort) else None

        # 寻找未执行的不确定任务并使其损坏 (时间增加)
        unexecuted_uncertain = [t for t in self.uncertain_tasks if
                                t not in self.executed_tasks and t != self.ongoing_task]
        changed_task = -1
        if unexecuted_uncertain:
            changed_task = np.random.choice(unexecuted_uncertain)
            # 损坏导致时间增加 (论文中为 1.4 ~ 2.5 倍)
            damage_factor = np.random.uniform(1.5, 2.5)
            self.current_times[changed_task] = self.base_times[changed_task] * damage_factor

        return changed_task, self.executed_tasks, self.ongoing_task

    def get_task_times(self):
        return self.current_times

    def get_predecessors(self, task_idx):
        return list(self.G.predecessors(task_idx))