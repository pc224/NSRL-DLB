import random
from taog_model import TAOG


class Instance:
    def __init__(self):
        self.tasks = {}  # task_id -> {th, tr, thr, type}
        self.taog = TAOG()
        self.CT = 30  # Cycle Time
        self.Wc = 30  # Workstation opening cost
        self.Ec = 5  # Robot energy cost per time
        self.Lc = 7  # Human labor cost per time


def create_mock_instance() -> Instance:
    """生成一个包含12个任务的TAOG测试实例"""
    inst = Instance()

    # 定义TAOG节点
    nodes = ['A0', 'B1', 'B2', 'A1', 'A2', 'A3', 'B3', 'B4', 'B5', 'B6', 'A4', 'A5', 'A6', 'A7', 'B7', 'B8', 'B9',
             'B10']
    types = {'A0': 'A', 'B1': 'B', 'B2': 'B', 'A1': 'A', 'A2': 'A', 'A3': 'A', 'B3': 'B', 'B4': 'B', 'B5': 'B',
             'B6': 'B',
             'A4': 'A', 'A5': 'A', 'A6': 'A', 'A7': 'A', 'B7': 'B', 'B8': 'B', 'B9': 'B', 'B10': 'B'}

    for n in nodes:
        inst.taog.add_node(n, types[n])

    # 定义边 (构建 AND/OR 逻辑)
    edges = [
        ('A0', 'B1'), ('A0', 'B2'),  # A0 OR (B1, B2)
        ('B1', 'A1'), ('B1', 'A2'),  # B1 AND (A1, A2)
        ('B2', 'A3'),  # B2 AND (A3)
        ('A1', 'B3'), ('A2', 'B4'), ('A2', 'B5'),  # A1->B3, A2 OR(B4, B5)
        ('A3', 'B6'),  # A3->B6
        ('B3', 'A4'), ('B4', 'A5'), ('B5', 'A6'), ('B6', 'A7'),
        ('A4', 'B7'), ('A5', 'B8'), ('A6', 'B9'), ('A7', 'B10')
    ]
    for u, v in edges:
        inst.taog.add_edge(u, v)

    inst.taog.set_root('A0')

    # 生成任务属性
    task_ids = [n for n in nodes if types[n] == 'B']
    task_types = ['Normal', 'Complex', 'Hazardous', 'Interactive']

    for tid in task_ids:
        t_type = random.choice(task_types)
        base_time = random.randint(5, 12)
        inst.tasks[tid] = {
            'type': t_type,
            'th': base_time if t_type in ['Normal', 'Complex', 'Interactive'] else 0,
            'tr': base_time if t_type in ['Normal', 'Hazardous', 'Interactive'] else 0,
            'thr': int(base_time * 0.8) if t_type == 'Interactive' else 0
        }

    return inst