import random
from typing import List, Dict
from problem_instance import Instance
from taog_model import Alternative


class Chromosome:
    def __init__(self, task_seq: List[str], resource_alloc: List[int]):
        self.task_seq = task_seq  # 第一层：任务序列
        self.resource_alloc = resource_alloc  # 第二层：资源分配 (0:Human, 1:Robot, 2:Interactive)
        self.f1 = float('inf')  # 目标1：总成本
        self.f2 = float('inf')  # 目标2：负载平滑指数 (Idle time squared)
        self.station_assign = {}  # 记录任务分配的工作站
        self.rank = 0
        self.crowding_dist = 0.0


def decode(ind: Chromosome, alt: Alternative, inst: Instance):

    # 构建前驱字典以便快速查找
    pred = {t: [] for t in alt.tasks}
    for u, v in alt.edges:
        pred[v].append(u)

    station = {}
    comp_time = {}

    w = 1
    h_time = {1: 0}  # 记录每个工作站Human的累计占用时间
    r_time = {1: 0}  # 记录每个工作站Robot的累计占用时间

    for i, task in enumerate(ind.task_seq):
        res = ind.resource_alloc[i]

        # 获取任务时间
        t_h = inst.tasks[task]['th'] if res in [0, 2] else 0
        t_r = inst.tasks[task]['tr'] if res in [1, 2] else 0
        if res == 2:  # 交互任务
            t_h = inst.tasks[task]['thr']
            t_r = inst.tasks[task]['thr']

        # 优先级约束：必须等所有前驱完成
        pred_comp = max([comp_time[p] for p in pred[task]] + [0])

        assigned = False
        curr_w = max(w, 1)

        while not assigned:
            if curr_w not in h_time:
                h_time[curr_w] = 0
                r_time[curr_w] = 0

            # 当前工作站资源的最早可用时间
            earliest_res = max(h_time[curr_w] if t_h > 0 else 0, r_time[curr_w] if t_r > 0 else 0)
            st = max(pred_comp, earliest_res)


            if (h_time[curr_w] + t_h <= inst.CT) and (r_time[curr_w] + t_r <= inst.CT):
                # 还需要确保开始时间+持续时间不超过当前工作站的绝对时间边界
                if st + max(t_h, t_r) <= curr_w * inst.CT:
                    station[task] = curr_w
                    comp_time[task] = st + max(t_h, t_r)

                    if t_h > 0: h_time[curr_w] = st + t_h
                    if t_r > 0: r_time[curr_w] = st + t_r
                    assigned = True
                else:
                    curr_w += 1
            else:
                curr_w += 1

        if curr_w > w:
            w = curr_w

    ind.station_assign = station

    # 计算目标 1: 总成本 (工作站开启成本 + 能耗 + 人工)
    num_stations = w
    cost_ws = num_stations * inst.Wc
    cost_energy = sum(inst.tasks[t]['tr'] if ind.resource_alloc[i] == 1 else (
        inst.tasks[t]['thr'] if ind.resource_alloc[i] == 2 else 0) for i, t in enumerate(ind.task_seq)) * inst.Ec
    cost_labor = sum(inst.tasks[t]['th'] if ind.resource_alloc[i] == 0 else (
        inst.tasks[t]['thr'] if ind.resource_alloc[i] == 2 else 0) for i, t in enumerate(ind.task_seq)) * inst.Lc
    ind.f1 = cost_ws + cost_energy + cost_labor

    # 计算目标 2: 负载平滑指数 (Idle Time Balance)
    f2 = 0
    for ws in range(1, w + 1):
        th_ws = sum(inst.tasks[t]['th'] if ind.resource_alloc[i] == 0 else (
            inst.tasks[t]['thr'] if ind.resource_alloc[i] == 2 else 0) for i, t in enumerate(ind.task_seq) if
                    station[t] == ws)
        tr_ws = sum(inst.tasks[t]['tr'] if ind.resource_alloc[i] == 1 else (
            inst.tasks[t]['thr'] if ind.resource_alloc[i] == 2 else 0) for i, t in enumerate(ind.task_seq) if
                    station[t] == ws)
        delta_h = inst.CT - th_ws
        delta_r = inst.CT - tr_ws
        f2 += (delta_h ** 2 + delta_r ** 2)
    ind.f2 = f2